// Copyright (c) 2025 Michael Ivertowski
// Licensed under the Apache License, Version 2.0. See LICENSE file for details.

using System.Diagnostics.CodeAnalysis;
using DotCompute.Abstractions.RingKernels;
using Microsoft.Extensions.Logging;
using Orleans.Runtime;
using Orleans.GpuBridge.Abstractions.Temporal;

namespace Orleans.GpuBridge.Runtime.RingKernels;

/// <summary>
/// Base class for Orleans grains that execute as GPU-native actors using persistent ring kernels.
/// </summary>
/// <remarks>
/// <para>
/// This class bridges Orleans grain lifecycle with DotCompute ring kernel lifecycle:
/// - OnActivateAsync() → LaunchAsync() + ActivateAsync() (GPU kernel starts)
/// - Method calls → SendMessage() → GPU processes → ReceiveMessage() → Return
/// - OnDeactivateAsync() → DeactivateAsync() (GPU kernel pauses)
/// - DisposeAsync() → TerminateAsync() (GPU kernel exits)
/// </para>
/// <para>
/// Performance characteristics:
/// - First activation: ~10-50ms (kernel compilation + launch)
/// - Subsequent calls: 100-500ns (GPU queue operations only)
/// - Reactivation: ~1μs (no compilation, just ActivateAsync)
/// </para>
/// <para>
/// The grain maintains GPU state across Orleans activation cycles. Ring kernel stays
/// launched but paused during grain idle periods, enabling instant reactivation.
/// </para>
/// </remarks>
public abstract class GpuNativeGrain : Grain, IGrainWithIntegerKey, IAsyncDisposable
{
    private readonly IRingKernelRuntime _runtime;
    private readonly ILogger<GpuNativeGrain> _logger;
    private string? _kernelId;
    private bool _isKernelLaunched;
    private bool _isKernelActive;
    private bool _disposed;
    private long _sequenceNumber;

    /// <summary>
    /// Hybrid Logical Clock for temporal ordering of GPU messages.
    /// </summary>
    /// <remarks>
    /// Provides causal ordering with sub-microsecond precision.
    /// GPU kernels maintain their own HLC state in GPU memory for 20ns updates.
    /// </remarks>
    private readonly HybridLogicalClock _hlcClock;

    /// <summary>
    /// Initializes a new instance of the <see cref="GpuNativeGrain"/> class.
    /// </summary>
    /// <param name="runtime">Ring kernel runtime for GPU operations.</param>
    /// <param name="logger">Logger instance.</param>
    protected GpuNativeGrain(IRingKernelRuntime runtime, ILogger<GpuNativeGrain> logger)
    {
        _runtime = runtime ?? throw new ArgumentNullException(nameof(runtime));
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));

        // Initialize HLC with grain's primary key as node ID (truncated to ushort)
        // This ensures unique timestamps across grains
        var nodeId = (ushort)(this.GetPrimaryKeyLong() & 0xFFFF);
        _hlcClock = new HybridLogicalClock(nodeId);

        _logger.LogDebug(
            "Initialized GPU-native grain with HLC node ID {NodeId} (from grain key {GrainKey})",
            nodeId,
            this.GetPrimaryKeyLong());
    }

    /// <summary>
    /// Gets the ring kernel ID for this grain.
    /// </summary>
    /// <remarks>
    /// Format: {GrainTypeName}_{PrimaryKey}
    /// Example: VectorAddActor_42
    /// </remarks>
    protected string KernelId => _kernelId ?? throw new InvalidOperationException("Kernel not initialized");

    /// <summary>
    /// Gets a value indicating whether the GPU kernel is currently active.
    /// </summary>
    protected bool IsKernelActive => _isKernelActive;

    /// <summary>
    /// Gets grid and block size configuration for kernel launch.
    /// </summary>
    /// <remarks>
    /// Override to customize GPU thread configuration.
    /// Default: 1 block × 256 threads (suitable for single-actor workloads).
    /// </remarks>
    protected virtual (int gridSize, int blockSize) GetKernelConfiguration()
    {
        return (gridSize: 1, blockSize: 256);
    }

    /// <summary>
    /// Called when the grain is activated by Orleans.
    /// Launches the persistent ring kernel on GPU.
    /// </summary>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <remarks>
    /// This method uses reflection for ring kernel queue creation and message type detection.
    /// AOT compilation may require additional configuration for proper operation.
    /// </remarks>
    [UnconditionalSuppressMessage("AOT", "IL2026:RequiresUnreferencedCode",
        Justification = "DotCompute ring kernel runtime uses reflection for queue creation. AOT scenarios require pre-generated message types.")]
    [UnconditionalSuppressMessage("AOT", "IL3050:RequiresDynamicCode",
        Justification = "DotCompute ring kernel runtime uses reflection for queue creation. AOT scenarios require pre-generated message types.")]
    public override async Task OnActivateAsync(CancellationToken cancellationToken)
    {
        await base.OnActivateAsync(cancellationToken);

        // Generate kernel ID from grain type and primary key
        _kernelId = $"{GetType().Name}_{this.GetPrimaryKeyLong()}";

        // Get kernel configuration
        var (gridSize, blockSize) = GetKernelConfiguration();

        _logger.LogInformation(
            "Activating GPU-native grain {GrainType} (ID: {GrainId}) with kernel {KernelId} (grid={Grid}, block={Block})",
            GetType().Name,
            this.GetPrimaryKeyLong(),
            _kernelId,
            gridSize,
            blockSize);

        try
        {
            // Launch persistent ring kernel
            var launchStart = DateTime.UtcNow;
            await _runtime.LaunchAsync(_kernelId, gridSize, blockSize, options: null, cancellationToken);
            var launchDuration = DateTime.UtcNow - launchStart;

            _isKernelLaunched = true;

            _logger.LogInformation(
                "Launched ring kernel {KernelId} in {LaunchTimeMs}ms",
                _kernelId,
                launchDuration.TotalMilliseconds);

            // Activate kernel immediately (ready to process messages)
            await _runtime.ActivateAsync(_kernelId, cancellationToken);
            _isKernelActive = true;

            _logger.LogInformation("Activated ring kernel {KernelId}", _kernelId);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex,
                "Failed to launch/activate ring kernel {KernelId} for grain {GrainType}",
                _kernelId,
                GetType().Name);
            throw;
        }
    }

    /// <summary>
    /// Called when the grain is deactivated by Orleans.
    /// Deactivates the ring kernel (pauses it) but keeps it launched for fast reactivation.
    /// </summary>
    /// <param name="reason">Deactivation reason.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    [UnconditionalSuppressMessage("AOT", "IL2026:RequiresUnreferencedCode",
        Justification = "DotCompute ring kernel runtime uses reflection for queue management. AOT scenarios require pre-generated message types.")]
    [UnconditionalSuppressMessage("AOT", "IL3050:RequiresDynamicCode",
        Justification = "DotCompute ring kernel runtime uses reflection for queue management. AOT scenarios require pre-generated message types.")]
    public override async Task OnDeactivateAsync(DeactivationReason reason, CancellationToken cancellationToken)
    {
        _logger.LogInformation(
            "Deactivating grain {GrainType} (kernel: {KernelId}, reason: {Reason})",
            GetType().Name,
            _kernelId,
            reason);

        if (_isKernelActive && _kernelId != null)
        {
            try
            {
                // Deactivate kernel (pauses message processing)
                await _runtime.DeactivateAsync(_kernelId, cancellationToken);
                _isKernelActive = false;

                _logger.LogInformation(
                    "Deactivated ring kernel {KernelId} (kernel remains launched for reactivation)",
                    _kernelId);
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex,
                    "Error deactivating ring kernel {KernelId}",
                    _kernelId);
            }
        }

        await base.OnDeactivateAsync(reason, cancellationToken);
    }

    /// <summary>
    /// Invokes a GPU kernel method with typed request/response.
    /// </summary>
    /// <typeparam name="TRequest">Unmanaged request type.</typeparam>
    /// <typeparam name="TResponse">Unmanaged response type.</typeparam>
    /// <param name="request">Request payload.</param>
    /// <param name="timeout">Response timeout (default: 5 seconds).</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Response from GPU kernel.</returns>
    /// <exception cref="InvalidOperationException">Kernel not launched.</exception>
    /// <exception cref="TimeoutException">GPU kernel did not respond within timeout.</exception>
    /// <remarks>
    /// <para>
    /// Message flow with HLC temporal ordering:
    /// 1. Generate HLC timestamp (CPU-side, ~50ns)
    /// 2. Wrap request + timestamp into ActorMessage
    /// 3. Send to GPU via ring kernel queue
    /// 4. GPU temporal kernel processes with HLC update (~20ns on GPU)
    /// 5. Response unwrapped and returned
    /// </para>
    /// <para>
    /// For small payloads (&lt;= 8 bytes), data is embedded directly in ActorMessage.
    /// Larger payloads require GPU memory management (Phase 4).
    /// </para>
    /// </remarks>
    protected async Task<TResponse> InvokeKernelAsync<TRequest, TResponse>(
        TRequest request,
        TimeSpan timeout = default,
        CancellationToken cancellationToken = default)
        where TRequest : unmanaged
        where TResponse : unmanaged
    {
        if (_kernelId == null)
        {
            throw new InvalidOperationException("Kernel not initialized");
        }

        // Ensure kernel is active
        if (!_isKernelActive)
        {
            await _runtime.ActivateAsync(_kernelId, cancellationToken);
            _isKernelActive = true;

            _logger.LogDebug("Reactivated ring kernel {KernelId} for method call", _kernelId);
        }

        // Generate HLC timestamp for this message (local event)
        var sendTimestamp = GetCurrentTimestamp();

        _logger.LogTrace(
            "Sending GPU message with HLC timestamp: Physical={Physical}ns, Logical={Logical}, Node={NodeId}",
            sendTimestamp.PhysicalTime,
            sendTimestamp.LogicalCounter,
            sendTimestamp.NodeId);

        // Wrap request with HLC timestamp into ActorMessage
        var actorMessage = Temporal.TemporalMessageAdapter.WrapWithTimestamp(
            senderId: 0, // Orleans runtime
            receiverId: (ulong)this.GetPrimaryKeyLong(),
            request: request,
            timestamp: sendTimestamp,
            sequenceNumber: (ulong)Interlocked.Increment(ref _sequenceNumber));

        _logger.LogTrace(
            "Created ActorMessage {MessageId} with embedded HLC timestamp",
            actorMessage.MessageId);

        // Convert to KernelMessage for DotCompute compatibility
        var kernelMessage = Temporal.TemporalMessageAdapter.ToKernelMessage(actorMessage);

        // Send to GPU
        var sendStart = DateTime.UtcNow;
        await _runtime.SendMessageAsync<ActorMessage>(_kernelId, kernelMessage, cancellationToken);

        _logger.LogTrace(
            "Sent temporal message to GPU kernel {KernelId}, awaiting response...",
            _kernelId);

        // Wait for response
        var effectiveTimeout = timeout == default ? TimeSpan.FromSeconds(5) : timeout;
        var responseMessage = await _runtime.ReceiveMessageAsync<ActorMessage>(
            _kernelId,
            effectiveTimeout,
            cancellationToken);

        var roundTripNs = (DateTime.UtcNow - sendStart).TotalNanoseconds;

        if (responseMessage == null)
        {
            _logger.LogError(
                "GPU kernel {KernelId} response timeout after {TimeoutMs}ms",
                _kernelId,
                effectiveTimeout.TotalMilliseconds);

            throw new TimeoutException(
                $"GPU kernel {_kernelId} did not respond within {effectiveTimeout.TotalMilliseconds}ms");
        }

        // Extract ActorMessage from kernel message wrapper
        var responseActorMessage = responseMessage.Value.Payload;

        // Update local HLC with received timestamp (Lamport clock update)
        UpdateTimestamp(responseActorMessage.Timestamp);

        _logger.LogTrace(
            "GPU kernel {KernelId} round-trip: {LatencyNs}ns, received HLC: Physical={Physical}ns, Logical={Logical}",
            _kernelId,
            roundTripNs,
            responseActorMessage.Timestamp.PhysicalTime,
            responseActorMessage.Timestamp.LogicalCounter);

        // Unwrap response payload
        var response = Temporal.TemporalMessageAdapter.UnwrapResponse<TResponse>(responseActorMessage);

        return response;
    }

    /// <summary>
    /// Invokes a GPU kernel method using Orleans message serialization.
    /// </summary>
    /// <typeparam name="TResponse">Expected return type.</typeparam>
    /// <param name="methodName">Method name to invoke.</param>
    /// <param name="args">Method arguments.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Response from GPU kernel.</returns>
    /// <exception cref="InvalidOperationException">Kernel not launched.</exception>
    /// <exception cref="TimeoutException">GPU kernel did not respond.</exception>
    protected async Task<TResponse> InvokeKernelMethodAsync<TResponse>(
        string methodName,
        object[] args,
        CancellationToken cancellationToken = default)
        where TResponse : unmanaged
    {
        if (_kernelId == null)
        {
            throw new InvalidOperationException("Kernel not initialized");
        }

        // Serialize Orleans method call to GPU message
        var gpuMessage = GpuMessageSerializer.Serialize(
            methodName,
            senderId: 0,
            targetId: (int)this.GetPrimaryKeyLong(),
            args);

        // Ensure kernel is active
        if (!_isKernelActive)
        {
            await _runtime.ActivateAsync(_kernelId, cancellationToken);
            _isKernelActive = true;
        }

        // Convert to KernelMessage format
        var kernelMessage = KernelMessage<OrleansGpuMessage>.Create(
            senderId: 0,
            receiverId: (int)this.GetPrimaryKeyLong(),
            type: DotCompute.Abstractions.RingKernels.MessageType.Data,
            payload: gpuMessage);

        // Send to GPU
        await _runtime.SendMessageAsync<OrleansGpuMessage>(_kernelId, kernelMessage, cancellationToken);

        // Wait for response
        var response = await _runtime.ReceiveMessageAsync<OrleansGpuMessage>(
            _kernelId,
            TimeSpan.FromSeconds(5),
            cancellationToken);

        if (response == null)
        {
            throw new TimeoutException($"GPU kernel {_kernelId} response timeout");
        }

        // Deserialize response
        return GpuMessageSerializer.Deserialize<TResponse>(response.Value.Payload);
    }

    /// <summary>
    /// Gets the current HLC timestamp for this grain.
    /// </summary>
    /// <remarks>
    /// This generates a new monotonic timestamp that incorporates:
    /// - Current physical time (nanosecond precision)
    /// - Logical counter for events at same physical time
    /// - Grain's unique node ID
    /// </remarks>
    /// <returns>New HLC timestamp for local event.</returns>
    protected HybridTimestamp GetCurrentTimestamp()
    {
        return _hlcClock.Now();
    }

    /// <summary>
    /// Updates HLC with a received timestamp (for causal ordering).
    /// </summary>
    /// <param name="receivedTimestamp">Timestamp from received message.</param>
    /// <remarks>
    /// Implements Lamport clock update rules to maintain happens-before relationships.
    /// Should be called when receiving messages from other actors.
    /// </remarks>
    protected void UpdateTimestamp(HybridTimestamp receivedTimestamp)
    {
        _hlcClock.Update(receivedTimestamp);

        _logger.LogTrace(
            "Updated HLC from received timestamp: Physical={Physical}ns, Logical={Logical}",
            receivedTimestamp.PhysicalTime,
            receivedTimestamp.LogicalCounter);
    }

    /// <summary>
    /// Gets the last generated HLC timestamp.
    /// </summary>
    protected HybridTimestamp LastTimestamp => _hlcClock.LastTimestamp;

    /// <summary>
    /// Gets current ring kernel status and metrics.
    /// </summary>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Kernel status information.</returns>
    protected async Task<RingKernelStatus> GetKernelStatusAsync(CancellationToken cancellationToken = default)
    {
        if (_kernelId == null)
        {
            throw new InvalidOperationException("Kernel not initialized");
        }

        return await _runtime.GetStatusAsync(_kernelId, cancellationToken);
    }

    /// <summary>
    /// Gets ring kernel performance metrics.
    /// </summary>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Performance metrics.</returns>
    protected async Task<RingKernelMetrics> GetKernelMetricsAsync(CancellationToken cancellationToken = default)
    {
        if (_kernelId == null)
        {
            throw new InvalidOperationException("Kernel not initialized");
        }

        return await _runtime.GetMetricsAsync(_kernelId, cancellationToken);
    }

    /// <summary>
    /// Disposes the GPU-native grain and terminates the ring kernel.
    /// </summary>
    public virtual async ValueTask DisposeAsync()
    {
        if (_disposed)
        {
            return;
        }

        _logger.LogInformation(
            "Disposing GPU-native grain {GrainType} (kernel: {KernelId})",
            GetType().Name,
            _kernelId);

        if (_kernelId != null && _isKernelLaunched)
        {
            try
            {
                // Terminate ring kernel gracefully
                await _runtime.TerminateAsync(_kernelId);

                _logger.LogInformation(
                    "Terminated ring kernel {KernelId}",
                    _kernelId);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex,
                    "Error terminating ring kernel {KernelId}",
                    _kernelId);
            }
        }

        _disposed = true;
        GC.SuppressFinalize(this);
    }
}

/// <summary>
/// Marker interface for grains that execute on GPU using ring kernels.
/// </summary>
/// <remarks>
/// Used by GPU-aware placement strategies to identify GPU-native grains.
/// </remarks>
public interface IGpuNativeGrain : IGrainWithIntegerKey
{
}

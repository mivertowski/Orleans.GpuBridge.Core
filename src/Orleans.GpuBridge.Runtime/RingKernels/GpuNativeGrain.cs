// Copyright (c) 2025 Michael Ivertowski
// Licensed under the MIT License.

using DotCompute.Abstractions.RingKernels;
using Microsoft.Extensions.Logging;
using Orleans.Runtime;

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

    /// <summary>
    /// Initializes a new instance of the <see cref="GpuNativeGrain"/> class.
    /// </summary>
    /// <param name="runtime">Ring kernel runtime for GPU operations.</param>
    /// <param name="logger">Logger instance.</param>
    protected GpuNativeGrain(IRingKernelRuntime runtime, ILogger<GpuNativeGrain> logger)
    {
        _runtime = runtime ?? throw new ArgumentNullException(nameof(runtime));
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
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
            await _runtime.LaunchAsync(_kernelId, gridSize, blockSize, cancellationToken);
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

        // Create message
        var message = KernelMessage<TRequest>.Create(
            senderId: 0, // Orleans runtime
            receiverId: (int)this.GetPrimaryKeyLong(),
            type: DotCompute.Abstractions.RingKernels.MessageType.Data,
            payload: request);

        // Send to GPU
        var sendStart = DateTime.UtcNow;
        await _runtime.SendMessageAsync<TRequest>(_kernelId, message, cancellationToken);

        // Wait for response
        var effectiveTimeout = timeout == default ? TimeSpan.FromSeconds(5) : timeout;
        var response = await _runtime.ReceiveMessageAsync<TResponse>(
            _kernelId,
            effectiveTimeout,
            cancellationToken);

        var roundTripNs = (DateTime.UtcNow - sendStart).TotalNanoseconds;

        if (response == null)
        {
            _logger.LogError(
                "GPU kernel {KernelId} response timeout after {TimeoutMs}ms",
                _kernelId,
                effectiveTimeout.TotalMilliseconds);

            throw new TimeoutException(
                $"GPU kernel {_kernelId} did not respond within {effectiveTimeout.TotalMilliseconds}ms");
        }

        _logger.LogTrace(
            "GPU kernel {KernelId} round-trip: {LatencyNs}ns",
            _kernelId,
            roundTripNs);

        return response.Value.Payload;
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

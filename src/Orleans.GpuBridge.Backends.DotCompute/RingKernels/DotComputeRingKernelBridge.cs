// Orleans.GpuBridge - GPU-native distributed computing for Microsoft Orleans
// Copyright (c) 2025 Michael Ivertowski

using System.Collections.Concurrent;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using DotComputeRingKernels = DotCompute.Abstractions.RingKernels;
using Microsoft.Extensions.Logging;
using Orleans.GpuBridge.Abstractions.RingKernels;
using Orleans.GpuBridge.Abstractions.Providers;
using Orleans.GpuBridge.Abstractions.Providers.Memory.Allocators;
using Orleans.GpuBridge.Abstractions.Providers.Memory.Options;
using Orleans.GpuBridge.Backends.DotCompute.Memory;

namespace Orleans.GpuBridge.Backends.DotCompute.RingKernels;

/// <summary>
/// DotCompute implementation of the ring kernel bridge for GPU-native actor execution.
/// </summary>
/// <remarks>
/// <para>
/// This bridge connects Orleans.GpuBridge generated actors to DotCompute's GPU execution infrastructure.
/// It handles:
/// </para>
/// <list type="bullet">
/// <item><description>GPU device discovery and placement</description></item>
/// <item><description>GPU memory allocation for actor state</description></item>
/// <item><description>Message serialization and ring kernel dispatch</description></item>
/// <item><description>Response collection and state synchronization</description></item>
/// </list>
/// <para>
/// Performance characteristics:
/// </para>
/// <list type="bullet">
/// <item><description>State allocation: ~1-10μs (one-time per actor)</description></item>
/// <item><description>Handler execution: 100-500ns (GPU queue operations)</description></item>
/// <item><description>State read/write: ~500ns-2μs (PCIe transfer)</description></item>
/// </list>
/// </remarks>
public sealed class DotComputeRingKernelBridge : IRingKernelBridge
{
    private readonly ILogger<DotComputeRingKernelBridge> _logger;
    private readonly DotComputeBackendProvider _backendProvider;
    private readonly DotComputeRingKernels.IRingKernelRuntime? _ringKernelRuntime;
    private readonly ConcurrentDictionary<string, StateHandleInfo> _stateHandles;
    private readonly object _telemetryLock = new();

    // Telemetry counters
    private long _totalExecutions;
    private long _gpuExecutions;
    private long _cpuFallbackExecutions;
    private long _bytesToGpu;
    private long _bytesFromGpu;
    private long _totalLatencyNs;

    /// <summary>
    /// Initializes a new instance of the <see cref="DotComputeRingKernelBridge"/> class.
    /// </summary>
    /// <param name="logger">Logger for bridge operations.</param>
    /// <param name="backendProvider">DotCompute backend provider.</param>
    /// <param name="ringKernelRuntime">Optional ring kernel runtime for persistent kernel mode.</param>
    public DotComputeRingKernelBridge(
        ILogger<DotComputeRingKernelBridge> logger,
        DotComputeBackendProvider backendProvider,
        DotComputeRingKernels.IRingKernelRuntime? ringKernelRuntime = null)
    {
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        _backendProvider = backendProvider ?? throw new ArgumentNullException(nameof(backendProvider));
        _ringKernelRuntime = ringKernelRuntime;
        _stateHandles = new ConcurrentDictionary<string, StateHandleInfo>();

        _logger.LogInformation("Initialized DotCompute ring kernel bridge");
    }

    /// <inheritdoc/>
    public Task<bool> IsGpuAvailableAsync(CancellationToken cancellationToken = default)
    {
        try
        {
            var isAvailable = _backendProvider.IsAvailable();

            _logger.LogDebug("GPU availability check: {IsAvailable}", isAvailable);

            return Task.FromResult(isAvailable);
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Error checking GPU availability, returning false");
            return Task.FromResult(false);
        }
    }

    /// <inheritdoc/>
    public Task<int> GetDevicePlacementAsync(string actorKey, CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(actorKey);

        try
        {
            var deviceManager = _backendProvider.GetDeviceManager();
            var devices = deviceManager.GetDevices();

            if (devices.Count == 0)
            {
                _logger.LogDebug("No GPU devices available for actor {ActorKey}, returning -1", actorKey);
                return Task.FromResult(-1);
            }

            // Simple placement strategy: hash actor key to device index
            // This ensures consistent placement for the same actor
            var hash = actorKey.GetHashCode();
            var deviceIndex = Math.Abs(hash) % devices.Count;

            _logger.LogDebug(
                "Placed actor {ActorKey} on device {DeviceIndex} of {DeviceCount}",
                actorKey,
                deviceIndex,
                devices.Count);

            return Task.FromResult(deviceIndex);
        }
        catch (InvalidOperationException)
        {
            // Backend not initialized
            _logger.LogDebug("Backend not initialized for actor {ActorKey}, returning -1", actorKey);
            return Task.FromResult(-1);
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Error getting device placement for actor {ActorKey}", actorKey);
            return Task.FromResult(-1);
        }
    }

    /// <inheritdoc/>
    public async Task<GpuStateHandle<TState>> AllocateStateAsync<TState>(
        string actorId,
        int deviceId,
        CancellationToken cancellationToken = default)
        where TState : unmanaged
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(actorId);

        var sizeBytes = Unsafe.SizeOf<TState>();

        _logger.LogDebug(
            "Allocating GPU state for actor {ActorId} on device {DeviceId} ({SizeBytes} bytes)",
            actorId,
            deviceId,
            sizeBytes);

        var handle = new GpuStateHandle<TState>(actorId, deviceId, sizeBytes);

        try
        {
            if (_backendProvider.IsAvailable() && deviceId >= 0)
            {
                var memoryAllocator = _backendProvider.GetMemoryAllocator();
                var deviceManager = _backendProvider.GetDeviceManager();

                var device = deviceManager.GetDevice(deviceId);
                var options = new MemoryAllocationOptions
                {
                    PreferredDevice = device
                };

                // Allocate GPU memory for state
                var deviceMemory = await memoryAllocator.AllocateAsync(
                    sizeBytes,
                    options,
                    cancellationToken);

                handle.GpuPointer = deviceMemory.DevicePointer;

                // Track the allocation
                _stateHandles[actorId] = new StateHandleInfo(deviceMemory, sizeBytes);

                Interlocked.Add(ref _bytesToGpu, sizeBytes);

                _logger.LogDebug(
                    "Allocated GPU state for actor {ActorId} at {GpuPointer:X16}",
                    actorId,
                    handle.GpuPointer.ToInt64());
            }
            else
            {
                // CPU fallback - state is stored in ShadowState
                _logger.LogDebug(
                    "Using CPU fallback for actor {ActorId} (GPU unavailable or deviceId={DeviceId})",
                    actorId,
                    deviceId);

                handle.GpuPointer = IntPtr.Zero;
            }

            return handle;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to allocate GPU state for actor {ActorId}", actorId);

            // Return handle with CPU fallback
            handle.GpuPointer = IntPtr.Zero;
            return handle;
        }
    }

    /// <inheritdoc/>
    public Task ReleaseStateAsync<TState>(
        GpuStateHandle<TState> handle,
        CancellationToken cancellationToken = default)
        where TState : unmanaged
    {
        ArgumentNullException.ThrowIfNull(handle);

        _logger.LogDebug("Releasing GPU state for actor {ActorId}", handle.ActorId);

        try
        {
            if (_stateHandles.TryRemove(handle.ActorId, out var info))
            {
                info.DeviceMemory?.Dispose();

                _logger.LogDebug(
                    "Released GPU state for actor {ActorId} ({SizeBytes} bytes)",
                    handle.ActorId,
                    info.SizeBytes);
            }

            handle.Dispose();
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Error releasing GPU state for actor {ActorId}", handle.ActorId);
        }

        return Task.CompletedTask;
    }

    /// <inheritdoc/>
    public async Task<RingKernelResult<TResponse, TState>> ExecuteHandlerAsync<TRequest, TResponse, TState>(
        string kernelId,
        int handlerId,
        TRequest request,
        GpuStateHandle<TState> stateHandle,
        CancellationToken cancellationToken = default)
        where TRequest : unmanaged
        where TResponse : unmanaged
        where TState : unmanaged
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(kernelId);
        ArgumentNullException.ThrowIfNull(stateHandle);

        var stopwatch = Stopwatch.StartNew();
        var wasGpuExecution = false;

        try
        {
            TResponse response;
            TState newState;

            if (stateHandle.IsValid && _ringKernelRuntime != null)
            {
                // GPU execution via ring kernel
                (response, newState) = await ExecuteOnGpuAsync<TRequest, TResponse, TState>(
                    kernelId,
                    handlerId,
                    request,
                    stateHandle,
                    cancellationToken);

                wasGpuExecution = true;
                Interlocked.Increment(ref _gpuExecutions);
            }
            else
            {
                // CPU fallback execution
                (response, newState) = ExecuteOnCpu<TRequest, TResponse, TState>(
                    handlerId,
                    request,
                    stateHandle.ShadowState);

                Interlocked.Increment(ref _cpuFallbackExecutions);
            }

            stopwatch.Stop();
            var latencyNs = stopwatch.ElapsedTicks * 1_000_000_000L / Stopwatch.Frequency;

            Interlocked.Increment(ref _totalExecutions);
            Interlocked.Add(ref _totalLatencyNs, latencyNs);

            // Update shadow state
            stateHandle.ShadowState = newState;

            _logger.LogTrace(
                "Executed handler {HandlerId} on kernel {KernelId} in {LatencyNs}ns ({ExecutionType})",
                handlerId,
                kernelId,
                latencyNs,
                wasGpuExecution ? "GPU" : "CPU");

            return new RingKernelResult<TResponse, TState>(
                response,
                newState,
                latencyNs,
                wasGpuExecution);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex,
                "Failed to execute handler {HandlerId} on kernel {KernelId}",
                handlerId,
                kernelId);

            return new RingKernelResult<TResponse, TState>(
                errorCode: -1,
                currentState: stateHandle.ShadowState);
        }
    }

    /// <inheritdoc/>
    public async Task<TState> ExecuteFireAndForgetAsync<TRequest, TState>(
        string kernelId,
        int handlerId,
        TRequest request,
        GpuStateHandle<TState> stateHandle,
        CancellationToken cancellationToken = default)
        where TRequest : unmanaged
        where TState : unmanaged
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(kernelId);
        ArgumentNullException.ThrowIfNull(stateHandle);

        try
        {
            TState newState;

            if (stateHandle.IsValid && _ringKernelRuntime != null)
            {
                // GPU execution - send message without waiting for response
                newState = await ExecuteFireAndForgetOnGpuAsync<TRequest, TState>(
                    kernelId,
                    handlerId,
                    request,
                    stateHandle,
                    cancellationToken);

                Interlocked.Increment(ref _gpuExecutions);
            }
            else
            {
                // CPU fallback
                newState = ExecuteFireAndForgetOnCpu<TRequest, TState>(
                    handlerId,
                    request,
                    stateHandle.ShadowState);

                Interlocked.Increment(ref _cpuFallbackExecutions);
            }

            Interlocked.Increment(ref _totalExecutions);

            // Update shadow state
            stateHandle.ShadowState = newState;

            return newState;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex,
                "Failed to execute fire-and-forget handler {HandlerId} on kernel {KernelId}",
                handlerId,
                kernelId);

            return stateHandle.ShadowState;
        }
    }

    /// <inheritdoc/>
    public async Task<TState> ReadStateAsync<TState>(
        GpuStateHandle<TState> handle,
        CancellationToken cancellationToken = default)
        where TState : unmanaged
    {
        ArgumentNullException.ThrowIfNull(handle);

        if (!handle.IsValid)
        {
            // Return shadow state for CPU fallback
            return handle.ShadowState;
        }

        try
        {
            // Try GPU memory read via DotCompute native buffer
            if (_stateHandles.TryGetValue(handle.ActorId, out var stateInfo) &&
                stateInfo.DeviceMemory is Memory.DotComputeDeviceMemoryWrapper<TState> typedMemory &&
                typedMemory.NativeBuffer is { } nativeBuffer)
            {
                // Use native buffer's CopyToAsync to read GPU memory to host
                var hostMemory = new TState[1];
                await nativeBuffer.CopyToAsync(hostMemory.AsMemory(), cancellationToken);

                var state = hostMemory[0];

                // Update shadow state with GPU value
                handle.ShadowState = state;

                Interlocked.Add(ref _bytesFromGpu, handle.SizeBytes);

                _logger.LogTrace(
                    "Read state for actor {ActorId} from GPU memory via DotCompute",
                    handle.ActorId);

                return state;
            }

            // CPU fallback: return shadow state
            _logger.LogTrace(
                "Reading state for actor {ActorId} from shadow copy (GPU unavailable)",
                handle.ActorId);

            return handle.ShadowState;
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Error reading GPU state for actor {ActorId}, falling back to shadow", handle.ActorId);
            return handle.ShadowState;
        }
    }

    /// <inheritdoc/>
    public async Task WriteStateAsync<TState>(
        GpuStateHandle<TState> handle,
        TState state,
        CancellationToken cancellationToken = default)
        where TState : unmanaged
    {
        ArgumentNullException.ThrowIfNull(handle);

        // Always update shadow state first
        handle.ShadowState = state;

        if (!handle.IsValid)
        {
            return;
        }

        try
        {
            // Try GPU memory write via DotCompute native buffer
            if (_stateHandles.TryGetValue(handle.ActorId, out var stateInfo) &&
                stateInfo.DeviceMemory is Memory.DotComputeDeviceMemoryWrapper<TState> typedMemory &&
                typedMemory.NativeBuffer is { } nativeBuffer)
            {
                // Use native buffer's CopyFromAsync to write from host to GPU memory
                var hostMemory = new TState[] { state };
                await nativeBuffer.CopyFromAsync(new ReadOnlyMemory<TState>(hostMemory), cancellationToken);

                Interlocked.Add(ref _bytesToGpu, handle.SizeBytes);

                _logger.LogTrace(
                    "Wrote state for actor {ActorId} to GPU memory via DotCompute",
                    handle.ActorId);

                return;
            }

            // CPU fallback: shadow state already updated
            _logger.LogTrace(
                "Writing state for actor {ActorId} to shadow copy (GPU unavailable)",
                handle.ActorId);
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Error writing GPU state for actor {ActorId}, shadow state updated", handle.ActorId);
        }
    }

    /// <inheritdoc/>
    public RingKernelTelemetry GetTelemetry()
    {
        lock (_telemetryLock)
        {
            var totalExecutions = Interlocked.Read(ref _totalExecutions);
            var avgLatencyNs = totalExecutions > 0
                ? (double)Interlocked.Read(ref _totalLatencyNs) / totalExecutions
                : 0;

            return new RingKernelTelemetry
            {
                TotalExecutions = totalExecutions,
                GpuExecutions = Interlocked.Read(ref _gpuExecutions),
                CpuFallbackExecutions = Interlocked.Read(ref _cpuFallbackExecutions),
                BytesToGpu = Interlocked.Read(ref _bytesToGpu),
                BytesFromGpu = Interlocked.Read(ref _bytesFromGpu),
                AverageLatencyNs = avgLatencyNs,
                ActiveStateHandles = _stateHandles.Count,
                AllocatedGpuMemoryBytes = _stateHandles.Values.Sum(h => h.SizeBytes)
            };
        }
    }

    #region GPU Execution

    private async Task<(TResponse response, TState newState)> ExecuteOnGpuAsync<TRequest, TResponse, TState>(
        string kernelId,
        int handlerId,
        TRequest request,
        GpuStateHandle<TState> stateHandle,
        CancellationToken cancellationToken)
        where TRequest : unmanaged
        where TResponse : unmanaged
        where TState : unmanaged
    {
        // Create kernel message with request payload
        // Note: handlerId is encoded in the receiver ID field for handler routing
        var message = DotComputeRingKernels.KernelMessage<TRequest>.CreateData(
            senderId: stateHandle.ActorId.GetHashCode(),
            receiverId: handlerId, // Use handlerId as receiver for routing
            payload: request);

        // Send to ring kernel
        await _ringKernelRuntime!.SendMessageAsync(kernelId, message, cancellationToken);

        // Receive response
        var responseMessage = await _ringKernelRuntime.ReceiveMessageAsync<TResponse>(
            kernelId,
            timeout: TimeSpan.FromSeconds(5),
            cancellationToken: cancellationToken);

        if (!responseMessage.HasValue)
        {
            throw new TimeoutException($"Timeout waiting for response from kernel {kernelId}");
        }

        // Read updated state
        var newState = await ReadStateAsync(stateHandle, cancellationToken);

        return (responseMessage.Value.Payload, newState);
    }

    private async Task<TState> ExecuteFireAndForgetOnGpuAsync<TRequest, TState>(
        string kernelId,
        int handlerId,
        TRequest request,
        GpuStateHandle<TState> stateHandle,
        CancellationToken cancellationToken)
        where TRequest : unmanaged
        where TState : unmanaged
    {
        // Create kernel message with request payload
        // Note: handlerId is encoded in the receiver ID field for handler routing
        var message = DotComputeRingKernels.KernelMessage<TRequest>.CreateData(
            senderId: stateHandle.ActorId.GetHashCode(),
            receiverId: handlerId, // Use handlerId as receiver for routing
            payload: request);

        // Send to ring kernel (fire and forget - don't wait for response)
        await _ringKernelRuntime!.SendMessageAsync(kernelId, message, cancellationToken);

        // Return current shadow state (updated async by GPU)
        return stateHandle.ShadowState;
    }

    #endregion

    #region CPU Fallback Execution

    private static (TResponse response, TState newState) ExecuteOnCpu<TRequest, TResponse, TState>(
        int handlerId,
        TRequest request,
        TState currentState)
        where TRequest : unmanaged
        where TResponse : unmanaged
        where TState : unmanaged
    {
        // CPU fallback: execute handler logic directly
        // This is a generic implementation - specific handlers override this
        // For now, return defaults and preserve state
        return (default, currentState);
    }

    private static TState ExecuteFireAndForgetOnCpu<TRequest, TState>(
        int handlerId,
        TRequest request,
        TState currentState)
        where TRequest : unmanaged
        where TState : unmanaged
    {
        // CPU fallback: fire-and-forget just returns current state
        return currentState;
    }

    #endregion

    /// <summary>
    /// Internal tracking for allocated state handles.
    /// </summary>
    private sealed record StateHandleInfo(
        Abstractions.Providers.Memory.Interfaces.IDeviceMemory? DeviceMemory,
        long SizeBytes);
}

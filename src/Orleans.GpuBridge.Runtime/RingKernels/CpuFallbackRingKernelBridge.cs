// Orleans.GpuBridge - GPU-native distributed computing for Microsoft Orleans
// Copyright (c) 2025 Michael Ivertowski

using System.Collections.Concurrent;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using Microsoft.Extensions.Logging;
using Orleans.GpuBridge.Abstractions.RingKernels;

namespace Orleans.GpuBridge.Runtime.RingKernels;

/// <summary>
/// CPU fallback implementation of the ring kernel bridge for when no GPU is available.
/// </summary>
/// <remarks>
/// <para>
/// This bridge provides a fully functional CPU-only implementation that allows
/// generated actors to work without GPU hardware. Useful for:
/// </para>
/// <list type="bullet">
/// <item><description>Development environments without GPU</description></item>
/// <item><description>Unit testing without GPU dependencies</description></item>
/// <item><description>Graceful degradation when GPU fails</description></item>
/// </list>
/// <para>
/// Performance characteristics (CPU fallback):
/// </para>
/// <list type="bullet">
/// <item><description>Handler execution: ~1-10μs (direct method call)</description></item>
/// <item><description>State access: immediate (in-memory)</description></item>
/// </list>
/// </remarks>
public sealed class CpuFallbackRingKernelBridge : IRingKernelBridge
{
    private readonly ILogger<CpuFallbackRingKernelBridge> _logger;
    private readonly ConcurrentDictionary<string, int> _stateHandleSizes;
    private readonly object _telemetryLock = new();

    // Telemetry counters
    private long _totalExecutions;
    private long _cpuFallbackExecutions;
    private long _totalLatencyNs;

    /// <summary>
    /// Initializes a new instance of the <see cref="CpuFallbackRingKernelBridge"/> class.
    /// </summary>
    /// <param name="logger">Logger for bridge operations.</param>
    public CpuFallbackRingKernelBridge(ILogger<CpuFallbackRingKernelBridge> logger)
    {
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        _stateHandleSizes = new ConcurrentDictionary<string, int>();

        _logger.LogInformation("Initialized CPU fallback ring kernel bridge");
    }

    /// <inheritdoc/>
    public Task<bool> IsGpuAvailableAsync(CancellationToken cancellationToken = default)
    {
        // CPU fallback always reports no GPU
        return Task.FromResult(false);
    }

    /// <inheritdoc/>
    public Task<int> GetDevicePlacementAsync(string actorKey, CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(actorKey);

        // CPU fallback returns -1 (no device)
        return Task.FromResult(-1);
    }

    /// <inheritdoc/>
    public Task<GpuStateHandle<TState>> AllocateStateAsync<TState>(
        string actorId,
        int deviceId,
        CancellationToken cancellationToken = default)
        where TState : unmanaged
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(actorId);

        var sizeBytes = Unsafe.SizeOf<TState>();

        _logger.LogDebug(
            "Allocating CPU state for actor {ActorId} ({SizeBytes} bytes)",
            actorId,
            sizeBytes);

        // CPU fallback: state stored in ShadowState, no GPU pointer
        var handle = new GpuStateHandle<TState>(actorId, -1, sizeBytes)
        {
            GpuPointer = IntPtr.Zero
        };

        _stateHandleSizes[actorId] = sizeBytes;

        return Task.FromResult(handle);
    }

    /// <inheritdoc/>
    public Task ReleaseStateAsync<TState>(
        GpuStateHandle<TState> handle,
        CancellationToken cancellationToken = default)
        where TState : unmanaged
    {
        ArgumentNullException.ThrowIfNull(handle);

        _logger.LogDebug("Releasing CPU state for actor {ActorId}", handle.ActorId);

        _stateHandleSizes.TryRemove(handle.ActorId, out _);
        handle.Dispose();

        return Task.CompletedTask;
    }

    /// <inheritdoc/>
    public Task<RingKernelResult<TResponse, TState>> ExecuteHandlerAsync<TRequest, TResponse, TState>(
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

        try
        {
            // CPU fallback: execute handler synchronously
            // The actual logic is in the generated actor - this just returns defaults
            var response = default(TResponse);
            var newState = stateHandle.ShadowState;

            stopwatch.Stop();
            var latencyNs = stopwatch.ElapsedTicks * 1_000_000_000L / Stopwatch.Frequency;

            Interlocked.Increment(ref _totalExecutions);
            Interlocked.Increment(ref _cpuFallbackExecutions);
            Interlocked.Add(ref _totalLatencyNs, latencyNs);

            _logger.LogTrace(
                "Executed handler {HandlerId} on kernel {KernelId} (CPU fallback) in {LatencyNs}ns",
                handlerId,
                kernelId,
                latencyNs);

            return Task.FromResult(new RingKernelResult<TResponse, TState>(
                response,
                newState,
                latencyNs,
                wasGpuExecution: false));
        }
        catch (Exception ex)
        {
            _logger.LogError(ex,
                "Failed to execute handler {HandlerId} on kernel {KernelId} (CPU fallback)",
                handlerId,
                kernelId);

            return Task.FromResult(new RingKernelResult<TResponse, TState>(
                errorCode: -1,
                currentState: stateHandle.ShadowState));
        }
    }

    /// <inheritdoc/>
    public Task<TState> ExecuteFireAndForgetAsync<TRequest, TState>(
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

        Interlocked.Increment(ref _totalExecutions);
        Interlocked.Increment(ref _cpuFallbackExecutions);

        _logger.LogTrace(
            "Executed fire-and-forget handler {HandlerId} on kernel {KernelId} (CPU fallback)",
            handlerId,
            kernelId);

        // CPU fallback: just return current state
        return Task.FromResult(stateHandle.ShadowState);
    }

    /// <inheritdoc/>
    public Task<TState> ReadStateAsync<TState>(
        GpuStateHandle<TState> handle,
        CancellationToken cancellationToken = default)
        where TState : unmanaged
    {
        ArgumentNullException.ThrowIfNull(handle);

        // CPU fallback: state is always in ShadowState
        return Task.FromResult(handle.ShadowState);
    }

    /// <inheritdoc/>
    public Task WriteStateAsync<TState>(
        GpuStateHandle<TState> handle,
        TState state,
        CancellationToken cancellationToken = default)
        where TState : unmanaged
    {
        ArgumentNullException.ThrowIfNull(handle);

        // CPU fallback: write directly to ShadowState
        handle.ShadowState = state;

        return Task.CompletedTask;
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
                GpuExecutions = 0,
                CpuFallbackExecutions = Interlocked.Read(ref _cpuFallbackExecutions),
                BytesToGpu = 0,
                BytesFromGpu = 0,
                AverageLatencyNs = avgLatencyNs,
                ActiveStateHandles = _stateHandleSizes.Count,
                AllocatedGpuMemoryBytes = 0
            };
        }
    }
}

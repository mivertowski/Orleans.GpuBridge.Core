using System;
using System.Collections.Concurrent;
using System.Threading;
using System.Threading.Tasks;
using DotCompute.Synchronization;
using Microsoft.Extensions.Logging;

namespace Orleans.GpuBridge.Backends.DotCompute.Temporal;

/// <summary>
/// Adapter for DotCompute's IBarrierProvider - enables device-wide synchronization.
/// Supports CUDA cooperative groups, OpenCL work-group barriers, and ring kernel coordination.
/// </summary>
public sealed class DotComputeBarrierProvider : IDisposable
{
    private readonly IBarrierProvider _dotComputeBarrier;
    private readonly ILogger<DotComputeBarrierProvider> _logger;
    private readonly ConcurrentDictionary<Guid, IBarrierHandle> _activeBarriers;
    private bool _disposed;

    public DotComputeBarrierProvider(
        IBarrierProvider dotComputeBarrier,
        ILogger<DotComputeBarrierProvider> logger)
    {
        _dotComputeBarrier = dotComputeBarrier ?? throw new ArgumentNullException(nameof(dotComputeBarrier));
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        _activeBarriers = new ConcurrentDictionary<Guid, IBarrierHandle>();

        _logger.LogInformation(
            "DotComputeBarrierProvider initialized - Hardware barriers: {HardwareSupport}, Max participants: {MaxParticipants}",
            _dotComputeBarrier.IsHardwareBarrierSupported,
            _dotComputeBarrier.MaxBarrierParticipants);
    }

    /// <summary>
    /// Checks if device supports hardware barriers (CUDA cooperative groups).
    /// Required for ring kernels and device-wide synchronization.
    /// </summary>
    public bool IsHardwareBarrierSupported => _dotComputeBarrier.IsHardwareBarrierSupported;

    /// <summary>
    /// Gets maximum number of threads that can participate in a barrier.
    /// Typical values: 1M+ threads for CUDA cooperative groups.
    /// </summary>
    public int MaxBarrierParticipants => _dotComputeBarrier.MaxBarrierParticipants;

    /// <summary>
    /// Creates a device-wide barrier for lockstep execution.
    /// CUDA: Uses cooperative groups (cudaLaunchCooperativeKernel)
    /// OpenCL: Uses work-group barriers (requires extension)
    /// CPU: Uses System.Threading.Barrier
    /// </summary>
    public IBarrierHandle CreateBarrier(
        int participantCount,
        BarrierOptions? options = null)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(participantCount);

        if (participantCount > MaxBarrierParticipants)
        {
            throw new ArgumentOutOfRangeException(
                nameof(participantCount),
                $"Participant count {participantCount} exceeds maximum {MaxBarrierParticipants}");
        }

        try
        {
            var barrier = _dotComputeBarrier.CreateBarrier(participantCount, options);
            _activeBarriers[barrier.BarrierId] = barrier;

            _logger.LogInformation(
                "Created barrier {BarrierId} with {ParticipantCount} participants (scope: {Scope})",
                barrier.BarrierId,
                participantCount,
                options?.Scope ?? BarrierScope.Device);

            return new BarrierHandleWrapper(barrier, this, _logger);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to create barrier with {ParticipantCount} participants", participantCount);
            throw;
        }
    }

    /// <summary>
    /// Creates a barrier specifically for ring kernel coordination.
    /// Ring kernels use infinite loops and require device-wide barriers.
    /// </summary>
    public IBarrierHandle CreateRingKernelBarrier(
        int actorCount,
        int threadsPerActor = 1)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        var totalThreads = actorCount * threadsPerActor;

        var options = new BarrierOptions
        {
            Scope = BarrierScope.Device,
            EnableArrivalCounting = true,
            Timeout = TimeSpan.FromSeconds(30) // Ring kernels should complete quickly
        };

        _logger.LogInformation(
            "Creating ring kernel barrier for {ActorCount} actors × {ThreadsPerActor} threads = {TotalThreads} total",
            actorCount, threadsPerActor, totalThreads);

        return CreateBarrier(totalThreads, options);
    }

    /// <summary>
    /// Launches kernel with device-wide barrier support.
    /// Requires cooperative launch for CUDA (cudaLaunchCooperativeKernel).
    /// Overhead: ~20-50μs for cooperative launch.
    /// </summary>
    public async Task ExecuteWithBarrierAsync(
        ICompiledKernel kernel,
        IBarrierHandle barrier,
        LaunchConfiguration config,
        object[] arguments,
        CancellationToken ct = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIf(kernel);
        ArgumentNullException.ThrowIf(barrier);
        ArgumentNullException.ThrowIf(config);

        try
        {
            _logger.LogDebug(
                "Executing kernel with barrier {BarrierId} - Grid: {GridDim}, Block: {BlockDim}",
                barrier.BarrierId,
                config.GridDim,
                config.BlockDim);

            await _dotComputeBarrier.ExecuteWithBarrierAsync(
                kernel, barrier, config, arguments, ct).ConfigureAwait(false);

            _logger.LogDebug(
                "Kernel execution complete - Barrier arrivals: {ArrivalCount}/{ParticipantCount}",
                barrier.ArrivalCount,
                barrier.ParticipantCount);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Kernel execution with barrier {BarrierId} failed", barrier.BarrierId);
            throw;
        }
    }

    /// <summary>
    /// Executes ring kernel with infinite loop and barrier coordination.
    /// Ring kernels run until explicitly stopped (via cancellation or signal).
    /// </summary>
    public async Task ExecuteRingKernelAsync(
        ICompiledKernel kernel,
        IBarrierHandle barrier,
        LaunchConfiguration config,
        object[] arguments,
        CancellationToken ct = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        _logger.LogInformation(
            "Launching ring kernel with barrier {BarrierId} - This kernel runs indefinitely until cancelled",
            barrier.BarrierId);

        try
        {
            // Ring kernels use cooperative launch and run until cancellation
            await ExecuteWithBarrierAsync(kernel, barrier, config, arguments, ct).ConfigureAwait(false);
        }
        catch (OperationCanceledException)
        {
            _logger.LogInformation("Ring kernel cancelled - Barrier {BarrierId}", barrier.BarrierId);
            throw;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Ring kernel execution failed - Barrier {BarrierId}", barrier.BarrierId);
            throw;
        }
    }

    internal void UnregisterBarrier(Guid barrierId)
    {
        _activeBarriers.TryRemove(barrierId, out _);
        _logger.LogDebug("Barrier {BarrierId} unregistered", barrierId);
    }

    public void Dispose()
    {
        if (_disposed)
            return;

        try
        {
            _logger.LogDebug(
                "Disposing DotComputeBarrierProvider with {ActiveBarriers} active barriers",
                _activeBarriers.Count);

            // Dispose all active barriers
            foreach (var barrier in _activeBarriers.Values)
            {
                try
                {
                    barrier.Dispose();
                }
                catch (Exception ex)
                {
                    _logger.LogWarning(ex, "Error disposing barrier {BarrierId}", barrier.BarrierId);
                }
            }

            _activeBarriers.Clear();
            _disposed = true;

            _logger.LogInformation("DotComputeBarrierProvider disposed");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error during DotComputeBarrierProvider disposal");
        }
    }

    /// <summary>
    /// Wrapper for IBarrierHandle that ensures cleanup.
    /// </summary>
    private sealed class BarrierHandleWrapper : IBarrierHandle
    {
        private readonly IBarrierHandle _inner;
        private readonly DotComputeBarrierProvider _provider;
        private readonly ILogger _logger;
        private bool _disposed;

        public BarrierHandleWrapper(
            IBarrierHandle inner,
            DotComputeBarrierProvider provider,
            ILogger logger)
        {
            _inner = inner;
            _provider = provider;
            _logger = logger;
        }

        public Guid BarrierId => _inner.BarrierId;
        public int ArrivalCount => _inner.ArrivalCount;
        public int ParticipantCount => _inner.ParticipantCount;
        public bool IsReady => _inner.IsReady;

        public Task WaitAsync(CancellationToken ct = default)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            return _inner.WaitAsync(ct);
        }

        public void Reset()
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            _inner.Reset();
            _logger.LogDebug("Barrier {BarrierId} reset", BarrierId);
        }

        public void Dispose()
        {
            if (_disposed)
                return;

            try
            {
                _inner.Dispose();
                _provider.UnregisterBarrier(BarrierId);
                _disposed = true;
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Error disposing barrier {BarrierId}", BarrierId);
            }
        }
    }
}

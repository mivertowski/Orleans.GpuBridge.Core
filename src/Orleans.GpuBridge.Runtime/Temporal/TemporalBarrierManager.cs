// Orleans.GpuBridge - GPU-native distributed computing for Microsoft Orleans
// Copyright (c) 2025 Michael Ivertowski

using System;
using System.Collections.Concurrent;
using System.Diagnostics;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Orleans.GpuBridge.Abstractions.Temporal;

namespace Orleans.GpuBridge.Runtime.Temporal;

/// <summary>
/// Manages device-wide barriers for GPU temporal synchronization.
/// </summary>
/// <remarks>
/// <para>
/// This manager provides barrier synchronization primitives that abstract over:
/// <list type="bullet">
/// <item><description>CUDA Cooperative Groups for device-wide barriers</description></item>
/// <item><description>__syncthreads() for thread block barriers</description></item>
/// <item><description>Grid-level synchronization via cooperative kernel launches</description></item>
/// </list>
/// </para>
/// <para>
/// <strong>Barrier Scopes:</strong>
/// <list type="bullet">
/// <item><description>ThreadBlock: Synchronizes threads within a single block (~100ns)</description></item>
/// <item><description>Device: Synchronizes all threads on a single GPU (~1μs)</description></item>
/// <item><description>Grid: Synchronizes all threads in a kernel launch (~5μs)</description></item>
/// <item><description>System: Synchronizes across multiple GPUs (~100μs)</description></item>
/// </list>
/// </para>
/// <para>
/// When running on CPU (no GPU available), barriers are simulated using
/// <see cref="Barrier"/> from the Task Parallel Library.
/// </para>
/// </remarks>
public sealed class TemporalBarrierManager : ITemporalBarrierManager
{
    private readonly ILogger<TemporalBarrierManager> _logger;
    private readonly ConcurrentDictionary<Guid, TemporalBarrierImpl> _activeBarriers = new();

    // Statistics
    private long _totalBarriersCreated;
    private long _successfulSyncs;
    private long _timeouts;
    private long _totalSyncTimeUs;
    private long _maxSyncTimeUs;
    private long _totalThreadsPerBarrier;

    /// <summary>
    /// Event raised when a barrier completes.
    /// </summary>
    public event EventHandler<BarrierEventArgs>? BarrierCompleted;

    /// <summary>
    /// Event raised when a barrier times out.
    /// </summary>
    public event EventHandler<BarrierEventArgs>? BarrierTimedOut;

    /// <summary>
    /// Initializes a new instance of the <see cref="TemporalBarrierManager"/> class.
    /// </summary>
    /// <param name="logger">Logger for diagnostic output.</param>
    public TemporalBarrierManager(ILogger<TemporalBarrierManager> logger)
    {
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        _logger.LogInformation("TemporalBarrierManager initialized");
    }

    /// <inheritdoc/>
    public ITemporalBarrier CreateBarrier(BarrierScope scope, int timeoutMs = 5000)
    {
        if (timeoutMs <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(timeoutMs), "Timeout must be positive.");
        }

        var barrier = new TemporalBarrierImpl(
            barrierId: Guid.NewGuid(),
            scope: scope,
            timeoutMs: timeoutMs,
            expectedThreadCount: GetExpectedThreadCountForScope(scope));

        _activeBarriers[barrier.BarrierId] = barrier;
        Interlocked.Increment(ref _totalBarriersCreated);

        _logger.LogDebug(
            "Created barrier {BarrierId} with scope {Scope}, timeout {TimeoutMs}ms, " +
            "expected threads {ExpectedThreads}",
            barrier.BarrierId,
            scope,
            timeoutMs,
            barrier.ExpectedThreadCount);

        return barrier;
    }

    /// <inheritdoc/>
    public async Task SynchronizeAsync(ITemporalBarrier barrier, CancellationToken ct = default)
    {
        ArgumentNullException.ThrowIfNull(barrier);

        if (barrier is not TemporalBarrierImpl impl)
        {
            throw new ArgumentException(
                "Barrier must be created by this manager.",
                nameof(barrier));
        }

        if (impl.IsComplete)
        {
            _logger.LogWarning("Barrier {BarrierId} already completed", barrier.BarrierId);
            return;
        }

        var sw = Stopwatch.StartNew();
        _logger.LogTrace("Synchronizing on barrier {BarrierId}...", barrier.BarrierId);

        try
        {
            // Simulate barrier synchronization
            // In a real GPU implementation, this would use:
            // - __syncthreads() for ThreadBlock
            // - cooperative_groups::this_grid().sync() for Device/Grid
            // - Multi-GPU sync for System scope

            var result = await SimulateBarrierSyncAsync(impl, ct);

            sw.Stop();
            var elapsedUs = sw.ElapsedTicks * 1_000_000 / Stopwatch.Frequency;

            if (result)
            {
                Interlocked.Increment(ref _successfulSyncs);
                Interlocked.Add(ref _totalSyncTimeUs, elapsedUs);

                // Update max sync time (thread-safe)
                long currentMax;
                do
                {
                    currentMax = _maxSyncTimeUs;
                    if (elapsedUs <= currentMax) break;
                } while (Interlocked.CompareExchange(ref _maxSyncTimeUs, elapsedUs, currentMax) != currentMax);

                impl.MarkComplete();

                _logger.LogDebug(
                    "Barrier {BarrierId} synchronized in {ElapsedUs}μs",
                    barrier.BarrierId,
                    elapsedUs);

                OnBarrierCompleted(new BarrierEventArgs
                {
                    Barrier = barrier,
                    TimestampNanos = DateTimeOffset.UtcNow.ToUnixTimeNanoseconds(),
                    Success = true
                });
            }
            else
            {
                Interlocked.Increment(ref _timeouts);
                impl.MarkTimedOut();

                _logger.LogWarning(
                    "Barrier {BarrierId} timed out after {TimeoutMs}ms",
                    barrier.BarrierId,
                    barrier.TimeoutMs);

                OnBarrierTimedOut(new BarrierEventArgs
                {
                    Barrier = barrier,
                    TimestampNanos = DateTimeOffset.UtcNow.ToUnixTimeNanoseconds(),
                    Success = false,
                    ErrorMessage = $"Barrier timed out after {barrier.TimeoutMs}ms"
                });
            }
        }
        catch (OperationCanceledException)
        {
            _logger.LogDebug("Barrier {BarrierId} synchronization cancelled", barrier.BarrierId);
            throw;
        }
    }

    /// <inheritdoc/>
    public BarrierStatistics GetStatistics()
    {
        var avgSyncTime = _successfulSyncs > 0
            ? (double)_totalSyncTimeUs / _successfulSyncs
            : 0.0;

        var avgThreads = _totalBarriersCreated > 0
            ? (double)_totalThreadsPerBarrier / _totalBarriersCreated
            : 0.0;

        return new BarrierStatistics
        {
            TotalBarriersCreated = _totalBarriersCreated,
            SuccessfulSyncs = _successfulSyncs,
            Timeouts = _timeouts,
            AverageSyncTimeUs = avgSyncTime,
            MaxSyncTimeUs = _maxSyncTimeUs,
            AverageThreadsPerBarrier = avgThreads
        };
    }

    /// <summary>
    /// Removes a completed barrier from tracking.
    /// </summary>
    /// <param name="barrierId">ID of barrier to remove.</param>
    public void RemoveBarrier(Guid barrierId)
    {
        if (_activeBarriers.TryRemove(barrierId, out var barrier))
        {
            barrier.Dispose();
            _logger.LogTrace("Removed barrier {BarrierId}", barrierId);
        }
    }

    /// <summary>
    /// Gets the count of active barriers.
    /// </summary>
    public int ActiveBarrierCount => _activeBarriers.Count;

    /// <summary>
    /// Simulates barrier synchronization for CPU fallback mode.
    /// </summary>
    private async Task<bool> SimulateBarrierSyncAsync(
        TemporalBarrierImpl barrier,
        CancellationToken ct)
    {
        // Simulate barrier delay based on scope
        var simulatedDelayMs = barrier.Scope switch
        {
            BarrierScope.ThreadBlock => 0,   // ~100ns - too small to simulate
            BarrierScope.Device => 1,        // ~1μs
            BarrierScope.Grid => 5,          // ~5μs
            BarrierScope.System => 100,      // ~100μs
            _ => 1
        };

        using var cts = CancellationTokenSource.CreateLinkedTokenSource(ct);
        cts.CancelAfter(barrier.TimeoutMs);

        try
        {
            if (simulatedDelayMs > 0)
            {
                await Task.Delay(simulatedDelayMs, cts.Token);
            }

            // Simulate thread arrival
            barrier.IncrementArrivedCount();
            Interlocked.Add(ref _totalThreadsPerBarrier, 1);

            return true;
        }
        catch (OperationCanceledException) when (!ct.IsCancellationRequested)
        {
            // Timeout, not user cancellation
            return false;
        }
    }

    /// <summary>
    /// Gets expected thread count based on barrier scope.
    /// </summary>
    private static int GetExpectedThreadCountForScope(BarrierScope scope)
    {
        // In a real GPU implementation, these would be actual thread counts
        // For simulation, we use representative values
        return scope switch
        {
            BarrierScope.ThreadBlock => 256,    // Typical warp size × warps per block
            BarrierScope.Device => 2048,        // Typical SM count × threads per SM
            BarrierScope.Grid => 65536,         // Maximum grid size
            BarrierScope.System => 131072,      // Multi-GPU configuration
            _ => 1
        };
    }

    /// <summary>
    /// Raises the BarrierCompleted event.
    /// </summary>
    private void OnBarrierCompleted(BarrierEventArgs e)
    {
        BarrierCompleted?.Invoke(this, e);
    }

    /// <summary>
    /// Raises the BarrierTimedOut event.
    /// </summary>
    private void OnBarrierTimedOut(BarrierEventArgs e)
    {
        BarrierTimedOut?.Invoke(this, e);
    }

    /// <summary>
    /// Resets statistics counters.
    /// </summary>
    public void ResetStatistics()
    {
        _totalBarriersCreated = 0;
        _successfulSyncs = 0;
        _timeouts = 0;
        _totalSyncTimeUs = 0;
        _maxSyncTimeUs = 0;
        _totalThreadsPerBarrier = 0;
    }

    /// <inheritdoc/>
    public override string ToString()
    {
        return $"TemporalBarrierManager(Active={_activeBarriers.Count}, " +
               $"Created={_totalBarriersCreated}, Syncs={_successfulSyncs})";
    }
}

/// <summary>
/// Internal implementation of <see cref="ITemporalBarrier"/>.
/// </summary>
internal sealed class TemporalBarrierImpl : ITemporalBarrier
{
    private int _arrivedThreadCount;
    private bool _isComplete;
    private bool _timedOut;
    private bool _disposed;

    /// <inheritdoc/>
    public Guid BarrierId { get; }

    /// <inheritdoc/>
    public BarrierScope Scope { get; }

    /// <inheritdoc/>
    public int ExpectedThreadCount { get; }

    /// <inheritdoc/>
    public int ArrivedThreadCount => _arrivedThreadCount;

    /// <inheritdoc/>
    public bool IsComplete => _isComplete;

    /// <summary>
    /// Gets whether the barrier timed out.
    /// </summary>
    public bool TimedOut => _timedOut;

    /// <inheritdoc/>
    public int TimeoutMs { get; }

    /// <summary>
    /// Initializes a new instance of the <see cref="TemporalBarrierImpl"/> class.
    /// </summary>
    public TemporalBarrierImpl(
        Guid barrierId,
        BarrierScope scope,
        int timeoutMs,
        int expectedThreadCount)
    {
        BarrierId = barrierId;
        Scope = scope;
        TimeoutMs = timeoutMs;
        ExpectedThreadCount = expectedThreadCount;
    }

    /// <summary>
    /// Increments the arrived thread count.
    /// </summary>
    /// <returns>The new arrived count.</returns>
    public int IncrementArrivedCount()
    {
        return Interlocked.Increment(ref _arrivedThreadCount);
    }

    /// <summary>
    /// Marks the barrier as complete.
    /// </summary>
    public void MarkComplete()
    {
        _isComplete = true;
    }

    /// <summary>
    /// Marks the barrier as timed out.
    /// </summary>
    public void MarkTimedOut()
    {
        _timedOut = true;
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        _disposed = true;
    }

    /// <inheritdoc/>
    public override string ToString()
    {
        var status = _isComplete ? "Complete" : (_timedOut ? "TimedOut" : "Active");
        return $"TemporalBarrier({BarrierId}, {Scope}, {status}, " +
               $"Arrived={_arrivedThreadCount}/{ExpectedThreadCount})";
    }
}

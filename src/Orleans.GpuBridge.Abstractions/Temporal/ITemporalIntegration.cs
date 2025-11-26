// Orleans.GpuBridge - GPU-native distributed computing for Microsoft Orleans
// Copyright (c) 2025 Michael Ivertowski

using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;

namespace Orleans.GpuBridge.Abstractions.Temporal;

/// <summary>
/// Provides GPU temporal integration capabilities for Orleans actors.
/// </summary>
/// <remarks>
/// <para>
/// This interface abstracts over DotCompute's temporal features:
/// <list type="bullet">
/// <item><description>GPU-side timestamp injection via [Kernel] attributes</description></item>
/// <item><description>Device-wide barriers via Cooperative Groups</description></item>
/// <item><description>Memory ordering via fence primitives</description></item>
/// </list>
/// </para>
/// <para>
/// Requires DotCompute 0.4.2-rc2 or later for full functionality.
/// </para>
/// </remarks>
public interface ITemporalIntegration
{
    /// <summary>
    /// Enables GPU-side timestamps for temporal actors.
    /// </summary>
    /// <param name="options">Configuration options for temporal kernels.</param>
    /// <param name="ct">Cancellation token.</param>
    /// <returns>A task that completes when configuration is applied.</returns>
    Task ConfigureTemporalKernelAsync(TemporalKernelOptions options, CancellationToken ct = default);

    /// <summary>
    /// Calibrates GPU clock against CPU time.
    /// </summary>
    /// <param name="sampleCount">Number of samples for calibration.</param>
    /// <param name="ct">Cancellation token.</param>
    /// <returns>Calibration result with offset and drift.</returns>
    Task<ClockCalibration> CalibrateGpuClockAsync(int sampleCount = 1000, CancellationToken ct = default);

    /// <summary>
    /// Gets the current GPU timestamp in nanoseconds.
    /// </summary>
    /// <param name="ct">Cancellation token.</param>
    /// <returns>GPU timestamp in nanoseconds since device initialization.</returns>
    Task<long> GetGpuTimestampAsync(CancellationToken ct = default);

    /// <summary>
    /// Gets the current calibration status.
    /// </summary>
    ClockCalibration? CurrentCalibration { get; }

    /// <summary>
    /// Gets the underlying timing provider.
    /// </summary>
    IGpuTimingProvider TimingProvider { get; }

    /// <summary>
    /// Gets whether GPU timing is available.
    /// </summary>
    bool IsGpuTimingAvailable { get; }

    /// <summary>
    /// Gets whether device-wide barriers are supported.
    /// </summary>
    bool AreBarriersSupported { get; }

    /// <summary>
    /// Gets whether memory ordering primitives are supported.
    /// </summary>
    bool IsMemoryOrderingSupported { get; }
}

/// <summary>
/// Configuration options for temporal kernel execution.
/// </summary>
public sealed class TemporalKernelOptions
{
    /// <summary>
    /// Enable automatic timestamp injection at kernel entry.
    /// </summary>
    public bool EnableTimestamps { get; set; } = true;

    /// <summary>
    /// Enable device-wide barriers for synchronization.
    /// </summary>
    public bool EnableBarriers { get; set; } = false;

    /// <summary>
    /// Barrier scope (Device, System, Grid).
    /// </summary>
    public BarrierScope BarrierScope { get; set; } = BarrierScope.Device;

    /// <summary>
    /// Memory ordering mode for causal correctness.
    /// </summary>
    public MemoryOrderingMode MemoryOrdering { get; set; } = MemoryOrderingMode.ReleaseAcquire;

    /// <summary>
    /// Maximum timeout for barrier operations in milliseconds.
    /// </summary>
    public int BarrierTimeoutMs { get; set; } = 5000;

    /// <summary>
    /// Enable fence insertion after temporal operations.
    /// </summary>
    public bool EnableFences { get; set; } = true;

    /// <summary>
    /// Fence scope for memory ordering.
    /// </summary>
    public FenceScope FenceScope { get; set; } = FenceScope.Device;
}

/// <summary>
/// Scope for device-wide barriers.
/// </summary>
public enum BarrierScope
{
    /// <summary>
    /// Thread block scope (intra-SM).
    /// </summary>
    ThreadBlock,

    /// <summary>
    /// Device scope (all SMs on one GPU).
    /// </summary>
    Device,

    /// <summary>
    /// Grid scope (entire kernel grid).
    /// </summary>
    Grid,

    /// <summary>
    /// System scope (multi-GPU, requires NVLINK or PCIe).
    /// </summary>
    System
}

/// <summary>
/// Memory ordering mode for GPU operations.
/// </summary>
public enum MemoryOrderingMode
{
    /// <summary>
    /// Relaxed ordering - no memory ordering guarantees.
    /// </summary>
    Relaxed,

    /// <summary>
    /// Acquire semantics - ensures all prior writes are visible.
    /// </summary>
    Acquire,

    /// <summary>
    /// Release semantics - ensures all prior writes are committed.
    /// </summary>
    Release,

    /// <summary>
    /// Acquire-release semantics - combines acquire and release.
    /// </summary>
    ReleaseAcquire,

    /// <summary>
    /// Sequentially consistent - strongest ordering guarantee.
    /// </summary>
    SequentiallyConsistent
}

/// <summary>
/// Scope for memory fence operations.
/// </summary>
public enum FenceScope
{
    /// <summary>
    /// Thread block fence (__threadfence_block).
    /// </summary>
    ThreadBlock,

    /// <summary>
    /// Device fence (__threadfence).
    /// </summary>
    Device,

    /// <summary>
    /// System fence (__threadfence_system).
    /// </summary>
    System
}

/// <summary>
/// Interface for managing GPU barriers in temporal kernels.
/// </summary>
public interface ITemporalBarrierManager
{
    /// <summary>
    /// Creates a device-wide barrier for synchronization.
    /// </summary>
    /// <param name="scope">Barrier scope.</param>
    /// <param name="timeoutMs">Timeout in milliseconds.</param>
    /// <returns>Barrier instance.</returns>
    ITemporalBarrier CreateBarrier(BarrierScope scope, int timeoutMs = 5000);

    /// <summary>
    /// Executes a device-wide barrier synchronization.
    /// </summary>
    /// <param name="barrier">Barrier to execute.</param>
    /// <param name="ct">Cancellation token.</param>
    /// <returns>A task that completes when all threads reach the barrier.</returns>
    Task SynchronizeAsync(ITemporalBarrier barrier, CancellationToken ct = default);

    /// <summary>
    /// Gets barrier statistics.
    /// </summary>
    BarrierStatistics GetStatistics();
}

/// <summary>
/// Represents a temporal barrier for GPU synchronization.
/// </summary>
public interface ITemporalBarrier : IDisposable
{
    /// <summary>
    /// Unique barrier identifier.
    /// </summary>
    Guid BarrierId { get; }

    /// <summary>
    /// Barrier scope.
    /// </summary>
    BarrierScope Scope { get; }

    /// <summary>
    /// Number of threads expected at this barrier.
    /// </summary>
    int ExpectedThreadCount { get; }

    /// <summary>
    /// Number of threads that have arrived at the barrier.
    /// </summary>
    int ArrivedThreadCount { get; }

    /// <summary>
    /// Whether the barrier has been reached by all threads.
    /// </summary>
    bool IsComplete { get; }

    /// <summary>
    /// Timeout in milliseconds.
    /// </summary>
    int TimeoutMs { get; }
}

/// <summary>
/// Statistics about barrier operations.
/// </summary>
public sealed record BarrierStatistics
{
    /// <summary>
    /// Total barriers created.
    /// </summary>
    public long TotalBarriersCreated { get; init; }

    /// <summary>
    /// Total successful synchronizations.
    /// </summary>
    public long SuccessfulSyncs { get; init; }

    /// <summary>
    /// Total timeouts.
    /// </summary>
    public long Timeouts { get; init; }

    /// <summary>
    /// Average sync time in microseconds.
    /// </summary>
    public double AverageSyncTimeUs { get; init; }

    /// <summary>
    /// Maximum sync time in microseconds.
    /// </summary>
    public double MaxSyncTimeUs { get; init; }

    /// <summary>
    /// Average threads per barrier.
    /// </summary>
    public double AverageThreadsPerBarrier { get; init; }

    /// <summary>
    /// Timeout rate.
    /// </summary>
    public double TimeoutRate => TotalBarriersCreated > 0
        ? (double)Timeouts / TotalBarriersCreated
        : 0;
}

/// <summary>
/// Interface for causal memory ordering operations.
/// </summary>
public interface ICausalMemoryOrdering
{
    /// <summary>
    /// Inserts a memory fence with the specified scope.
    /// </summary>
    /// <param name="scope">Fence scope.</param>
    /// <returns>Fence identifier for tracking.</returns>
    long InsertFence(FenceScope scope);

    /// <summary>
    /// Gets the current memory ordering mode.
    /// </summary>
    MemoryOrderingMode CurrentMode { get; }

    /// <summary>
    /// Sets the memory ordering mode for subsequent operations.
    /// </summary>
    /// <param name="mode">Memory ordering mode.</param>
    void SetMode(MemoryOrderingMode mode);

    /// <summary>
    /// Ensures all prior stores are visible to subsequent loads.
    /// </summary>
    void AcquireSemantics();

    /// <summary>
    /// Ensures all prior stores are committed before subsequent operations.
    /// </summary>
    void ReleaseSemantics();

    /// <summary>
    /// Gets memory ordering statistics.
    /// </summary>
    MemoryOrderingStatistics GetStatistics();
}

/// <summary>
/// Statistics about memory ordering operations.
/// </summary>
public sealed record MemoryOrderingStatistics
{
    /// <summary>
    /// Total fences inserted.
    /// </summary>
    public long TotalFencesInserted { get; init; }

    /// <summary>
    /// Fences by scope.
    /// </summary>
    public IReadOnlyDictionary<FenceScope, long> FencesByScope { get; init; } =
        new Dictionary<FenceScope, long>();

    /// <summary>
    /// Current memory ordering mode.
    /// </summary>
    public MemoryOrderingMode CurrentMode { get; init; }

    /// <summary>
    /// Average fence overhead in nanoseconds.
    /// </summary>
    public double AverageFenceOverheadNanos { get; init; }

    /// <summary>
    /// Total acquire operations.
    /// </summary>
    public long TotalAcquires { get; init; }

    /// <summary>
    /// Total release operations.
    /// </summary>
    public long TotalReleases { get; init; }
}

/// <summary>
/// Event arguments for temporal barrier events.
/// </summary>
public sealed class BarrierEventArgs : EventArgs
{
    /// <summary>
    /// The barrier that triggered the event.
    /// </summary>
    public required ITemporalBarrier Barrier { get; init; }

    /// <summary>
    /// Timestamp when the event occurred.
    /// </summary>
    public required long TimestampNanos { get; init; }

    /// <summary>
    /// Whether the barrier completed successfully.
    /// </summary>
    public required bool Success { get; init; }

    /// <summary>
    /// Error message if the barrier failed.
    /// </summary>
    public string? ErrorMessage { get; init; }
}

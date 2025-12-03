// Orleans.GpuBridge - GPU-native distributed computing for Microsoft Orleans
// Copyright (c) 2025 Michael Ivertowski

namespace Orleans.GpuBridge.Diagnostics.Interfaces;

/// <summary>
/// Interface for per-grain GPU memory telemetry collection and monitoring.
/// This service provides detailed tracking of GPU memory allocations at the grain level,
/// enabling fine-grained resource monitoring for enterprise observability dashboards.
/// </summary>
/// <remarks>
/// <para>
/// This interface complements <see cref="IGpuTelemetry"/> by providing grain-level granularity:
/// </para>
/// <list type="bullet">
/// <item><description>Per-grain memory allocation tracking</description></item>
/// <item><description>Memory usage aggregation by grain type</description></item>
/// <item><description>Memory pool utilization metrics</description></item>
/// <item><description>Real-time memory event streaming</description></item>
/// </list>
/// </remarks>
public interface IGpuMemoryTelemetryProvider
{
    /// <summary>
    /// Records a GPU memory allocation for a specific grain.
    /// </summary>
    /// <param name="grainType">The type name of the grain (e.g., "OrderMatchingGrain").</param>
    /// <param name="grainId">The unique identifier of the grain instance.</param>
    /// <param name="deviceIndex">The GPU device index where memory was allocated.</param>
    /// <param name="bytes">The number of bytes allocated.</param>
    void RecordGrainMemoryAllocation(
        string grainType,
        string grainId,
        int deviceIndex,
        long bytes);

    /// <summary>
    /// Records the release of GPU memory for a specific grain.
    /// </summary>
    /// <param name="grainType">The type name of the grain.</param>
    /// <param name="grainId">The unique identifier of the grain instance.</param>
    /// <param name="deviceIndex">The GPU device index where memory was released.</param>
    /// <param name="bytes">The number of bytes released.</param>
    void RecordGrainMemoryRelease(
        string grainType,
        string grainId,
        int deviceIndex,
        long bytes);

    /// <summary>
    /// Gets a snapshot of memory usage for a specific grain.
    /// </summary>
    /// <param name="grainType">The type name of the grain.</param>
    /// <param name="grainId">The unique identifier of the grain instance.</param>
    /// <returns>A snapshot of the grain's memory usage, or null if not found.</returns>
    GrainMemorySnapshot? GetGrainMemorySnapshot(string grainType, string grainId);

    /// <summary>
    /// Gets aggregated memory statistics for all grains of a specific type.
    /// </summary>
    /// <param name="grainType">The type name of the grain.</param>
    /// <returns>Aggregated memory statistics for the grain type, or null if no grains found.</returns>
    GrainTypeMemoryStats? GetMemoryStatsByGrainType(string grainType);

    /// <summary>
    /// Gets memory statistics for all tracked grain types.
    /// </summary>
    /// <returns>A dictionary mapping grain type names to their memory statistics.</returns>
    IReadOnlyDictionary<string, GrainTypeMemoryStats> GetAllGrainTypeMemoryStats();

    /// <summary>
    /// Gets the total GPU memory allocated across all grains.
    /// </summary>
    /// <returns>Total allocated memory in bytes.</returns>
    long GetTotalAllocatedMemory();

    /// <summary>
    /// Gets the number of active grains with GPU memory allocations.
    /// </summary>
    /// <returns>The count of active GPU grains.</returns>
    int GetActiveGrainCount();

    /// <summary>
    /// Records memory pool statistics for monitoring pool utilization.
    /// </summary>
    /// <param name="deviceIndex">The GPU device index.</param>
    /// <param name="poolUsedBytes">Bytes currently used from the pool.</param>
    /// <param name="poolTotalBytes">Total pool capacity in bytes.</param>
    /// <param name="fragmentationPercent">Fragmentation percentage (0-100).</param>
    void RecordMemoryPoolStats(
        int deviceIndex,
        long poolUsedBytes,
        long poolTotalBytes,
        int fragmentationPercent);

    /// <summary>
    /// Gets the current memory pool statistics for a device.
    /// </summary>
    /// <param name="deviceIndex">The GPU device index.</param>
    /// <returns>Memory pool statistics, or null if not available.</returns>
    MemoryPoolStats? GetMemoryPoolStats(int deviceIndex);

    /// <summary>
    /// Streams GPU memory events asynchronously.
    /// </summary>
    /// <param name="cancellationToken">Cancellation token to stop streaming.</param>
    /// <returns>An async enumerable of GPU memory events.</returns>
    IAsyncEnumerable<GpuMemoryEvent> StreamEventsAsync(CancellationToken cancellationToken = default);
}

/// <summary>
/// Represents a point-in-time snapshot of a grain's GPU memory usage.
/// </summary>
/// <param name="GrainType">The type name of the grain.</param>
/// <param name="GrainId">The unique identifier of the grain instance.</param>
/// <param name="DeviceIndex">The GPU device index where memory is allocated.</param>
/// <param name="AllocatedBytes">Currently allocated memory in bytes.</param>
/// <param name="PeakBytes">Peak memory allocation since grain activation.</param>
/// <param name="AllocationTime">When the memory was first allocated.</param>
/// <param name="LastAccessTime">When the memory was last accessed.</param>
public sealed record GrainMemorySnapshot(
    string GrainType,
    string GrainId,
    int DeviceIndex,
    long AllocatedBytes,
    long PeakBytes,
    DateTimeOffset AllocationTime,
    DateTimeOffset LastAccessTime);

/// <summary>
/// Aggregated memory statistics for all grains of a specific type.
/// </summary>
/// <param name="GrainType">The type name of the grain.</param>
/// <param name="GrainCount">Number of active grains of this type.</param>
/// <param name="TotalAllocatedBytes">Total memory allocated across all grains.</param>
/// <param name="AveragePerGrain">Average memory per grain in bytes.</param>
/// <param name="PeakPerGrain">Maximum memory allocated by any single grain.</param>
/// <param name="MinPerGrain">Minimum memory allocated by any single grain.</param>
public sealed record GrainTypeMemoryStats(
    string GrainType,
    int GrainCount,
    long TotalAllocatedBytes,
    long AveragePerGrain,
    long PeakPerGrain,
    long MinPerGrain);

/// <summary>
/// Memory pool statistics for a GPU device.
/// </summary>
/// <param name="DeviceIndex">The GPU device index.</param>
/// <param name="PoolUsedBytes">Bytes currently used from the pool.</param>
/// <param name="PoolTotalBytes">Total pool capacity in bytes.</param>
/// <param name="PoolAvailableBytes">Available bytes in the pool.</param>
/// <param name="UtilizationPercent">Pool utilization percentage (0-100).</param>
/// <param name="FragmentationPercent">Fragmentation percentage (0-100).</param>
/// <param name="AllocationCount">Number of allocations from this pool.</param>
/// <param name="LastUpdated">When these statistics were last updated.</param>
public sealed record MemoryPoolStats(
    int DeviceIndex,
    long PoolUsedBytes,
    long PoolTotalBytes,
    long PoolAvailableBytes,
    double UtilizationPercent,
    int FragmentationPercent,
    int AllocationCount,
    DateTimeOffset LastUpdated);

/// <summary>
/// Represents a GPU memory event for real-time monitoring.
/// </summary>
/// <param name="EventType">The type of memory event.</param>
/// <param name="GrainType">The grain type involved in the event.</param>
/// <param name="GrainId">The grain instance involved in the event.</param>
/// <param name="DeviceIndex">The GPU device index.</param>
/// <param name="Bytes">The number of bytes involved.</param>
/// <param name="Timestamp">When the event occurred.</param>
public sealed record GpuMemoryEvent(
    GpuMemoryEventType EventType,
    string GrainType,
    string GrainId,
    int DeviceIndex,
    long Bytes,
    DateTimeOffset Timestamp);

/// <summary>
/// Types of GPU memory events.
/// </summary>
public enum GpuMemoryEventType
{
    /// <summary>Memory was allocated for a grain.</summary>
    Allocated,

    /// <summary>Memory was released by a grain.</summary>
    Released,

    /// <summary>Memory pool statistics were updated.</summary>
    PoolStatsUpdated,

    /// <summary>Memory allocation failed.</summary>
    AllocationFailed,

    /// <summary>Memory pool threshold exceeded.</summary>
    PoolThresholdExceeded
}

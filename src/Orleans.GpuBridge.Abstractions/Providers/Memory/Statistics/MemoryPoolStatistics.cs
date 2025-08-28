using System.Collections.Generic;

namespace Orleans.GpuBridge.Abstractions.Providers.Memory.Statistics;

/// <summary>
/// Memory pool statistics
/// </summary>
public sealed record MemoryPoolStatistics(
    long TotalBytesAllocated,
    long TotalBytesInUse,
    long TotalBytesFree,
    int AllocationCount,
    int FreeBlockCount,
    long LargestFreeBlock,
    double FragmentationPercent,
    long PeakUsageBytes,
    IReadOnlyDictionary<string, object>? ExtendedStats = null);
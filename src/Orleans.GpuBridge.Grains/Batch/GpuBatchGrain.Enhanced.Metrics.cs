using System;
using Orleans.GpuBridge.Abstractions;
using Orleans;

namespace Orleans.GpuBridge.Grains.Batch;

/// <summary>
/// Comprehensive performance metrics for GPU batch execution
/// </summary>
[GenerateSerializer]
public sealed record GpuBatchMetrics(
    [property: Id(0)] int TotalItems,
    [property: Id(1)] int SubBatchCount,
    [property: Id(2)] int SuccessfulSubBatches,
    [property: Id(3)] TimeSpan TotalExecutionTime,
    [property: Id(4)] TimeSpan KernelExecutionTime,
    [property: Id(5)] TimeSpan MemoryTransferTime,
    [property: Id(6)] double Throughput,
    [property: Id(7)] long MemoryAllocated,
    [property: Id(8)] string DeviceType,
    [property: Id(9)] string DeviceName)
{
    /// <summary>
    /// Percentage of time spent in actual kernel execution (vs memory transfer)
    /// </summary>
    public double KernelEfficiency =>
        (KernelExecutionTime.TotalMilliseconds / TotalExecutionTime.TotalMilliseconds) * 100;

    /// <summary>
    /// Items processed per millisecond
    /// </summary>
    public double ItemsPerMillisecond =>
        TotalItems / TotalExecutionTime.TotalMilliseconds;

    /// <summary>
    /// Memory bandwidth in MB/s
    /// </summary>
    public double MemoryBandwidthMBps =>
        (MemoryAllocated / (1024.0 * 1024.0)) / TotalExecutionTime.TotalSeconds;
}

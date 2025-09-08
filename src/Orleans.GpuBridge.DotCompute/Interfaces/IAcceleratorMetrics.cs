using System;
using System.Collections.Generic;

namespace Orleans.GpuBridge.DotCompute.Interfaces;

/// <summary>
/// Provides metrics and performance data for an accelerator
/// </summary>
public interface IAcceleratorMetrics
{
    /// <summary>
    /// Gets the total number of kernels executed
    /// </summary>
    long TotalKernelsExecuted { get; }
    
    /// <summary>
    /// Gets the total execution time in milliseconds
    /// </summary>
    double TotalExecutionTimeMs { get; }
    
    /// <summary>
    /// Gets the average kernel execution time in milliseconds
    /// </summary>
    double AverageKernelExecutionTimeMs { get; }
    
    /// <summary>
    /// Gets the total memory allocated in bytes
    /// </summary>
    long TotalMemoryAllocatedBytes { get; }
    
    /// <summary>
    /// Gets the total memory freed in bytes
    /// </summary>
    long TotalMemoryFreedBytes { get; }
    
    /// <summary>
    /// Gets the current memory usage in bytes
    /// </summary>
    long CurrentMemoryUsageBytes { get; }
    
    /// <summary>
    /// Gets the peak memory usage in bytes
    /// </summary>
    long PeakMemoryUsageBytes { get; }
    
    /// <summary>
    /// Gets the number of memory allocations
    /// </summary>
    long MemoryAllocationCount { get; }
    
    /// <summary>
    /// Gets the number of memory deallocations
    /// </summary>
    long MemoryDeallocationCount { get; }
    
    /// <summary>
    /// Gets the total data transferred to the device in bytes
    /// </summary>
    long TotalDataTransferredToDeviceBytes { get; }
    
    /// <summary>
    /// Gets the total data transferred from the device in bytes
    /// </summary>
    long TotalDataTransferredFromDeviceBytes { get; }
    
    /// <summary>
    /// Gets the average data transfer rate to the device in MB/s
    /// </summary>
    double AverageTransferRateToDeviceMBps { get; }
    
    /// <summary>
    /// Gets the average data transfer rate from the device in MB/s
    /// </summary>
    double AverageTransferRateFromDeviceMBps { get; }
    
    /// <summary>
    /// Gets kernel-specific metrics
    /// </summary>
    IReadOnlyDictionary<string, KernelMetrics> KernelMetrics { get; }
    
    /// <summary>
    /// Gets the timestamp when metrics collection started
    /// </summary>
    DateTime MetricsStartTime { get; }
    
    /// <summary>
    /// Gets the timestamp of the last metric update
    /// </summary>
    DateTime LastUpdateTime { get; }
    
    /// <summary>
    /// Resets all metrics
    /// </summary>
    void Reset();
    
    /// <summary>
    /// Gets a snapshot of current metrics
    /// </summary>
    /// <returns>A snapshot of the current metrics</returns>
    IAcceleratorMetrics GetSnapshot();
}

/// <summary>
/// Metrics for a specific kernel
/// </summary>
[GenerateSerializer]
[Immutable]
public sealed record KernelMetrics
{
    /// <summary>
    /// Gets the kernel name
    /// </summary>
    [Id(0)]
    public string KernelName { get; init; } = string.Empty;
    
    /// <summary>
    /// Gets the number of times this kernel was executed
    /// </summary>
    [Id(1)]
    public long ExecutionCount { get; init; }
    
    /// <summary>
    /// Gets the total execution time in milliseconds
    /// </summary>
    [Id(2)]
    public double TotalExecutionTimeMs { get; init; }
    
    /// <summary>
    /// Gets the average execution time in milliseconds
    /// </summary>
    [Id(3)]
    public double AverageExecutionTimeMs { get; init; }
    
    /// <summary>
    /// Gets the minimum execution time in milliseconds
    /// </summary>
    [Id(4)]
    public double MinExecutionTimeMs { get; init; }
    
    /// <summary>
    /// Gets the maximum execution time in milliseconds
    /// </summary>
    [Id(5)]
    public double MaxExecutionTimeMs { get; init; }
    
    /// <summary>
    /// Gets the standard deviation of execution times
    /// </summary>
    [Id(6)]
    public double StandardDeviationMs { get; init; }
    
    /// <summary>
    /// Gets the last execution time in milliseconds
    /// </summary>
    [Id(7)]
    public double LastExecutionTimeMs { get; init; }
    
    /// <summary>
    /// Gets the timestamp of the last execution
    /// </summary>
    [Id(8)]
    public DateTime LastExecutionTime { get; init; }
    
    /// <summary>
    /// Gets the average occupancy percentage
    /// </summary>
    [Id(9)]
    public double AverageOccupancyPercentage { get; init; }
    
    /// <summary>
    /// Gets the average memory throughput in GB/s
    /// </summary>
    [Id(10)]
    public double AverageMemoryThroughputGBps { get; init; }
}
using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using Orleans.GpuBridge.Abstractions.Models;

namespace Orleans.GpuBridge.Abstractions.Metrics;

/// <summary>
/// Interface for collecting GPU metrics and performance data
/// </summary>
public interface IGpuMetricsCollector
{
    /// <summary>
    /// Gets memory information for a specific GPU device
    /// </summary>
    /// <param name="deviceIndex">The index of the GPU device</param>
    /// <param name="cancellationToken">Cancellation token</param>
    /// <returns>GPU memory information</returns>
    Task<GpuMemoryInfo> GetMemoryInfoAsync(int deviceIndex = 0, CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Gets memory information for all available GPU devices
    /// </summary>
    /// <param name="cancellationToken">Cancellation token</param>
    /// <returns>Collection of GPU memory information for all devices</returns>
    Task<IReadOnlyList<GpuMemoryInfo>> GetAllMemoryInfoAsync(CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Gets GPU utilization metrics
    /// </summary>
    /// <param name="deviceIndex">The index of the GPU device</param>
    /// <param name="cancellationToken">Cancellation token</param>
    /// <returns>GPU utilization metrics</returns>
    Task<GpuUtilizationMetrics> GetUtilizationAsync(int deviceIndex = 0, CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Gets performance counters for the GPU
    /// </summary>
    /// <param name="deviceIndex">The index of the GPU device</param>
    /// <param name="cancellationToken">Cancellation token</param>
    /// <returns>Dictionary of performance counter names and values</returns>
    Task<IReadOnlyDictionary<string, double>> GetPerformanceCountersAsync(
        int deviceIndex = 0, 
        CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Gets temperature information for the GPU
    /// </summary>
    /// <param name="deviceIndex">The index of the GPU device</param>
    /// <param name="cancellationToken">Cancellation token</param>
    /// <returns>Temperature in Celsius</returns>
    Task<double> GetTemperatureAsync(int deviceIndex = 0, CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Gets power consumption information for the GPU
    /// </summary>
    /// <param name="deviceIndex">The index of the GPU device</param>
    /// <param name="cancellationToken">Cancellation token</param>
    /// <returns>Power consumption in watts</returns>
    Task<double> GetPowerConsumptionAsync(int deviceIndex = 0, CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Gets clock speeds for the GPU
    /// </summary>
    /// <param name="deviceIndex">The index of the GPU device</param>
    /// <param name="cancellationToken">Cancellation token</param>
    /// <returns>Clock speed information</returns>
    Task<GpuClockSpeeds> GetClockSpeedsAsync(int deviceIndex = 0, CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Starts continuous metric collection
    /// </summary>
    /// <param name="interval">Collection interval</param>
    /// <param name="cancellationToken">Cancellation token</param>
    Task StartCollectionAsync(TimeSpan interval, CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Stops continuous metric collection
    /// </summary>
    Task StopCollectionAsync();
    
    /// <summary>
    /// Gets aggregated metrics over a time period
    /// </summary>
    /// <param name="duration">Duration to aggregate over</param>
    /// <param name="deviceIndex">The index of the GPU device</param>
    /// <param name="cancellationToken">Cancellation token</param>
    /// <returns>Aggregated metrics</returns>
    Task<AggregatedGpuMetrics> GetAggregatedMetricsAsync(
        TimeSpan duration,
        int deviceIndex = 0,
        CancellationToken cancellationToken = default);
}

/// <summary>
/// GPU utilization metrics
/// </summary>
[GenerateSerializer]
[Immutable]
public sealed record GpuUtilizationMetrics
{
    /// <summary>
    /// Gets the GPU core utilization percentage (0-100)
    /// </summary>
    [Id(0)]
    public double GpuUtilizationPercentage { get; init; }
    
    /// <summary>
    /// Gets the memory controller utilization percentage (0-100)
    /// </summary>
    [Id(1)]
    public double MemoryUtilizationPercentage { get; init; }
    
    /// <summary>
    /// Gets the encoder utilization percentage (0-100)
    /// </summary>
    [Id(2)]
    public double EncoderUtilizationPercentage { get; init; }
    
    /// <summary>
    /// Gets the decoder utilization percentage (0-100)
    /// </summary>
    [Id(3)]
    public double DecoderUtilizationPercentage { get; init; }
    
    /// <summary>
    /// Gets the timestamp when these metrics were captured
    /// </summary>
    [Id(4)]
    public DateTime Timestamp { get; init; }
}

/// <summary>
/// GPU clock speed information
/// </summary>
[GenerateSerializer]
[Immutable]
public sealed record GpuClockSpeeds
{
    /// <summary>
    /// Gets the current graphics clock speed in MHz
    /// </summary>
    [Id(0)]
    public int GraphicsClockMHz { get; init; }
    
    /// <summary>
    /// Gets the current memory clock speed in MHz
    /// </summary>
    [Id(1)]
    public int MemoryClockMHz { get; init; }
    
    /// <summary>
    /// Gets the maximum graphics clock speed in MHz
    /// </summary>
    [Id(2)]
    public int MaxGraphicsClockMHz { get; init; }
    
    /// <summary>
    /// Gets the maximum memory clock speed in MHz
    /// </summary>
    [Id(3)]
    public int MaxMemoryClockMHz { get; init; }
    
    /// <summary>
    /// Gets the timestamp when these speeds were captured
    /// </summary>
    [Id(4)]
    public DateTime Timestamp { get; init; }
}

/// <summary>
/// Aggregated GPU metrics over a time period
/// </summary>
[GenerateSerializer]
[Immutable]
public sealed record AggregatedGpuMetrics
{
    /// <summary>
    /// Gets the average GPU utilization percentage
    /// </summary>
    [Id(0)]
    public double AverageGpuUtilization { get; init; }
    
    /// <summary>
    /// Gets the peak GPU utilization percentage
    /// </summary>
    [Id(1)]
    public double PeakGpuUtilization { get; init; }
    
    /// <summary>
    /// Gets the average memory utilization percentage
    /// </summary>
    [Id(2)]
    public double AverageMemoryUtilization { get; init; }
    
    /// <summary>
    /// Gets the peak memory utilization percentage
    /// </summary>
    [Id(3)]
    public double PeakMemoryUtilization { get; init; }
    
    /// <summary>
    /// Gets the average temperature in Celsius
    /// </summary>
    [Id(4)]
    public double AverageTemperature { get; init; }
    
    /// <summary>
    /// Gets the peak temperature in Celsius
    /// </summary>
    [Id(5)]
    public double PeakTemperature { get; init; }
    
    /// <summary>
    /// Gets the average power consumption in watts
    /// </summary>
    [Id(6)]
    public double AveragePowerConsumption { get; init; }
    
    /// <summary>
    /// Gets the total energy consumed in watt-hours
    /// </summary>
    [Id(7)]
    public double TotalEnergyConsumed { get; init; }
    
    /// <summary>
    /// Gets the number of samples collected
    /// </summary>
    [Id(8)]
    public int SampleCount { get; init; }
    
    /// <summary>
    /// Gets the duration of the aggregation period
    /// </summary>
    [Id(9)]
    public TimeSpan Duration { get; init; }
    
    /// <summary>
    /// Gets the start time of the aggregation period
    /// </summary>
    [Id(10)]
    public DateTime StartTime { get; init; }
    
    /// <summary>
    /// Gets the end time of the aggregation period
    /// </summary>
    [Id(11)]
    public DateTime EndTime { get; init; }
}
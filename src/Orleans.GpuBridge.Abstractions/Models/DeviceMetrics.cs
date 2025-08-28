using System;
using System.Collections.Generic;

namespace Orleans.GpuBridge.Abstractions.Models;

/// <summary>
/// Represents current device metrics and utilization statistics.
/// This record provides a snapshot of device performance characteristics,
/// resource utilization, and operational statistics that can be used for
/// monitoring, load balancing, and performance optimization decisions.
/// </summary>
/// <param name="GpuUtilizationPercent">
/// The current GPU utilization as a percentage (0.0 to 100.0).
/// This represents how busy the compute units are with active work.
/// High values indicate the device is actively processing computations.
/// </param>
/// <param name="MemoryUtilizationPercent">
/// The current memory utilization as a percentage (0.0 to 100.0).
/// This represents the percentage of device memory that is currently allocated.
/// Values approaching 100% may indicate memory pressure and potential
/// performance degradation due to memory management overhead.
/// </param>
/// <param name="UsedMemoryBytes">
/// The amount of device memory currently in use, measured in bytes.
/// This represents the absolute amount of memory allocated for kernels,
/// buffers, and other device resources.
/// </param>
/// <param name="TemperatureCelsius">
/// The current device temperature in degrees Celsius.
/// Higher temperatures may indicate heavy utilization and can affect
/// device performance through thermal throttling. Monitoring temperature
/// is important for maintaining optimal performance and device longevity.
/// </param>
/// <param name="PowerWatts">
/// The current power consumption of the device in watts.
/// This metric is useful for understanding energy usage and can be
/// important in power-constrained environments or for cost optimization
/// in cloud deployments.
/// </param>
/// <param name="FanSpeedPercent">
/// The current fan speed as a percentage of maximum (0 to 100).
/// Higher fan speeds typically indicate higher temperatures or power draw.
/// This metric is primarily useful for diagnostic purposes and understanding
/// device thermal management behavior.
/// </param>
/// <param name="KernelsExecuted">
/// The total number of kernels executed on this device since initialization.
/// This cumulative counter provides insight into device activity levels
/// and can be useful for load balancing decisions and performance analysis.
/// </param>
/// <param name="BytesTransferred">
/// The total number of bytes transferred to/from device memory since initialization.
/// This cumulative counter includes all memory operations and can help identify
/// memory-intensive workloads and potential bottlenecks in data movement.
/// </param>
/// <param name="Uptime">
/// The duration since the device was initialized or last reset.
/// This provides context for understanding the cumulative metrics and
/// can be useful for calculating rates (e.g., kernels per second).
/// </param>
/// <param name="ExtendedMetrics">
/// Backend-specific extended metrics as key-value pairs.
/// This allows backend providers to expose additional device metrics
/// that may not be covered by the standard properties. Examples might include
/// specific hardware counters, vendor-specific performance indicators,
/// or specialized diagnostic information.
/// </param>
public sealed record DeviceMetrics(
    double GpuUtilizationPercent,
    double MemoryUtilizationPercent,
    long UsedMemoryBytes,
    double TemperatureCelsius,
    double PowerWatts,
    int FanSpeedPercent,
    long KernelsExecuted,
    long BytesTransferred,
    TimeSpan Uptime,
    IReadOnlyDictionary<string, object>? ExtendedMetrics = null);
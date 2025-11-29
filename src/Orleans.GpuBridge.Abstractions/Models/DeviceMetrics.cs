using System;
using System.Collections.Generic;

namespace Orleans.GpuBridge.Abstractions.Models;

/// <summary>
/// Represents runtime metrics for a compute device
/// </summary>
[GenerateSerializer]
public sealed record DeviceMetrics
{
    /// <summary>
    /// GPU utilization percentage (0-100)
    /// </summary>
    [Id(0)]
    public float GpuUtilizationPercent { get; init; }

    /// <summary>
    /// Memory utilization percentage (0-100)
    /// </summary>
    [Id(1)]
    public float MemoryUtilizationPercent { get; init; }

    /// <summary>
    /// Used memory in bytes
    /// </summary>
    [Id(2)]
    public long UsedMemoryBytes { get; init; }

    /// <summary>
    /// Device temperature in Celsius
    /// </summary>
    [Id(3)]
    public float TemperatureCelsius { get; init; }

    /// <summary>
    /// Power consumption in watts
    /// </summary>
    [Id(4)]
    public float PowerWatts { get; init; }

    /// <summary>
    /// Fan speed percentage (0-100)
    /// </summary>
    [Id(5)]
    public float FanSpeedPercent { get; init; }

    /// <summary>
    /// Number of kernels executed
    /// </summary>
    [Id(6)]
    public long KernelsExecuted { get; init; }

    /// <summary>
    /// Total bytes transferred
    /// </summary>
    [Id(7)]
    public long BytesTransferred { get; init; }

    /// <summary>
    /// Device uptime
    /// </summary>
    [Id(8)]
    public TimeSpan Uptime { get; init; }

    /// <summary>
    /// Additional metrics
    /// </summary>
    [Id(9)]
    public Dictionary<string, object>? ExtendedMetrics { get; init; }
}
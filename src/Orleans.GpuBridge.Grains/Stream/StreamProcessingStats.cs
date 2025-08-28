using System;
using Orleans;

namespace Orleans.GpuBridge.Grains.Stream;

/// <summary>
/// Stream processing statistics
/// </summary>
[GenerateSerializer]
public sealed record StreamProcessingStats(
    [property: Id(0)] long ItemsProcessed,
    [property: Id(1)] long ItemsFailed,
    [property: Id(2)] TimeSpan TotalProcessingTime,
    [property: Id(3)] double AverageLatencyMs,
    [property: Id(4)] DateTime StartTime,
    [property: Id(5)] DateTime? LastProcessedTime);
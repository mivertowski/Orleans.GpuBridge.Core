using System;
using System.Collections.Generic;
using Orleans;
using Orleans.GpuBridge.Abstractions;

namespace Orleans.GpuBridge.Grains.Batch;

/// <summary>
/// Result from batch execution with optional performance metrics
/// </summary>
[GenerateSerializer]
public sealed record GpuBatchResult<TOut>(
    [property: Id(0)] IReadOnlyList<TOut> Results,
    [property: Id(1)] TimeSpan ExecutionTime,
    [property: Id(2)] string HandleId,
    [property: Id(3)] KernelId KernelId,
    [property: Id(4)] string? Error = null,
    [property: Id(5)] GpuBatchMetrics? Metrics = null) where TOut : notnull
{
    public bool Success => Error == null;
}
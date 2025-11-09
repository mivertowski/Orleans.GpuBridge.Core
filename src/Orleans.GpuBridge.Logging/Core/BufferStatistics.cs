namespace Orleans.GpuBridge.Logging.Core;

/// <summary>
/// Statistics about buffer performance.
/// </summary>
public sealed record BufferStatistics
{
    public long TotalEnqueued { get; init; }
    public long TotalProcessed { get; init; }
    public long TotalDropped { get; init; }
    public long TotalErrors { get; init; }
    public long TotalBatches { get; init; }
    public int MaxBatchSize { get; init; }
    public TimeSpan AverageProcessingTime { get; init; }
    public DateTimeOffset LastEnqueueTime { get; init; }
    public DateTimeOffset LastProcessTime { get; init; }
    public DateTimeOffset LastDropTime { get; init; }
    public DateTimeOffset LastErrorTime { get; init; }
    public string? LastError { get; init; }
}

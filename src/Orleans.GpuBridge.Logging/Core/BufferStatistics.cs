namespace Orleans.GpuBridge.Logging.Core;

/// <summary>
/// Statistics about buffer performance.
/// </summary>
public sealed record BufferStatistics
{
    /// <summary>
    /// Gets the total number of log entries enqueued.
    /// </summary>
    public long TotalEnqueued { get; init; }

    /// <summary>
    /// Gets the total number of log entries successfully processed.
    /// </summary>
    public long TotalProcessed { get; init; }

    /// <summary>
    /// Gets the total number of log entries dropped due to overflow.
    /// </summary>
    public long TotalDropped { get; init; }

    /// <summary>
    /// Gets the total number of processing errors encountered.
    /// </summary>
    public long TotalErrors { get; init; }

    /// <summary>
    /// Gets the total number of batches processed.
    /// </summary>
    public long TotalBatches { get; init; }

    /// <summary>
    /// Gets the maximum batch size observed.
    /// </summary>
    public int MaxBatchSize { get; init; }

    /// <summary>
    /// Gets the average time taken to process a batch.
    /// </summary>
    public TimeSpan AverageProcessingTime { get; init; }

    /// <summary>
    /// Gets the timestamp of the last enqueue operation.
    /// </summary>
    public DateTimeOffset LastEnqueueTime { get; init; }

    /// <summary>
    /// Gets the timestamp of the last processing operation.
    /// </summary>
    public DateTimeOffset LastProcessTime { get; init; }

    /// <summary>
    /// Gets the timestamp when the last entry was dropped.
    /// </summary>
    public DateTimeOffset LastDropTime { get; init; }

    /// <summary>
    /// Gets the timestamp of the last error.
    /// </summary>
    public DateTimeOffset LastErrorTime { get; init; }

    /// <summary>
    /// Gets the last error message encountered.
    /// </summary>
    public string? LastError { get; init; }
}

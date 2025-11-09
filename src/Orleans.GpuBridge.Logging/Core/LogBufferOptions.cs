namespace Orleans.GpuBridge.Logging.Core;

/// <summary>
/// Configuration options for the log buffer.
/// </summary>
public sealed record LogBufferOptions
{
    /// <summary>
    /// Maximum number of log entries to buffer.
    /// </summary>
    public int Capacity { get; init; } = 10000;

    /// <summary>
    /// Maximum number of entries per batch.
    /// </summary>
    public int MaxBatchSize { get; init; } = 100;

    /// <summary>
    /// Maximum time to wait before flushing a partial batch.
    /// </summary>
    public TimeSpan FlushInterval { get; init; } = TimeSpan.FromSeconds(1);

    /// <summary>
    /// Whether to drop old entries when buffer is full.
    /// </summary>
    public bool DropOnOverflow { get; init; } = true;

    /// <summary>
    /// Whether to prioritize high-severity entries.
    /// </summary>
    public bool PrioritizeHighSeverity { get; init; } = true;
}

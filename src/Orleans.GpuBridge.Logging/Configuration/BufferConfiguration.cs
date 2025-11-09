namespace Orleans.GpuBridge.Logging.Configuration;

/// <summary>
/// Buffer configuration options.
/// </summary>
public sealed class BufferConfiguration
{
    /// <summary>
    /// Buffer capacity for log entries.
    /// </summary>
    public int Capacity { get; set; } = 10000;

    /// <summary>
    /// Maximum batch size for processing.
    /// </summary>
    public int MaxBatchSize { get; set; } = 100;

    /// <summary>
    /// Flush interval for batched processing.
    /// </summary>
    public TimeSpan FlushInterval { get; set; } = TimeSpan.FromSeconds(1);

    /// <summary>
    /// Whether to drop entries when buffer is full.
    /// </summary>
    public bool DropOnFull { get; set; } = true;

    /// <summary>
    /// Whether to prioritize high-severity entries.
    /// </summary>
    public bool PrioritizeHighSeverity { get; set; } = true;
}

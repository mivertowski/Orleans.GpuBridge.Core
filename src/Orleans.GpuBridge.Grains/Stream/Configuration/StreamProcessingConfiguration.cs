namespace Orleans.GpuBridge.Grains.Stream.Configuration;

/// <summary>
/// Complete configuration for stream processing
/// </summary>
public sealed class StreamProcessingConfiguration
{
    /// <summary>
    /// Batch accumulation configuration
    /// </summary>
    public BatchAccumulationConfig BatchConfig { get; init; } = BatchAccumulationConfig.Default;

    /// <summary>
    /// Backpressure management configuration
    /// </summary>
    public BackpressureConfig BackpressureConfig { get; init; } = BackpressureConfig.Default;

    /// <summary>
    /// Default configuration (reasonable defaults for most scenarios)
    /// </summary>
    public static StreamProcessingConfiguration Default => new();

    /// <summary>
    /// Low-latency configuration (prioritizes latency over throughput)
    /// </summary>
    public static StreamProcessingConfiguration LowLatency => new()
    {
        BatchConfig = new BatchAccumulationConfig
        {
            MinBatchSize = 16,
            MaxBatchSize = 1_000,
            MaxBatchWaitTime = System.TimeSpan.FromMilliseconds(10), // 10ms SLA
            EnableAdaptiveBatching = false
        },
        BackpressureConfig = new BackpressureConfig
        {
            BufferCapacity = 10_000,
            PauseThreshold = 0.8,
            ResumeThreshold = 0.4
        }
    };

    /// <summary>
    /// High-throughput configuration (prioritizes throughput over latency)
    /// </summary>
    public static StreamProcessingConfiguration HighThroughput => new()
    {
        BatchConfig = new BatchAccumulationConfig
        {
            MinBatchSize = 256,
            MaxBatchSize = 100_000,
            MaxBatchWaitTime = System.TimeSpan.FromSeconds(1), // 1s batch window
            EnableAdaptiveBatching = true,
            GpuMemoryUtilizationTarget = 0.85 // Use more GPU memory
        },
        BackpressureConfig = new BackpressureConfig
        {
            BufferCapacity = 1_000_000,
            PauseThreshold = 0.95,
            ResumeThreshold = 0.7,
            DropOldestOnFull = true // Drop old to maintain flow
        }
    };
}

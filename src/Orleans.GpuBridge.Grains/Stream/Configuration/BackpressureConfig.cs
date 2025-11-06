namespace Orleans.GpuBridge.Grains.Stream.Configuration;

/// <summary>
/// Configuration for backpressure management
/// </summary>
public sealed class BackpressureConfig
{
    /// <summary>
    /// Maximum buffer capacity (number of items)
    /// </summary>
    public int BufferCapacity { get; init; } = 100_000;

    /// <summary>
    /// Pause stream when buffer reaches this threshold (0.0 - 1.0)
    /// </summary>
    public double PauseThreshold { get; init; } = 0.9; // 90% full

    /// <summary>
    /// Resume stream when buffer drops below this threshold (0.0 - 1.0)
    /// </summary>
    public double ResumeThreshold { get; init; } = 0.5; // 50% full

    /// <summary>
    /// Drop oldest items when buffer full (vs blocking producer)
    /// </summary>
    public bool DropOldestOnFull { get; init; } = false;

    /// <summary>
    /// Default configuration
    /// </summary>
    public static BackpressureConfig Default => new();
}

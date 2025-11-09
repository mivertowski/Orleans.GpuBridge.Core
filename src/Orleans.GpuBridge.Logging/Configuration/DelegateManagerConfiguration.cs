namespace Orleans.GpuBridge.Logging.Configuration;

/// <summary>
/// Delegate manager configuration options.
/// </summary>
public sealed class DelegateManagerConfiguration
{
    /// <summary>
    /// Maximum queue size for delegate manager.
    /// </summary>
    public int MaxQueueSize { get; set; } = 10000;

    /// <summary>
    /// Processing interval for delegate manager.
    /// </summary>
    public TimeSpan ProcessingInterval { get; set; } = TimeSpan.FromMilliseconds(100);

    /// <summary>
    /// Whether to drop entries when queue is full.
    /// </summary>
    public bool DropOnQueueFull { get; set; } = true;
}

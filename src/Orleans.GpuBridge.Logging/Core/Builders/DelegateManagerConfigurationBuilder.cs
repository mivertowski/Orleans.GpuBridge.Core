namespace Orleans.GpuBridge.Logging.Core;

/// <summary>
/// Builder for configuring delegate manager settings.
/// </summary>
public sealed class DelegateManagerConfigurationBuilder
{
    private readonly LoggerFactoryOptions _options;

    internal DelegateManagerConfigurationBuilder(LoggerFactoryOptions options)
    {
        _options = options;
    }

    /// <summary>
    /// Sets the maximum queue size for the delegate manager.
    /// </summary>
    /// <param name="size">Maximum number of entries in the queue before dropping or blocking.</param>
    /// <returns>This builder for method chaining.</returns>
    public DelegateManagerConfigurationBuilder WithMaxQueueSize(int size)
    {
        _options.MaxQueueSize = size;
        return this;
    }

    /// <summary>
    /// Sets the processing interval for batch processing.
    /// </summary>
    /// <param name="interval">Time interval between batch processing operations.</param>
    /// <returns>This builder for method chaining.</returns>
    public DelegateManagerConfigurationBuilder WithProcessingInterval(TimeSpan interval)
    {
        _options.ProcessingInterval = interval;
        return this;
    }

    /// <summary>
    /// Configures whether to drop entries when the queue is full.
    /// </summary>
    /// <param name="drop">If true, entries are dropped when queue is full; otherwise, blocks until space is available.</param>
    /// <returns>This builder for method chaining.</returns>
    public DelegateManagerConfigurationBuilder DropOnQueueFull(bool drop = true)
    {
        _options.DropOnQueueFull = drop;
        return this;
    }
}

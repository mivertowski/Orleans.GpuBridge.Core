namespace Orleans.GpuBridge.Logging.Core;

/// <summary>
/// Builder for configuring buffer settings.
/// </summary>
public sealed class BufferConfigurationBuilder
{
    private readonly LoggerFactoryOptions _options;

    internal BufferConfigurationBuilder(LoggerFactoryOptions options)
    {
        _options = options;
    }

    /// <summary>
    /// Sets the buffer capacity.
    /// </summary>
    /// <param name="capacity">Maximum number of entries the buffer can hold.</param>
    /// <returns>This builder for method chaining.</returns>
    public BufferConfigurationBuilder WithCapacity(int capacity)
    {
        _options.BufferCapacity = capacity;
        return this;
    }

    /// <summary>
    /// Sets the maximum batch size for processing.
    /// </summary>
    /// <param name="batchSize">Maximum number of entries to process in a single batch.</param>
    /// <returns>This builder for method chaining.</returns>
    public BufferConfigurationBuilder WithMaxBatchSize(int batchSize)
    {
        _options.MaxBatchSize = batchSize;
        return this;
    }

    /// <summary>
    /// Sets the flush interval for automatic buffer flushing.
    /// </summary>
    /// <param name="interval">Time interval between automatic flushes.</param>
    /// <returns>This builder for method chaining.</returns>
    public BufferConfigurationBuilder WithFlushInterval(TimeSpan interval)
    {
        _options.FlushInterval = interval;
        return this;
    }

    /// <summary>
    /// Configures whether to drop entries when the buffer is full.
    /// </summary>
    /// <param name="drop">If true, oldest entries are dropped when buffer is full; otherwise, blocks until space is available.</param>
    /// <returns>This builder for method chaining.</returns>
    public BufferConfigurationBuilder DropOnFull(bool drop = true)
    {
        _options.DropOnBufferFull = drop;
        return this;
    }
}

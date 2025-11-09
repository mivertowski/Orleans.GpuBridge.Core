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

    public BufferConfigurationBuilder WithCapacity(int capacity)
    {
        _options.BufferCapacity = capacity;
        return this;
    }

    public BufferConfigurationBuilder WithMaxBatchSize(int batchSize)
    {
        _options.MaxBatchSize = batchSize;
        return this;
    }

    public BufferConfigurationBuilder WithFlushInterval(TimeSpan interval)
    {
        _options.FlushInterval = interval;
        return this;
    }

    public BufferConfigurationBuilder DropOnFull(bool drop = true)
    {
        _options.DropOnBufferFull = drop;
        return this;
    }
}

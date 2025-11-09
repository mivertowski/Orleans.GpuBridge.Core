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

    public DelegateManagerConfigurationBuilder WithMaxQueueSize(int size)
    {
        _options.MaxQueueSize = size;
        return this;
    }

    public DelegateManagerConfigurationBuilder WithProcessingInterval(TimeSpan interval)
    {
        _options.ProcessingInterval = interval;
        return this;
    }

    public DelegateManagerConfigurationBuilder DropOnQueueFull(bool drop = true)
    {
        _options.DropOnQueueFull = drop;
        return this;
    }
}

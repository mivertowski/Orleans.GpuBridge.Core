using Orleans.GpuBridge.Logging.Abstractions;
using Orleans.GpuBridge.Logging.Delegates;

namespace Orleans.GpuBridge.Logging.Core;

/// <summary>
/// Statistics about the logger factory.
/// </summary>
public sealed record LoggerFactoryStatistics
{
    /// <summary>
    /// Gets the number of registered loggers.
    /// </summary>
    public int RegisteredLoggers { get; init; }

    /// <summary>
    /// Gets the number of registered logger delegates.
    /// </summary>
    public int RegisteredDelegates { get; init; }

    /// <summary>
    /// Gets the buffer health status.
    /// </summary>
    public BufferHealth BufferHealth { get; init; } = new();

    /// <summary>
    /// Gets the delegate manager statistics.
    /// </summary>
    public LoggingStatistics DelegateManagerStats { get; init; } = new();
}

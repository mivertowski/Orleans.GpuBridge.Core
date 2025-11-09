using Orleans.GpuBridge.Logging.Abstractions;
using Orleans.GpuBridge.Logging.Delegates;

namespace Orleans.GpuBridge.Logging.Core;

/// <summary>
/// Statistics about the logger factory.
/// </summary>
public sealed record LoggerFactoryStatistics
{
    public int RegisteredLoggers { get; init; }
    public int RegisteredDelegates { get; init; }
    public BufferHealth BufferHealth { get; init; } = new();
    public LoggingStatistics DelegateManagerStats { get; init; } = new();
}

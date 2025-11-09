using Orleans.GpuBridge.Logging.Abstractions;

namespace Orleans.GpuBridge.Logging.Core;

/// <summary>
/// Event arguments for buffer overflow.
/// </summary>
public sealed class BufferOverflowEventArgs : EventArgs
{
    public LogEntry DroppedEntry { get; }
    public DateTimeOffset Timestamp { get; } = DateTimeOffset.UtcNow;

    public BufferOverflowEventArgs(LogEntry droppedEntry)
    {
        DroppedEntry = droppedEntry ?? throw new ArgumentNullException(nameof(droppedEntry));
    }
}

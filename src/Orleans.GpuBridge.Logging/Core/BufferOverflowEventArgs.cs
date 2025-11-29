using Orleans.GpuBridge.Logging.Abstractions;

namespace Orleans.GpuBridge.Logging.Core;

/// <summary>
/// Event arguments for buffer overflow.
/// </summary>
public sealed class BufferOverflowEventArgs : EventArgs
{
    /// <summary>
    /// Gets the log entry that was dropped due to buffer overflow.
    /// </summary>
    public LogEntry DroppedEntry { get; }

    /// <summary>
    /// Gets the timestamp when the overflow occurred.
    /// </summary>
    public DateTimeOffset Timestamp { get; } = DateTimeOffset.UtcNow;

    /// <summary>
    /// Initializes a new instance of the <see cref="BufferOverflowEventArgs"/> class.
    /// </summary>
    /// <param name="droppedEntry">The log entry that was dropped due to buffer overflow.</param>
    public BufferOverflowEventArgs(LogEntry droppedEntry)
    {
        DroppedEntry = droppedEntry ?? throw new ArgumentNullException(nameof(droppedEntry));
    }
}

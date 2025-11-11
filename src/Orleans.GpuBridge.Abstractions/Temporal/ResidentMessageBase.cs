using System;
using Orleans;

namespace Orleans.GpuBridge.Abstractions.Temporal;

/// <summary>
/// Base message type for Ring Kernel communication.
/// Ring Kernels process these messages in a persistent GPU-resident loop.
/// </summary>
public abstract record ResidentMessage
{
    /// <summary>
    /// Unique identifier for this message request.
    /// Used to match responses back to pending grain operations.
    /// </summary>
    public Guid RequestId { get; init; } = Guid.NewGuid();

    /// <summary>
    /// Timestamp when the message was created (for latency tracking).
    /// </summary>
    public long TimestampTicks { get; init; } = DateTime.UtcNow.Ticks;
}

using System;
using System.Runtime.InteropServices;

namespace Orleans.GpuBridge.Abstractions.Temporal;

/// <summary>
/// Actor message with temporal metadata for GPU processing.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public struct ActorMessage
{
    /// <summary>
    /// Message identifier for dependency tracking.
    /// </summary>
    public Guid MessageId;

    /// <summary>
    /// Source actor identifier.
    /// </summary>
    public ulong SourceActorId;

    /// <summary>
    /// Target actor identifier.
    /// </summary>
    public ulong TargetActorId;

    /// <summary>
    /// HLC timestamp when message was sent.
    /// </summary>
    public HybridTimestamp Timestamp;

    /// <summary>
    /// Message type identifier.
    /// </summary>
    public MessageType Type;

    /// <summary>
    /// Message payload (simplified - in real system would be variant/union).
    /// </summary>
    public long Payload;

    /// <summary>
    /// Sequence number for ordering.
    /// </summary>
    public ulong SequenceNumber;

    /// <summary>
    /// Priority level for message processing.
    /// </summary>
    public byte Priority;

    public override string ToString() =>
        $"Message(Id={MessageId}, From={SourceActorId}, To={TargetActorId}, Type={Type}, Seq={SequenceNumber})";
}

/// <summary>
/// Message type enumeration.
/// </summary>
public enum MessageType : byte
{
    StateUpdate = 0,
    Query = 1,
    Command = 2,
    Event = 3,
    Response = 4
}

/// <summary>
/// Actor state stored in GPU memory.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public struct ActorState
{
    /// <summary>
    /// Actor identifier.
    /// </summary>
    public ulong ActorId;

    /// <summary>
    /// Current HLC physical time.
    /// </summary>
    public long HLCPhysical;

    /// <summary>
    /// Current HLC logical counter.
    /// </summary>
    public long HLCLogical;

    /// <summary>
    /// Timestamp of last processed message.
    /// </summary>
    public HybridTimestamp LastProcessedTimestamp;

    /// <summary>
    /// Total messages processed by this actor.
    /// </summary>
    public ulong MessageCount;

    /// <summary>
    /// Actor data (simplified - would be custom struct in real system).
    /// </summary>
    public long Data;

    /// <summary>
    /// Actor status flags.
    /// </summary>
    public ActorStatusFlags Status;

    /// <summary>
    /// Reserved for future use.
    /// </summary>
    public long Reserved;

    public override string ToString() =>
        $"ActorState(Id={ActorId}, HLC=({HLCPhysical},{HLCLogical}), Messages={MessageCount}, Status={Status})";
}

/// <summary>
/// Actor status flags.
/// </summary>
[Flags]
public enum ActorStatusFlags : byte
{
    Active = 0x01,
    Processing = 0x02,
    Blocked = 0x04,
    Error = 0x08
}

/// <summary>
/// Temporal event for pattern detection.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public struct TemporalEvent
{
    public Guid EventId;
    public ulong ActorId;
    public HybridTimestamp Timestamp;
    public long PhysicalTimeNanos;
    public EventType Type;
    public long Value;

    public override string ToString() =>
        $"Event(Id={EventId}, Actor={ActorId}, Type={Type}, Time={PhysicalTimeNanos}ns)";
}

public enum EventType : byte
{
    Transaction = 0,
    StateChange = 1,
    Query = 2,
    Alert = 3
}

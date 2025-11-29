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

    /// <summary>
    /// Returns a string representation of the message showing ID, source, target, type, and sequence.
    /// </summary>
    /// <returns>A formatted string describing the message.</returns>
    public override string ToString() =>
        $"Message(Id={MessageId}, From={SourceActorId}, To={TargetActorId}, Type={Type}, Seq={SequenceNumber})";
}

/// <summary>
/// Message type enumeration.
/// </summary>
public enum MessageType : byte
{
    /// <summary>
    /// State update message.
    /// </summary>
    StateUpdate = 0,

    /// <summary>
    /// Query message requesting information.
    /// </summary>
    Query = 1,

    /// <summary>
    /// Command message requesting an action.
    /// </summary>
    Command = 2,

    /// <summary>
    /// Event notification message.
    /// </summary>
    Event = 3,

    /// <summary>
    /// Response message to a query or command.
    /// </summary>
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

    /// <summary>
    /// Returns a string representation of the actor state showing ID, HLC, message count, and status.
    /// </summary>
    /// <returns>A formatted string describing the actor state.</returns>
    public override string ToString() =>
        $"ActorState(Id={ActorId}, HLC=({HLCPhysical},{HLCLogical}), Messages={MessageCount}, Status={Status})";
}

/// <summary>
/// Actor status flags.
/// </summary>
[Flags]
public enum ActorStatusFlags : byte
{
    /// <summary>
    /// Actor is active and ready to process messages.
    /// </summary>
    Active = 0x01,

    /// <summary>
    /// Actor is currently processing a message.
    /// </summary>
    Processing = 0x02,

    /// <summary>
    /// Actor is blocked waiting for a resource or condition.
    /// </summary>
    Blocked = 0x04,

    /// <summary>
    /// Actor encountered an error during processing.
    /// </summary>
    Error = 0x08
}

/// <summary>
/// Temporal event for pattern detection.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public struct TemporalEvent
{
    /// <summary>
    /// Unique identifier for this event.
    /// </summary>
    public Guid EventId;

    /// <summary>
    /// Actor that generated this event.
    /// </summary>
    public ulong ActorId;

    /// <summary>
    /// HLC timestamp when event occurred.
    /// </summary>
    public HybridTimestamp Timestamp;

    /// <summary>
    /// Physical time in nanoseconds when event occurred.
    /// </summary>
    public long PhysicalTimeNanos;

    /// <summary>
    /// Type of event.
    /// </summary>
    public EventType Type;

    /// <summary>
    /// Event value or payload.
    /// </summary>
    public long Value;

    /// <summary>
    /// Returns a string representation of the event showing ID, actor, type, and time.
    /// </summary>
    /// <returns>A formatted string describing the event.</returns>
    public override string ToString() =>
        $"Event(Id={EventId}, Actor={ActorId}, Type={Type}, Time={PhysicalTimeNanos}ns)";
}

/// <summary>
/// Event type enumeration.
/// </summary>
public enum EventType : byte
{
    /// <summary>
    /// Transaction event.
    /// </summary>
    Transaction = 0,

    /// <summary>
    /// State change event.
    /// </summary>
    StateChange = 1,

    /// <summary>
    /// Query event.
    /// </summary>
    Query = 2,

    /// <summary>
    /// Alert event.
    /// </summary>
    Alert = 3
}

using System;
using System.Collections.Immutable;
using Orleans;
using Orleans.GpuBridge.Abstractions.Temporal;

namespace Orleans.GpuBridge.Abstractions.Temporal;

/// <summary>
/// Base message type for Ring Kernel communication with temporal correctness metadata.
/// </summary>
/// <remarks>
/// <para>
/// Extends <see cref="ResidentMessage"/> with temporal information for:
/// </para>
/// <list type="bullet">
///   <item><description>Total event ordering via Hybrid Logical Clocks</description></item>
///   <item><description>Causal dependency tracking</description></item>
///   <item><description>Physical timestamp with error bounds</description></item>
///   <item><description>Message sequencing per sender</description></item>
/// </list>
/// <para>
/// Use temporal messages when you need:
/// - Guaranteed message ordering across distributed actors
/// - Detection of concurrent events
/// - Temporal pattern matching (e.g., "events within 5 seconds")
/// - Causal relationship tracking (A caused B caused C)
/// </para>
/// </remarks>
public abstract record TemporalResidentMessage : ResidentMessage
{
    /// <summary>
    /// Hybrid Logical Clock timestamp providing total ordering of events.
    /// </summary>
    /// <remarks>
    /// The HLC timestamp ensures:
    /// - If event A happens-before event B, then HLC(A) &lt; HLC(B)
    /// - Bounded drift from physical time
    /// - Deterministic ordering across distributed actors
    /// </remarks>
    public HybridTimestamp HLC { get; init; }

    /// <summary>
    /// Physical timestamp in nanoseconds since Unix epoch (1970-01-01 00:00:00 UTC).
    /// </summary>
    /// <remarks>
    /// This is the wall-clock time when the message was created,
    /// independent of the HLC logical counter.
    /// Used for:
    /// - Measuring actual elapsed time
    /// - Temporal queries ("find events between T1 and T2")
    /// - Latency measurement
    /// </remarks>
    public long PhysicalTimeNanos { get; init; }

    /// <summary>
    /// Error bound for physical timestamp in nanoseconds (±).
    /// </summary>
    /// <remarks>
    /// <para>
    /// Indicates the uncertainty in the physical timestamp due to:
    /// - Clock synchronization error (NTP: ±1-10ms, PTP: ±10-100ns)
    /// - Clock drift since last synchronization
    /// - Measurement overhead
    /// </para>
    /// <para>
    /// The actual physical time is within:
    /// [PhysicalTimeNanos - TimestampErrorBoundNanos, PhysicalTimeNanos + TimestampErrorBoundNanos]
    /// </para>
    /// </remarks>
    public long TimestampErrorBoundNanos { get; init; }

    /// <summary>
    /// List of message IDs that must be processed before this message (causal dependencies).
    /// </summary>
    /// <remarks>
    /// <para>
    /// Enforces causal ordering: if message A is a dependency of message B,
    /// then B will not be processed until A completes.
    /// </para>
    /// <para>
    /// Example: Transaction graph
    /// <code>
    /// // Node A sends $1000 to Node B
    /// var msg1 = new TransferMessage { ... };
    ///
    /// // Node B splits money: $500 to C, $500 to D
    /// // Both depend on receiving msg1 first
    /// var msg2 = new TransferMessage
    /// {
    ///     CausalDependencies = ImmutableArray.Create(msg1.RequestId)
    /// };
    /// var msg3 = new TransferMessage
    /// {
    ///     CausalDependencies = ImmutableArray.Create(msg1.RequestId)
    /// };
    /// </code>
    /// </para>
    /// </remarks>
    public ImmutableArray<Guid> CausalDependencies { get; init; } = ImmutableArray<Guid>.Empty;

    /// <summary>
    /// Sequence number for messages from this sender (monotonically increasing).
    /// </summary>
    /// <remarks>
    /// <para>
    /// Provides FIFO ordering guarantee within a single sender:
    /// if sender X sends messages M1, M2, M3 with sequence numbers 1, 2, 3,
    /// the receiver will process them in that order.
    /// </para>
    /// <para>
    /// Combined with HLC timestamps, this provides both:
    /// - Local FIFO ordering (sequence numbers)
    /// - Global causal ordering (HLC + dependencies)
    /// </para>
    /// </remarks>
    public ulong SequenceNumber { get; init; }

    /// <summary>
    /// Optional: Valid time range for processing this message.
    /// </summary>
    /// <remarks>
    /// <para>
    /// If specified, the message should only be processed if the current time
    /// falls within [ValidFrom, ValidTo]. Messages outside this range may be:
    /// - Dropped (if too late)
    /// - Queued (if too early)
    /// - Logged as deadline misses (for monitoring)
    /// </para>
    /// <para>
    /// Use for:
    /// - Real-time systems with deadlines
    /// - Time-sensitive trading operations
    /// - Event windows ("process all events between T1 and T2")
    /// </para>
    /// </remarks>
    public TimeRange? ValidityWindow { get; init; }

    /// <summary>
    /// Optional: Priority level for message processing.
    /// </summary>
    /// <remarks>
    /// Higher priority messages are processed before lower priority messages,
    /// subject to HLC ordering constraints and causal dependencies.
    /// </remarks>
    public MessagePriority Priority { get; init; } = MessagePriority.Normal;

    /// <summary>
    /// Checks if this message's validity window contains the specified time.
    /// </summary>
    public bool IsValidAt(long timeNanos)
    {
        return ValidityWindow?.Contains(timeNanos) ?? true;
    }

    /// <summary>
    /// Checks if this message has expired (past the validity window).
    /// </summary>
    public bool IsExpired(long currentTimeNanos)
    {
        return ValidityWindow.HasValue && currentTimeNanos > ValidityWindow.Value.EndNanos;
    }

    /// <summary>
    /// Gets the age of this message in nanoseconds.
    /// </summary>
    public long GetAgeNanos(long currentTimeNanos)
    {
        return currentTimeNanos - PhysicalTimeNanos;
    }

    /// <inheritdoc/>
    public override string ToString()
    {
        var deps = CausalDependencies.IsEmpty ? "none" : $"{CausalDependencies.Length}";
        return $"{GetType().Name}(RequestId={RequestId:N}, HLC={HLC}, Seq={SequenceNumber}, Deps={deps})";
    }
}

/// <summary>
/// Represents a time range for message validity.
/// </summary>
public readonly record struct TimeRange
{
    /// <summary>
    /// Start of the time range (inclusive) in nanoseconds since Unix epoch.
    /// </summary>
    public long StartNanos { get; init; }

    /// <summary>
    /// End of the time range (inclusive) in nanoseconds since Unix epoch.
    /// </summary>
    public long EndNanos { get; init; }

    /// <summary>
    /// Duration of the time range in nanoseconds.
    /// </summary>
    public long DurationNanos => EndNanos - StartNanos;

    /// <summary>
    /// Checks if the time range contains the specified time.
    /// </summary>
    public bool Contains(long timeNanos)
    {
        return timeNanos >= StartNanos && timeNanos <= EndNanos;
    }

    /// <summary>
    /// Checks if this time range overlaps with another.
    /// </summary>
    public bool Overlaps(TimeRange other)
    {
        return StartNanos <= other.EndNanos && EndNanos >= other.StartNanos;
    }

    /// <inheritdoc/>
    public override string ToString()
    {
        var durationMs = DurationNanos / 1_000_000.0;
        return $"TimeRange({durationMs:F1}ms)";
    }
}

/// <summary>
/// Message priority levels for temporal processing.
/// </summary>
public enum MessagePriority
{
    /// <summary>
    /// Low priority: background tasks, analytics.
    /// </summary>
    Low = 0,

    /// <summary>
    /// Normal priority: regular operations (default).
    /// </summary>
    Normal = 1,

    /// <summary>
    /// High priority: time-sensitive operations.
    /// </summary>
    High = 2,

    /// <summary>
    /// Critical priority: real-time operations with strict deadlines.
    /// </summary>
    Critical = 3
}

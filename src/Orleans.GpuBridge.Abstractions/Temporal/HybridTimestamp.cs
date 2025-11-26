using System;
using System.Runtime.InteropServices;
using Orleans;

namespace Orleans.GpuBridge.Abstractions.Temporal;

/// <summary>
/// Hybrid Logical Clock timestamp combining physical time and logical counter.
/// Provides happens-before ordering for distributed events.
/// </summary>
/// <remarks>
/// HLC timestamps ensure:
/// 1. Monotonicity: Timestamps never decrease for an actor
/// 2. Causality: If event A happens-before event B, then HLC(A) &lt; HLC(B)
/// 3. Physical time approximation: HLC tracks physical time closely
/// </remarks>
[StructLayout(LayoutKind.Sequential)]
[GenerateSerializer]
[Immutable]
public readonly struct HybridTimestamp : IComparable<HybridTimestamp>, IEquatable<HybridTimestamp>
{
    /// <summary>
    /// Physical time component in nanoseconds since Unix epoch.
    /// </summary>
    [Id(0)]
    public long PhysicalTime { get; init; }

    /// <summary>
    /// Logical counter for events occurring at the same physical time.
    /// </summary>
    [Id(1)]
    public long LogicalCounter { get; init; }

    /// <summary>
    /// Node identifier for tie-breaking concurrent events (optional).
    /// </summary>
    [Id(2)]
    public ushort NodeId { get; init; }

    /// <summary>
    /// Creates a new HLC timestamp.
    /// </summary>
    public HybridTimestamp(long physicalTime, long logicalCounter, ushort nodeId = 0)
    {
        PhysicalTime = physicalTime;
        LogicalCounter = logicalCounter;
        NodeId = nodeId;
    }

    /// <summary>
    /// Creates an HLC timestamp from current system time.
    /// </summary>
    public static HybridTimestamp Now(ushort nodeId = 0)
    {
        long physicalTime = DateTimeOffset.UtcNow.ToUnixTimeNanoseconds();
        return new HybridTimestamp(physicalTime, 0, nodeId);
    }

    /// <summary>
    /// Gets current physical time in nanoseconds since Unix epoch.
    /// </summary>
    public static long GetCurrentPhysicalTimeNanos()
    {
        return DateTimeOffset.UtcNow.ToUnixTimeNanoseconds();
    }

    /// <summary>
    /// Updates HLC timestamp based on received timestamp (Lamport clock algorithm).
    /// </summary>
    /// <param name="local">Local HLC timestamp.</param>
    /// <param name="received">Received HLC timestamp from remote actor.</param>
    /// <param name="physicalTime">Current physical time in nanoseconds.</param>
    /// <returns>Updated HLC timestamp maintaining causal ordering.</returns>
    public static HybridTimestamp Update(
        HybridTimestamp local,
        HybridTimestamp received,
        long physicalTime)
    {
        // Take maximum of all three times
        long maxPhysical = Math.Max(Math.Max(local.PhysicalTime, received.PhysicalTime), physicalTime);
        long newLogical;

        if (maxPhysical == local.PhysicalTime && maxPhysical == received.PhysicalTime)
        {
            // Both local and received have same physical time
            newLogical = Math.Max(local.LogicalCounter, received.LogicalCounter) + 1;
        }
        else if (maxPhysical == local.PhysicalTime)
        {
            // Local time is ahead
            newLogical = local.LogicalCounter + 1;
        }
        else if (maxPhysical == received.PhysicalTime)
        {
            // Received time is ahead
            newLogical = received.LogicalCounter + 1;
        }
        else
        {
            // Physical time is ahead of both
            newLogical = 0;
        }

        return new HybridTimestamp(maxPhysical, newLogical, local.NodeId);
    }

    /// <summary>
    /// Increments the logical counter for a local event.
    /// </summary>
    public HybridTimestamp Increment(long physicalTime)
    {
        if (physicalTime > PhysicalTime)
        {
            // Physical time advanced - reset logical counter
            return new HybridTimestamp(physicalTime, 0, NodeId);
        }
        else
        {
            // Same physical time - increment logical counter
            return new HybridTimestamp(PhysicalTime, LogicalCounter + 1, NodeId);
        }
    }

    /// <summary>
    /// Compares two HLC timestamps for happens-before ordering.
    /// </summary>
    /// <returns>
    /// -1 if this &lt; other (this happens-before other)
    /// 0 if this == other (concurrent or same event)
    /// 1 if this &gt; other (other happens-before this)
    /// </returns>
    public int CompareTo(HybridTimestamp other)
    {
        // Compare physical time first
        int physicalComparison = PhysicalTime.CompareTo(other.PhysicalTime);
        if (physicalComparison != 0)
            return physicalComparison;

        // If physical times are equal, compare logical counters
        int logicalComparison = LogicalCounter.CompareTo(other.LogicalCounter);
        if (logicalComparison != 0)
            return logicalComparison;

        // If both are equal, use node ID for total ordering
        return NodeId.CompareTo(other.NodeId);
    }

    public bool Equals(HybridTimestamp other) =>
        PhysicalTime == other.PhysicalTime &&
        LogicalCounter == other.LogicalCounter &&
        NodeId == other.NodeId;

    public override bool Equals(object? obj) =>
        obj is HybridTimestamp timestamp && Equals(timestamp);

    public override int GetHashCode() =>
        HashCode.Combine(PhysicalTime, LogicalCounter, NodeId);

    public override string ToString() =>
        $"HLC({PhysicalTime}, {LogicalCounter}, Node={NodeId})";

    public static bool operator ==(HybridTimestamp left, HybridTimestamp right) =>
        left.Equals(right);

    public static bool operator !=(HybridTimestamp left, HybridTimestamp right) =>
        !left.Equals(right);

    public static bool operator <(HybridTimestamp left, HybridTimestamp right) =>
        left.CompareTo(right) < 0;

    public static bool operator <=(HybridTimestamp left, HybridTimestamp right) =>
        left.CompareTo(right) <= 0;

    public static bool operator >(HybridTimestamp left, HybridTimestamp right) =>
        left.CompareTo(right) > 0;

    public static bool operator >=(HybridTimestamp left, HybridTimestamp right) =>
        left.CompareTo(right) >= 0;

    /// <summary>
    /// Converts this HLC timestamp to a 64-bit integer representation.
    /// Uses physical time for the conversion, providing nanosecond precision.
    /// </summary>
    /// <returns>The physical time component in nanoseconds since Unix epoch.</returns>
    public long ToInt64() => PhysicalTime;

    /// <summary>
    /// Creates an HLC timestamp from a 64-bit integer representation.
    /// </summary>
    /// <param name="value">Physical time in nanoseconds since Unix epoch.</param>
    /// <param name="nodeId">Optional node identifier for distributed ordering.</param>
    /// <returns>A new HLC timestamp with the specified physical time.</returns>
    public static HybridTimestamp FromInt64(long value, ushort nodeId = 0) =>
        new(value, 0, nodeId);
}

/// <summary>
/// Utility methods for Unix nanosecond timestamps.
/// </summary>
public static class DateTimeOffsetExtensions
{
    // Unix epoch: January 1, 1970 00:00:00 UTC in ticks
    private const long UnixEpochTicks = 621355968000000000L;

    /// <summary>
    /// Converts DateTimeOffset to Unix nanoseconds.
    /// </summary>
    public static long ToUnixTimeNanoseconds(this DateTimeOffset dateTimeOffset)
    {
        long ticks = dateTimeOffset.UtcDateTime.Ticks;
        long elapsedTicks = ticks - UnixEpochTicks;
        // Convert ticks (100ns) to nanoseconds by multiplying by 100
        return elapsedTicks * 100L;
    }

    /// <summary>
    /// Converts Unix nanoseconds to DateTimeOffset.
    /// </summary>
    public static DateTimeOffset FromUnixTimeNanoseconds(long nanos)
    {
        // Convert nanoseconds to ticks (divide by 100)
        long elapsedTicks = nanos / 100L;
        long ticks = UnixEpochTicks + elapsedTicks;
        return new DateTimeOffset(ticks, TimeSpan.Zero);
    }
}

using System;
using System.Diagnostics;
using System.Runtime.InteropServices;

namespace Orleans.GpuBridge.Abstractions.Temporal;

/// <summary>
/// Represents a Hybrid Logical Clock (HLC) timestamp combining physical time with logical ordering.
/// </summary>
/// <remarks>
/// <para>
/// Hybrid Logical Clocks provide total ordering of events across distributed systems while
/// maintaining bounded drift from physical time. The timestamp consists of:
/// </para>
/// <list type="bullet">
///   <item><description>Physical time component (nanoseconds since epoch)</description></item>
///   <item><description>Logical counter (increments on concurrent events)</description></item>
///   <item><description>Node identifier (for tie-breaking)</description></item>
/// </list>
/// <para>
/// Properties:
/// - If event A happens-before event B, then HLC(A) &lt; HLC(B)
/// - Bounded drift from physical time (within clock synchronization error)
/// - Total ordering of all events (no ambiguity)
/// </para>
/// <para>
/// Reference: "Logical Physical Clocks and Consistent Snapshots in Globally Distributed Databases"
/// by Kulkarni et al., 2014
/// </para>
/// </remarks>
[StructLayout(LayoutKind.Sequential, Pack = 8)]
public readonly struct HybridTimestamp : IEquatable<HybridTimestamp>, IComparable<HybridTimestamp>
{
    /// <summary>
    /// Physical time component in nanoseconds since Unix epoch (1970-01-01 00:00:00 UTC).
    /// </summary>
    public long PhysicalTime { get; init; }

    /// <summary>
    /// Logical counter that increments when physical times are equal.
    /// Used to establish ordering for concurrent events.
    /// </summary>
    public long LogicalCounter { get; init; }

    /// <summary>
    /// Node identifier for deterministic tie-breaking when physical time and logical counter are equal.
    /// </summary>
    public ushort NodeId { get; init; }

    /// <summary>
    /// Creates a new hybrid timestamp.
    /// </summary>
    /// <param name="physicalTime">Physical time in nanoseconds since Unix epoch</param>
    /// <param name="logicalCounter">Logical counter for ordering</param>
    /// <param name="nodeId">Node identifier</param>
    public HybridTimestamp(long physicalTime, long logicalCounter, ushort nodeId)
    {
        PhysicalTime = physicalTime;
        LogicalCounter = logicalCounter;
        NodeId = nodeId;
    }

    /// <summary>
    /// Creates a timestamp from current physical time.
    /// </summary>
    /// <param name="nodeId">Node identifier</param>
    /// <returns>New timestamp with physical time set to current time</returns>
    public static HybridTimestamp FromCurrentTime(ushort nodeId)
    {
        return new HybridTimestamp(GetCurrentPhysicalTimeNanos(), 0, nodeId);
    }

    /// <summary>
    /// Gets current physical time in nanoseconds since Unix epoch.
    /// </summary>
    /// <returns>Current time in nanoseconds</returns>
    public static long GetCurrentPhysicalTimeNanos()
    {
        // Use DateTimeOffset for cross-platform compatibility
        var utcNow = DateTimeOffset.UtcNow;
        var epochTicks = new DateTimeOffset(1970, 1, 1, 0, 0, 0, TimeSpan.Zero).Ticks;
        var elapsedTicks = utcNow.Ticks - epochTicks;

        // Convert ticks (100ns resolution) to nanoseconds
        return elapsedTicks * 100;
    }

    /// <summary>
    /// Compares two hybrid timestamps for ordering.
    /// </summary>
    /// <remarks>
    /// Comparison order:
    /// 1. Physical time (most significant)
    /// 2. Logical counter
    /// 3. Node ID (for deterministic tie-breaking)
    /// </remarks>
    public int CompareTo(HybridTimestamp other)
    {
        // Compare physical time first
        var physicalComparison = PhysicalTime.CompareTo(other.PhysicalTime);
        if (physicalComparison != 0)
            return physicalComparison;

        // Physical times equal, compare logical counter
        var logicalComparison = LogicalCounter.CompareTo(other.LogicalCounter);
        if (logicalComparison != 0)
            return logicalComparison;

        // Both equal, use node ID for deterministic ordering
        return NodeId.CompareTo(other.NodeId);
    }

    /// <summary>
    /// Checks if this timestamp equals another timestamp.
    /// </summary>
    public bool Equals(HybridTimestamp other)
    {
        return PhysicalTime == other.PhysicalTime &&
               LogicalCounter == other.LogicalCounter &&
               NodeId == other.NodeId;
    }

    /// <inheritdoc/>
    public override bool Equals(object? obj)
    {
        return obj is HybridTimestamp other && Equals(other);
    }

    /// <inheritdoc/>
    public override int GetHashCode()
    {
        return HashCode.Combine(PhysicalTime, LogicalCounter, NodeId);
    }

    /// <inheritdoc/>
    public override string ToString()
    {
        return $"HLC({PhysicalTime}ns, L{LogicalCounter}, N{NodeId})";
    }

    /// <summary>
    /// Converts the timestamp to a human-readable string with date/time.
    /// </summary>
    public string ToDetailedString()
    {
        var epochTicks = new DateTimeOffset(1970, 1, 1, 0, 0, 0, TimeSpan.Zero).Ticks;
        var timestampTicks = epochTicks + (PhysicalTime / 100); // Convert nanos to ticks
        var dateTime = new DateTimeOffset(timestampTicks, TimeSpan.Zero);

        return $"HLC({dateTime:yyyy-MM-dd HH:mm:ss.fffffff}, L{LogicalCounter}, N{NodeId})";
    }

    /// <summary>
    /// Checks if this timestamp happens-before another timestamp.
    /// </summary>
    /// <param name="other">The other timestamp</param>
    /// <returns>True if this timestamp happens before the other</returns>
    public bool HappensBefore(HybridTimestamp other)
    {
        return CompareTo(other) < 0;
    }

    /// <summary>
    /// Checks if this timestamp is concurrent with another timestamp.
    /// </summary>
    /// <remarks>
    /// Two timestamps are concurrent if they have the same physical time and logical counter
    /// but different node IDs.
    /// </remarks>
    public bool IsConcurrentWith(HybridTimestamp other)
    {
        return PhysicalTime == other.PhysicalTime &&
               LogicalCounter == other.LogicalCounter &&
               NodeId != other.NodeId;
    }

    /// <summary>
    /// Gets the elapsed time in nanoseconds since this timestamp.
    /// </summary>
    public long GetElapsedNanos()
    {
        return GetCurrentPhysicalTimeNanos() - PhysicalTime;
    }

    /// <summary>
    /// Gets the time difference in nanoseconds between this and another timestamp.
    /// </summary>
    public long GetDifferenceNanos(HybridTimestamp other)
    {
        return PhysicalTime - other.PhysicalTime;
    }

    // Comparison operators
    public static bool operator ==(HybridTimestamp left, HybridTimestamp right) => left.Equals(right);
    public static bool operator !=(HybridTimestamp left, HybridTimestamp right) => !left.Equals(right);
    public static bool operator <(HybridTimestamp left, HybridTimestamp right) => left.CompareTo(right) < 0;
    public static bool operator >(HybridTimestamp left, HybridTimestamp right) => left.CompareTo(right) > 0;
    public static bool operator <=(HybridTimestamp left, HybridTimestamp right) => left.CompareTo(right) <= 0;
    public static bool operator >=(HybridTimestamp left, HybridTimestamp right) => left.CompareTo(right) >= 0;
}

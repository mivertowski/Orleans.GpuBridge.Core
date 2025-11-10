using System;
using System.Threading;

namespace Orleans.GpuBridge.Abstractions.Temporal;

/// <summary>
/// Implements a Hybrid Logical Clock for distributed event ordering.
/// </summary>
/// <remarks>
/// <para>
/// The Hybrid Logical Clock (HLC) combines physical time with logical counters to provide:
/// </para>
/// <list type="bullet">
///   <item><description>Total ordering of events (no ambiguity)</description></item>
///   <item><description>Bounded drift from physical time</description></item>
///   <item><description>Causality preservation (if A→B then HLC(A) &lt; HLC(B))</description></item>
/// </list>
/// <para>
/// Thread-safe implementation using interlocked operations for lock-free performance.
/// </para>
/// <para>
/// Usage:
/// <code>
/// var clock = new HybridLogicalClock(nodeId: 1);
///
/// // Generate timestamp for local event
/// var timestamp = clock.Now();
///
/// // Update clock on message receipt
/// clock.Update(receivedTimestamp);
/// </code>
/// </para>
/// </remarks>
public sealed class HybridLogicalClock
{
    private long _lastPhysicalTime;
    private long _lastLogicalCounter;
    private readonly ushort _nodeId;
    private readonly IPhysicalClockSource? _clockSource;

    /// <summary>
    /// Gets the node identifier for this clock.
    /// </summary>
    public ushort NodeId => _nodeId;

    /// <summary>
    /// Gets the current physical time in nanoseconds.
    /// </summary>
    public long CurrentPhysicalTime => GetPhysicalTime();

    /// <summary>
    /// Gets the last generated timestamp.
    /// </summary>
    public HybridTimestamp LastTimestamp => new(
        Interlocked.Read(ref _lastPhysicalTime),
        Interlocked.Read(ref _lastLogicalCounter),
        _nodeId);

    /// <summary>
    /// Creates a new Hybrid Logical Clock.
    /// </summary>
    /// <param name="nodeId">Unique node identifier (0-65535)</param>
    /// <param name="clockSource">Optional custom physical clock source (defaults to system clock)</param>
    public HybridLogicalClock(ushort nodeId, IPhysicalClockSource? clockSource = null)
    {
        _nodeId = nodeId;
        _clockSource = clockSource;
        _lastPhysicalTime = GetPhysicalTime();
        _lastLogicalCounter = 0;
    }

    /// <summary>
    /// Generates a new timestamp for a local event.
    /// </summary>
    /// <remarks>
    /// <para>
    /// HLC update rules for local events:
    /// </para>
    /// <code>
    /// physical_time = max(last_physical_time, system_time)
    /// if (physical_time == last_physical_time)
    ///     logical_counter = last_logical_counter + 1
    /// else
    ///     logical_counter = 0
    /// </code>
    /// <para>
    /// Thread-safe using compare-and-swap to avoid locks.
    /// </para>
    /// </remarks>
    /// <returns>New hybrid timestamp</returns>
    public HybridTimestamp Now()
    {
        while (true)
        {
            // Read current clock state
            var lastPhysical = Interlocked.Read(ref _lastPhysicalTime);
            var lastLogical = Interlocked.Read(ref _lastLogicalCounter);

            // Get current physical time
            var currentPhysical = GetPhysicalTime();

            // Calculate new timestamp
            var newPhysical = Math.Max(lastPhysical, currentPhysical);
            var newLogical = (newPhysical == lastPhysical) ? lastLogical + 1 : 0;

            // Try to update atomically
            var originalPhysical = Interlocked.CompareExchange(
                ref _lastPhysicalTime, newPhysical, lastPhysical);

            if (originalPhysical == lastPhysical)
            {
                // Physical time update succeeded, now update logical counter
                Interlocked.Exchange(ref _lastLogicalCounter, newLogical);
                return new HybridTimestamp(newPhysical, newLogical, _nodeId);
            }

            // Conflict detected, retry
            // This happens when another thread updated the clock concurrently
            Thread.SpinWait(1);
        }
    }

    /// <summary>
    /// Updates the clock based on a received timestamp.
    /// </summary>
    /// <remarks>
    /// <para>
    /// HLC update rules for message receipt:
    /// </para>
    /// <code>
    /// physical_time = max(last_physical_time, system_time, received_physical_time)
    /// if (physical_time == last_physical_time AND physical_time == received_physical_time)
    ///     logical_counter = max(last_logical_counter, received_logical_counter) + 1
    /// else if (physical_time == last_physical_time)
    ///     logical_counter = last_logical_counter + 1
    /// else if (physical_time == received_physical_time)
    ///     logical_counter = received_logical_counter + 1
    /// else
    ///     logical_counter = 0
    /// </code>
    /// <para>
    /// This ensures causality: if message M was sent with timestamp T_send,
    /// the recipient's clock will be updated such that all subsequent events
    /// have timestamps greater than T_send.
    /// </para>
    /// </remarks>
    /// <param name="receivedTimestamp">Timestamp from received message</param>
    /// <returns>New timestamp after update</returns>
    public HybridTimestamp Update(HybridTimestamp receivedTimestamp)
    {
        while (true)
        {
            // Read current clock state
            var lastPhysical = Interlocked.Read(ref _lastPhysicalTime);
            var lastLogical = Interlocked.Read(ref _lastLogicalCounter);

            // Get current physical time
            var currentPhysical = GetPhysicalTime();

            // Calculate new physical time (max of all three)
            var newPhysical = Math.Max(Math.Max(lastPhysical, currentPhysical), receivedTimestamp.PhysicalTime);

            // Calculate new logical counter based on HLC update rules
            long newLogical;
            if (newPhysical == lastPhysical && newPhysical == receivedTimestamp.PhysicalTime)
            {
                // All three equal: increment max of both logical counters
                newLogical = Math.Max(lastLogical, receivedTimestamp.LogicalCounter) + 1;
            }
            else if (newPhysical == lastPhysical)
            {
                // Physical time matches last: increment logical counter
                newLogical = lastLogical + 1;
            }
            else if (newPhysical == receivedTimestamp.PhysicalTime)
            {
                // Physical time matches received: increment received logical counter
                newLogical = receivedTimestamp.LogicalCounter + 1;
            }
            else
            {
                // Physical time advanced: reset logical counter
                newLogical = 0;
            }

            // Try to update atomically
            var originalPhysical = Interlocked.CompareExchange(
                ref _lastPhysicalTime, newPhysical, lastPhysical);

            if (originalPhysical == lastPhysical)
            {
                // Physical time update succeeded, now update logical counter
                Interlocked.Exchange(ref _lastLogicalCounter, newLogical);
                return new HybridTimestamp(newPhysical, newLogical, _nodeId);
            }

            // Conflict detected, retry
            Thread.SpinWait(1);
        }
    }

    /// <summary>
    /// Gets the clock drift in nanoseconds relative to another timestamp.
    /// </summary>
    /// <remarks>
    /// Positive drift means this clock is ahead of the other timestamp.
    /// Negative drift means this clock is behind.
    /// </remarks>
    public long GetClockDriftNanos(HybridTimestamp other)
    {
        var currentPhysical = GetPhysicalTime();
        return currentPhysical - other.PhysicalTime;
    }

    /// <summary>
    /// Resets the clock to a specific timestamp.
    /// </summary>
    /// <remarks>
    /// WARNING: This violates monotonicity and should only be used for testing
    /// or clock synchronization recovery scenarios.
    /// </remarks>
    public void Reset(HybridTimestamp timestamp)
    {
        Interlocked.Exchange(ref _lastPhysicalTime, timestamp.PhysicalTime);
        Interlocked.Exchange(ref _lastLogicalCounter, timestamp.LogicalCounter);
    }

    /// <summary>
    /// Gets the physical time from the clock source or system clock.
    /// </summary>
    private long GetPhysicalTime()
    {
        return _clockSource?.GetCurrentTimeNanos() ?? HybridTimestamp.GetCurrentPhysicalTimeNanos();
    }

    /// <inheritdoc/>
    public override string ToString()
    {
        var last = LastTimestamp;
        return $"HybridLogicalClock(NodeId={_nodeId}, Last={last})";
    }
}

/// <summary>
/// Provides physical clock time for Hybrid Logical Clocks.
/// </summary>
/// <remarks>
/// Implementations can provide different clock sources:
/// <list type="bullet">
///   <item><description>System clock (default)</description></item>
///   <item><description>NTP synchronized clock</description></item>
///   <item><description>PTP (Precision Time Protocol) clock</description></item>
///   <item><description>GPS synchronized clock</description></item>
/// </list>
/// </remarks>
public interface IPhysicalClockSource
{
    /// <summary>
    /// Gets current physical time in nanoseconds since Unix epoch (1970-01-01 00:00:00 UTC).
    /// </summary>
    long GetCurrentTimeNanos();

    /// <summary>
    /// Gets the estimated error bound in nanoseconds (±).
    /// </summary>
    /// <remarks>
    /// For example, if error bound is 1,000,000 (1ms), the actual time is within
    /// [GetCurrentTimeNanos() - 1ms, GetCurrentTimeNanos() + 1ms].
    /// </remarks>
    long GetErrorBound();

    /// <summary>
    /// Indicates whether the clock is synchronized with an external time source.
    /// </summary>
    bool IsSynchronized { get; }

    /// <summary>
    /// Gets the clock drift rate in parts per million (PPM).
    /// </summary>
    /// <remarks>
    /// Positive values indicate the clock runs faster than real time.
    /// For example, +10 PPM means the clock gains 10 microseconds per second.
    /// </remarks>
    double GetClockDrift();
}

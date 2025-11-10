using System;
using System.Collections.Generic;

namespace Orleans.GpuBridge.Abstractions.Temporal;

/// <summary>
/// Combines Hybrid Logical Clock (HLC) with Vector Clock for both total ordering and causality tracking.
/// </summary>
/// <remarks>
/// <para>
/// This clock provides the best of both worlds:
/// - **HLC**: Total ordering of all events, close to physical time
/// - **Vector Clock**: Precise causal dependency tracking
/// </para>
/// <para>
/// Use HLC for:
/// - Timestamp-based queries (e.g., "events in last 5 seconds")
/// - Total ordering for deterministic replay
/// - Deadlines and timeouts
/// </para>
/// <para>
/// Use Vector Clock for:
/// - Detecting causal dependencies between actors
/// - Identifying concurrent operations (conflicts)
/// - Causal message delivery
/// </para>
/// <para>
/// Typical workflow:
/// 1. Local event: Increment both HLC and vector clock
/// 2. Send message: Attach both timestamps
/// 3. Receive message: Update both clocks, check causality
/// 4. Query: Use HLC for time-based, VC for causality-based
/// </para>
/// </remarks>
public sealed class HybridCausalClock
{
    private readonly HybridLogicalClock _hlc;
    private readonly ushort _actorId;
    private VectorClock _vectorClock;
    private readonly object _lock = new();

    /// <summary>
    /// Gets the actor ID for this clock.
    /// </summary>
    public ushort ActorId => _actorId;

    /// <summary>
    /// Gets the current vector clock (snapshot).
    /// </summary>
    public VectorClock VectorClock
    {
        get
        {
            lock (_lock)
            {
                return _vectorClock;
            }
        }
    }

    /// <summary>
    /// Creates a new hybrid causal clock.
    /// </summary>
    /// <param name="actorId">Unique identifier for this actor</param>
    /// <param name="clockSource">Physical clock source (optional)</param>
    public HybridCausalClock(ushort actorId, IPhysicalClockSource? clockSource = null)
    {
        _actorId = actorId;
        _hlc = new HybridLogicalClock(actorId, clockSource);
        _vectorClock = VectorClock.Create(actorId, 0);
    }

    /// <summary>
    /// Generates a timestamp for a local event.
    /// </summary>
    /// <returns>Combined HLC and vector clock timestamp</returns>
    /// <remarks>
    /// Atomically increments both HLC and vector clock.
    /// </remarks>
    public HybridCausalTimestamp Now()
    {
        lock (_lock)
        {
            var hlcTimestamp = _hlc.Now();
            _vectorClock = _vectorClock.Increment(_actorId);

            return new HybridCausalTimestamp
            {
                HLC = hlcTimestamp,
                VectorClock = _vectorClock
            };
        }
    }

    /// <summary>
    /// Updates the clock upon receiving a message.
    /// </summary>
    /// <param name="receivedTimestamp">Timestamp from received message</param>
    /// <returns>New timestamp after update</returns>
    /// <remarks>
    /// Updates both HLC (for total ordering) and vector clock (for causality).
    /// The returned timestamp should be used as the receive event timestamp.
    /// </remarks>
    public HybridCausalTimestamp Update(HybridCausalTimestamp receivedTimestamp)
    {
        ArgumentNullException.ThrowIfNull(receivedTimestamp);

        lock (_lock)
        {
            // Update HLC
            var hlcTimestamp = _hlc.Update(receivedTimestamp.HLC);

            // Update vector clock: merge + increment
            _vectorClock = _vectorClock.Update(receivedTimestamp.VectorClock, _actorId);

            return new HybridCausalTimestamp
            {
                HLC = hlcTimestamp,
                VectorClock = _vectorClock
            };
        }
    }

    /// <summary>
    /// Checks if a message should be delivered based on causal dependencies.
    /// </summary>
    /// <param name="messageTimestamp">Timestamp of the message to deliver</param>
    /// <param name="dependencies">Causal dependencies (other messages that must be delivered first)</param>
    /// <returns>True if all dependencies are satisfied</returns>
    /// <remarks>
    /// A message can be delivered if all its causal dependencies have been observed.
    /// This is checked using vector clock comparison.
    /// </remarks>
    public bool CanDeliver(HybridCausalTimestamp messageTimestamp, IEnumerable<HybridCausalTimestamp>? dependencies = null)
    {
        ArgumentNullException.ThrowIfNull(messageTimestamp);

        lock (_lock)
        {
            // Check if we've observed all events in the message's vector clock
            // (except the sender's own event)
            var messageVC = messageTimestamp.VectorClock;
            var currentVC = _vectorClock;

            foreach (var actorId in messageVC.ActorIds)
            {
                // Skip the sender's entry (we haven't delivered the message yet)
                if (actorId == messageTimestamp.SenderId)
                    continue;

                // Check if we've seen all events from this actor that the sender saw
                if (currentVC[actorId] < messageVC[actorId])
                {
                    return false; // Missing causal dependencies
                }
            }

            // Check explicit dependencies if provided
            if (dependencies != null)
            {
                foreach (var dependency in dependencies)
                {
                    if (!dependency.VectorClock.IsDominatedBy(currentVC))
                    {
                        return false; // Dependency not yet delivered
                    }
                }
            }

            return true;
        }
    }

    /// <summary>
    /// Gets the causal relationship between two timestamps.
    /// </summary>
    /// <param name="t1">First timestamp</param>
    /// <param name="t2">Second timestamp</param>
    /// <returns>Causal relationship</returns>
    public static CausalRelationship GetCausalRelationship(
        HybridCausalTimestamp t1,
        HybridCausalTimestamp t2)
    {
        ArgumentNullException.ThrowIfNull(t1);
        ArgumentNullException.ThrowIfNull(t2);

        return t1.VectorClock.CompareTo(t2.VectorClock);
    }

    /// <summary>
    /// Resets the clock (for testing purposes).
    /// </summary>
    internal void Reset()
    {
        lock (_lock)
        {
            _vectorClock = VectorClock.Create(_actorId, 0);
        }
    }

    /// <inheritdoc/>
    public override string ToString()
    {
        lock (_lock)
        {
            return $"HybridCausalClock(Actor={_actorId}, VC={_vectorClock})";
        }
    }
}

/// <summary>
/// Represents a timestamp that combines HLC and Vector Clock.
/// </summary>
public sealed record HybridCausalTimestamp
{
    /// <summary>
    /// Hybrid Logical Clock timestamp (total ordering, physical time approximation).
    /// </summary>
    public required HybridTimestamp HLC { get; init; }

    /// <summary>
    /// Vector Clock timestamp (causal dependencies).
    /// </summary>
    public required VectorClock VectorClock { get; init; }

    /// <summary>
    /// The actor ID that generated this timestamp (from HLC).
    /// </summary>
    public ushort SenderId => HLC.NodeId;

    /// <summary>
    /// Physical time in nanoseconds (from HLC).
    /// </summary>
    public long PhysicalTimeNanos => HLC.PhysicalTime;

    /// <summary>
    /// Compares this timestamp with another for total ordering (using HLC).
    /// </summary>
    /// <param name="other">Timestamp to compare with</param>
    /// <returns>Comparison result</returns>
    /// <remarks>
    /// Uses HLC for total ordering. For causal relationships, use GetCausalRelationship().
    /// </remarks>
    public int CompareHLC(HybridCausalTimestamp other)
    {
        ArgumentNullException.ThrowIfNull(other);
        return HLC.CompareTo(other.HLC);
    }

    /// <summary>
    /// Gets the causal relationship with another timestamp (using Vector Clock).
    /// </summary>
    /// <param name="other">Timestamp to compare with</param>
    /// <returns>Causal relationship</returns>
    public CausalRelationship GetCausalRelationship(HybridCausalTimestamp other)
    {
        ArgumentNullException.ThrowIfNull(other);
        return VectorClock.CompareTo(other.VectorClock);
    }

    /// <summary>
    /// Checks if this timestamp causally precedes another.
    /// </summary>
    /// <param name="other">Timestamp to compare with</param>
    /// <returns>True if this happens-before other</returns>
    public bool HappensBefore(HybridCausalTimestamp other)
    {
        return GetCausalRelationship(other) == CausalRelationship.HappensBefore;
    }

    /// <summary>
    /// Checks if this timestamp is concurrent with another.
    /// </summary>
    /// <param name="other">Timestamp to compare with</param>
    /// <returns>True if concurrent</returns>
    public bool IsConcurrentWith(HybridCausalTimestamp other)
    {
        return GetCausalRelationship(other) == CausalRelationship.Concurrent;
    }

    /// <inheritdoc/>
    public override string ToString()
    {
        return $"HCT(HLC={HLC}, VC={VectorClock})";
    }

    /// <summary>
    /// Converts to compact byte array for serialization.
    /// </summary>
    /// <returns>Byte array representation</returns>
    public byte[] ToBytes()
    {
        var hlcBytes = new byte[18]; // HLC: 8 (physical) + 8 (logical) + 2 (node)
        var span = hlcBytes.AsSpan();
        BitConverter.TryWriteBytes(span, HLC.PhysicalTime);
        BitConverter.TryWriteBytes(span[8..], HLC.LogicalCounter);
        BitConverter.TryWriteBytes(span[16..], HLC.NodeId);

        var vcBytes = VectorClock.ToBytes();

        // Combine: [HLC length:2] [HLC data] [VC data]
        var result = new byte[2 + hlcBytes.Length + vcBytes.Length];
        BitConverter.TryWriteBytes(result, (ushort)hlcBytes.Length);
        hlcBytes.CopyTo(result, 2);
        vcBytes.CopyTo(result, 2 + hlcBytes.Length);

        return result;
    }

    /// <summary>
    /// Deserializes from byte array.
    /// </summary>
    /// <param name="bytes">Byte array from ToBytes()</param>
    /// <returns>Deserialized timestamp</returns>
    public static HybridCausalTimestamp FromBytes(ReadOnlySpan<byte> bytes)
    {
        if (bytes.Length < 2)
            throw new ArgumentException("Invalid hybrid causal timestamp bytes", nameof(bytes));

        var hlcLength = BitConverter.ToUInt16(bytes);
        if (bytes.Length < 2 + hlcLength)
            throw new ArgumentException("Incomplete hybrid causal timestamp bytes", nameof(bytes));

        // Deserialize HLC
        var hlcSpan = bytes.Slice(2, hlcLength);
        var physicalTime = BitConverter.ToInt64(hlcSpan);
        var logicalCounter = BitConverter.ToInt64(hlcSpan[8..]);
        var nodeId = BitConverter.ToUInt16(hlcSpan[16..]);
        var hlc = new HybridTimestamp(physicalTime, logicalCounter, nodeId);

        // Deserialize VC
        var vcSpan = bytes[(2 + hlcLength)..];
        var vc = VectorClock.FromBytes(vcSpan);

        return new HybridCausalTimestamp
        {
            HLC = hlc,
            VectorClock = vc
        };
    }
}

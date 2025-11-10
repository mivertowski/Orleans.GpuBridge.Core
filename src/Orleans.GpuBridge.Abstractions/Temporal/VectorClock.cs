using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Text;

namespace Orleans.GpuBridge.Abstractions.Temporal;

/// <summary>
/// Vector clock for tracking causal relationships in distributed systems.
/// </summary>
/// <remarks>
/// <para>
/// Vector clocks provide partial ordering of events across distributed actors.
/// Each actor maintains a vector of logical times, one entry per actor.
/// </para>
/// <para>
/// Key properties:
/// - If A happens-before B, then VC(A) &lt; VC(B)
/// - If VC(A) &lt; VC(B), then A happens-before B
/// - If neither VC(A) &lt; VC(B) nor VC(B) &lt; VC(A), then A and B are concurrent
/// </para>
/// <para>
/// Use cases:
/// - Multi-actor causal dependency tracking
/// - Detecting concurrent operations (conflicts)
/// - Causal message delivery ordering
/// - Distributed debugging and replay
/// </para>
/// </remarks>
public sealed class VectorClock : IEquatable<VectorClock>, IComparable<VectorClock>
{
    private readonly ImmutableDictionary<ushort, long> _clocks;

    /// <summary>
    /// Gets the clock value for a specific actor.
    /// </summary>
    /// <param name="actorId">Actor identifier</param>
    /// <returns>Clock value (0 if actor not in vector)</returns>
    public long this[ushort actorId] => _clocks.TryGetValue(actorId, out var value) ? value : 0;

    /// <summary>
    /// Gets all actor IDs in this vector clock.
    /// </summary>
    public IEnumerable<ushort> ActorIds => _clocks.Keys;

    /// <summary>
    /// Gets the number of actors in this vector clock.
    /// </summary>
    public int Count => _clocks.Count;

    /// <summary>
    /// Creates an empty vector clock.
    /// </summary>
    public VectorClock()
    {
        _clocks = ImmutableDictionary<ushort, long>.Empty;
    }

    /// <summary>
    /// Creates a vector clock with the specified clock values.
    /// </summary>
    /// <param name="clocks">Dictionary of actor IDs to clock values</param>
    public VectorClock(ImmutableDictionary<ushort, long> clocks)
    {
        ArgumentNullException.ThrowIfNull(clocks);
        _clocks = clocks;
    }

    /// <summary>
    /// Creates a vector clock from a dictionary.
    /// </summary>
    /// <param name="clocks">Dictionary of actor IDs to clock values</param>
    public VectorClock(IDictionary<ushort, long> clocks)
    {
        ArgumentNullException.ThrowIfNull(clocks);
        _clocks = clocks.ToImmutableDictionary();
    }

    /// <summary>
    /// Increments the clock for the specified actor.
    /// </summary>
    /// <param name="actorId">Actor to increment</param>
    /// <returns>New vector clock with incremented value</returns>
    /// <remarks>
    /// Called when a local event occurs at the actor.
    /// </remarks>
    public VectorClock Increment(ushort actorId)
    {
        var currentValue = this[actorId];
        var newClocks = _clocks.SetItem(actorId, currentValue + 1);
        return new VectorClock(newClocks);
    }

    /// <summary>
    /// Merges this vector clock with another, taking the maximum of each entry.
    /// </summary>
    /// <param name="other">Vector clock to merge with</param>
    /// <returns>New vector clock with merged values</returns>
    /// <remarks>
    /// Called when receiving a message from another actor.
    /// The result represents knowledge of both vector clocks combined.
    /// </remarks>
    public VectorClock Merge(VectorClock other)
    {
        ArgumentNullException.ThrowIfNull(other);

        // Get union of all actor IDs
        var allActorIds = _clocks.Keys.Union(other._clocks.Keys).Distinct();

        var builder = ImmutableDictionary.CreateBuilder<ushort, long>();
        foreach (var actorId in allActorIds)
        {
            var maxValue = Math.Max(this[actorId], other[actorId]);
            builder.Add(actorId, maxValue);
        }

        return new VectorClock(builder.ToImmutable());
    }

    /// <summary>
    /// Updates this vector clock upon receiving a message.
    /// </summary>
    /// <param name="receivedClock">Vector clock from received message</param>
    /// <param name="localActorId">Local actor ID</param>
    /// <returns>New vector clock after update</returns>
    /// <remarks>
    /// Equivalent to: Merge(receivedClock).Increment(localActorId)
    /// This is the typical operation when processing a received message.
    /// </remarks>
    public VectorClock Update(VectorClock receivedClock, ushort localActorId)
    {
        return Merge(receivedClock).Increment(localActorId);
    }

    /// <summary>
    /// Determines the causal relationship between this and another vector clock.
    /// </summary>
    /// <param name="other">Vector clock to compare with</param>
    /// <returns>Causal relationship</returns>
    public CausalRelationship CompareTo(VectorClock other)
    {
        ArgumentNullException.ThrowIfNull(other);

        var allActorIds = _clocks.Keys.Union(other._clocks.Keys).Distinct();

        bool thisLessOrEqual = true;
        bool otherLessOrEqual = true;
        bool strictlyLess = false;
        bool strictlyGreater = false;

        foreach (var actorId in allActorIds)
        {
            var thisValue = this[actorId];
            var otherValue = other[actorId];

            if (thisValue < otherValue)
            {
                otherLessOrEqual = false;
                strictlyLess = true;
            }
            else if (thisValue > otherValue)
            {
                thisLessOrEqual = false;
                strictlyGreater = true;
            }
        }

        // Determine relationship
        if (thisLessOrEqual && otherLessOrEqual)
        {
            // All entries equal
            return CausalRelationship.Equal;
        }
        else if (thisLessOrEqual && strictlyLess)
        {
            // This happens-before other
            return CausalRelationship.HappensBefore;
        }
        else if (otherLessOrEqual && strictlyGreater)
        {
            // Other happens-before this
            return CausalRelationship.HappensAfter;
        }
        else
        {
            // Neither less than the other - concurrent events
            return CausalRelationship.Concurrent;
        }
    }

    /// <summary>
    /// Checks if this vector clock happens-before another.
    /// </summary>
    /// <param name="other">Vector clock to compare with</param>
    /// <returns>True if this happens-before other</returns>
    public bool HappensBefore(VectorClock other)
    {
        return CompareTo(other) == CausalRelationship.HappensBefore;
    }

    /// <summary>
    /// Checks if this vector clock happens-after another.
    /// </summary>
    /// <param name="other">Vector clock to compare with</param>
    /// <returns>True if this happens-after other</returns>
    public bool HappensAfter(VectorClock other)
    {
        return CompareTo(other) == CausalRelationship.HappensAfter;
    }

    /// <summary>
    /// Checks if this vector clock is concurrent with another.
    /// </summary>
    /// <param name="other">Vector clock to compare with</param>
    /// <returns>True if concurrent (neither happens-before nor happens-after)</returns>
    public bool IsConcurrentWith(VectorClock other)
    {
        return CompareTo(other) == CausalRelationship.Concurrent;
    }

    /// <summary>
    /// Checks if this vector clock is causally dominated by another.
    /// </summary>
    /// <param name="other">Vector clock to compare with</param>
    /// <returns>True if this ≤ other (happens-before or equal)</returns>
    /// <remarks>
    /// Used to check if all events in this clock are known by other.
    /// </remarks>
    public bool IsDominatedBy(VectorClock other)
    {
        var relationship = CompareTo(other);
        return relationship == CausalRelationship.HappensBefore ||
               relationship == CausalRelationship.Equal;
    }

    #region IComparable Implementation

    /// <summary>
    /// Compares this vector clock with another for ordering.
    /// </summary>
    /// <param name="other">Vector clock to compare with</param>
    /// <returns>
    /// -1 if this happens-before other,
    /// 0 if equal,
    /// 1 if this happens-after other or concurrent
    /// </returns>
    /// <remarks>
    /// Note: This provides a total ordering for sorting, but concurrent
    /// events are arbitrarily ordered. Use CompareTo(VectorClock) for
    /// precise causal relationship determination.
    /// </remarks>
    int IComparable<VectorClock>.CompareTo(VectorClock? other)
    {
        if (other is null) return 1;

        var relationship = CompareTo(other);
        return relationship switch
        {
            CausalRelationship.HappensBefore => -1,
            CausalRelationship.Equal => 0,
            CausalRelationship.HappensAfter => 1,
            CausalRelationship.Concurrent => CompareArbitrarily(other), // Arbitrary but consistent
            _ => 0
        };
    }

    /// <summary>
    /// Provides arbitrary but consistent ordering for concurrent events.
    /// </summary>
    private int CompareArbitrarily(VectorClock other)
    {
        // Use lexicographic ordering of (count, sum of values, first actor ID, ...)
        if (Count != other.Count)
            return Count.CompareTo(other.Count);

        var thisSum = _clocks.Values.Sum();
        var otherSum = other._clocks.Values.Sum();
        if (thisSum != otherSum)
            return thisSum.CompareTo(otherSum);

        // Compare individual entries
        var allActorIds = _clocks.Keys.Union(other._clocks.Keys).OrderBy(id => id);
        foreach (var actorId in allActorIds)
        {
            var cmp = this[actorId].CompareTo(other[actorId]);
            if (cmp != 0) return cmp;
        }

        return 0;
    }

    #endregion

    #region Equality

    /// <inheritdoc/>
    public bool Equals(VectorClock? other)
    {
        if (other is null) return false;
        if (ReferenceEquals(this, other)) return true;

        if (_clocks.Count != other._clocks.Count)
            return false;

        foreach (var kvp in _clocks)
        {
            if (!other._clocks.TryGetValue(kvp.Key, out var otherValue))
                return false;
            if (kvp.Value != otherValue)
                return false;
        }

        return true;
    }

    /// <inheritdoc/>
    public override bool Equals(object? obj) => obj is VectorClock other && Equals(other);

    /// <inheritdoc/>
    public override int GetHashCode()
    {
        var hash = new HashCode();
        foreach (var kvp in _clocks.OrderBy(kvp => kvp.Key))
        {
            hash.Add(kvp.Key);
            hash.Add(kvp.Value);
        }
        return hash.ToHashCode();
    }

    /// <summary>
    /// Equality operator.
    /// </summary>
    public static bool operator ==(VectorClock? left, VectorClock? right)
    {
        if (left is null) return right is null;
        return left.Equals(right);
    }

    /// <summary>
    /// Inequality operator.
    /// </summary>
    public static bool operator !=(VectorClock? left, VectorClock? right) => !(left == right);

    #endregion

    #region Serialization Helpers

    /// <summary>
    /// Converts vector clock to compact byte array.
    /// </summary>
    /// <returns>Byte array representation</returns>
    /// <remarks>
    /// Format: [count:2 bytes] ([actorId:2 bytes][value:8 bytes])*
    /// </remarks>
    public byte[] ToBytes()
    {
        var buffer = new byte[2 + _clocks.Count * 10]; // 2 for count, 10 per entry
        var span = buffer.AsSpan();

        // Write count
        BitConverter.TryWriteBytes(span, (ushort)_clocks.Count);
        var offset = 2;

        // Write entries (sorted for deterministic serialization)
        foreach (var kvp in _clocks.OrderBy(kvp => kvp.Key))
        {
            BitConverter.TryWriteBytes(span[offset..], kvp.Key);
            offset += 2;
            BitConverter.TryWriteBytes(span[offset..], kvp.Value);
            offset += 8;
        }

        return buffer;
    }

    /// <summary>
    /// Creates vector clock from byte array.
    /// </summary>
    /// <param name="bytes">Byte array from ToBytes()</param>
    /// <returns>Deserialized vector clock</returns>
    public static VectorClock FromBytes(ReadOnlySpan<byte> bytes)
    {
        if (bytes.Length < 2)
            throw new ArgumentException("Invalid vector clock bytes", nameof(bytes));

        var count = BitConverter.ToUInt16(bytes);
        var expectedLength = 2 + count * 10;
        if (bytes.Length != expectedLength)
            throw new ArgumentException($"Invalid vector clock bytes length: expected {expectedLength}, got {bytes.Length}", nameof(bytes));

        var builder = ImmutableDictionary.CreateBuilder<ushort, long>();
        var offset = 2;

        for (int i = 0; i < count; i++)
        {
            var actorId = BitConverter.ToUInt16(bytes[offset..]);
            offset += 2;
            var value = BitConverter.ToInt64(bytes[offset..]);
            offset += 8;

            builder.Add(actorId, value);
        }

        return new VectorClock(builder.ToImmutable());
    }

    #endregion

    /// <inheritdoc/>
    public override string ToString()
    {
        if (_clocks.Count == 0)
            return "VC{}";

        var sb = new StringBuilder("VC{");
        bool first = true;
        foreach (var kvp in _clocks.OrderBy(kvp => kvp.Key))
        {
            if (!first) sb.Append(", ");
            sb.Append($"A{kvp.Key}:{kvp.Value}");
            first = false;
        }
        sb.Append('}');
        return sb.ToString();
    }

    /// <summary>
    /// Creates a vector clock with a single actor.
    /// </summary>
    /// <param name="actorId">Actor ID</param>
    /// <param name="value">Clock value</param>
    /// <returns>Vector clock with single entry</returns>
    public static VectorClock Create(ushort actorId, long value = 0)
    {
        var clocks = ImmutableDictionary.Create<ushort, long>().Add(actorId, value);
        return new VectorClock(clocks);
    }
}

/// <summary>
/// Represents the causal relationship between two vector clocks.
/// </summary>
public enum CausalRelationship
{
    /// <summary>
    /// The vector clocks are equal (same events observed).
    /// </summary>
    Equal,

    /// <summary>
    /// This vector clock happens-before the other (causally precedes).
    /// </summary>
    HappensBefore,

    /// <summary>
    /// This vector clock happens-after the other (causally follows).
    /// </summary>
    HappensAfter,

    /// <summary>
    /// The vector clocks are concurrent (independent events).
    /// </summary>
    Concurrent
}

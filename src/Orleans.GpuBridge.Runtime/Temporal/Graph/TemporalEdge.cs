using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using Orleans.GpuBridge.Abstractions.Temporal;

namespace Orleans.GpuBridge.Runtime.Temporal.Graph;

/// <summary>
/// Represents a temporal edge in a graph with a validity time range.
/// </summary>
/// <remarks>
/// <para>
/// Temporal edges exist only during a specific time range, enabling queries like:
/// - "What edges existed between time T₁ and T₂?"
/// - "Find all paths that occurred within a 5-second window"
/// - "Which connections were active during the transaction?"
/// </para>
/// <para>
/// Use cases:
/// - Financial transaction graphs (money transfers)
/// - Social network graphs (friendship changes over time)
/// - Communication graphs (who talked to whom and when)
/// - Supply chain graphs (goods movement over time)
/// </para>
/// </remarks>
public readonly struct TemporalEdge : IEquatable<TemporalEdge>, IComparable<TemporalEdge>
{
    /// <summary>
    /// Source node identifier.
    /// </summary>
    public ulong SourceId { get; init; }

    /// <summary>
    /// Target node identifier.
    /// </summary>
    public ulong TargetId { get; init; }

    /// <summary>
    /// Start of the time range when this edge was valid (inclusive).
    /// </summary>
    public long ValidFrom { get; init; }

    /// <summary>
    /// End of the time range when this edge was valid (inclusive).
    /// </summary>
    public long ValidTo { get; init; }

    /// <summary>
    /// Hybrid Logical Clock timestamp when the edge was created.
    /// </summary>
    public HybridTimestamp HLC { get; init; }

    /// <summary>
    /// Edge weight (e.g., transaction amount, connection strength).
    /// </summary>
    public double Weight { get; init; }

    /// <summary>
    /// Edge type identifier (e.g., "transfer", "friendship", "message").
    /// </summary>
    public string EdgeType { get; init; }

    /// <summary>
    /// Additional edge metadata (optional).
    /// </summary>
    public ImmutableDictionary<string, object>? Properties { get; init; }

    /// <summary>
    /// Duration of the edge's validity in nanoseconds.
    /// </summary>
    public long DurationNanos => ValidTo - ValidFrom;

    /// <summary>
    /// Creates a new temporal edge.
    /// </summary>
    public TemporalEdge(
        ulong sourceId,
        ulong targetId,
        long validFrom,
        long validTo,
        HybridTimestamp hlc,
        double weight = 1.0,
        string? edgeType = null,
        ImmutableDictionary<string, object>? properties = null)
    {
        if (validTo < validFrom)
            throw new ArgumentException("ValidTo must be >= ValidFrom");

        SourceId = sourceId;
        TargetId = targetId;
        ValidFrom = validFrom;
        ValidTo = validTo;
        HLC = hlc;
        Weight = weight;
        EdgeType = edgeType ?? "default";
        Properties = properties;
    }

    /// <summary>
    /// Checks if this edge was valid at the specified time.
    /// </summary>
    public bool IsValidAt(long timeNanos)
    {
        return timeNanos >= ValidFrom && timeNanos <= ValidTo;
    }

    /// <summary>
    /// Checks if this edge overlaps with the specified time range.
    /// </summary>
    public bool OverlapsWith(long startTime, long endTime)
    {
        return ValidFrom <= endTime && ValidTo >= startTime;
    }

    /// <summary>
    /// Gets the intersection of this edge's validity with the specified range.
    /// </summary>
    public (long start, long end)? GetIntersection(long startTime, long endTime)
    {
        if (!OverlapsWith(startTime, endTime))
            return null;

        return (Math.Max(ValidFrom, startTime), Math.Min(ValidTo, endTime));
    }

    /// <summary>
    /// Compares edges by ValidFrom time (for sorting).
    /// </summary>
    public int CompareTo(TemporalEdge other)
    {
        var timeComparison = ValidFrom.CompareTo(other.ValidFrom);
        if (timeComparison != 0)
            return timeComparison;

        // Secondary sort by HLC
        return HLC.CompareTo(other.HLC);
    }

    /// <inheritdoc/>
    public bool Equals(TemporalEdge other)
    {
        return SourceId == other.SourceId &&
               TargetId == other.TargetId &&
               ValidFrom == other.ValidFrom &&
               ValidTo == other.ValidTo &&
               HLC.Equals(other.HLC);
    }

    /// <inheritdoc/>
    public override bool Equals(object? obj)
    {
        return obj is TemporalEdge other && Equals(other);
    }

    /// <inheritdoc/>
    public override int GetHashCode()
    {
        return HashCode.Combine(SourceId, TargetId, ValidFrom, ValidTo, HLC);
    }

    /// <inheritdoc/>
    public override string ToString()
    {
        var duration = DurationNanos / 1_000_000.0; // Convert to milliseconds
        return $"Edge({SourceId}→{TargetId}, {ValidFrom}ns-{ValidTo}ns, {duration:F1}ms, weight={Weight})";
    }

    public static bool operator ==(TemporalEdge left, TemporalEdge right) => left.Equals(right);
    public static bool operator !=(TemporalEdge left, TemporalEdge right) => !left.Equals(right);
}

/// <summary>
/// Represents a path through a temporal graph.
/// </summary>
/// <remarks>
/// A temporal path is a sequence of edges where each edge's ValidFrom time
/// is within an acceptable time window from the previous edge.
/// </remarks>
public sealed class TemporalPath : IEquatable<TemporalPath>
{
    private readonly List<TemporalEdge> _edges;

    /// <summary>
    /// Edges in the path (ordered by time).
    /// </summary>
    public IReadOnlyList<TemporalEdge> Edges => _edges;

    /// <summary>
    /// Number of edges in the path.
    /// </summary>
    public int Length => _edges.Count;

    /// <summary>
    /// Total weight of the path (sum of edge weights).
    /// </summary>
    public double TotalWeight { get; private set; }

    /// <summary>
    /// Start time of the path (first edge's ValidFrom).
    /// </summary>
    public long StartTime => _edges.Count > 0 ? _edges[0].ValidFrom : 0;

    /// <summary>
    /// End time of the path (last edge's ValidFrom).
    /// </summary>
    public long EndTime => _edges.Count > 0 ? _edges[^1].ValidFrom : 0;

    /// <summary>
    /// Total duration of the path in nanoseconds.
    /// </summary>
    public long TotalDurationNanos => EndTime - StartTime;

    /// <summary>
    /// Source node of the path.
    /// </summary>
    public ulong SourceNode => _edges.Count > 0 ? _edges[0].SourceId : 0;

    /// <summary>
    /// Target node of the path.
    /// </summary>
    public ulong TargetNode => _edges.Count > 0 ? _edges[^1].TargetId : 0;

    /// <summary>
    /// Creates an empty temporal path.
    /// </summary>
    public TemporalPath()
    {
        _edges = new List<TemporalEdge>();
        TotalWeight = 0;
    }

    /// <summary>
    /// Creates a temporal path from a list of edges.
    /// </summary>
    public TemporalPath(IEnumerable<TemporalEdge> edges)
    {
        _edges = new List<TemporalEdge>(edges);
        TotalWeight = _edges.Sum(e => e.Weight);
    }

    /// <summary>
    /// Adds an edge to the path.
    /// </summary>
    /// <remarks>
    /// The edge must connect to the last node in the path and occur after the last edge.
    /// </remarks>
    public void AddEdge(TemporalEdge edge)
    {
        if (_edges.Count > 0)
        {
            var lastEdge = _edges[^1];

            // Validate connectivity
            if (lastEdge.TargetId != edge.SourceId)
            {
                throw new ArgumentException(
                    $"Edge does not connect: last edge ends at {lastEdge.TargetId}, new edge starts at {edge.SourceId}");
            }

            // Validate temporal ordering
            if (edge.ValidFrom < lastEdge.ValidFrom)
            {
                throw new ArgumentException(
                    $"Edge violates temporal ordering: new edge starts at {edge.ValidFrom}, but last edge started at {lastEdge.ValidFrom}");
            }
        }

        _edges.Add(edge);
        TotalWeight += edge.Weight;
    }

    /// <summary>
    /// Creates a new path by appending an edge to this path.
    /// </summary>
    public TemporalPath Append(TemporalEdge edge)
    {
        var newPath = new TemporalPath(_edges);
        newPath.AddEdge(edge);
        return newPath;
    }

    /// <summary>
    /// Gets all nodes in the path (including source and target).
    /// </summary>
    public IEnumerable<ulong> GetNodes()
    {
        if (_edges.Count == 0)
            yield break;

        yield return _edges[0].SourceId;

        foreach (var edge in _edges)
        {
            yield return edge.TargetId;
        }
    }

    /// <summary>
    /// Checks if the path visits a specific node.
    /// </summary>
    public bool ContainsNode(ulong nodeId)
    {
        if (_edges.Count == 0)
            return false;

        if (_edges[0].SourceId == nodeId)
            return true;

        return _edges.Any(e => e.TargetId == nodeId);
    }

    /// <summary>
    /// Gets the time delay between two consecutive edges in the path.
    /// </summary>
    public IEnumerable<long> GetEdgeDelays()
    {
        for (int i = 1; i < _edges.Count; i++)
        {
            yield return _edges[i].ValidFrom - _edges[i - 1].ValidFrom;
        }
    }

    /// <summary>
    /// Checks if all edges in the path occur within the specified time window.
    /// </summary>
    public bool FitsWithinTimeWindow(long maxDurationNanos)
    {
        return TotalDurationNanos <= maxDurationNanos;
    }

    /// <inheritdoc/>
    public bool Equals(TemporalPath? other)
    {
        if (other is null || _edges.Count != other._edges.Count)
            return false;

        for (int i = 0; i < _edges.Count; i++)
        {
            if (!_edges[i].Equals(other._edges[i]))
                return false;
        }

        return true;
    }

    /// <inheritdoc/>
    public override bool Equals(object? obj) => obj is TemporalPath other && Equals(other);

    /// <inheritdoc/>
    public override int GetHashCode()
    {
        var hash = new HashCode();
        foreach (var edge in _edges)
        {
            hash.Add(edge);
        }
        return hash.ToHashCode();
    }

    /// <inheritdoc/>
    public override string ToString()
    {
        if (_edges.Count == 0)
            return "TemporalPath(empty)";

        var nodes = string.Join("→", GetNodes());
        var duration = TotalDurationNanos / 1_000_000.0; // Convert to milliseconds
        return $"TemporalPath({nodes}, {_edges.Count} edges, {duration:F1}ms, weight={TotalWeight:F1})";
    }
}

/// <summary>
/// Stores edges for a single node with efficient temporal queries.
/// </summary>
internal sealed class TemporalEdgeList
{
    // Edges sorted by ValidFrom time for efficient temporal queries
    private readonly SortedList<long, List<TemporalEdge>> _edgesByTime = new();
    private int _totalEdges;

    /// <summary>
    /// Gets the total number of edges.
    /// </summary>
    public int Count => _totalEdges;

    /// <summary>
    /// Adds an edge to the list.
    /// </summary>
    public void Add(TemporalEdge edge)
    {
        if (!_edgesByTime.TryGetValue(edge.ValidFrom, out var list))
        {
            list = new List<TemporalEdge>();
            _edgesByTime[edge.ValidFrom] = list;
        }

        list.Add(edge);
        _totalEdges++;
    }

    /// <summary>
    /// Queries edges that overlap with the specified time range.
    /// </summary>
    /// <remarks>
    /// <para>
    /// For overlap queries, we must check all edges from the beginning because
    /// an edge with ValidFrom &lt; startTime may still overlap if ValidTo &gt;= startTime.
    /// </para>
    /// <para>
    /// The iteration stops when we reach edges where ValidFrom &gt; endTime (these cannot overlap).
    /// Complexity: O(N) where N is the number of edges. For truly efficient interval queries,
    /// consider using the interval tree index instead.
    /// </para>
    /// </remarks>
    public IEnumerable<TemporalEdge> Query(long startTime, long endTime)
    {
        // For overlap queries, we must start from the beginning because edges
        // with ValidFrom < startTime may still be active at startTime if their ValidTo >= startTime.
        // We can stop early when ValidFrom > endTime (no future edges can overlap).
        foreach (var (time, edges) in _edgesByTime)
        {
            // Stop if we've passed the end time - no future edges can overlap
            if (time > endTime)
                break;

            // Check each edge in this time bucket for overlap
            foreach (var edge in edges)
            {
                if (edge.OverlapsWith(startTime, endTime))
                    yield return edge;
            }
        }
    }

    /// <summary>
    /// Gets all edges (in temporal order).
    /// </summary>
    public IEnumerable<TemporalEdge> GetAll()
    {
        foreach (var (_, edges) in _edgesByTime)
        {
            foreach (var edge in edges)
            {
                yield return edge;
            }
        }
    }

    /// <summary>
    /// Binary search to find the largest index where key[index] is less than or equal to target.
    /// </summary>
    private static int BinarySearchFloor(IList<long> keys, long target)
    {
        int left = 0, right = keys.Count - 1;
        int result = 0;

        while (left <= right)
        {
            int mid = left + (right - left) / 2;

            if (keys[mid] <= target)
            {
                result = mid;
                left = mid + 1;
            }
            else
            {
                right = mid - 1;
            }
        }

        return result;
    }
}

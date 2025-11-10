using System;
using System.Collections.Generic;
using System.Linq;

namespace Orleans.GpuBridge.Runtime.Temporal.Graph;

/// <summary>
/// Interval tree for efficient temporal range queries.
/// </summary>
/// <typeparam name="TKey">The interval boundary type (typically long for nanosecond timestamps)</typeparam>
/// <typeparam name="TValue">The value type stored in intervals</typeparam>
/// <remarks>
/// <para>
/// An interval tree stores intervals [start, end] and efficiently answers queries like:
/// "Find all intervals that overlap with [query_start, query_end]"
/// </para>
/// <para>
/// Performance:
/// - Construction: O(N log N) where N is the number of intervals
/// - Query: O(log N + M) where M is the number of matching intervals
/// - Space: O(N)
/// </para>
/// <para>
/// Use cases:
/// - Find all temporal edges active during a time range
/// - Find all events that occurred in a time window
/// - Detect overlapping time intervals
/// </para>
/// </remarks>
public sealed class IntervalTree<TKey, TValue> where TKey : IComparable<TKey>
{
    private IntervalNode? _root;
    private int _count;

    /// <summary>
    /// Gets the number of intervals in the tree.
    /// </summary>
    public int Count => _count;

    /// <summary>
    /// Adds an interval to the tree.
    /// </summary>
    /// <param name="start">Start of the interval (inclusive)</param>
    /// <param name="end">End of the interval (inclusive)</param>
    /// <param name="value">Value associated with the interval</param>
    public void Add(TKey start, TKey end, TValue value)
    {
        if (start.CompareTo(end) > 0)
            throw new ArgumentException("Start must be <= End");

        var interval = new Interval(start, end, value);
        _root = Insert(_root, interval);
        _count++;
    }

    /// <summary>
    /// Queries all intervals that overlap with the specified range.
    /// </summary>
    /// <param name="start">Start of the query range</param>
    /// <param name="end">End of the query range</param>
    /// <returns>Values of all overlapping intervals</returns>
    public IEnumerable<TValue> Query(TKey start, TKey end)
    {
        if (start.CompareTo(end) > 0)
            throw new ArgumentException("Start must be <= End");

        var results = new List<TValue>();
        QueryInternal(_root, start, end, results);
        return results;
    }

    /// <summary>
    /// Queries all intervals that contain the specified point.
    /// </summary>
    public IEnumerable<TValue> QueryPoint(TKey point)
    {
        return Query(point, point);
    }

    /// <summary>
    /// Clears all intervals from the tree.
    /// </summary>
    public void Clear()
    {
        _root = null;
        _count = 0;
    }

    /// <summary>
    /// Gets all intervals in the tree (in no particular order).
    /// </summary>
    public IEnumerable<(TKey start, TKey end, TValue value)> GetAll()
    {
        var results = new List<(TKey, TKey, TValue)>();
        GetAllInternal(_root, results);
        return results;
    }

    /// <summary>
    /// Represents an interval with associated value.
    /// </summary>
    private sealed class Interval
    {
        public TKey Start { get; }
        public TKey End { get; }
        public TValue Value { get; }

        public Interval(TKey start, TKey end, TValue value)
        {
            Start = start;
            End = end;
            Value = value;
        }

        /// <summary>
        /// Checks if this interval overlaps with [start, end].
        /// </summary>
        public bool Overlaps(TKey start, TKey end)
        {
            // Two intervals [a1, a2] and [b1, b2] overlap if:
            // a1 <= b2 AND a2 >= b1
            return Start.CompareTo(end) <= 0 && End.CompareTo(start) >= 0;
        }
    }

    /// <summary>
    /// Node in the interval tree.
    /// </summary>
    private sealed class IntervalNode
    {
        public Interval Interval { get; }
        public TKey Max { get; set; } // Maximum End value in subtree
        public IntervalNode? Left { get; set; }
        public IntervalNode? Right { get; set; }

        public IntervalNode(Interval interval)
        {
            Interval = interval;
            Max = interval.End;
        }
    }

    /// <summary>
    /// Inserts an interval into the tree.
    /// </summary>
    private static IntervalNode Insert(IntervalNode? node, Interval interval)
    {
        // Base case: empty tree
        if (node == null)
            return new IntervalNode(interval);

        // Choose subtree based on start time (use as BST key)
        var comparison = interval.Start.CompareTo(node.Interval.Start);

        if (comparison < 0)
        {
            node.Left = Insert(node.Left, interval);
        }
        else
        {
            node.Right = Insert(node.Right, interval);
        }

        // Update max value in this subtree
        node.Max = MaxOf(node.Max, interval.End);
        if (node.Left != null)
            node.Max = MaxOf(node.Max, node.Left.Max);
        if (node.Right != null)
            node.Max = MaxOf(node.Max, node.Right.Max);

        return node;
    }

    /// <summary>
    /// Queries the tree for overlapping intervals.
    /// </summary>
    private static void QueryInternal(
        IntervalNode? node,
        TKey start,
        TKey end,
        List<TValue> results)
    {
        if (node == null)
            return;

        // Check if current interval overlaps
        if (node.Interval.Overlaps(start, end))
        {
            results.Add(node.Interval.Value);
        }

        // Traverse left subtree if it might contain overlapping intervals
        // Left subtree might overlap if its maximum end >= query start
        if (node.Left != null && node.Left.Max.CompareTo(start) >= 0)
        {
            QueryInternal(node.Left, start, end, results);
        }

        // Traverse right subtree if it might contain overlapping intervals
        // Right subtree might overlap if its minimum start <= query end
        // Since we don't track minimum start, we check if node.Interval.Start <= end
        if (node.Right != null && node.Interval.Start.CompareTo(end) <= 0)
        {
            QueryInternal(node.Right, start, end, results);
        }
    }

    /// <summary>
    /// Gets all intervals in the tree.
    /// </summary>
    private static void GetAllInternal(
        IntervalNode? node,
        List<(TKey, TKey, TValue)> results)
    {
        if (node == null)
            return;

        results.Add((node.Interval.Start, node.Interval.End, node.Interval.Value));

        GetAllInternal(node.Left, results);
        GetAllInternal(node.Right, results);
    }

    /// <summary>
    /// Returns the maximum of two comparable values.
    /// </summary>
    private static TKey MaxOf(TKey a, TKey b)
    {
        return a.CompareTo(b) >= 0 ? a : b;
    }

    /// <inheritdoc/>
    public override string ToString()
    {
        return $"IntervalTree({_count} intervals)";
    }
}

/// <summary>
/// Extension methods for interval tree queries.
/// </summary>
public static class IntervalTreeExtensions
{
    /// <summary>
    /// Queries intervals using a time range specification.
    /// </summary>
    public static IEnumerable<TValue> QueryTimeRange<TValue>(
        this IntervalTree<long, TValue> tree,
        long startNanos,
        long endNanos)
    {
        return tree.Query(startNanos, endNanos);
    }

    /// <summary>
    /// Queries intervals that were active at a specific point in time.
    /// </summary>
    public static IEnumerable<TValue> QueryAtTime<TValue>(
        this IntervalTree<long, TValue> tree,
        long timeNanos)
    {
        return tree.QueryPoint(timeNanos);
    }
}

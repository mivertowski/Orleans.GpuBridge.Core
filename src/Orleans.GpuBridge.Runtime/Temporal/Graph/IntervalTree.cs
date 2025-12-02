using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;

namespace Orleans.GpuBridge.Runtime.Temporal.Graph;

/// <summary>
/// Thread-safe interval tree for efficient temporal range queries.
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
/// Thread Safety:
/// - Uses ReaderWriterLockSlim for concurrent read access and exclusive write access
/// - Multiple readers can query simultaneously
/// - Writers get exclusive access for Add/Clear operations
/// </para>
/// <para>
/// Use cases:
/// - Find all temporal edges active during a time range
/// - Find all events that occurred in a time window
/// - Detect overlapping time intervals
/// </para>
/// </remarks>
public sealed class IntervalTree<TKey, TValue> : IDisposable where TKey : IComparable<TKey>
{
    private readonly ReaderWriterLockSlim _lock = new(LockRecursionPolicy.NoRecursion);
    private IntervalNode? _root;
    private int _count;
    private long _insertionSequence; // Monotonic counter for guaranteed uniqueness
    private bool _disposed;

    /// <summary>
    /// Gets the number of intervals in the tree.
    /// </summary>
    public int Count
    {
        get
        {
            ThrowIfDisposed();
            _lock.EnterReadLock();
            try
            {
                return _count;
            }
            finally
            {
                _lock.ExitReadLock();
            }
        }
    }

    /// <summary>
    /// Adds an interval to the tree.
    /// </summary>
    /// <param name="start">Start of the interval (inclusive)</param>
    /// <param name="end">End of the interval (inclusive)</param>
    /// <param name="value">Value associated with the interval</param>
    public void Add(TKey start, TKey end, TValue value)
    {
        ThrowIfDisposed();

        if (start.CompareTo(end) > 0)
            throw new ArgumentException("Start must be <= End");

        _lock.EnterWriteLock();
        try
        {
            var insertionId = _insertionSequence++;
            var interval = new Interval(start, end, value, insertionId);
            _root = Insert(_root, interval);
            _count++;
        }
        finally
        {
            _lock.ExitWriteLock();
        }
    }

    /// <summary>
    /// Queries all intervals that overlap with the specified range.
    /// </summary>
    /// <param name="start">Start of the query range</param>
    /// <param name="end">End of the query range</param>
    /// <returns>Values of all overlapping intervals</returns>
    public IEnumerable<TValue> Query(TKey start, TKey end)
    {
        ThrowIfDisposed();

        if (start.CompareTo(end) > 0)
            throw new ArgumentException("Start must be <= End");

        _lock.EnterReadLock();
        try
        {
            var results = new List<TValue>();
            QueryInternal(_root, start, end, results);
            return results;
        }
        finally
        {
            _lock.ExitReadLock();
        }
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
        ThrowIfDisposed();

        _lock.EnterWriteLock();
        try
        {
            _root = null;
            _count = 0;
        }
        finally
        {
            _lock.ExitWriteLock();
        }
    }

    /// <summary>
    /// Gets all intervals in the tree (in no particular order).
    /// </summary>
    public IEnumerable<(TKey start, TKey end, TValue value)> GetAll()
    {
        ThrowIfDisposed();

        _lock.EnterReadLock();
        try
        {
            var results = new List<(TKey, TKey, TValue)>();
            GetAllInternal(_root, results);
            return results;
        }
        finally
        {
            _lock.ExitReadLock();
        }
    }

    /// <summary>
    /// Represents an interval with associated value.
    /// </summary>
    private sealed class Interval
    {
        public TKey Start { get; }
        public TKey End { get; }
        public TValue Value { get; }
        public long InsertionId { get; }

        public Interval(TKey start, TKey end, TValue value, long insertionId)
        {
            Start = start;
            End = end;
            Value = value;
            InsertionId = insertionId;
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
    /// Node in the AVL-balanced interval tree.
    /// </summary>
    private sealed class IntervalNode
    {
        public Interval Interval { get; }
        public TKey Max { get; set; } // Maximum End value in subtree
        public int Height { get; set; } // Height for AVL balancing
        public IntervalNode? Left { get; set; }
        public IntervalNode? Right { get; set; }

        public IntervalNode(Interval interval)
        {
            Interval = interval;
            Max = interval.End;
            Height = 1; // New node has height 1
        }
    }

    /// <summary>
    /// Inserts an interval into the AVL-balanced tree.
    /// </summary>
    private static IntervalNode Insert(IntervalNode? node, Interval interval)
    {
        // Base case: empty tree
        if (node == null)
            return new IntervalNode(interval);

        // Choose subtree based on start time (primary key) and end time (tie-breaker)
        // This ensures total ordering and prevents infinite recursion with duplicate start times
        var startComparison = interval.Start.CompareTo(node.Interval.Start);

        // Use end time as tie-breaker when start times are equal
        int comparison;
        if (startComparison == 0)
        {
            comparison = interval.End.CompareTo(node.Interval.End);
            // If both start AND end are equal, use insertion sequence as final tie-breaker
            // This guarantees total ordering with O(1) comparison and no hash collisions
            if (comparison == 0)
            {
                comparison = interval.InsertionId.CompareTo(node.Interval.InsertionId);
            }
        }
        else
        {
            comparison = startComparison;
        }

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

        // Update height of current node
        node.Height = 1 + Math.Max(GetHeight(node.Left), GetHeight(node.Right));

        // Get balance factor to check if rebalancing is needed
        int balance = GetBalance(node);

        // Left-Left case: Right rotation
        if (balance > 1 && GetBalance(node.Left) >= 0)
        {
            return RotateRight(node);
        }

        // Left-Right case: Left rotation on left child, then right rotation on node
        if (balance > 1 && GetBalance(node.Left) < 0)
        {
            node.Left = RotateLeft(node.Left!);
            return RotateRight(node);
        }

        // Right-Right case: Left rotation
        if (balance < -1 && GetBalance(node.Right) <= 0)
        {
            return RotateLeft(node);
        }

        // Right-Left case: Right rotation on right child, then left rotation on node
        if (balance < -1 && GetBalance(node.Right) > 0)
        {
            node.Right = RotateRight(node.Right!);
            return RotateLeft(node);
        }

        return node;
    }

    /// <summary>
    /// Gets the height of a node (0 for null nodes).
    /// </summary>
    private static int GetHeight(IntervalNode? node)
    {
        return node?.Height ?? 0;
    }

    /// <summary>
    /// Gets the balance factor of a node.
    /// Positive means left-heavy, negative means right-heavy.
    /// </summary>
    private static int GetBalance(IntervalNode? node)
    {
        if (node == null)
            return 0;
        return GetHeight(node.Left) - GetHeight(node.Right);
    }

    /// <summary>
    /// Performs a right rotation on the subtree rooted at node.
    /// </summary>
    private static IntervalNode RotateRight(IntervalNode node)
    {
        var newRoot = node.Left!;
        var temp = newRoot.Right;

        // Perform rotation
        newRoot.Right = node;
        node.Left = temp;

        // Update heights
        node.Height = 1 + Math.Max(GetHeight(node.Left), GetHeight(node.Right));
        newRoot.Height = 1 + Math.Max(GetHeight(newRoot.Left), GetHeight(newRoot.Right));

        // Update max values
        node.Max = node.Interval.End;
        if (node.Left != null)
            node.Max = MaxOf(node.Max, node.Left.Max);
        if (node.Right != null)
            node.Max = MaxOf(node.Max, node.Right.Max);

        newRoot.Max = newRoot.Interval.End;
        if (newRoot.Left != null)
            newRoot.Max = MaxOf(newRoot.Max, newRoot.Left.Max);
        if (newRoot.Right != null)
            newRoot.Max = MaxOf(newRoot.Max, newRoot.Right.Max);

        return newRoot;
    }

    /// <summary>
    /// Performs a left rotation on the subtree rooted at node.
    /// </summary>
    private static IntervalNode RotateLeft(IntervalNode node)
    {
        var newRoot = node.Right!;
        var temp = newRoot.Left;

        // Perform rotation
        newRoot.Left = node;
        node.Right = temp;

        // Update heights
        node.Height = 1 + Math.Max(GetHeight(node.Left), GetHeight(node.Right));
        newRoot.Height = 1 + Math.Max(GetHeight(newRoot.Left), GetHeight(newRoot.Right));

        // Update max values
        node.Max = node.Interval.End;
        if (node.Left != null)
            node.Max = MaxOf(node.Max, node.Left.Max);
        if (node.Right != null)
            node.Max = MaxOf(node.Max, node.Right.Max);

        newRoot.Max = newRoot.Interval.End;
        if (newRoot.Left != null)
            newRoot.Max = MaxOf(newRoot.Max, newRoot.Left.Max);
        if (newRoot.Right != null)
            newRoot.Max = MaxOf(newRoot.Max, newRoot.Right.Max);

        return newRoot;
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
        var count = 0;
        _lock.EnterReadLock();
        try
        {
            count = _count;
        }
        finally
        {
            _lock.ExitReadLock();
        }
        return $"IntervalTree({count} intervals)";
    }

    /// <summary>
    /// Disposes the interval tree and releases the lock resources.
    /// </summary>
    public void Dispose()
    {
        if (_disposed)
            return;

        _disposed = true;
        _lock.Dispose();
    }

    private void ThrowIfDisposed()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
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

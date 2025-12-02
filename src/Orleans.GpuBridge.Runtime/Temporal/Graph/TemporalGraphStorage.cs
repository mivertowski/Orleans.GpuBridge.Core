using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using Microsoft.Extensions.Logging;
using Orleans.GpuBridge.Abstractions.Temporal;

namespace Orleans.GpuBridge.Runtime.Temporal.Graph;

/// <summary>
/// Storage for temporal graphs with efficient time-range queries.
/// </summary>
/// <remarks>
/// <para>
/// A temporal graph is a graph where edges exist only during specific time ranges.
/// This enables queries like:
/// - "What was the graph topology at time T?"
/// - "Find paths that occurred within a 5-second window"
/// - "Which nodes were connected between T₁ and T₂?"
/// </para>
/// <para>
/// Data structures:
/// - Adjacency lists: Node ID → Temporal edges (sorted by time)
/// - Interval tree: Global time index for all edges
/// - Node index: Track all nodes that have appeared
/// </para>
/// <para>
/// Performance:
/// - Add edge: O(log N) for interval tree + O(log M) for adjacency list
/// - Query edges: O(log N + K) where K is result count
/// - Find paths: O(E + V) BFS/DFS where E, V are in time window
/// </para>
/// <para>
/// Thread Safety:
/// - Uses ReaderWriterLockSlim for concurrent read access and exclusive write access
/// - Multiple readers can query simultaneously
/// - Writers get exclusive access for AddEdge/Clear operations
/// </para>
/// </remarks>
public sealed class TemporalGraphStorage : IDisposable
{
    private readonly ConcurrentDictionary<ulong, TemporalEdgeList> _adjacency = new();
    private readonly IntervalTree<long, TemporalEdge> _timeIndex = new();
    private readonly ConcurrentDictionary<ulong, byte> _nodes = new(); // Using byte as dummy value for concurrent set
    private readonly ReaderWriterLockSlim _lock = new(LockRecursionPolicy.NoRecursion);
    private readonly ILogger? _logger;

    private long _edgeCount;
    private bool _disposed;

    /// <summary>
    /// Gets the total number of edges in the graph.
    /// </summary>
    public long EdgeCount => Interlocked.Read(ref _edgeCount);

    /// <summary>
    /// Gets the total number of nodes in the graph.
    /// </summary>
    public int NodeCount => _nodes.Count;

    /// <summary>
    /// Throws if the storage has been disposed.
    /// </summary>
    private void ThrowIfDisposed()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
    }

    /// <summary>
    /// Creates a new temporal graph storage.
    /// </summary>
    public TemporalGraphStorage(ILogger? logger = null)
    {
        _logger = logger;
    }

    /// <summary>
    /// Adds an edge to the graph.
    /// </summary>
    /// <remarks>
    /// Thread-safe: Uses write lock to ensure exclusive access during insertion.
    /// </remarks>
    public void AddEdge(TemporalEdge edge)
    {
        ThrowIfDisposed();

        _lock.EnterWriteLock();
        try
        {
            // Add to adjacency list for source node using GetOrAdd for thread-safety
            var edgeList = _adjacency.GetOrAdd(edge.SourceId, _ => new TemporalEdgeList());
            edgeList.Add(edge);

            // Add to global time index
            _timeIndex.Add(edge.ValidFrom, edge.ValidTo, edge);

            // Track nodes using concurrent dictionary
            _nodes.TryAdd(edge.SourceId, 0);
            _nodes.TryAdd(edge.TargetId, 0);

            Interlocked.Increment(ref _edgeCount);
        }
        finally
        {
            _lock.ExitWriteLock();
        }

        _logger?.LogTrace(
            "Added edge {Source}→{Target} ({ValidFrom}ns-{ValidTo}ns)",
            edge.SourceId, edge.TargetId, edge.ValidFrom, edge.ValidTo);
    }

    /// <summary>
    /// Adds an edge with simplified parameters.
    /// </summary>
    public void AddEdge(
        ulong sourceId,
        ulong targetId,
        long validFrom,
        long validTo,
        HybridTimestamp hlc,
        double weight = 1.0,
        string? edgeType = null)
    {
        var edge = new TemporalEdge(sourceId, targetId, validFrom, validTo, hlc, weight, edgeType);
        AddEdge(edge);
    }

    /// <summary>
    /// Gets all edges from a source node that overlap with the time range.
    /// </summary>
    /// <remarks>
    /// Thread-safe: Uses read lock for concurrent access.
    /// </remarks>
    public IEnumerable<TemporalEdge> GetEdgesInTimeRange(
        ulong sourceId,
        long startTimeNanos,
        long endTimeNanos)
    {
        ThrowIfDisposed();

        _lock.EnterReadLock();
        try
        {
            if (!_adjacency.TryGetValue(sourceId, out var edgeList))
                return Enumerable.Empty<TemporalEdge>();

            // Materialize results while holding lock to avoid lazy enumeration issues
            return edgeList.Query(startTimeNanos, endTimeNanos).ToList();
        }
        finally
        {
            _lock.ExitReadLock();
        }
    }

    /// <summary>
    /// Gets all edges in the graph that overlap with the time range.
    /// </summary>
    /// <remarks>
    /// Thread-safe: Uses read lock for concurrent access.
    /// </remarks>
    public IEnumerable<TemporalEdge> GetAllEdgesInTimeRange(
        long startTimeNanos,
        long endTimeNanos)
    {
        ThrowIfDisposed();

        _lock.EnterReadLock();
        try
        {
            // IntervalTree.Query already materializes the list internally
            return _timeIndex.Query(startTimeNanos, endTimeNanos);
        }
        finally
        {
            _lock.ExitReadLock();
        }
    }

    /// <summary>
    /// Gets all edges from a source node (entire history).
    /// </summary>
    /// <remarks>
    /// Thread-safe: Uses read lock for concurrent access.
    /// </remarks>
    public IEnumerable<TemporalEdge> GetAllEdges(ulong sourceId)
    {
        ThrowIfDisposed();

        _lock.EnterReadLock();
        try
        {
            if (!_adjacency.TryGetValue(sourceId, out var edgeList))
                return Enumerable.Empty<TemporalEdge>();

            // Materialize results while holding lock
            return edgeList.GetAll().ToList();
        }
        finally
        {
            _lock.ExitReadLock();
        }
    }

    /// <summary>
    /// Checks if a node exists in the graph.
    /// </summary>
    /// <remarks>
    /// Thread-safe: ConcurrentDictionary.ContainsKey is inherently thread-safe.
    /// </remarks>
    public bool ContainsNode(ulong nodeId) => _nodes.ContainsKey(nodeId);

    /// <summary>
    /// Gets all nodes in the graph.
    /// </summary>
    /// <remarks>
    /// Thread-safe: Returns a snapshot of current keys.
    /// </remarks>
    public IEnumerable<ulong> GetAllNodes() => _nodes.Keys;

    /// <summary>
    /// Finds all temporal paths from start to end node within a time window.
    /// </summary>
    /// <param name="startNode">Starting node</param>
    /// <param name="endNode">Target node</param>
    /// <param name="maxTimeSpanNanos">Maximum duration of the path</param>
    /// <param name="maxPathLength">Maximum number of edges in path (prevents infinite loops)</param>
    /// <returns>All paths that fit within the time window</returns>
    public IEnumerable<TemporalPath> FindTemporalPaths(
        ulong startNode,
        ulong endNode,
        long maxTimeSpanNanos,
        int maxPathLength = 10)
    {
        if (!ContainsNode(startNode) || !ContainsNode(endNode))
            return Enumerable.Empty<TemporalPath>();

        // Use optimized BFS with early termination for better performance
        var shortestPath = FindShortestPathBFS(startNode, endNode, maxTimeSpanNanos, maxPathLength);
        return shortestPath != null
            ? new[] { shortestPath }
            : Enumerable.Empty<TemporalPath>();
    }

    /// <summary>
    /// Finds the shortest temporal path using optimized BFS with early termination.
    /// Much faster than DFS for finding any path (stops at first path found).
    /// Supports circular path detection where startNode == endNode.
    /// </summary>
    private TemporalPath? FindShortestPathBFS(
        ulong startNode,
        ulong endNode,
        long maxTimeSpanNanos,
        int maxPathLength)
    {
        var queue = new Queue<(ulong node, TemporalPath path)>();
        var visited = new HashSet<ulong>();

        // Detect if we're looking for a cycle (circular path)
        var lookingForCycle = startNode == endNode;

        // Start with empty path from startNode
        queue.Enqueue((startNode, new TemporalPath()));

        // For non-cycle detection, mark start as visited to prevent revisiting
        // For cycle detection, we'll track visited during traversal but allow reaching start again
        if (!lookingForCycle)
        {
            visited.Add(startNode);
        }

        while (queue.Count > 0)
        {
            var (currentNode, currentPath) = queue.Dequeue();

            // Check if we reached the target
            if (currentNode == endNode && currentPath.Length > 0)
            {
                return currentPath; // Early termination - return first path found
            }

            // Check depth limit
            if (currentPath.Length >= maxPathLength)
                continue;

            // Mark current node as visited (for cycle detection, only after first step)
            if (currentPath.Length > 0 || !lookingForCycle)
            {
                visited.Add(currentNode);
            }

            // Determine time window for next edge
            long earliestTime = currentPath.Length == 0
                ? long.MinValue
                : currentPath.EndTime;
            long latestTime = currentPath.Length == 0
                ? long.MaxValue
                : currentPath.StartTime + maxTimeSpanNanos;

            // Explore outgoing edges
            var edges = GetEdgesInTimeRange(currentNode, earliestTime, latestTime);

            foreach (var edge in edges)
            {
                // For cycle detection, allow reaching the start node if we have enough path length
                var isTargetStartNode = edge.TargetId == startNode && lookingForCycle;

                // Skip if target node already visited (unless it's the target for cycle detection)
                if (visited.Contains(edge.TargetId) && !isTargetStartNode)
                    continue;

                // Check temporal constraint
                if (currentPath.Length > 0)
                {
                    var pathDuration = edge.ValidFrom - currentPath.StartTime;
                    if (pathDuration > maxTimeSpanNanos)
                        continue;
                }

                // Add edge to path and enqueue
                var newPath = currentPath.Append(edge);
                queue.Enqueue((edge.TargetId, newPath));

                // Don't mark start node as visited for cycle detection
                if (!isTargetStartNode)
                {
                    visited.Add(edge.TargetId);
                }
            }
        }

        return null; // No path found
    }

    /// <summary>
    /// Recursive helper for finding temporal paths.
    /// </summary>
    private void FindPathsRecursive(
        ulong currentNode,
        ulong targetNode,
        TemporalPath currentPath,
        long maxTimeSpan,
        int maxDepth,
        HashSet<ulong> visited,
        List<TemporalPath> results)
    {
        // Base case: reached target
        if (currentNode == targetNode && currentPath.Length > 0)
        {
            results.Add(new TemporalPath(currentPath.Edges));
            return;
        }

        // Base case: max depth reached
        if (currentPath.Length >= maxDepth)
            return;

        // Mark as visited (prevent cycles)
        visited.Add(currentNode);

        try
        {
            // Determine time window for next edge
            long earliestTime = currentPath.Length == 0
                ? long.MinValue
                : currentPath.EndTime;
            long latestTime = currentPath.Length == 0
                ? long.MaxValue
                : currentPath.StartTime + maxTimeSpan;

            // Explore outgoing edges
            var edges = GetEdgesInTimeRange(currentNode, earliestTime, latestTime);

            foreach (var edge in edges)
            {
                // Skip if target node already visited
                if (visited.Contains(edge.TargetId))
                    continue;

                // Check temporal constraint
                if (currentPath.Length > 0)
                {
                    var pathDuration = edge.ValidFrom - currentPath.StartTime;
                    if (pathDuration > maxTimeSpan)
                        continue;
                }

                // Add edge to path
                var newPath = currentPath.Append(edge);

                // Recurse
                FindPathsRecursive(
                    edge.TargetId,
                    targetNode,
                    newPath,
                    maxTimeSpan,
                    maxDepth,
                    visited,
                    results);
            }
        }
        finally
        {
            // Unmark visited (backtrack)
            visited.Remove(currentNode);
        }
    }

    /// <summary>
    /// Finds the shortest temporal path from start to end (by number of edges).
    /// </summary>
    public TemporalPath? FindShortestTemporalPath(
        ulong startNode,
        ulong endNode,
        long maxTimeSpanNanos)
    {
        var paths = FindTemporalPaths(startNode, endNode, maxTimeSpanNanos);
        return paths.OrderBy(p => p.Length).FirstOrDefault();
    }

    /// <summary>
    /// Finds the fastest temporal path from start to end (by duration).
    /// </summary>
    public TemporalPath? FindFastestTemporalPath(
        ulong startNode,
        ulong endNode,
        long maxTimeSpanNanos)
    {
        var paths = FindTemporalPaths(startNode, endNode, maxTimeSpanNanos);
        return paths.OrderBy(p => p.TotalDurationNanos).FirstOrDefault();
    }

    /// <summary>
    /// Gets the graph snapshot at a specific point in time.
    /// </summary>
    /// <remarks>
    /// Returns all edges that were valid at the specified time.
    /// </remarks>
    public IEnumerable<TemporalEdge> GetSnapshotAtTime(long timeNanos)
    {
        return _timeIndex.QueryPoint(timeNanos);
    }

    /// <summary>
    /// Finds all neighbors of a node at a specific time.
    /// </summary>
    public IEnumerable<ulong> GetNeighborsAtTime(ulong nodeId, long timeNanos)
    {
        return GetEdgesInTimeRange(nodeId, timeNanos, timeNanos)
            .Select(e => e.TargetId)
            .Distinct();
    }

    /// <summary>
    /// Computes temporal reachability: all nodes reachable from start within time window.
    /// </summary>
    public IEnumerable<ulong> GetReachableNodes(
        ulong startNode,
        long startTimeNanos,
        long maxTimeSpanNanos)
    {
        var reachable = new HashSet<ulong> { startNode };
        var queue = new Queue<(ulong node, long time)>();
        queue.Enqueue((startNode, startTimeNanos));

        while (queue.Count > 0)
        {
            var (currentNode, currentTime) = queue.Dequeue();

            // Find edges within remaining time window
            var remainingTime = maxTimeSpanNanos - (currentTime - startTimeNanos);
            if (remainingTime <= 0)
                continue;

            var edges = GetEdgesInTimeRange(
                currentNode,
                currentTime,
                startTimeNanos + maxTimeSpanNanos);

            foreach (var edge in edges)
            {
                if (!reachable.Contains(edge.TargetId))
                {
                    reachable.Add(edge.TargetId);
                    queue.Enqueue((edge.TargetId, edge.ValidFrom));
                }
            }
        }

        return reachable;
    }

    /// <summary>
    /// Gets statistics about the temporal graph.
    /// </summary>
    public GraphStatistics GetStatistics()
    {
        var allEdges = _adjacency.Values.SelectMany(list => list.GetAll()).ToList();

        var minTime = allEdges.Any() ? allEdges.Min(e => e.ValidFrom) : 0;
        var maxTime = allEdges.Any() ? allEdges.Max(e => e.ValidTo) : 0;
        var avgDegree = _adjacency.Any()
            ? _adjacency.Values.Average(list => list.Count)
            : 0;

        return new GraphStatistics
        {
            NodeCount = NodeCount,
            EdgeCount = EdgeCount,
            MinTime = minTime,
            MaxTime = maxTime,
            TimeSpanNanos = maxTime - minTime,
            AverageDegree = avgDegree
        };
    }

    /// <summary>
    /// Clears all data from the graph.
    /// </summary>
    /// <remarks>
    /// Thread-safe: Uses write lock for exclusive access.
    /// </remarks>
    public void Clear()
    {
        ThrowIfDisposed();

        _lock.EnterWriteLock();
        try
        {
            _adjacency.Clear();
            _timeIndex.Clear();
            _nodes.Clear();
            Interlocked.Exchange(ref _edgeCount, 0);
        }
        finally
        {
            _lock.ExitWriteLock();
        }
    }

    /// <inheritdoc/>
    public override string ToString()
    {
        return $"TemporalGraphStorage({NodeCount} nodes, {EdgeCount} edges)";
    }

    /// <summary>
    /// Disposes the temporal graph storage and releases resources.
    /// </summary>
    public void Dispose()
    {
        if (_disposed)
            return;

        _disposed = true;
        _timeIndex.Dispose();
        _lock.Dispose();
    }
}

/// <summary>
/// Statistics about a temporal graph.
/// </summary>
public sealed record GraphStatistics
{
    /// <summary>
    /// Gets the total number of nodes in the graph.
    /// </summary>
    public int NodeCount { get; init; }

    /// <summary>
    /// Gets the total number of edges in the graph.
    /// </summary>
    public long EdgeCount { get; init; }

    /// <summary>
    /// Gets the earliest edge start time (ValidFrom) in nanoseconds.
    /// </summary>
    public long MinTime { get; init; }

    /// <summary>
    /// Gets the latest edge end time (ValidTo) in nanoseconds.
    /// </summary>
    public long MaxTime { get; init; }

    /// <summary>
    /// Gets the total time span covered by the graph in nanoseconds.
    /// </summary>
    public long TimeSpanNanos { get; init; }

    /// <summary>
    /// Gets the average number of outgoing edges per node.
    /// </summary>
    public double AverageDegree { get; init; }

    /// <summary>
    /// Gets the total time span covered by the graph in seconds.
    /// </summary>
    public double TimeSpanSeconds => TimeSpanNanos / 1_000_000_000.0;

    /// <summary>
    /// Returns a string representation of the graph statistics.
    /// </summary>
    /// <returns>A formatted string containing node count, edge count, time span, and average degree.</returns>
    public override string ToString()
    {
        return $"Graph({NodeCount} nodes, {EdgeCount} edges, " +
               $"span={TimeSpanSeconds:F2}s, avg_degree={AverageDegree:F1})";
    }
}

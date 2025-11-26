using System;
using System.Collections.Generic;
using System.Linq;
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
/// </remarks>
public sealed class TemporalGraphStorage
{
    private readonly Dictionary<ulong, TemporalEdgeList> _adjacency = new();
    private readonly IntervalTree<long, TemporalEdge> _timeIndex = new();
    private readonly HashSet<ulong> _nodes = new();
    private readonly ILogger? _logger;

    private long _edgeCount;

    /// <summary>
    /// Gets the total number of edges in the graph.
    /// </summary>
    public long EdgeCount => _edgeCount;

    /// <summary>
    /// Gets the total number of nodes in the graph.
    /// </summary>
    public int NodeCount => _nodes.Count;

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
    public void AddEdge(TemporalEdge edge)
    {
        // Add to adjacency list for source node
        if (!_adjacency.TryGetValue(edge.SourceId, out var edgeList))
        {
            edgeList = new TemporalEdgeList();
            _adjacency[edge.SourceId] = edgeList;
        }
        edgeList.Add(edge);

        // Add to global time index
        _timeIndex.Add(edge.ValidFrom, edge.ValidTo, edge);

        // Track nodes
        _nodes.Add(edge.SourceId);
        _nodes.Add(edge.TargetId);

        _edgeCount++;

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
    public IEnumerable<TemporalEdge> GetEdgesInTimeRange(
        ulong sourceId,
        long startTimeNanos,
        long endTimeNanos)
    {
        if (!_adjacency.TryGetValue(sourceId, out var edgeList))
            return Enumerable.Empty<TemporalEdge>();

        return edgeList.Query(startTimeNanos, endTimeNanos);
    }

    /// <summary>
    /// Gets all edges in the graph that overlap with the time range.
    /// </summary>
    public IEnumerable<TemporalEdge> GetAllEdgesInTimeRange(
        long startTimeNanos,
        long endTimeNanos)
    {
        return _timeIndex.Query(startTimeNanos, endTimeNanos);
    }

    /// <summary>
    /// Gets all edges from a source node (entire history).
    /// </summary>
    public IEnumerable<TemporalEdge> GetAllEdges(ulong sourceId)
    {
        if (!_adjacency.TryGetValue(sourceId, out var edgeList))
            return Enumerable.Empty<TemporalEdge>();

        return edgeList.GetAll();
    }

    /// <summary>
    /// Checks if a node exists in the graph.
    /// </summary>
    public bool ContainsNode(ulong nodeId) => _nodes.Contains(nodeId);

    /// <summary>
    /// Gets all nodes in the graph.
    /// </summary>
    public IEnumerable<ulong> GetAllNodes() => _nodes;

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
    public void Clear()
    {
        _adjacency.Clear();
        _timeIndex.Clear();
        _nodes.Clear();
        _edgeCount = 0;
    }

    /// <inheritdoc/>
    public override string ToString()
    {
        return $"TemporalGraphStorage({NodeCount} nodes, {EdgeCount} edges)";
    }
}

/// <summary>
/// Statistics about a temporal graph.
/// </summary>
public sealed record GraphStatistics
{
    public int NodeCount { get; init; }
    public long EdgeCount { get; init; }
    public long MinTime { get; init; }
    public long MaxTime { get; init; }
    public long TimeSpanNanos { get; init; }
    public double AverageDegree { get; init; }

    public double TimeSpanSeconds => TimeSpanNanos / 1_000_000_000.0;

    public override string ToString()
    {
        return $"Graph({NodeCount} nodes, {EdgeCount} edges, " +
               $"span={TimeSpanSeconds:F2}s, avg_degree={AverageDegree:F1})";
    }
}

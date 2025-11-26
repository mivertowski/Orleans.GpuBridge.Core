// Orleans.GpuBridge - GPU-native distributed computing for Microsoft Orleans
// Copyright (c) 2025 Michael Ivertowski

using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;

namespace Orleans.GpuBridge.Abstractions.Temporal.Graph;

/// <summary>
/// Interface for temporal graph storage with efficient time-range queries.
/// </summary>
/// <remarks>
/// <para>
/// A temporal graph is a graph where edges exist only during specific time ranges.
/// This enables queries like:
/// <list type="bullet">
/// <item><description>"What was the graph topology at time T?"</description></item>
/// <item><description>"Find paths that occurred within a 5-second window"</description></item>
/// <item><description>"Which nodes were connected between T₁ and T₂?"</description></item>
/// </list>
/// </para>
/// <para>
/// Performance targets:
/// <list type="bullet">
/// <item><description>Add edge: O(log N) for interval tree + O(log M) for adjacency list</description></item>
/// <item><description>Query edges: O(log N + K) where K is result count</description></item>
/// <item><description>Find paths: O(E + V) BFS/DFS where E, V are in time window</description></item>
/// </list>
/// </para>
/// </remarks>
public interface ITemporalGraphStorage
{
    /// <summary>
    /// Gets the total number of edges in the graph.
    /// </summary>
    long EdgeCount { get; }

    /// <summary>
    /// Gets the total number of nodes in the graph.
    /// </summary>
    int NodeCount { get; }

    /// <summary>
    /// Adds an edge to the graph.
    /// </summary>
    /// <param name="edge">The temporal edge to add.</param>
    void AddEdge(TemporalEdgeData edge);

    /// <summary>
    /// Adds an edge with simplified parameters.
    /// </summary>
    /// <param name="sourceId">Source node identifier.</param>
    /// <param name="targetId">Target node identifier.</param>
    /// <param name="validFrom">Start of validity period (nanoseconds).</param>
    /// <param name="validTo">End of validity period (nanoseconds).</param>
    /// <param name="hlc">Hybrid logical clock timestamp.</param>
    /// <param name="weight">Edge weight (default 1.0).</param>
    /// <param name="edgeType">Edge type identifier.</param>
    void AddEdge(
        ulong sourceId,
        ulong targetId,
        long validFrom,
        long validTo,
        HybridTimestamp hlc,
        double weight = 1.0,
        string? edgeType = null);

    /// <summary>
    /// Gets all edges from a source node that overlap with the time range.
    /// </summary>
    /// <param name="sourceId">Source node identifier.</param>
    /// <param name="startTimeNanos">Start of query range (nanoseconds).</param>
    /// <param name="endTimeNanos">End of query range (nanoseconds).</param>
    /// <returns>Edges that overlap with the specified time range.</returns>
    IEnumerable<TemporalEdgeData> GetEdgesInTimeRange(
        ulong sourceId,
        long startTimeNanos,
        long endTimeNanos);

    /// <summary>
    /// Gets all edges in the graph that overlap with the time range.
    /// </summary>
    /// <param name="startTimeNanos">Start of query range (nanoseconds).</param>
    /// <param name="endTimeNanos">End of query range (nanoseconds).</param>
    /// <returns>All edges that overlap with the specified time range.</returns>
    IEnumerable<TemporalEdgeData> GetAllEdgesInTimeRange(
        long startTimeNanos,
        long endTimeNanos);

    /// <summary>
    /// Gets all edges from a source node (entire history).
    /// </summary>
    /// <param name="sourceId">Source node identifier.</param>
    /// <returns>All edges from the source node.</returns>
    IEnumerable<TemporalEdgeData> GetAllEdges(ulong sourceId);

    /// <summary>
    /// Checks if a node exists in the graph.
    /// </summary>
    /// <param name="nodeId">Node identifier to check.</param>
    /// <returns>True if the node exists in the graph.</returns>
    bool ContainsNode(ulong nodeId);

    /// <summary>
    /// Gets all nodes in the graph.
    /// </summary>
    /// <returns>All node identifiers in the graph.</returns>
    IEnumerable<ulong> GetAllNodes();

    /// <summary>
    /// Finds all temporal paths from start to end node within a time window.
    /// </summary>
    /// <param name="startNode">Starting node identifier.</param>
    /// <param name="endNode">Target node identifier.</param>
    /// <param name="maxTimeSpanNanos">Maximum duration of the path in nanoseconds.</param>
    /// <param name="maxPathLength">Maximum number of edges in path (prevents infinite loops).</param>
    /// <returns>All paths that fit within the time window.</returns>
    IEnumerable<TemporalPathData> FindTemporalPaths(
        ulong startNode,
        ulong endNode,
        long maxTimeSpanNanos,
        int maxPathLength = 10);

    /// <summary>
    /// Finds the shortest temporal path from start to end (by number of edges).
    /// </summary>
    /// <param name="startNode">Starting node identifier.</param>
    /// <param name="endNode">Target node identifier.</param>
    /// <param name="maxTimeSpanNanos">Maximum duration of the path in nanoseconds.</param>
    /// <returns>The shortest path, or null if no path exists.</returns>
    TemporalPathData? FindShortestTemporalPath(
        ulong startNode,
        ulong endNode,
        long maxTimeSpanNanos);

    /// <summary>
    /// Finds the fastest temporal path from start to end (by duration).
    /// </summary>
    /// <param name="startNode">Starting node identifier.</param>
    /// <param name="endNode">Target node identifier.</param>
    /// <param name="maxTimeSpanNanos">Maximum duration of the path in nanoseconds.</param>
    /// <returns>The fastest path, or null if no path exists.</returns>
    TemporalPathData? FindFastestTemporalPath(
        ulong startNode,
        ulong endNode,
        long maxTimeSpanNanos);

    /// <summary>
    /// Gets the graph snapshot at a specific point in time.
    /// </summary>
    /// <param name="timeNanos">Point in time (nanoseconds).</param>
    /// <returns>All edges that were valid at the specified time.</returns>
    IEnumerable<TemporalEdgeData> GetSnapshotAtTime(long timeNanos);

    /// <summary>
    /// Finds all neighbors of a node at a specific time.
    /// </summary>
    /// <param name="nodeId">Node identifier.</param>
    /// <param name="timeNanos">Point in time (nanoseconds).</param>
    /// <returns>Neighbor node identifiers.</returns>
    IEnumerable<ulong> GetNeighborsAtTime(ulong nodeId, long timeNanos);

    /// <summary>
    /// Computes temporal reachability: all nodes reachable from start within time window.
    /// </summary>
    /// <param name="startNode">Starting node identifier.</param>
    /// <param name="startTimeNanos">Start time in nanoseconds.</param>
    /// <param name="maxTimeSpanNanos">Maximum time span in nanoseconds.</param>
    /// <returns>All reachable node identifiers.</returns>
    IEnumerable<ulong> GetReachableNodes(
        ulong startNode,
        long startTimeNanos,
        long maxTimeSpanNanos);

    /// <summary>
    /// Gets statistics about the temporal graph.
    /// </summary>
    /// <returns>Graph statistics including node/edge counts, time span, etc.</returns>
    TemporalGraphStatistics GetStatistics();

    /// <summary>
    /// Clears all data from the graph.
    /// </summary>
    void Clear();
}

/// <summary>
/// Async extension interface for temporal graph storage supporting distributed operations.
/// </summary>
public interface ITemporalGraphStorageAsync : ITemporalGraphStorage
{
    /// <summary>
    /// Asynchronously adds an edge to the graph with optional persistence.
    /// </summary>
    /// <param name="edge">The temporal edge to add.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>True if the edge was added successfully.</returns>
    Task<bool> AddEdgeAsync(TemporalEdgeData edge, CancellationToken cancellationToken = default);

    /// <summary>
    /// Asynchronously finds temporal paths with progress reporting.
    /// </summary>
    /// <param name="startNode">Starting node identifier.</param>
    /// <param name="endNode">Target node identifier.</param>
    /// <param name="maxTimeSpanNanos">Maximum duration of the path in nanoseconds.</param>
    /// <param name="maxPathLength">Maximum number of edges in path.</param>
    /// <param name="progress">Progress reporter.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Paths found within the time window.</returns>
    Task<IReadOnlyList<TemporalPathData>> FindTemporalPathsAsync(
        ulong startNode,
        ulong endNode,
        long maxTimeSpanNanos,
        int maxPathLength = 10,
        IProgress<PathSearchProgress>? progress = null,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Asynchronously persists the graph to durable storage.
    /// </summary>
    /// <param name="cancellationToken">Cancellation token.</param>
    Task PersistAsync(CancellationToken cancellationToken = default);

    /// <summary>
    /// Asynchronously loads the graph from durable storage.
    /// </summary>
    /// <param name="cancellationToken">Cancellation token.</param>
    Task LoadAsync(CancellationToken cancellationToken = default);
}

/// <summary>
/// Represents a temporal edge with validity time range.
/// </summary>
/// <remarks>
/// This is the abstraction-layer data transfer object for temporal edges.
/// The runtime layer has a corresponding struct optimized for storage.
/// </remarks>
[GenerateSerializer]
[Immutable]
public sealed record TemporalEdgeData
{
    /// <summary>
    /// Source node identifier.
    /// </summary>
    [Id(0)]
    public ulong SourceId { get; init; }

    /// <summary>
    /// Target node identifier.
    /// </summary>
    [Id(1)]
    public ulong TargetId { get; init; }

    /// <summary>
    /// Start of the time range when this edge was valid (inclusive, nanoseconds).
    /// </summary>
    [Id(2)]
    public long ValidFrom { get; init; }

    /// <summary>
    /// End of the time range when this edge was valid (inclusive, nanoseconds).
    /// </summary>
    [Id(3)]
    public long ValidTo { get; init; }

    /// <summary>
    /// Hybrid Logical Clock timestamp when the edge was created.
    /// </summary>
    [Id(4)]
    public HybridTimestamp HLC { get; init; }

    /// <summary>
    /// Edge weight (e.g., transaction amount, connection strength).
    /// </summary>
    [Id(5)]
    public double Weight { get; init; } = 1.0;

    /// <summary>
    /// Edge type identifier (e.g., "transfer", "friendship", "message").
    /// </summary>
    [Id(6)]
    public string EdgeType { get; init; } = "default";

    /// <summary>
    /// Additional edge metadata.
    /// </summary>
    [Id(7)]
    public IReadOnlyDictionary<string, object>? Properties { get; init; }

    /// <summary>
    /// Duration of the edge's validity in nanoseconds.
    /// </summary>
    public long DurationNanos => ValidTo - ValidFrom;

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
}

/// <summary>
/// Represents a path through a temporal graph.
/// </summary>
[GenerateSerializer]
public sealed record TemporalPathData
{
    /// <summary>
    /// Edges in the path (ordered by time).
    /// </summary>
    [Id(0)]
    public IReadOnlyList<TemporalEdgeData> Edges { get; init; } = Array.Empty<TemporalEdgeData>();

    /// <summary>
    /// Number of edges in the path.
    /// </summary>
    public int Length => Edges.Count;

    /// <summary>
    /// Total weight of the path (sum of edge weights).
    /// </summary>
    [Id(1)]
    public double TotalWeight { get; init; }

    /// <summary>
    /// Start time of the path (first edge's ValidFrom).
    /// </summary>
    public long StartTime => Edges.Count > 0 ? Edges[0].ValidFrom : 0;

    /// <summary>
    /// End time of the path (last edge's ValidFrom).
    /// </summary>
    public long EndTime => Edges.Count > 0 ? Edges[^1].ValidFrom : 0;

    /// <summary>
    /// Total duration of the path in nanoseconds.
    /// </summary>
    public long TotalDurationNanos => EndTime - StartTime;

    /// <summary>
    /// Source node of the path.
    /// </summary>
    public ulong SourceNode => Edges.Count > 0 ? Edges[0].SourceId : 0;

    /// <summary>
    /// Target node of the path.
    /// </summary>
    public ulong TargetNode => Edges.Count > 0 ? Edges[^1].TargetId : 0;

    /// <summary>
    /// Gets all nodes in the path (including source and target).
    /// </summary>
    public IEnumerable<ulong> GetNodes()
    {
        if (Edges.Count == 0)
            yield break;

        yield return Edges[0].SourceId;

        foreach (var edge in Edges)
        {
            yield return edge.TargetId;
        }
    }
}

/// <summary>
/// Statistics about a temporal graph.
/// </summary>
[GenerateSerializer]
public sealed record TemporalGraphStatistics
{
    /// <summary>
    /// Total number of nodes.
    /// </summary>
    [Id(0)]
    public int NodeCount { get; init; }

    /// <summary>
    /// Total number of edges.
    /// </summary>
    [Id(1)]
    public long EdgeCount { get; init; }

    /// <summary>
    /// Minimum timestamp in the graph (nanoseconds).
    /// </summary>
    [Id(2)]
    public long MinTime { get; init; }

    /// <summary>
    /// Maximum timestamp in the graph (nanoseconds).
    /// </summary>
    [Id(3)]
    public long MaxTime { get; init; }

    /// <summary>
    /// Time span of the graph in nanoseconds.
    /// </summary>
    [Id(4)]
    public long TimeSpanNanos { get; init; }

    /// <summary>
    /// Average out-degree per node.
    /// </summary>
    [Id(5)]
    public double AverageDegree { get; init; }

    /// <summary>
    /// Time span in seconds.
    /// </summary>
    public double TimeSpanSeconds => TimeSpanNanos / 1_000_000_000.0;
}

/// <summary>
/// Progress information for path search operations.
/// </summary>
public sealed record PathSearchProgress
{
    /// <summary>
    /// Number of nodes visited so far.
    /// </summary>
    public int NodesVisited { get; init; }

    /// <summary>
    /// Number of edges examined so far.
    /// </summary>
    public int EdgesExamined { get; init; }

    /// <summary>
    /// Number of paths found so far.
    /// </summary>
    public int PathsFound { get; init; }

    /// <summary>
    /// Current search depth (hops from start).
    /// </summary>
    public int CurrentDepth { get; init; }

    /// <summary>
    /// Elapsed time in nanoseconds.
    /// </summary>
    public long ElapsedNanos { get; init; }
}

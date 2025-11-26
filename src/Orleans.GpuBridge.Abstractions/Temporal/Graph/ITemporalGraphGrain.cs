// Orleans.GpuBridge - GPU-native distributed computing for Microsoft Orleans
// Copyright (c) 2025 Michael Ivertowski

using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using Orleans;

namespace Orleans.GpuBridge.Abstractions.Temporal.Graph;

/// <summary>
/// Orleans grain interface for distributed temporal graph storage.
/// </summary>
/// <remarks>
/// <para>
/// This grain provides distributed, persistent temporal graph storage with
/// Orleans-native replication and fault tolerance.
/// </para>
/// <para>
/// <strong>Use Cases:</strong>
/// <list type="bullet">
/// <item><description>Financial transaction graphs with time-windowed queries</description></item>
/// <item><description>Social network evolution over time</description></item>
/// <item><description>Communication pattern analysis</description></item>
/// <item><description>Causally-ordered event graphs</description></item>
/// </list>
/// </para>
/// </remarks>
public interface ITemporalGraphGrain : IGrainWithStringKey
{
    /// <summary>
    /// Adds a temporal edge to the graph.
    /// </summary>
    /// <param name="edge">The temporal edge to add.</param>
    /// <returns>Result of the add operation.</returns>
    Task<EdgeAddResult> AddEdgeAsync(TemporalEdgeData edge);

    /// <summary>
    /// Adds multiple edges in a batch operation.
    /// </summary>
    /// <param name="edges">Edges to add.</param>
    /// <returns>Result of the batch operation.</returns>
    Task<BatchEdgeResult> AddEdgesBatchAsync(IReadOnlyList<TemporalEdgeData> edges);

    /// <summary>
    /// Gets edges from a source node within a time range.
    /// </summary>
    /// <param name="sourceId">Source node identifier.</param>
    /// <param name="startTimeNanos">Start of query range (nanoseconds).</param>
    /// <param name="endTimeNanos">End of query range (nanoseconds).</param>
    /// <returns>Query result with matching edges.</returns>
    Task<EdgeQueryResult> GetEdgesAsync(
        ulong sourceId,
        long startTimeNanos,
        long endTimeNanos);

    /// <summary>
    /// Gets all edges in a time range across all nodes.
    /// </summary>
    /// <param name="startTimeNanos">Start of query range (nanoseconds).</param>
    /// <param name="endTimeNanos">End of query range (nanoseconds).</param>
    /// <param name="maxResults">Maximum results to return.</param>
    /// <returns>Query result with matching edges.</returns>
    Task<EdgeQueryResult> GetAllEdgesInRangeAsync(
        long startTimeNanos,
        long endTimeNanos,
        int maxResults = 1000);

    /// <summary>
    /// Finds temporal paths between two nodes.
    /// </summary>
    /// <param name="request">Path finding request with constraints.</param>
    /// <returns>Result containing found paths.</returns>
    Task<PathFindingResult> FindPathsAsync(PathFindingRequest request);

    /// <summary>
    /// Gets the graph snapshot at a specific time.
    /// </summary>
    /// <param name="timeNanos">Point in time (nanoseconds).</param>
    /// <param name="maxEdges">Maximum edges to return.</param>
    /// <returns>Snapshot result with active edges.</returns>
    Task<SnapshotResult> GetSnapshotAsync(long timeNanos, int maxEdges = 10000);

    /// <summary>
    /// Computes reachable nodes from a starting point.
    /// </summary>
    /// <param name="startNode">Starting node identifier.</param>
    /// <param name="startTimeNanos">Start time in nanoseconds.</param>
    /// <param name="maxTimeSpanNanos">Maximum time span in nanoseconds.</param>
    /// <param name="maxNodes">Maximum nodes to return.</param>
    /// <returns>Reachability result.</returns>
    Task<ReachabilityResult> GetReachableNodesAsync(
        ulong startNode,
        long startTimeNanos,
        long maxTimeSpanNanos,
        int maxNodes = 10000);

    /// <summary>
    /// Gets graph statistics.
    /// </summary>
    /// <returns>Current graph statistics.</returns>
    Task<TemporalGraphStatistics> GetStatisticsAsync();

    /// <summary>
    /// Checks if a node exists in the graph.
    /// </summary>
    /// <param name="nodeId">Node identifier.</param>
    /// <returns>True if the node exists.</returns>
    Task<bool> ContainsNodeAsync(ulong nodeId);

    /// <summary>
    /// Gets all node identifiers in the graph.
    /// </summary>
    /// <param name="maxNodes">Maximum nodes to return.</param>
    /// <returns>Node identifiers.</returns>
    Task<IReadOnlyList<ulong>> GetNodesAsync(int maxNodes = 10000);

    /// <summary>
    /// Clears all data from the graph.
    /// </summary>
    Task ClearAsync();

    /// <summary>
    /// Forces persistence of in-memory data.
    /// </summary>
    Task PersistAsync();
}

/// <summary>
/// Result of an edge add operation.
/// </summary>
[GenerateSerializer]
public sealed record EdgeAddResult
{
    /// <summary>
    /// Whether the operation succeeded.
    /// </summary>
    [Id(0)]
    public bool Success { get; init; }

    /// <summary>
    /// New edge count after addition.
    /// </summary>
    [Id(1)]
    public long EdgeCount { get; init; }

    /// <summary>
    /// Operation timestamp (nanoseconds).
    /// </summary>
    [Id(2)]
    public long TimestampNanos { get; init; }

    /// <summary>
    /// Error message if operation failed.
    /// </summary>
    [Id(3)]
    public string? ErrorMessage { get; init; }
}

/// <summary>
/// Result of a batch edge operation.
/// </summary>
[GenerateSerializer]
public sealed record BatchEdgeResult
{
    /// <summary>
    /// Whether the operation succeeded.
    /// </summary>
    [Id(0)]
    public bool Success { get; init; }

    /// <summary>
    /// Number of edges added.
    /// </summary>
    [Id(1)]
    public int EdgesAdded { get; init; }

    /// <summary>
    /// Number of edges that failed.
    /// </summary>
    [Id(2)]
    public int EdgesFailed { get; init; }

    /// <summary>
    /// New total edge count.
    /// </summary>
    [Id(3)]
    public long TotalEdgeCount { get; init; }

    /// <summary>
    /// Operation duration in nanoseconds.
    /// </summary>
    [Id(4)]
    public long DurationNanos { get; init; }
}

/// <summary>
/// Result of an edge query operation.
/// </summary>
[GenerateSerializer]
public sealed record EdgeQueryResult
{
    /// <summary>
    /// Matching edges.
    /// </summary>
    [Id(0)]
    public IReadOnlyList<TemporalEdgeData> Edges { get; init; } = Array.Empty<TemporalEdgeData>();

    /// <summary>
    /// Whether results were truncated due to limits.
    /// </summary>
    [Id(1)]
    public bool IsTruncated { get; init; }

    /// <summary>
    /// Total matching edges (may be more than returned).
    /// </summary>
    [Id(2)]
    public int TotalCount { get; init; }

    /// <summary>
    /// Query duration in nanoseconds.
    /// </summary>
    [Id(3)]
    public long QueryTimeNanos { get; init; }
}

/// <summary>
/// Request for path finding operation.
/// </summary>
[GenerateSerializer]
public sealed record PathFindingRequest
{
    /// <summary>
    /// Starting node identifier.
    /// </summary>
    [Id(0)]
    public ulong StartNode { get; init; }

    /// <summary>
    /// Target node identifier.
    /// </summary>
    [Id(1)]
    public ulong EndNode { get; init; }

    /// <summary>
    /// Maximum time span for paths in nanoseconds.
    /// </summary>
    [Id(2)]
    public long MaxTimeSpanNanos { get; init; }

    /// <summary>
    /// Maximum path length (number of edges).
    /// </summary>
    [Id(3)]
    public int MaxPathLength { get; init; } = 10;

    /// <summary>
    /// Maximum number of paths to return.
    /// </summary>
    [Id(4)]
    public int MaxResults { get; init; } = 100;

    /// <summary>
    /// Whether to find shortest path only.
    /// </summary>
    [Id(5)]
    public bool ShortestOnly { get; init; }

    /// <summary>
    /// Whether to find fastest path only (by duration).
    /// </summary>
    [Id(6)]
    public bool FastestOnly { get; init; }

    /// <summary>
    /// Filter by edge types (null = all types).
    /// </summary>
    [Id(7)]
    public IReadOnlyList<string>? EdgeTypeFilter { get; init; }
}

/// <summary>
/// Result of a path finding operation.
/// </summary>
[GenerateSerializer]
public sealed record PathFindingResult
{
    /// <summary>
    /// Found paths.
    /// </summary>
    [Id(0)]
    public IReadOnlyList<TemporalPathData> Paths { get; init; } = Array.Empty<TemporalPathData>();

    /// <summary>
    /// Whether search was completed or truncated.
    /// </summary>
    [Id(1)]
    public bool IsComplete { get; init; }

    /// <summary>
    /// Number of nodes visited during search.
    /// </summary>
    [Id(2)]
    public int NodesVisited { get; init; }

    /// <summary>
    /// Number of edges examined during search.
    /// </summary>
    [Id(3)]
    public int EdgesExamined { get; init; }

    /// <summary>
    /// Search duration in nanoseconds.
    /// </summary>
    [Id(4)]
    public long SearchTimeNanos { get; init; }
}

/// <summary>
/// Result of a graph snapshot operation.
/// </summary>
[GenerateSerializer]
public sealed record SnapshotResult
{
    /// <summary>
    /// Edges active at the snapshot time.
    /// </summary>
    [Id(0)]
    public IReadOnlyList<TemporalEdgeData> Edges { get; init; } = Array.Empty<TemporalEdgeData>();

    /// <summary>
    /// Snapshot timestamp.
    /// </summary>
    [Id(1)]
    public long TimeNanos { get; init; }

    /// <summary>
    /// Whether results were truncated.
    /// </summary>
    [Id(2)]
    public bool IsTruncated { get; init; }

    /// <summary>
    /// Total active edges at this time.
    /// </summary>
    [Id(3)]
    public int TotalActiveEdges { get; init; }

    /// <summary>
    /// Query duration in nanoseconds.
    /// </summary>
    [Id(4)]
    public long QueryTimeNanos { get; init; }
}

/// <summary>
/// Result of a reachability query.
/// </summary>
[GenerateSerializer]
public sealed record ReachabilityResult
{
    /// <summary>
    /// Reachable node identifiers.
    /// </summary>
    [Id(0)]
    public IReadOnlyList<ulong> ReachableNodes { get; init; } = Array.Empty<ulong>();

    /// <summary>
    /// Starting node.
    /// </summary>
    [Id(1)]
    public ulong StartNode { get; init; }

    /// <summary>
    /// Whether results were truncated.
    /// </summary>
    [Id(2)]
    public bool IsTruncated { get; init; }

    /// <summary>
    /// Query duration in nanoseconds.
    /// </summary>
    [Id(3)]
    public long QueryTimeNanos { get; init; }
}

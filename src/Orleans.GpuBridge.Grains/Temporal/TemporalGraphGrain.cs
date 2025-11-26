// Orleans.GpuBridge - GPU-native distributed computing for Microsoft Orleans
// Copyright (c) 2025 Michael Ivertowski

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Orleans;
using Orleans.Runtime;
using Orleans.GpuBridge.Abstractions.Temporal;
using Orleans.GpuBridge.Abstractions.Temporal.Graph;
using Orleans.GpuBridge.Runtime.Temporal.Graph;

namespace Orleans.GpuBridge.Grains.Temporal;

/// <summary>
/// Orleans grain implementation for distributed temporal graph storage.
/// </summary>
/// <remarks>
/// <para>
/// This grain provides persistent, Orleans-integrated temporal graph storage with:
/// <list type="bullet">
/// <item><description>Automatic state persistence via Orleans storage providers</description></item>
/// <item><description>HLC timestamps for causal ordering across distributed nodes</description></item>
/// <item><description>Efficient time-range queries using interval trees</description></item>
/// <item><description>GPU-ready data structures for future GPU acceleration</description></item>
/// </list>
/// </para>
/// </remarks>
public sealed class TemporalGraphGrain : Grain, ITemporalGraphGrain
{
    private readonly ILogger<TemporalGraphGrain> _logger;
    private readonly IPersistentState<TemporalGraphState> _state;

    // In-memory graph storage (backed by Orleans persistence)
    private TemporalGraphStorage _graphStorage = null!;
    private string _graphId = string.Empty;
    private HybridTimestamp _hlcTimestamp;
    private long _operationCount;

    public TemporalGraphGrain(
        ILogger<TemporalGraphGrain> logger,
        [PersistentState("temporalGraph", "GpuBridgeStore")]
        IPersistentState<TemporalGraphState> state)
    {
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        _state = state ?? throw new ArgumentNullException(nameof(state));
    }

    public override async Task OnActivateAsync(CancellationToken cancellationToken)
    {
        _graphId = this.GetPrimaryKeyString();
        _hlcTimestamp = HybridTimestamp.Now();

        // Initialize graph storage
        _graphStorage = new TemporalGraphStorage(_logger);

        // Restore from persisted state if available
        if (_state.State.Edges is { Count: > 0 })
        {
            foreach (var edgeData in _state.State.Edges)
            {
                var edge = ConvertFromData(edgeData);
                _graphStorage.AddEdge(edge);
            }

            _logger.LogInformation(
                "Temporal graph {GraphId} restored with {EdgeCount} edges and {NodeCount} nodes",
                _graphId,
                _graphStorage.EdgeCount,
                _graphStorage.NodeCount);
        }
        else
        {
            _state.State.Edges = new List<TemporalEdgeData>();
            _state.State.GraphId = _graphId;
            _state.State.CreatedAtNanos = GetNanoseconds();
        }

        await base.OnActivateAsync(cancellationToken);
    }

    /// <inheritdoc />
    public Task<EdgeAddResult> AddEdgeAsync(TemporalEdgeData edge)
    {
        ArgumentNullException.ThrowIfNull(edge);

        try
        {
            var sw = Stopwatch.StartNew();

            // Validate edge
            if (edge.ValidTo < edge.ValidFrom)
            {
                return Task.FromResult(new EdgeAddResult
                {
                    Success = false,
                    EdgeCount = _graphStorage.EdgeCount,
                    TimestampNanos = GetNanoseconds(),
                    ErrorMessage = "ValidTo must be >= ValidFrom"
                });
            }

            // Add to in-memory storage
            var internalEdge = ConvertFromData(edge);
            _graphStorage.AddEdge(internalEdge);

            // Add to persistent state
            _state.State.Edges.Add(edge);
            _state.State.ModifiedAtNanos = GetNanoseconds();

            // Update HLC
            _hlcTimestamp = _hlcTimestamp.Increment(GetNanoseconds());
            _operationCount++;

            sw.Stop();

            _logger.LogDebug(
                "Added edge {Source}→{Target} to graph {GraphId} ({ValidFrom}ns-{ValidTo}ns, {TimeUs}μs)",
                edge.SourceId,
                edge.TargetId,
                _graphId,
                edge.ValidFrom,
                edge.ValidTo,
                sw.ElapsedTicks * 100 / 1000);

            return Task.FromResult(new EdgeAddResult
            {
                Success = true,
                EdgeCount = _graphStorage.EdgeCount,
                TimestampNanos = _hlcTimestamp.PhysicalTime
            });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to add edge to graph {GraphId}", _graphId);
            return Task.FromResult(new EdgeAddResult
            {
                Success = false,
                EdgeCount = _graphStorage.EdgeCount,
                TimestampNanos = GetNanoseconds(),
                ErrorMessage = ex.Message
            });
        }
    }

    /// <inheritdoc />
    public Task<BatchEdgeResult> AddEdgesBatchAsync(IReadOnlyList<TemporalEdgeData> edges)
    {
        ArgumentNullException.ThrowIfNull(edges);

        var sw = Stopwatch.StartNew();
        int added = 0;
        int failed = 0;

        foreach (var edge in edges)
        {
            try
            {
                if (edge.ValidTo < edge.ValidFrom)
                {
                    failed++;
                    continue;
                }

                var internalEdge = ConvertFromData(edge);
                _graphStorage.AddEdge(internalEdge);
                _state.State.Edges.Add(edge);
                added++;
            }
            catch
            {
                failed++;
            }
        }

        if (added > 0)
        {
            _state.State.ModifiedAtNanos = GetNanoseconds();
            _hlcTimestamp = _hlcTimestamp.Increment(GetNanoseconds());
            _operationCount++;
        }

        sw.Stop();

        _logger.LogInformation(
            "Batch added {Added} edges to graph {GraphId} ({Failed} failed, {DurationUs}μs)",
            added,
            _graphId,
            failed,
            sw.ElapsedTicks * 100 / 1000);

        return Task.FromResult(new BatchEdgeResult
        {
            Success = failed == 0,
            EdgesAdded = added,
            EdgesFailed = failed,
            TotalEdgeCount = _graphStorage.EdgeCount,
            DurationNanos = sw.ElapsedTicks * 100
        });
    }

    /// <inheritdoc />
    public Task<EdgeQueryResult> GetEdgesAsync(
        ulong sourceId,
        long startTimeNanos,
        long endTimeNanos)
    {
        var sw = Stopwatch.StartNew();

        var edges = _graphStorage.GetEdgesInTimeRange(sourceId, startTimeNanos, endTimeNanos)
            .Select(ConvertToData)
            .ToList();

        sw.Stop();

        return Task.FromResult(new EdgeQueryResult
        {
            Edges = edges,
            IsTruncated = false,
            TotalCount = edges.Count,
            QueryTimeNanos = sw.ElapsedTicks * 100
        });
    }

    /// <inheritdoc />
    public Task<EdgeQueryResult> GetAllEdgesInRangeAsync(
        long startTimeNanos,
        long endTimeNanos,
        int maxResults = 1000)
    {
        var sw = Stopwatch.StartNew();

        var allEdges = _graphStorage.GetAllEdgesInTimeRange(startTimeNanos, endTimeNanos).ToList();
        var isTruncated = allEdges.Count > maxResults;
        var edges = allEdges.Take(maxResults).Select(ConvertToData).ToList();

        sw.Stop();

        return Task.FromResult(new EdgeQueryResult
        {
            Edges = edges,
            IsTruncated = isTruncated,
            TotalCount = allEdges.Count,
            QueryTimeNanos = sw.ElapsedTicks * 100
        });
    }

    /// <inheritdoc />
    public Task<PathFindingResult> FindPathsAsync(PathFindingRequest request)
    {
        ArgumentNullException.ThrowIfNull(request);

        var sw = Stopwatch.StartNew();

        try
        {
            IEnumerable<TemporalPath> paths;

            if (request.ShortestOnly)
            {
                var shortestPath = _graphStorage.FindShortestTemporalPath(
                    request.StartNode,
                    request.EndNode,
                    request.MaxTimeSpanNanos);

                paths = shortestPath != null ? [shortestPath] : [];
            }
            else if (request.FastestOnly)
            {
                var fastestPath = _graphStorage.FindFastestTemporalPath(
                    request.StartNode,
                    request.EndNode,
                    request.MaxTimeSpanNanos);

                paths = fastestPath != null ? [fastestPath] : [];
            }
            else
            {
                paths = _graphStorage.FindTemporalPaths(
                    request.StartNode,
                    request.EndNode,
                    request.MaxTimeSpanNanos,
                    request.MaxPathLength);
            }

            var pathList = paths.Take(request.MaxResults)
                .Select(p => new TemporalPathData
                {
                    Edges = p.Edges.Select(ConvertToData).ToList(),
                    TotalWeight = p.TotalWeight
                })
                .ToList();

            sw.Stop();

            return Task.FromResult(new PathFindingResult
            {
                Paths = pathList,
                IsComplete = pathList.Count < request.MaxResults,
                NodesVisited = 0, // Would need to instrument BFS
                EdgesExamined = 0,
                SearchTimeNanos = sw.ElapsedTicks * 100
            });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Path finding failed in graph {GraphId}", _graphId);
            sw.Stop();

            return Task.FromResult(new PathFindingResult
            {
                Paths = [],
                IsComplete = false,
                SearchTimeNanos = sw.ElapsedTicks * 100
            });
        }
    }

    /// <inheritdoc />
    public Task<SnapshotResult> GetSnapshotAsync(long timeNanos, int maxEdges = 10000)
    {
        var sw = Stopwatch.StartNew();

        var allEdges = _graphStorage.GetSnapshotAtTime(timeNanos).ToList();
        var isTruncated = allEdges.Count > maxEdges;
        var edges = allEdges.Take(maxEdges).Select(ConvertToData).ToList();

        sw.Stop();

        return Task.FromResult(new SnapshotResult
        {
            Edges = edges,
            TimeNanos = timeNanos,
            IsTruncated = isTruncated,
            TotalActiveEdges = allEdges.Count,
            QueryTimeNanos = sw.ElapsedTicks * 100
        });
    }

    /// <inheritdoc />
    public Task<ReachabilityResult> GetReachableNodesAsync(
        ulong startNode,
        long startTimeNanos,
        long maxTimeSpanNanos,
        int maxNodes = 10000)
    {
        var sw = Stopwatch.StartNew();

        var allReachable = _graphStorage.GetReachableNodes(startNode, startTimeNanos, maxTimeSpanNanos).ToList();
        var isTruncated = allReachable.Count > maxNodes;
        var nodes = allReachable.Take(maxNodes).ToList();

        sw.Stop();

        return Task.FromResult(new ReachabilityResult
        {
            ReachableNodes = nodes,
            StartNode = startNode,
            IsTruncated = isTruncated,
            QueryTimeNanos = sw.ElapsedTicks * 100
        });
    }

    /// <inheritdoc />
    public Task<TemporalGraphStatistics> GetStatisticsAsync()
    {
        var stats = _graphStorage.GetStatistics();

        return Task.FromResult(new TemporalGraphStatistics
        {
            NodeCount = stats.NodeCount,
            EdgeCount = stats.EdgeCount,
            MinTime = stats.MinTime,
            MaxTime = stats.MaxTime,
            TimeSpanNanos = stats.TimeSpanNanos,
            AverageDegree = stats.AverageDegree
        });
    }

    /// <inheritdoc />
    public Task<bool> ContainsNodeAsync(ulong nodeId)
    {
        return Task.FromResult(_graphStorage.ContainsNode(nodeId));
    }

    /// <inheritdoc />
    public Task<IReadOnlyList<ulong>> GetNodesAsync(int maxNodes = 10000)
    {
        var nodes = _graphStorage.GetAllNodes().Take(maxNodes).ToList();
        return Task.FromResult<IReadOnlyList<ulong>>(nodes);
    }

    /// <inheritdoc />
    public async Task ClearAsync()
    {
        _graphStorage.Clear();
        _state.State.Edges.Clear();
        _state.State.ModifiedAtNanos = GetNanoseconds();

        await _state.WriteStateAsync();

        _logger.LogInformation("Cleared temporal graph {GraphId}", _graphId);
    }

    /// <inheritdoc />
    public async Task PersistAsync()
    {
        await _state.WriteStateAsync();

        _logger.LogDebug(
            "Persisted temporal graph {GraphId} ({EdgeCount} edges)",
            _graphId,
            _graphStorage.EdgeCount);
    }

    private static TemporalEdge ConvertFromData(TemporalEdgeData data)
    {
        return new TemporalEdge(
            data.SourceId,
            data.TargetId,
            data.ValidFrom,
            data.ValidTo,
            data.HLC,
            data.Weight,
            data.EdgeType);
    }

    private static TemporalEdgeData ConvertToData(TemporalEdge edge)
    {
        return new TemporalEdgeData
        {
            SourceId = edge.SourceId,
            TargetId = edge.TargetId,
            ValidFrom = edge.ValidFrom,
            ValidTo = edge.ValidTo,
            HLC = edge.HLC,
            Weight = edge.Weight,
            EdgeType = edge.EdgeType
        };
    }

    private static long GetNanoseconds()
    {
        return DateTimeOffset.UtcNow.ToUnixTimeMilliseconds() * 1_000_000 +
               (Stopwatch.GetTimestamp() % 1_000_000);
    }
}

/// <summary>
/// Persistent state for temporal graph grain.
/// </summary>
[GenerateSerializer]
public sealed class TemporalGraphState
{
    /// <summary>
    /// Graph identifier.
    /// </summary>
    [Id(0)]
    public string GraphId { get; set; } = string.Empty;

    /// <summary>
    /// All edges in the graph.
    /// </summary>
    [Id(1)]
    public List<TemporalEdgeData> Edges { get; set; } = new();

    /// <summary>
    /// Creation timestamp (nanoseconds).
    /// </summary>
    [Id(2)]
    public long CreatedAtNanos { get; set; }

    /// <summary>
    /// Last modification timestamp (nanoseconds).
    /// </summary>
    [Id(3)]
    public long ModifiedAtNanos { get; set; }

    /// <summary>
    /// Graph version for optimistic concurrency.
    /// </summary>
    [Id(4)]
    public long Version { get; set; }
}

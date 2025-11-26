using System;
using System.Linq;
using Orleans.GpuBridge.Abstractions.Temporal;
using Orleans.GpuBridge.Runtime.Temporal.Graph;
using Xunit;

namespace Orleans.GpuBridge.Temporal.Tests.Unit;

public class TemporalGraphTests
{
    [Fact]
    public void TemporalEdge_CreatesValidEdge()
    {
        var hlc = new HybridTimestamp(1000, 0, 1);
        var edge = new TemporalEdge(
            sourceId: 1,
            targetId: 2,
            validFrom: 100,
            validTo: 200,
            hlc: hlc,
            weight: 10.5);

        Assert.Equal(1ul, edge.SourceId);
        Assert.Equal(2ul, edge.TargetId);
        Assert.Equal(100, edge.ValidFrom);
        Assert.Equal(200, edge.ValidTo);
        Assert.Equal(100, edge.DurationNanos);
        Assert.Equal(10.5, edge.Weight);
    }

    [Fact]
    public void TemporalEdge_ThrowsOnInvalidTimeRange()
    {
        var hlc = new HybridTimestamp(1000, 0, 1);

        Assert.Throws<ArgumentException>(() => new TemporalEdge(
            1, 2,
            validFrom: 200,
            validTo: 100, // Invalid: ValidTo < ValidFrom
            hlc: hlc));
    }

    [Fact]
    public void TemporalEdge_IsValidAt_WorksCorrectly()
    {
        var hlc = new HybridTimestamp(1000, 0, 1);
        var edge = new TemporalEdge(1, 2, 100, 200, hlc);

        Assert.True(edge.IsValidAt(100));  // Start (inclusive)
        Assert.True(edge.IsValidAt(150));  // Middle
        Assert.True(edge.IsValidAt(200));  // End (inclusive)
        Assert.False(edge.IsValidAt(99));  // Before
        Assert.False(edge.IsValidAt(201)); // After
    }

    [Fact]
    public void TemporalEdge_OverlapsWith_DetectsOverlaps()
    {
        var hlc = new HybridTimestamp(1000, 0, 1);
        var edge = new TemporalEdge(1, 2, 100, 200, hlc);

        Assert.True(edge.OverlapsWith(50, 150));   // Partial overlap (start)
        Assert.True(edge.OverlapsWith(150, 250));  // Partial overlap (end)
        Assert.True(edge.OverlapsWith(100, 200));  // Exact match
        Assert.True(edge.OverlapsWith(50, 250));   // Fully contains
        Assert.True(edge.OverlapsWith(120, 180));  // Fully contained
        Assert.False(edge.OverlapsWith(50, 99));   // Before
        Assert.False(edge.OverlapsWith(201, 300)); // After
    }

    [Fact]
    public void TemporalPath_BuildsPathCorrectly()
    {
        var hlc = new HybridTimestamp(1000, 0, 1);
        var path = new TemporalPath();

        var edge1 = new TemporalEdge(1, 2, 100, 110, hlc, weight: 5.0);
        var edge2 = new TemporalEdge(2, 3, 120, 130, hlc, weight: 3.0);
        var edge3 = new TemporalEdge(3, 4, 140, 150, hlc, weight: 2.0);

        path.AddEdge(edge1);
        path.AddEdge(edge2);
        path.AddEdge(edge3);

        Assert.Equal(3, path.Length);
        Assert.Equal(10.0, path.TotalWeight); // 5 + 3 + 2
        Assert.Equal(1ul, path.SourceNode);
        Assert.Equal(4ul, path.TargetNode);
        Assert.Equal(100, path.StartTime);
        Assert.Equal(140, path.EndTime);
        Assert.Equal(40, path.TotalDurationNanos);
    }

    [Fact]
    public void TemporalPath_RejectsDisconnectedEdge()
    {
        var hlc = new HybridTimestamp(1000, 0, 1);
        var path = new TemporalPath();

        var edge1 = new TemporalEdge(1, 2, 100, 110, hlc);
        var edge2 = new TemporalEdge(3, 4, 120, 130, hlc); // Not connected to edge1

        path.AddEdge(edge1);

        Assert.Throws<ArgumentException>(() => path.AddEdge(edge2));
    }

    [Fact]
    public void TemporalPath_RejectsTemporallyInvalidEdge()
    {
        var hlc = new HybridTimestamp(1000, 0, 1);
        var path = new TemporalPath();

        var edge1 = new TemporalEdge(1, 2, 100, 110, hlc);
        var edge2 = new TemporalEdge(2, 3, 90, 95, hlc); // Starts before edge1

        path.AddEdge(edge1);

        Assert.Throws<ArgumentException>(() => path.AddEdge(edge2));
    }

    [Fact]
    public void TemporalPath_GetNodes_ReturnsAllNodes()
    {
        var hlc = new HybridTimestamp(1000, 0, 1);
        var path = new TemporalPath();

        path.AddEdge(new TemporalEdge(1, 2, 100, 110, hlc));
        path.AddEdge(new TemporalEdge(2, 3, 120, 130, hlc));
        path.AddEdge(new TemporalEdge(3, 4, 140, 150, hlc));

        var nodes = path.GetNodes().ToList();

        Assert.Equal(new[] { 1ul, 2ul, 3ul, 4ul }, nodes);
    }

    [Fact]
    public void IntervalTree_AddsAndQueriesIntervals()
    {
        var tree = new IntervalTree<long, string>();

        tree.Add(100, 200, "interval1");
        tree.Add(150, 250, "interval2");
        tree.Add(300, 400, "interval3");

        Assert.Equal(3, tree.Count);

        // Query overlapping intervals
        var results = tree.Query(120, 180).ToList();
        Assert.Equal(2, results.Count);
        Assert.Contains("interval1", results);
        Assert.Contains("interval2", results);
    }

    [Fact]
    public void IntervalTree_QueryPoint_FindsContainingIntervals()
    {
        var tree = new IntervalTree<long, string>();

        tree.Add(100, 200, "interval1");
        tree.Add(150, 250, "interval2");
        tree.Add(300, 400, "interval3");

        var results = tree.QueryPoint(175).ToList();

        Assert.Equal(2, results.Count);
        Assert.Contains("interval1", results);
        Assert.Contains("interval2", results);
    }

    [Fact]
    public void IntervalTree_QueryNoOverlap_ReturnsEmpty()
    {
        var tree = new IntervalTree<long, string>();

        tree.Add(100, 200, "interval1");
        tree.Add(300, 400, "interval2");

        var results = tree.Query(210, 290);

        Assert.Empty(results);
    }

    [Fact]
    public void TemporalGraphStorage_AddsAndQueriesEdges()
    {
        var graph = new TemporalGraphStorage();
        var hlc = new HybridTimestamp(1000, 0, 1);

        graph.AddEdge(1, 2, 100, 200, hlc);
        graph.AddEdge(2, 3, 150, 250, hlc);
        graph.AddEdge(1, 3, 300, 400, hlc);

        Assert.Equal(3, graph.EdgeCount);
        Assert.Equal(3, graph.NodeCount);
        Assert.True(graph.ContainsNode(1));
        Assert.True(graph.ContainsNode(2));
        Assert.True(graph.ContainsNode(3));
    }

    [Fact]
    public void TemporalGraphStorage_GetEdgesInTimeRange_ReturnsOverlappingEdges()
    {
        var graph = new TemporalGraphStorage();
        var hlc = new HybridTimestamp(1000, 0, 1);

        graph.AddEdge(1, 2, 100, 200, hlc, edgeType: "edge1");
        graph.AddEdge(1, 3, 150, 250, hlc, edgeType: "edge2");
        graph.AddEdge(1, 4, 300, 400, hlc, edgeType: "edge3");

        var edges = graph.GetEdgesInTimeRange(1, 120, 180).ToList();

        Assert.Equal(2, edges.Count);
        Assert.Contains(edges, e => e.EdgeType == "edge1");
        Assert.Contains(edges, e => e.EdgeType == "edge2");
    }

    [Fact]
    public void TemporalGraphStorage_FindsSimplePath()
    {
        var graph = new TemporalGraphStorage();
        var hlc = new HybridTimestamp(1000, 0, 1);

        // Create path: 1 → 2 → 3
        graph.AddEdge(1, 2, 100, 110, hlc);
        graph.AddEdge(2, 3, 120, 130, hlc);

        var paths = graph.FindTemporalPaths(
            startNode: 1,
            endNode: 3,
            maxTimeSpanNanos: 1000).ToList();

        Assert.Single(paths);
        Assert.Equal(2, paths[0].Length);
        Assert.Equal(1ul, paths[0].SourceNode);
        Assert.Equal(3ul, paths[0].TargetNode);
    }

    [Fact]
    public void TemporalGraphStorage_FindsShortestPath()
    {
        // Note: FindTemporalPaths uses BFS with early termination for performance,
        // returning only the shortest path rather than all possible paths.
        var graph = new TemporalGraphStorage();
        var hlc = new HybridTimestamp(1000, 0, 1);

        // Create diamond pattern: 1 → {2,3} → 4
        graph.AddEdge(1, 2, 100, 110, hlc);
        graph.AddEdge(1, 3, 100, 110, hlc);
        graph.AddEdge(2, 4, 120, 130, hlc);
        graph.AddEdge(3, 4, 120, 130, hlc);

        var paths = graph.FindTemporalPaths(
            startNode: 1,
            endNode: 4,
            maxTimeSpanNanos: 1000).ToList();

        // BFS finds the first shortest path (optimized for performance)
        Assert.Single(paths); // Returns shortest path (one of: 1→2→4 or 1→3→4)
        Assert.Equal(2, paths[0].Length);
    }

    [Fact]
    public void TemporalGraphStorage_RespectsTimeWindow()
    {
        var graph = new TemporalGraphStorage();
        var hlc = new HybridTimestamp(1000, 0, 1);

        // Create path that exceeds time window (nanosecond timestamps)
        // 5 seconds = 5_000_000_000 nanoseconds
        graph.AddEdge(1, 2, 100_000_000L, 110_000_000L, hlc); // 100ms-110ms
        graph.AddEdge(2, 3, 5_100_000_000L, 5_110_000_000L, hlc); // 5.1 seconds - gap is 5 seconds

        // Query with 1-second window (1_000_000_000 nanoseconds)
        var paths = graph.FindTemporalPaths(
            startNode: 1,
            endNode: 3,
            maxTimeSpanNanos: 1_000_000_000L).ToList(); // 1 second window

        Assert.Empty(paths); // Path exceeds time window (5 seconds > 1 second)
    }

    [Fact]
    public void TemporalGraphStorage_GetSnapshotAtTime_ReturnsActiveEdges()
    {
        var graph = new TemporalGraphStorage();
        var hlc = new HybridTimestamp(1000, 0, 1);

        graph.AddEdge(1, 2, 100, 200, hlc);
        graph.AddEdge(2, 3, 150, 250, hlc);
        graph.AddEdge(3, 4, 300, 400, hlc);

        var snapshot = graph.GetSnapshotAtTime(175).ToList();

        Assert.Equal(2, snapshot.Count); // Edges 1→2 and 2→3 are active at t=175
    }

    [Fact]
    public void TemporalGraphStorage_GetReachableNodes_FindsAllReachable()
    {
        var graph = new TemporalGraphStorage();
        var hlc = new HybridTimestamp(1000, 0, 1);

        // Create graph: 1 → 2 → 3 → 4
        graph.AddEdge(1, 2, 100, 110, hlc);
        graph.AddEdge(2, 3, 120, 130, hlc);
        graph.AddEdge(3, 4, 140, 150, hlc);

        var reachable = graph.GetReachableNodes(
            startNode: 1,
            startTimeNanos: 100,
            maxTimeSpanNanos: 1000).ToHashSet();

        Assert.Equal(4, reachable.Count);
        Assert.Contains(1ul, reachable);
        Assert.Contains(2ul, reachable);
        Assert.Contains(3ul, reachable);
        Assert.Contains(4ul, reachable);
    }

    [Fact]
    public void TemporalGraphStorage_FindShortestPath_ReturnsMinimumEdges()
    {
        var graph = new TemporalGraphStorage();
        var hlc = new HybridTimestamp(1000, 0, 1);

        // Create two paths: 1 → 2 → 3 (2 edges) and 1 → 3 (1 edge)
        graph.AddEdge(1, 2, 100, 110, hlc);
        graph.AddEdge(2, 3, 120, 130, hlc);
        graph.AddEdge(1, 3, 100, 110, hlc);

        var shortestPath = graph.FindShortestTemporalPath(1, 3, 1000);

        Assert.NotNull(shortestPath);
        Assert.Equal(1, shortestPath!.Length); // Direct edge 1→3
    }

    [Fact]
    public void TemporalGraphStorage_GetStatistics_ReturnsCorrectCounts()
    {
        var graph = new TemporalGraphStorage();
        var hlc = new HybridTimestamp(1000, 0, 1);

        graph.AddEdge(1, 2, 100, 200, hlc);
        graph.AddEdge(2, 3, 150, 250, hlc);
        graph.AddEdge(1, 3, 300, 400, hlc);

        var stats = graph.GetStatistics();

        Assert.Equal(3, stats.NodeCount);
        Assert.Equal(3, stats.EdgeCount);
        Assert.Equal(100, stats.MinTime);
        Assert.Equal(400, stats.MaxTime);
        Assert.Equal(300, stats.TimeSpanNanos);
    }

    [Fact]
    public void TemporalGraphStorage_Performance_HandlesLargeGraph()
    {
        var graph = new TemporalGraphStorage();
        var hlc = new HybridTimestamp(1000, 0, 1);

        // Add 10,000 edges
        var sw = System.Diagnostics.Stopwatch.StartNew();
        for (int i = 0; i < 10_000; i++)
        {
            graph.AddEdge((ulong)i, (ulong)(i + 1), i * 100, (i + 1) * 100, hlc);
        }
        sw.Stop();

        var addTimePerEdge = sw.Elapsed.TotalMicroseconds / 10_000;

        // Query edges
        sw.Restart();
        for (int i = 0; i < 1000; i++)
        {
            var edges = graph.GetEdgesInTimeRange((ulong)i, 0, long.MaxValue).ToList();
        }
        sw.Stop();

        var queryTimePerOp = sw.Elapsed.TotalMicroseconds / 1000;

        // Performance targets
        Assert.True(addTimePerEdge < 100, $"Add time: {addTimePerEdge:F2}μs (target: <100μs)");
        Assert.True(queryTimePerOp < 100, $"Query time: {queryTimePerOp:F2}μs (target: <100μs)");
    }
}

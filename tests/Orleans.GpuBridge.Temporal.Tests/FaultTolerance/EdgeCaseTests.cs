using System;
using System.Linq;
using Orleans.GpuBridge.Abstractions.Temporal;
using Orleans.GpuBridge.Runtime.Temporal;
using Orleans.GpuBridge.Runtime.Temporal.Graph;
using Xunit;
using Xunit.Abstractions;

namespace Orleans.GpuBridge.Temporal.Tests.FaultTolerance;

/// <summary>
/// Tests for edge cases and boundary conditions in temporal components.
/// Validates behavior at extremes and unusual scenarios.
/// </summary>
public class EdgeCaseTests
{
    private readonly ITestOutputHelper _output;

    public EdgeCaseTests(ITestOutputHelper output)
    {
        _output = output;
    }

    #region HLC Edge Cases

    [Fact]
    public void HLC_MinimumTimestamp()
    {
        // Arrange & Act: Create timestamp at minimum values
        var minTimestamp = new HybridTimestamp(
            physicalTime: 0L,
            logicalCounter: 0L,
            nodeId: 1);

        var hlc = new HybridLogicalClock(nodeId: 1);
        var updated = hlc.Update(minTimestamp);

        // Assert: Handles minimum values gracefully
        Assert.True(updated.PhysicalTime >= 0);
        Assert.True(updated.LogicalCounter >= 0);

        _output.WriteLine($"Minimum timestamp handling:");
        _output.WriteLine($"  Input: {minTimestamp}");
        _output.WriteLine($"  Updated: {updated}");
    }

    [Fact]
    public void HLC_MaximumTimestamp()
    {
        // Arrange & Act: Create timestamp near maximum values
        var maxTime = long.MaxValue / 2; // Avoid overflow in calculations
        var maxTimestamp = new HybridTimestamp(
            physicalTime: maxTime,
            logicalCounter: long.MaxValue / 2,
            nodeId: ushort.MaxValue);

        var hlc = new HybridLogicalClock(nodeId: 1);
        var updated = hlc.Update(maxTimestamp);

        // Assert: Handles large values gracefully
        Assert.True(updated.PhysicalTime >= maxTimestamp.PhysicalTime);

        _output.WriteLine($"Maximum timestamp handling:");
        _output.WriteLine($"  Physical time: {maxTime:N0} ns");
        _output.WriteLine($"  Logical counter: {maxTimestamp.LogicalCounter:N0}");
        _output.WriteLine($"  Node ID: {maxTimestamp.NodeId:N0}");
    }

    [Fact]
    public void HLC_LogicalCounterOverflow()
    {
        // Arrange: Force logical counter to near maximum
        var hlc = new HybridLogicalClock(nodeId: 1);
        var baseTime = DateTimeOffset.UtcNow.ToUnixTimeNanoseconds();

        // Create timestamp with high logical counter
        var highCounter = new HybridTimestamp(baseTime, long.MaxValue - 10, 1);

        // Act: Update multiple times at same physical time
        var timestamps = new System.Collections.Generic.List<HybridTimestamp>();
        for (int i = 0; i < 20; i++)
        {
            var updated = hlc.Update(highCounter);
            timestamps.Add(updated);
        }

        // Assert: All timestamps remain ordered despite high counter
        for (int i = 1; i < timestamps.Count; i++)
        {
            Assert.True(timestamps[i].CompareTo(timestamps[i - 1]) >= 0);
        }

        _output.WriteLine($"Logical counter near overflow:");
        _output.WriteLine($"  Starting counter: {long.MaxValue - 10:N0}");
        _output.WriteLine($"  Updates: {timestamps.Count}");
        _output.WriteLine($"  Monotonicity: MAINTAINED");
    }

    [Fact]
    public void HLC_NodeIdZero()
    {
        // Arrange & Act: Node ID of zero (edge case)
        var hlc = new HybridLogicalClock(nodeId: 0);
        var ts1 = hlc.Now();
        var ts2 = hlc.Now();

        // Assert: Still functions correctly
        Assert.Equal((ushort)0, ts1.NodeId);
        Assert.Equal((ushort)0, ts2.NodeId);
        Assert.True(ts2.CompareTo(ts1) > 0);

        _output.WriteLine($"Node ID zero handling:");
        _output.WriteLine($"  Timestamp 1: {ts1}");
        _output.WriteLine($"  Timestamp 2: {ts2}");
        _output.WriteLine($"  Ordering: CORRECT");
    }

    #endregion

    #region Temporal Graph Edge Cases

    [Fact]
    public void TemporalGraph_EmptyGraph()
    {
        // Arrange: Empty graph
        var graph = new TemporalGraphStorage();

        // Act & Assert: All operations handle empty graph
        Assert.Equal(0, graph.NodeCount);
        Assert.Equal(0, graph.EdgeCount);
        Assert.Empty(graph.GetAllNodes());
        Assert.False(graph.ContainsNode(1));

        var edges = graph.GetEdgesInTimeRange(1, 0, long.MaxValue);
        Assert.Empty(edges);

        var paths = graph.FindTemporalPaths(1, 2, long.MaxValue);
        Assert.Empty(paths);

        _output.WriteLine($"Empty graph handling: VERIFIED");
    }

    [Fact]
    public void TemporalGraph_SingleNode()
    {
        // Arrange: Graph with single node (self-loop)
        var graph = new TemporalGraphStorage();
        var baseTime = DateTimeOffset.UtcNow.ToUnixTimeNanoseconds();

        graph.AddEdge(
            sourceId: 1,
            targetId: 1, // Self-loop
            validFrom: baseTime,
            validTo: baseTime + 1_000_000_000L,
            hlc: new HybridTimestamp(baseTime, 0, 1));

        // Act & Assert: Handles self-loop correctly
        Assert.Equal(1, graph.NodeCount); // Only one unique node
        Assert.Equal(1, graph.EdgeCount);
        Assert.True(graph.ContainsNode(1));

        var edges = graph.GetEdgesInTimeRange(1, baseTime, baseTime + 2_000_000_000L);
        Assert.Single(edges);

        _output.WriteLine($"Self-loop handling: VERIFIED");
    }

    [Fact]
    public void TemporalGraph_DisconnectedComponents()
    {
        // Arrange: Graph with multiple disconnected components
        var graph = new TemporalGraphStorage();
        var baseTime = DateTimeOffset.UtcNow.ToUnixTimeNanoseconds();

        // Component 1: nodes {1, 2, 3}
        graph.AddEdge(1, 2, baseTime, baseTime + 1_000_000_000L, new HybridTimestamp(baseTime, 0L, 1));
        graph.AddEdge(2, 3, baseTime, baseTime + 1_000_000_000L, new HybridTimestamp(baseTime, 1L, 1));

        // Component 2: nodes {10, 11, 12}
        graph.AddEdge(10, 11, baseTime, baseTime + 1_000_000_000L, new HybridTimestamp(baseTime, 2L, 1));
        graph.AddEdge(11, 12, baseTime, baseTime + 1_000_000_000L, new HybridTimestamp(baseTime, 3L, 1));

        // Act: Try to find path between disconnected components
        var paths = graph.FindTemporalPaths(1, 10, long.MaxValue);

        // Assert: No path exists
        Assert.Empty(paths);
        Assert.Equal(6, graph.NodeCount); // Both components counted
        Assert.Equal(4, graph.EdgeCount);

        _output.WriteLine($"Disconnected components:");
        _output.WriteLine($"  Total nodes: {graph.NodeCount}");
        _output.WriteLine($"  Total edges: {graph.EdgeCount}");
        _output.WriteLine($"  Path 1→10: None (disconnected)");
    }

    [Fact]
    public void TemporalGraph_ZeroTimeRange()
    {
        // Arrange: Query with zero time range (point query)
        var graph = new TemporalGraphStorage();
        var baseTime = DateTimeOffset.UtcNow.ToUnixTimeNanoseconds();

        graph.AddEdge(1, 2, baseTime, baseTime + 1_000_000_000L,
            new HybridTimestamp(baseTime, 0, 1));

        // Act: Query at exact point in time
        var edges = graph.GetEdgesInTimeRange(1, baseTime, baseTime);

        // Assert: Finds edges valid at that instant
        Assert.NotEmpty(edges); // Edge valid at baseTime

        _output.WriteLine($"Zero time range query: HANDLED");
    }

    [Fact]
    public void TemporalGraph_InfiniteTimeRange()
    {
        // Arrange: Query with maximum time range
        var graph = new TemporalGraphStorage();
        var baseTime = DateTimeOffset.UtcNow.ToUnixTimeNanoseconds();

        for (ulong i = 1; i <= 10; i++)
        {
            graph.AddEdge(i, i + 1, baseTime + (long)i,
                baseTime + (long)i + 1_000_000_000L,
                new HybridTimestamp(baseTime, (long)i, 1));
        }

        // Act: Query entire history
        var allEdges = graph.GetAllEdgesInTimeRange(long.MinValue, long.MaxValue);

        // Assert: Returns all edges
        Assert.Equal(10, allEdges.Count());

        _output.WriteLine($"Infinite time range query:");
        _output.WriteLine($"  Edges found: {allEdges.Count()}");
        _output.WriteLine($"  Time range: [MIN, MAX]");
    }

    [Fact]
    public void TemporalGraph_OverlappingIntervals()
    {
        // Arrange: Multiple edges with overlapping time ranges
        var graph = new TemporalGraphStorage();
        var baseTime = DateTimeOffset.UtcNow.ToUnixTimeNanoseconds();

        // All edges from same source, overlapping times
        for (int i = 0; i < 10; i++)
        {
            graph.AddEdge(
                sourceId: 1,
                targetId: (ulong)(i + 2),
                validFrom: baseTime + i * 100_000_000L, // Staggered starts
                validTo: baseTime + 2_000_000_000L, // All end at same time
                hlc: new HybridTimestamp(baseTime, (long)i, 1));
        }

        // Act: Query overlapping period
        var overlapping = graph.GetEdgesInTimeRange(1,
            baseTime + 500_000_000L,
            baseTime + 1_500_000_000L);

        // Assert: Finds all overlapping edges
        Assert.Equal(10, overlapping.Count()); // All edges overlap this range

        _output.WriteLine($"Overlapping intervals:");
        _output.WriteLine($"  Total edges: 10");
        _output.WriteLine($"  Overlapping edges found: {overlapping.Count()}");
    }

    [Fact]
    public void TemporalGraph_DegeneratePath()
    {
        // Arrange: Path where start == end (zero-length path)
        var graph = new TemporalGraphStorage();
        var baseTime = DateTimeOffset.UtcNow.ToUnixTimeNanoseconds();

        graph.AddEdge(1, 2, baseTime, baseTime + 1_000_000_000L,
            new HybridTimestamp(baseTime, 0, 1));

        // Act: Find path from node to itself
        var paths = graph.FindTemporalPaths(
            startNode: 1,
            endNode: 1,
            maxTimeSpanNanos: long.MaxValue);

        // Assert: No self-path (path must have length > 0)
        Assert.Empty(paths);

        _output.WriteLine($"Degenerate path (self): CORRECTLY REJECTED");
    }

    [Fact]
    public void TemporalGraph_DeepPathTraversal()
    {
        // Arrange: Create deep linear path
        var graph = new TemporalGraphStorage();
        var baseTime = DateTimeOffset.UtcNow.ToUnixTimeNanoseconds();
        var pathDepth = 100;

        for (int i = 0; i < pathDepth; i++)
        {
            graph.AddEdge(
                sourceId: (ulong)i,
                targetId: (ulong)(i + 1),
                validFrom: baseTime + (long)i * 1_000_000_000L,
                validTo: baseTime + (long)(i + 10) * 1_000_000_000L,
                hlc: new HybridTimestamp(baseTime, (long)i, 1));
        }

        // Act: Find deep path
        var paths = graph.FindTemporalPaths(
            startNode: 0UL,
            endNode: (ulong)pathDepth,
            maxTimeSpanNanos: pathDepth * 1_000_000_000L,
            maxPathLength: pathDepth + 10);

        // Assert: Finds path efficiently (BFS from Week 14 optimization)
        Assert.NotEmpty(paths);
        var path = paths.First();
        Assert.True(path.Length <= pathDepth);

        _output.WriteLine($"Deep path traversal:");
        _output.WriteLine($"  Path depth: {pathDepth}");
        _output.WriteLine($"  Path found: Length {path.Length}");
        _output.WriteLine($"  Optimization: BFS early termination (Week 14)");
    }

    #endregion

    #region IntervalTree Edge Cases

    [Fact]
    public void IntervalTree_PointInterval()
    {
        // Arrange: Interval where start == end (point)
        var tree = new IntervalTree<long, string>();

        // Act: Add point intervals
        tree.Add(100, 100, "point_1");
        tree.Add(200, 200, "point_2");
        tree.Add(100, 100, "point_3"); // Duplicate point

        // Assert: Handles point intervals
        var results = tree.Query(100, 100);
        Assert.Equal(2, results.Count()); // Two points at 100

        _output.WriteLine($"Point interval handling: VERIFIED");
    }

    [Fact]
    public void IntervalTree_EmptyTree()
    {
        // Arrange: Empty tree
        var tree = new IntervalTree<long, int>();

        // Act & Assert: Queries on empty tree
        var results = tree.Query(0, 1000);
        Assert.Empty(results);

        var pointResults = tree.QueryPoint(500);
        Assert.Empty(pointResults);

        _output.WriteLine($"Empty tree handling: VERIFIED");
    }

    [Fact]
    public void IntervalTree_NegativeIntervals()
    {
        // Arrange: Intervals with negative coordinates
        var tree = new IntervalTree<long, string>();

        tree.Add(-1000, -500, "negative_1");
        tree.Add(-500, 0, "span_zero");
        tree.Add(0, 500, "positive_1");

        // Act: Query negative range
        var negResults = tree.Query(-750, -250);

        // Assert: Handles negative coordinates
        Assert.Equal(2, negResults.Count()); // negative_1 and span_zero

        _output.WriteLine($"Negative interval handling: VERIFIED");
    }

    [Fact]
    public void IntervalTree_IdenticalIntervals()
    {
        // Arrange: Multiple identical intervals
        var tree = new IntervalTree<long, int>();

        for (int i = 0; i < 100; i++)
        {
            tree.Add(0, 1000, i); // Same interval, different values
        }

        // Act: Query identical intervals
        var results = tree.Query(500, 600);

        // Assert: Finds all identical intervals
        Assert.Equal(100, results.Count());

        _output.WriteLine($"Identical intervals:");
        _output.WriteLine($"  Count: 100");
        _output.WriteLine($"  Query results: {results.Count()}");
    }

    [Fact]
    public void IntervalTree_NestedIntervals()
    {
        // Arrange: Fully nested intervals
        var tree = new IntervalTree<long, string>();

        tree.Add(0, 1000, "outer");
        tree.Add(100, 900, "middle");
        tree.Add(200, 800, "inner");
        tree.Add(300, 700, "innermost");

        // Act: Query innermost region
        var results = tree.Query(400, 600);

        // Assert: Finds all nested intervals
        Assert.Equal(4, results.Count());

        _output.WriteLine($"Nested intervals: ALL FOUND");
    }

    [Fact]
    public void IntervalTree_AVLBalance_AfterManyInsertions()
    {
        // Arrange: Insert many intervals in various orders
        var tree = new IntervalTree<long, int>();
        var count = 10000;

        // Sequential insertion (tests AVL balancing)
        for (int i = 0; i < count; i++)
        {
            tree.Add(i * 100L, i * 100L + 50L, i);
        }

        // Act: Query should be fast (O(log N))
        var startQuery = DateTime.UtcNow;
        var results = tree.Query(5000L, 10000L);
        var queryTime = (DateTime.UtcNow - startQuery).TotalMilliseconds;

        // Assert: Query remains fast due to AVL balancing
        Assert.NotEmpty(results);
        Assert.True(queryTime < 10, // Should be sub-millisecond, but allow margin
            $"Query took {queryTime}ms - AVL balance may be degraded");

        _output.WriteLine($"AVL balance after {count} insertions:");
        _output.WriteLine($"  Query time: {queryTime:F3}ms");
        _output.WriteLine($"  Expected: O(log {count}) ≈ {Math.Log2(count):F1} levels");
        _output.WriteLine($"  Performance: {(queryTime < 1 ? "EXCELLENT" : "ACCEPTABLE")}");
    }

    #endregion

    [Fact]
    public void TemporalGraph_Statistics_EmptyGraph()
    {
        // Arrange: Empty graph
        var graph = new TemporalGraphStorage();

        // Act: Get statistics
        var stats = graph.GetStatistics();

        // Assert: Handles empty graph statistics
        Assert.Equal(0, stats.NodeCount);
        Assert.Equal(0, stats.EdgeCount);
        Assert.Equal(0, stats.MinTime);
        Assert.Equal(0, stats.MaxTime);
        Assert.Equal(0, stats.TimeSpanNanos);
        Assert.Equal(0, stats.AverageDegree);

        _output.WriteLine($"Empty graph statistics: HANDLED");
    }
}

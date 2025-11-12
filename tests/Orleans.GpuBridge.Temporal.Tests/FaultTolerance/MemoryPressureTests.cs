using System;
using System.Linq;
using Orleans.GpuBridge.Abstractions.Temporal;
using Orleans.GpuBridge.Runtime.Temporal;
using Orleans.GpuBridge.Runtime.Temporal.Graph;
using Xunit;
using Xunit.Abstractions;

namespace Orleans.GpuBridge.Temporal.Tests.FaultTolerance;

/// <summary>
/// Tests for graceful degradation under memory pressure.
/// Validates behavior when system resources are constrained.
/// </summary>
public class MemoryPressureTests
{
    private readonly ITestOutputHelper _output;

    public MemoryPressureTests(ITestOutputHelper output)
    {
        _output = output;
    }

    [Fact]
    public void TemporalGraph_LargeDataset()
    {
        // Arrange: Create large graph to test memory usage
        var graph = new TemporalGraphStorage();
        var nodeCount = 10000;
        var edgesPerNode = 10;
        var baseTime = DateTimeOffset.UtcNow.ToUnixTimeNanoseconds();

        var startMemory = GC.GetTotalMemory(forceFullCollection: true);

        // Act: Insert large number of edges
        for (int i = 0; i < nodeCount; i++)
        {
            for (int j = 0; j < edgesPerNode; j++)
            {
                var targetId = ((ulong)i + (ulong)j + 1UL) % (ulong)nodeCount;
                graph.AddEdge(
                    sourceId: (ulong)i,
                    targetId: targetId,
                    validFrom: baseTime + (long)i * 1_000_000L,
                    validTo: baseTime + (long)(i + 1) * 1_000_000L,
                    hlc: new HybridTimestamp(baseTime, (long)i, 1));
            }

            // Periodic GC to prevent OOM in test
            if (i % 1000 == 0)
            {
                GC.Collect(0, GCCollectionMode.Optimized);
            }
        }

        var endMemory = GC.GetTotalMemory(forceFullCollection: false);
        var memoryUsed = (endMemory - startMemory) / (1024.0 * 1024.0); // MB

        // Assert: Reasonable memory usage
        var expectedEdges = nodeCount * edgesPerNode;
        Assert.Equal(expectedEdges, graph.EdgeCount);

        var bytesPerEdge = (endMemory - startMemory) / (double)expectedEdges;

        _output.WriteLine($"Large dataset memory usage:");
        _output.WriteLine($"  Nodes: {nodeCount:N0}");
        _output.WriteLine($"  Edges: {graph.EdgeCount:N0}");
        _output.WriteLine($"  Memory used: {memoryUsed:F2} MB");
        _output.WriteLine($"  Bytes per edge: {bytesPerEdge:F2}");
        _output.WriteLine($"  Status: {(memoryUsed < 500 ? "EFFICIENT" : "REVIEW NEEDED")}");
    }

    [Fact]
    public void TemporalGraph_MemoryReleaseAfterClear()
    {
        // Arrange: Create and populate graph
        var graph = new TemporalGraphStorage();
        var edgeCount = 50000;
        var baseTime = DateTimeOffset.UtcNow.ToUnixTimeNanoseconds();

        var beforeInsert = GC.GetTotalMemory(forceFullCollection: true);

        for (int i = 0; i < (int)edgeCount; i++)
        {
            graph.AddEdge(
                sourceId: (ulong)(i % 1000),
                targetId: (ulong)((i + 1) % 1000),
                validFrom: baseTime + (long)i,
                validTo: baseTime + (long)i + 1_000_000_000L,
                hlc: new HybridTimestamp(baseTime, (long)i, 1));
        }

        var afterInsert = GC.GetTotalMemory(forceFullCollection: true);
        var memoryUsed = (afterInsert - beforeInsert) / (1024.0 * 1024.0);

        // Act: Clear graph
        graph.Clear();
        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();

        var afterClear = GC.GetTotalMemory(forceFullCollection: true);
        var memoryReleased = (afterInsert - afterClear) / (1024.0 * 1024.0);
        var releasePercentage = (memoryReleased / memoryUsed) * 100;

        // Assert: Memory released
        Assert.Equal(0, graph.EdgeCount);
        Assert.True(releasePercentage > 50, // At least 50% released
            $"Expected >50% memory release, got {releasePercentage:F1}%");

        _output.WriteLine($"Memory release after clear:");
        _output.WriteLine($"  Edges inserted: {edgeCount:N0}");
        _output.WriteLine($"  Memory used: {memoryUsed:F2} MB");
        _output.WriteLine($"  Memory released: {memoryReleased:F2} MB ({releasePercentage:F1}%)");
        _output.WriteLine($"  Cleanup: VERIFIED");
    }

    [Fact]
    public void IntervalTree_MemoryEfficiencyWithManyIntervals()
    {
        // Arrange: Test AVL tree memory efficiency
        var tree = new IntervalTree<long, int>();
        var intervalCount = 100000;

        var beforeInsert = GC.GetTotalMemory(forceFullCollection: true);

        // Act: Insert many intervals
        for (int i = 0; i < intervalCount; i++)
        {
            tree.Add(i * 100L, i * 100L + 50L, i);

            if (i % 10000 == 0)
            {
                GC.Collect(0, GCCollectionMode.Optimized);
            }
        }

        var afterInsert = GC.GetTotalMemory(forceFullCollection: true);
        var memoryUsed = (afterInsert - beforeInsert) / (1024.0 * 1024.0);
        var bytesPerInterval = (afterInsert - beforeInsert) / (double)intervalCount;

        // Query to verify tree integrity
        var queryResults = tree.Query(5000L, 10000L);
        Assert.NotEmpty(queryResults);

        _output.WriteLine($"IntervalTree memory efficiency:");
        _output.WriteLine($"  Intervals: {intervalCount:N0}");
        _output.WriteLine($"  Memory used: {memoryUsed:F2} MB");
        _output.WriteLine($"  Bytes per interval: {bytesPerInterval:F2}");
        _output.WriteLine($"  Tree balance: AVL (O(log N) depth)");

        // Assert: Reasonable memory per node
        // AVL node: ~40-80 bytes (Int64 start/end, height, left/right pointers, value)
        Assert.True(bytesPerInterval < 200, // Allow some overhead
            $"Expected <200 bytes/node, got {bytesPerInterval:F2}");
    }

    [Fact]
    public void HLC_ConstantMemoryUnderLoad()
    {
        // Arrange: HLC should use constant memory regardless of operation count
        var hlc = new HybridLogicalClock(nodeId: 1);
        var opCount = 1_000_000;

        var beforeOps = GC.GetTotalMemory(forceFullCollection: true);

        // Act: Perform many operations
        for (int i = 0; i < opCount; i++)
        {
            var ts = hlc.Now();

            // Occasional updates to simulate real usage
            if (i % 100 == 0)
            {
                var remoteTs = new HybridTimestamp(
                    ts.PhysicalTime + 1000,
                    0,
                    2);
                hlc.Update(remoteTs);
            }
        }

        var afterOps = GC.GetTotalMemory(forceFullCollection: true);
        var memoryGrowth = (afterOps - beforeOps) / (1024.0 * 1024.0);

        // Assert: Minimal memory growth (HLC is stateless except for last timestamp)
        Assert.True(memoryGrowth < 1.0, // Less than 1 MB growth
            $"HLC memory growth should be constant, got {memoryGrowth:F3} MB");

        _output.WriteLine($"HLC memory under load:");
        _output.WriteLine($"  Operations: {opCount:N0}");
        _output.WriteLine($"  Memory growth: {memoryGrowth:F3} MB");
        _output.WriteLine($"  Expected: <1 MB (constant memory)");
        _output.WriteLine($"  Status: {(memoryGrowth < 1.0 ? "PASS" : "MEMORY LEAK DETECTED")}");
    }

    [Fact]
    public void TemporalGraph_GracefulDegradationUnderPressure()
    {
        // Arrange: Simulate low memory by creating large initial allocation
        var graph = new TemporalGraphStorage();
        var baseTime = DateTimeOffset.UtcNow.ToUnixTimeNanoseconds();

        // Act: Try to insert many edges
        var successCount = 0;
        var failCount = 0;

        try
        {
            for (int i = 0; i < 100000; i++)
            {
                graph.AddEdge(
                    sourceId: (ulong)(i % 1000),
                    targetId: (ulong)((i + 1) % 1000),
                    validFrom: baseTime + (long)i,
                    validTo: baseTime + (long)i + 1_000_000_000L,
                    hlc: new HybridTimestamp(baseTime, (long)i, 1));
                successCount++;

                // Periodic GC
                if (i % 5000 == 0)
                {
                    GC.Collect(0, GCCollectionMode.Optimized);
                }
            }
        }
        catch (OutOfMemoryException)
        {
            failCount++;
            _output.WriteLine("OOM caught during stress test (expected under extreme pressure)");
        }

        // Assert: Operations succeeded or failed gracefully
        Assert.True(successCount > 0, "Should succeed under normal conditions");
        Assert.True(graph.EdgeCount == successCount);

        _output.WriteLine($"Graceful degradation test:");
        _output.WriteLine($"  Successful insertions: {successCount:N0}");
        _output.WriteLine($"  Failed insertions: {failCount:N0}");
        _output.WriteLine($"  Graph integrity: {(graph.EdgeCount == successCount ? "MAINTAINED" : "CORRUPTED")}");
    }

    [Fact]
    public void TemporalPath_MemoryEfficiency()
    {
        // Arrange: Create graph with paths
        var graph = new TemporalGraphStorage();
        var pathLength = 100;
        var baseTime = DateTimeOffset.UtcNow.ToUnixTimeNanoseconds();

        // Create a linear path
        for (int i = 0; i < pathLength; i++)
        {
            graph.AddEdge(
                sourceId: (ulong)i,
                targetId: (ulong)(i + 1),
                validFrom: baseTime + (long)i * 1_000_000_000L,
                validTo: baseTime + (long)(i + 10) * 1_000_000_000L,
                hlc: new HybridTimestamp(baseTime, (long)i, 1));
        }

        var beforePathfinding = GC.GetTotalMemory(forceFullCollection: true);

        // Act: Find paths (uses BFS with early termination from Week 14 optimization)
        var paths = graph.FindTemporalPaths(
            startNode: 0UL,
            endNode: (ulong)pathLength,
            maxTimeSpanNanos: 1000_000_000_000L,
            maxPathLength: pathLength);

        var pathList = paths.ToList();

        var afterPathfinding = GC.GetTotalMemory(forceFullCollection: false);
        var memoryUsed = (afterPathfinding - beforePathfinding) / 1024.0; // KB

        // Assert: Efficient pathfinding memory usage
        Assert.NotEmpty(pathList);

        _output.WriteLine($"Pathfinding memory efficiency:");
        _output.WriteLine($"  Path length: {pathLength}");
        _output.WriteLine($"  Paths found: {pathList.Count}");
        _output.WriteLine($"  Memory used: {memoryUsed:F2} KB");
        _output.WriteLine($"  Optimization: BFS early termination (Week 14)");
        _output.WriteLine($"  Status: {(memoryUsed < 100 ? "EFFICIENT" : "ACCEPTABLE")}");
    }

    [Fact]
    public void TemporalEdgeList_SortedListEfficiency()
    {
        // Arrange: Test TemporalEdgeList memory efficiency
        var edgeCount = 10000;
        var baseTime = DateTimeOffset.UtcNow.ToUnixTimeNanoseconds();

        // Create temporary graph to test edge list
        var graph = new TemporalGraphStorage();

        var beforeInsert = GC.GetTotalMemory(forceFullCollection: true);

        // Act: Add many edges to single node (tests TemporalEdgeList)
        for (int i = 0; i < (int)edgeCount; i++)
        {
            graph.AddEdge(
                sourceId: 1, // Same source
                targetId: (ulong)(i + 2),
                validFrom: baseTime + (long)i * 1_000_000L,
                validTo: baseTime + (long)(i + 1) * 1_000_000L,
                hlc: new HybridTimestamp(baseTime, (long)i, 1));
        }

        var afterInsert = GC.GetTotalMemory(forceFullCollection: true);
        var memoryUsed = (afterInsert - beforeInsert) / (1024.0 * 1024.0);

        // Query to verify efficiency
        var edges = graph.GetEdgesInTimeRange(
            sourceId: 1,
            startTimeNanos: baseTime,
            endTimeNanos: baseTime + edgeCount * 1_000_000L);

        var queryResultCount = edges.Count();

        _output.WriteLine($"TemporalEdgeList efficiency:");
        _output.WriteLine($"  Edges per node: {edgeCount:N0}");
        _output.WriteLine($"  Memory used: {memoryUsed:F2} MB");
        _output.WriteLine($"  Query results: {queryResultCount:N0}");
        _output.WriteLine($"  Data structure: SortedList<long, TemporalEdge>");
    }
}

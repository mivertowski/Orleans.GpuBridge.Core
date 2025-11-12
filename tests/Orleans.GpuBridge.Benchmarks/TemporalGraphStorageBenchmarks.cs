using BenchmarkDotNet.Attributes;
using Orleans.GpuBridge.Abstractions.Temporal;
using Orleans.GpuBridge.Runtime.Temporal;
using Orleans.GpuBridge.Runtime.Temporal.Graph;

namespace Orleans.GpuBridge.Benchmarks;

/// <summary>
/// Baseline performance benchmarks for TemporalGraphStorage.
/// Target: <100μs for path search, efficient time-range queries.
/// </summary>
[MemoryDiagnoser]
[MinColumn, MaxColumn, MeanColumn, MedianColumn]
[MarkdownExporter, HtmlExporter, CsvExporter]
public class TemporalGraphStorageBenchmarks
{
    private TemporalGraphStorage _smallGraph = null!; // 100 nodes
    private TemporalGraphStorage _mediumGraph = null!; // 1,000 nodes
    private TemporalGraphStorage _largeGraph = null!; // 10,000 nodes
    private HybridLogicalClock _hlc = null!;
    private long _baseTime;

    [GlobalSetup]
    public void Setup()
    {
        _hlc = new HybridLogicalClock(nodeId: 1);
        _baseTime = DateTimeOffset.UtcNow.ToUnixTimeNanoseconds();

        // Create graphs of different sizes
        _smallGraph = CreateGraph(100, avgDegree: 5);
        _mediumGraph = CreateGraph(1_000, avgDegree: 10);
        _largeGraph = CreateGraph(10_000, avgDegree: 15);
    }

    private TemporalGraphStorage CreateGraph(int nodeCount, int avgDegree)
    {
        var graph = new TemporalGraphStorage();
        var random = new Random(42); // Deterministic

        for (int i = 0; i < nodeCount; i++)
        {
            // Create edges to random neighbors
            var degree = random.Next(avgDegree / 2, avgDegree * 2);
            for (int j = 0; j < degree; j++)
            {
                var target = (ulong)random.Next(0, nodeCount);
                if (target == (ulong)i) continue; // No self-loops

                var validFrom = _baseTime + (long)i * 1_000_000L;
                var validTo = validFrom + random.NextInt64(1_000_000_000L, 10_000_000_000L);

                graph.AddEdge(
                    sourceId: (ulong)i,
                    targetId: target,
                    validFrom: validFrom,
                    validTo: validTo,
                    hlc: new HybridTimestamp(validFrom, (long)i, 1));
            }
        }

        return graph;
    }

    #region Edge Operations

    /// <summary>
    /// Baseline: Single edge insertion.
    /// Measures IntervalTree Add() performance.
    /// </summary>
    [Benchmark(Baseline = true)]
    public void AddEdge_Single()
    {
        var graph = new TemporalGraphStorage();
        graph.AddEdge(
            sourceId: 1,
            targetId: 2,
            validFrom: _baseTime,
            validTo: _baseTime + 1_000_000_000L,
            hlc: _hlc.Now());
    }

    /// <summary>
    /// Throughput: 1000 edge insertions.
    /// Measures sustained insertion rate.
    /// </summary>
    [Benchmark]
    public void AddEdge_Batch1000()
    {
        var graph = new TemporalGraphStorage();
        for (int i = 0; i < 1000; i++)
        {
            graph.AddEdge(
                sourceId: (ulong)(i % 100),
                targetId: (ulong)((i + 1) % 100),
                validFrom: _baseTime + (long)i,
                validTo: _baseTime + (long)i + 1_000_000_000L,
                hlc: new HybridTimestamp(_baseTime, (long)i, 1));
        }
    }

    #endregion

    #region Time-Range Query Benchmarks

    /// <summary>
    /// Baseline: Time-range query on small graph (100 nodes).
    /// Target: Sub-microsecond query for small graphs.
    /// </summary>
    [Benchmark]
    public int GetEdgesInTimeRange_SmallGraph()
    {
        var edges = _smallGraph.GetEdgesInTimeRange(
            sourceId: 50,
            startTimeNanos: _baseTime,
            endTimeNanos: _baseTime + 5_000_000_000L);
        return edges.Count();
    }

    /// <summary>
    /// Baseline: Time-range query on medium graph (1K nodes).
    /// Target: Single-digit microseconds.
    /// </summary>
    [Benchmark]
    public int GetEdgesInTimeRange_MediumGraph()
    {
        var edges = _mediumGraph.GetEdgesInTimeRange(
            sourceId: 500,
            startTimeNanos: _baseTime,
            endTimeNanos: _baseTime + 5_000_000_000L);
        return edges.Count();
    }

    /// <summary>
    /// Baseline: Time-range query on large graph (10K nodes).
    /// Target: <10μs despite scale.
    /// </summary>
    [Benchmark]
    public int GetEdgesInTimeRange_LargeGraph()
    {
        var edges = _largeGraph.GetEdgesInTimeRange(
            sourceId: 5000,
            startTimeNanos: _baseTime,
            endTimeNanos: _baseTime + 5_000_000_000L);
        return edges.Count();
    }

    /// <summary>
    /// Baseline: Global time-range query (all edges in range).
    /// Measures multi-node query aggregation.
    /// </summary>
    [Benchmark]
    public int GetAllEdgesInTimeRange_MediumGraph()
    {
        var edges = _mediumGraph.GetAllEdgesInTimeRange(
            startTimeNanos: _baseTime,
            endTimeNanos: _baseTime + 5_000_000_000L);
        return edges.Count();
    }

    #endregion

    #region Path Search Benchmarks

    /// <summary>
    /// Baseline: Temporal path search on small graph.
    /// Target: <10μs for short paths in small graphs.
    /// </summary>
    [Benchmark]
    public int FindTemporalPaths_SmallGraph_ShortPath()
    {
        var paths = _smallGraph.FindTemporalPaths(
            startNode: 10,
            endNode: 20,
            maxTimeSpanNanos: 10_000_000_000L,
            maxPathLength: 10);
        return paths.Count();
    }

    /// <summary>
    /// Baseline: Temporal path search on medium graph.
    /// Target: <100μs for medium-length paths.
    /// </summary>
    [Benchmark]
    public int FindTemporalPaths_MediumGraph_MediumPath()
    {
        var paths = _mediumGraph.FindTemporalPaths(
            startNode: 100,
            endNode: 900,
            maxTimeSpanNanos: 50_000_000_000L,
            maxPathLength: 20);
        return paths.Count();
    }

    /// <summary>
    /// Baseline: Temporal path search on large graph.
    /// Target: <500μs for complex path searches.
    /// Validates BFS optimization (Week 14).
    /// </summary>
    [Benchmark]
    public int FindTemporalPaths_LargeGraph_LongPath()
    {
        var paths = _largeGraph.FindTemporalPaths(
            startNode: 1000,
            endNode: 9000,
            maxTimeSpanNanos: 100_000_000_000L,
            maxPathLength: 50);
        return paths.Count();
    }

    /// <summary>
    /// Worst-case: Path search with no path exists.
    /// Measures early termination efficiency.
    /// </summary>
    [Benchmark]
    public int FindTemporalPaths_NoPath()
    {
        var paths = _mediumGraph.FindTemporalPaths(
            startNode: 1,
            endNode: 999,
            maxTimeSpanNanos: 1L, // Too short for any path
            maxPathLength: 100);
        return paths.Count();
    }

    #endregion

    #region Statistics and Traversal

    /// <summary>
    /// Baseline: Graph statistics computation.
    /// Measures metadata aggregation performance.
    /// </summary>
    [Benchmark]
    public GraphStatistics GetStatistics_MediumGraph()
    {
        return _mediumGraph.GetStatistics();
    }

    /// <summary>
    /// Baseline: Node containment check.
    /// Should be O(1) with hash-based lookup.
    /// </summary>
    [Benchmark]
    public bool ContainsNode_MediumGraph()
    {
        return _mediumGraph.ContainsNode(500);
    }

    /// <summary>
    /// Baseline: Get all nodes in graph.
    /// Measures full traversal cost.
    /// </summary>
    [Benchmark]
    public int GetAllNodes_MediumGraph()
    {
        return _mediumGraph.GetAllNodes().Count();
    }

    #endregion

    #region Memory Benchmarks

    /// <summary>
    /// Memory: Graph construction with 1K nodes.
    /// Validates memory efficiency and allocation patterns.
    /// </summary>
    [Benchmark]
    public TemporalGraphStorage MemoryConstruction_1K()
    {
        return CreateGraph(1_000, avgDegree: 10);
    }

    /// <summary>
    /// Memory: Graph construction with 10K nodes.
    /// Measures large-scale memory footprint.
    /// </summary>
    [Benchmark]
    public TemporalGraphStorage MemoryConstruction_10K()
    {
        return CreateGraph(10_000, avgDegree: 15);
    }

    #endregion
}

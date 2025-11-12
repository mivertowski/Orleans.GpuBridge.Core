using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Configs;
using BenchmarkDotNet.Diagnosers;
using BenchmarkDotNet.Jobs;
using Orleans.GpuBridge.Abstractions.Temporal;
using Orleans.GpuBridge.Runtime.Temporal;
using Orleans.GpuBridge.Runtime.Temporal.Graph;

namespace Orleans.GpuBridge.Benchmarks;

/// <summary>
/// Simplified profiling harness for temporal components using BenchmarkDotNet.
/// Establishes baseline performance metrics for Phase 7 optimization.
/// </summary>
[MemoryDiagnoser]
[Config(typeof(Config))]
public class TemporalProfilingHarness
{
    private HybridLogicalClock? _hlc;
    private TemporalGraphStorage? _graph;
    private TemporalGraphStorage? _iterationGraph; // Fresh graph for Add edge benchmark
    private HybridTimestamp _sampleTimestamp;

    private class Config : ManualConfig
    {
        public Config()
        {
            AddJob(Job.Default
                .WithIterationCount(100)
                .WithWarmupCount(10));

            AddDiagnoser(MemoryDiagnoser.Default);
        }
    }

    [GlobalSetup]
    public void Setup()
    {
        _hlc = new HybridLogicalClock(nodeId: 1);
        _graph = new TemporalGraphStorage();
        _sampleTimestamp = new HybridTimestamp(1000000000L, 0, 1);

        // Pre-populate graph with test data for query benchmarks
        for (ulong i = 1; i <= 100; i++)
        {
            for (ulong j = i + 1; j <= Math.Min(i + 10, 100); j++)
            {
                _graph.AddEdge(
                    sourceId: i,
                    targetId: j,
                    validFrom: (long)(i * 1_000_000_000L),
                    validTo: (long)((i + 10) * 1_000_000_000L),
                    hlc: _sampleTimestamp);
            }
        }
    }

    /// <summary>
    /// Reset state before each iteration to prevent memory accumulation.
    /// </summary>
    [IterationSetup(Target = nameof(GraphAddEdge))]
    public void IterationSetup()
    {
        // Create fresh graph for each iteration to prevent OOM
        _iterationGraph = new TemporalGraphStorage();
    }

    #region HLC Benchmarks

    /// <summary>
    /// Benchmark: Generate HLC timestamp for local event.
    /// Target: &lt;40ns (baseline from Phase 6: 42ns).
    /// </summary>
    [Benchmark(Description = "HLC: Generate timestamp")]
    public HybridTimestamp HlcGenerate()
    {
        return _hlc!.Now();
    }

    /// <summary>
    /// Benchmark: Update HLC with received timestamp.
    /// Target: &lt;70ns.
    /// </summary>
    [Benchmark(Description = "HLC: Update with received timestamp")]
    public HybridTimestamp HlcUpdate()
    {
        var received = new HybridTimestamp(
            physicalTime: DateTimeOffset.UtcNow.ToUnixTimeNanoseconds(),
            logicalCounter: 42,
            nodeId: 2);
        return _hlc!.Update(received);
    }

    /// <summary>
    /// Benchmark: Compare two HLC timestamps.
    /// Target: &lt;20ns.
    /// </summary>
    [Benchmark(Description = "HLC: Compare timestamps")]
    public int HlcCompare()
    {
        var ts1 = new HybridTimestamp(1000000000L, 42, 1);
        var ts2 = new HybridTimestamp(1000000001L, 43, 2);
        return ts1.CompareTo(ts2);
    }

    #endregion

    #region Temporal Graph Benchmarks

    /// <summary>
    /// Benchmark: Add edge to temporal graph.
    /// Current: 319μs per edge (10,000 edges test)
    /// Target: &lt;100μs
    /// </summary>
    [Benchmark(Description = "Graph: Add edge")]
    public long GraphAddEdge()
    {
        var edge = new TemporalEdge(
            sourceId: 200,
            targetId: 201,
            validFrom: DateTimeOffset.UtcNow.ToUnixTimeNanoseconds(),
            validTo: DateTimeOffset.UtcNow.ToUnixTimeNanoseconds() + 1_000_000_000L,
            hlc: _sampleTimestamp,
            weight: 1.0);
        _iterationGraph!.AddEdge(edge); // Use iteration-specific graph
        return _iterationGraph.EdgeCount; // Return edge count to prevent JIT optimization
    }

    /// <summary>
    /// Benchmark: Query edges in time range.
    /// Current: Unknown (baseline to be established)
    /// Target: &lt;200ns (from Phase 6: 457ns baseline)
    /// </summary>
    [Benchmark(Description = "Graph: Query time range")]
    public int GraphQueryTimeRange()
    {
        var edges = _graph!.GetEdgesInTimeRange(
            sourceId: 50,
            startTimeNanos: 45_000_000_000L,
            endTimeNanos: 55_000_000_000L);
        return edges.Count(); // Materialize to measure actual query execution
    }

    /// <summary>
    /// Benchmark: Find temporal paths between nodes.
    /// Target: &lt;1ms for simple 2-hop paths
    /// </summary>
    [Benchmark(Description = "Graph: Find temporal paths")]
    public int GraphFindPaths()
    {
        // Find paths from node 1 to node 5 (shorter path, more realistic)
        // With setup: node 1 → {2,3,4,5,...,11}
        // So there's a direct path 1→5 and multi-hop paths 1→2→5, 1→3→5, etc.
        var paths = _graph!.FindTemporalPaths(
            startNode: 1,
            endNode: 5,
            maxTimeSpanNanos: 50_000_000_000L, // 50 seconds
            maxPathLength: 5); // Limit path complexity
        return paths.Count(); // Materialize to measure actual pathfinding
    }

    /// <summary>
    /// Benchmark: Get reachable nodes from starting point.
    /// Target: &lt;500μs for 100-node graph
    /// </summary>
    [Benchmark(Description = "Graph: Get reachable nodes")]
    public int GraphGetReachableNodes()
    {
        var nodes = _graph!.GetReachableNodes(
            startNode: 1,
            startTimeNanos: 0,
            maxTimeSpanNanos: 1_000_000_000_000L);
        return nodes.Count(); // Materialize to measure actual reachability analysis
    }

    /// <summary>
    /// Benchmark: Get graph statistics.
    /// Target: &lt;50ns (property access).
    /// </summary>
    [Benchmark(Description = "Graph: Get statistics")]
    public (int nodes, long edges) GraphGetStats()
    {
        return (_graph!.NodeCount, _graph.EdgeCount);
    }

    #endregion

    #region Combined Operations Benchmarks

    /// <summary>
    /// Benchmark: Complete workflow (HLC + Graph).
    /// Simulates real-world actor message processing pattern.
    /// Target: &lt;500ns for HLC + graph query.
    /// </summary>
    [Benchmark(Description = "Combined: HLC + Graph query")]
    public void CombinedWorkflow()
    {
        // Generate timestamp (actor receives message)
        var timestamp = _hlc!.Now();

        // Query graph for related edges (actor checks relationships)
        var edges = _graph!.GetEdgesInTimeRange(
            sourceId: 50,
            startTimeNanos: timestamp.PhysicalTime - 1_000_000_000L,
            endTimeNanos: timestamp.PhysicalTime + 1_000_000_000L);

        // Consume result to prevent optimization
        _ = edges.Count();
    }

    #endregion

    [GlobalCleanup]
    public void Cleanup()
    {
        // Cleanup if needed
    }
}

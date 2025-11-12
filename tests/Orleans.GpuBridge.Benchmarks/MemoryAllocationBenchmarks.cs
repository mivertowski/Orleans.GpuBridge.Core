using BenchmarkDotNet.Attributes;
using Orleans.GpuBridge.Abstractions.Temporal;
using Orleans.GpuBridge.Runtime.Temporal;
using Orleans.GpuBridge.Runtime.Temporal.Graph;

namespace Orleans.GpuBridge.Benchmarks;

/// <summary>
/// Memory allocation and GC pressure benchmarks.
/// Target: Minimize allocations, validate zero-allocation paths for hot code.
/// </summary>
[MemoryDiagnoser]
[MinColumn, MaxColumn, MeanColumn, MedianColumn]
[MarkdownExporter, HtmlExporter, CsvExporter]
public class MemoryAllocationBenchmarks
{
    private HybridLogicalClock _hlc = null!;
    private TemporalGraphStorage _graph = null!;
    private long _baseTime;

    [GlobalSetup]
    public void Setup()
    {
        _hlc = new HybridLogicalClock(nodeId: 1);
        _baseTime = DateTimeOffset.UtcNow.ToUnixTimeNanoseconds();

        // Pre-populate graph for query benchmarks
        _graph = new TemporalGraphStorage();
        for (int i = 0; i < 1000; i++)
        {
            _graph.AddEdge(
                sourceId: (ulong)(i % 100),
                targetId: (ulong)((i + 1) % 100),
                validFrom: _baseTime + (long)i,
                validTo: _baseTime + (long)i + 1_000_000_000L,
                hlc: new HybridTimestamp(_baseTime, (long)i, 1));
        }
    }

    #region HLC Allocation Tests

    /// <summary>
    /// Zero-allocation test: HybridTimestamp is a struct.
    /// Target: 0 bytes allocated per operation.
    /// </summary>
    [Benchmark(Baseline = true)]
    public HybridTimestamp HLC_Now_ZeroAllocation()
    {
        // HybridTimestamp is a struct - should not allocate
        return _hlc.Now();
    }

    /// <summary>
    /// Allocation test: 1000 timestamp generations.
    /// Target: 0 bytes allocated (all stack-based structs).
    /// </summary>
    [Benchmark]
    public void HLC_Now_Batch1000_AllocationTest()
    {
        for (int i = 0; i < 1000; i++)
        {
            _ = _hlc.Now();
        }
    }

    /// <summary>
    /// Allocation test: Timestamp comparison.
    /// Target: 0 bytes allocated (struct comparison).
    /// </summary>
    [Benchmark]
    public int HLC_CompareTo_ZeroAllocation()
    {
        var ts1 = _hlc.Now();
        var ts2 = _hlc.Now();
        return ts1.CompareTo(ts2);
    }

    #endregion

    #region IntervalTree Allocation Tests

    /// <summary>
    /// Allocation test: Single interval insertion.
    /// Measures node allocation overhead.
    /// </summary>
    [Benchmark]
    public void IntervalTree_Add_AllocationTest()
    {
        var tree = new IntervalTree<long, string>();
        tree.Add(0, 1000, "test");
    }

    /// <summary>
    /// Allocation test: 1000 insertions.
    /// Measures cumulative allocation for tree construction.
    /// </summary>
    [Benchmark]
    public void IntervalTree_Add_Batch1000_AllocationTest()
    {
        var tree = new IntervalTree<long, string>();
        for (int i = 0; i < 1000; i++)
        {
            tree.Add(i * 100L, i * 100L + 50L, $"interval_{i}");
        }
    }

    /// <summary>
    /// Allocation test: Query operation.
    /// Measures enumerable allocation overhead.
    /// </summary>
    [Benchmark]
    public int IntervalTree_Query_AllocationTest()
    {
        var tree = new IntervalTree<long, string>();
        for (int i = 0; i < 100; i++)
        {
            tree.Add(i * 100L, i * 100L + 50L, $"interval_{i}");
        }

        var results = tree.Query(400L, 600L);
        return results.Count(); // Materialize enumerable
    }

    #endregion

    #region TemporalGraphStorage Allocation Tests

    /// <summary>
    /// Allocation test: Edge insertion.
    /// Measures combined HLC + IntervalTree allocations.
    /// </summary>
    [Benchmark]
    public void TemporalGraph_AddEdge_AllocationTest()
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
    /// Allocation test: Time-range query.
    /// Measures enumerable and edge structure allocations.
    /// </summary>
    [Benchmark]
    public int TemporalGraph_GetEdgesInTimeRange_AllocationTest()
    {
        var edges = _graph.GetEdgesInTimeRange(
            sourceId: 50,
            startTimeNanos: _baseTime,
            endTimeNanos: _baseTime + 5_000_000_000L);
        return edges.Count();
    }

    /// <summary>
    /// Allocation test: Path search.
    /// Measures BFS queue and path list allocations.
    /// </summary>
    [Benchmark]
    public int TemporalGraph_FindTemporalPaths_AllocationTest()
    {
        var paths = _graph.FindTemporalPaths(
            startNode: 10,
            endNode: 90,
            maxTimeSpanNanos: 100_000_000_000L,
            maxPathLength: 20);
        return paths.Count();
    }

    #endregion

    #region GC Pressure Tests

    /// <summary>
    /// GC pressure test: Rapid graph construction and disposal.
    /// Measures Gen0/Gen1/Gen2 collection frequency.
    /// </summary>
    [Benchmark]
    public void GCPressure_GraphConstruction()
    {
        for (int i = 0; i < 100; i++)
        {
            var graph = new TemporalGraphStorage();
            for (int j = 0; j < 100; j++)
            {
                graph.AddEdge(
                    sourceId: (ulong)(j % 10),
                    targetId: (ulong)((j + 1) % 10),
                    validFrom: _baseTime + (long)j,
                    validTo: _baseTime + (long)j + 1_000_000_000L,
                    hlc: new HybridTimestamp(_baseTime, (long)j, 1));
            }
        }
    }

    /// <summary>
    /// GC pressure test: Large object heap allocations.
    /// Measures LOH pressure from large graphs.
    /// </summary>
    [Benchmark]
    public void GCPressure_LargeGraph()
    {
        var graph = new TemporalGraphStorage();
        for (int i = 0; i < 100_000; i++)
        {
            graph.AddEdge(
                sourceId: (ulong)(i % 1000),
                targetId: (ulong)((i + 1) % 1000),
                validFrom: _baseTime + (long)i,
                validTo: _baseTime + (long)i + 1_000_000_000L,
                hlc: new HybridTimestamp(_baseTime, (long)i, 1));
        }
    }

    #endregion

    #region String Allocation Tests

    /// <summary>
    /// Allocation test: Edge metadata with strings.
    /// Validates metadata field allocation overhead.
    /// </summary>
    [Benchmark]
    public void TemporalGraph_AddEdge_WithMetadata_AllocationTest()
    {
        var graph = new TemporalGraphStorage();
        graph.AddEdge(
            sourceId: 1,
            targetId: 2,
            validFrom: _baseTime,
            validTo: _baseTime + 1_000_000_000L,
            hlc: _hlc.Now(),
            weight: 1.0,
            edgeType: "test_metadata_string");
    }

    #endregion
}

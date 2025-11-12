using BenchmarkDotNet.Attributes;
using Orleans.GpuBridge.Runtime.Temporal.Graph;

namespace Orleans.GpuBridge.Benchmarks;

/// <summary>
/// Baseline performance benchmarks for IntervalTree<TKey, TValue>.
/// Target: O(log N) query performance with AVL balancing.
/// </summary>
[MemoryDiagnoser]
[MinColumn, MaxColumn, MeanColumn, MedianColumn]
[MarkdownExporter, HtmlExporter, CsvExporter]
public class IntervalTreeBenchmarks
{
    private IntervalTree<long, string> _tree = null!;
    private IntervalTree<long, string> _tree1K = null!;
    private IntervalTree<long, string> _tree10K = null!;
    private IntervalTree<long, string> _tree100K = null!;
    private IntervalTree<long, string> _tree1M = null!;

    [GlobalSetup]
    public void Setup()
    {
        // Empty tree for insertion benchmarks
        _tree = new IntervalTree<long, string>();

        // Pre-populated trees for query benchmarks
        _tree1K = CreatePopulatedTree(1_000);
        _tree10K = CreatePopulatedTree(10_000);
        _tree100K = CreatePopulatedTree(100_000);
        _tree1M = CreatePopulatedTree(1_000_000);
    }

    private static IntervalTree<long, string> CreatePopulatedTree(int count)
    {
        var tree = new IntervalTree<long, string>();
        var random = new Random(42); // Deterministic seed

        for (int i = 0; i < count; i++)
        {
            var start = random.NextInt64(0, 1_000_000_000L);
            var end = start + random.NextInt64(1, 10_000_000L);
            tree.Add(start, end, $"interval_{i}");
        }

        return tree;
    }

    #region Insertion Benchmarks

    /// <summary>
    /// Baseline: Single interval insertion into empty tree.
    /// Measures AVL insertion performance.
    /// </summary>
    [Benchmark(Baseline = true)]
    public void Add_Single()
    {
        var tree = new IntervalTree<long, string>();
        tree.Add(0, 1000, "test");
    }

    /// <summary>
    /// Throughput: 1000 sequential insertions (worst case for AVL balance).
    /// Validates O(log N) insertion with rotations.
    /// </summary>
    [Benchmark]
    public void Add_Sequential_1K()
    {
        var tree = new IntervalTree<long, string>();
        for (int i = 0; i < 1000; i++)
        {
            tree.Add(i * 100L, i * 100L + 50L, $"interval_{i}");
        }
    }

    /// <summary>
    /// Throughput: 10K sequential insertions (worst case for AVL balance).
    /// Validates O(log N) insertion with rotations at scale.
    /// </summary>
    [Benchmark]
    public void Add_Sequential_10K()
    {
        var tree = new IntervalTree<long, string>();
        for (int i = 0; i < 10000; i++)
        {
            tree.Add(i * 100L, i * 100L + 50L, $"interval_{i}");
        }
    }

    /// <summary>
    /// Throughput: 1000 random insertions (average case for AVL balance).
    /// More realistic workload with mixed rotations.
    /// </summary>
    [Benchmark]
    public void Add_Random_1K()
    {
        var tree = new IntervalTree<long, string>();
        var random = new Random(42);
        for (int i = 0; i < 1000; i++)
        {
            var start = random.NextInt64(0, 1_000_000_000L);
            var end = start + random.NextInt64(1, 10_000_000L);
            tree.Add(start, end, $"interval_{i}");
        }
    }

    /// <summary>
    /// Throughput: 10K random insertions (average case for AVL balance).
    /// Validates performance at scale.
    /// </summary>
    [Benchmark]
    public void Add_Random_10K()
    {
        var tree = new IntervalTree<long, string>();
        var random = new Random(42);
        for (int i = 0; i < 10000; i++)
        {
            var start = random.NextInt64(0, 1_000_000_000L);
            var end = start + random.NextInt64(1, 10_000_000L);
            tree.Add(start, end, $"interval_{i}");
        }
    }

    #endregion

    #region Query Benchmarks

    /// <summary>
    /// Baseline: Point query on tree with 1K intervals.
    /// Target: O(log 1000) ≈ 10 comparisons.
    /// </summary>
    [Benchmark]
    public int QueryPoint_1K()
    {
        var results = _tree1K.QueryPoint(500_000L);
        return results.Count();
    }

    /// <summary>
    /// Baseline: Point query on tree with 10K intervals.
    /// Target: O(log 10000) ≈ 13 comparisons.
    /// </summary>
    [Benchmark]
    public int QueryPoint_10K()
    {
        var results = _tree10K.QueryPoint(500_000L);
        return results.Count();
    }

    /// <summary>
    /// Baseline: Point query on tree with 100K intervals.
    /// Target: O(log 100000) ≈ 17 comparisons.
    /// </summary>
    [Benchmark]
    public int QueryPoint_100K()
    {
        var results = _tree100K.QueryPoint(500_000L);
        return results.Count();
    }

    /// <summary>
    /// Baseline: Point query on tree with 1M intervals.
    /// Target: O(log 1000000) ≈ 20 comparisons.
    /// Validates AVL balance at scale.
    /// </summary>
    [Benchmark]
    public int QueryPoint_1M()
    {
        var results = _tree1M.QueryPoint(500_000L);
        return results.Count();
    }

    /// <summary>
    /// Baseline: Range query on tree with 10K intervals.
    /// Measures multi-result query performance.
    /// </summary>
    [Benchmark]
    public int QueryRange_10K()
    {
        var results = _tree10K.Query(400_000L, 600_000L);
        return results.Count();
    }

    /// <summary>
    /// Baseline: Range query on tree with 100K intervals.
    /// Validates performance with larger result sets.
    /// </summary>
    [Benchmark]
    public int QueryRange_100K()
    {
        var results = _tree100K.Query(400_000L, 600_000L);
        return results.Count();
    }

    /// <summary>
    /// Worst-case: Full range query (returns all intervals).
    /// Measures maximum traversal cost.
    /// </summary>
    [Benchmark]
    public int QueryRange_Full_10K()
    {
        var results = _tree10K.Query(long.MinValue, long.MaxValue);
        return results.Count();
    }

    #endregion

    #region Memory Benchmarks

    /// <summary>
    /// Memory: Tree construction with 10K intervals.
    /// Validates memory efficiency and GC pressure.
    /// </summary>
    [Benchmark]
    public IntervalTree<long, string> MemoryConstruction_10K()
    {
        return CreatePopulatedTree(10_000);
    }

    /// <summary>
    /// Memory: Tree construction with 100K intervals.
    /// Measures large-scale memory allocation patterns.
    /// </summary>
    [Benchmark]
    public IntervalTree<long, string> MemoryConstruction_100K()
    {
        return CreatePopulatedTree(100_000);
    }

    #endregion

    #region AVL Balance Verification

    /// <summary>
    /// Validation: Query performance after sequential insertions.
    /// Verifies AVL rotations maintain O(log N) despite worst-case input.
    /// </summary>
    [Benchmark]
    public int AVLBalance_SequentialInsert_Query()
    {
        var tree = new IntervalTree<long, string>();

        // Sequential insertions (worst case for unbalanced tree)
        for (int i = 0; i < 10_000; i++)
        {
            tree.Add(i * 100L, i * 100L + 50L, $"interval_{i}");
        }

        // Query should still be O(log N) due to AVL balance
        var results = tree.QueryPoint(500_000L);
        return results.Count();
    }

    #endregion
}

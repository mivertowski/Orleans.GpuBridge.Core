using BenchmarkDotNet.Attributes;
using Orleans.GpuBridge.Abstractions.Temporal;
using Orleans.GpuBridge.Runtime.Temporal;

namespace Orleans.GpuBridge.Benchmarks;

/// <summary>
/// Baseline performance benchmarks for HybridLogicalClock (HLC).
/// Target: <50ns per timestamp generation (CPU baseline before GPU optimization).
/// </summary>
[MemoryDiagnoser]
[MinColumn, MaxColumn, MeanColumn, MedianColumn]
[MarkdownExporter, HtmlExporter, CsvExporter]
public class HybridLogicalClockBenchmarks
{
    private HybridLogicalClock _hlc = null!;
    private HybridTimestamp _remoteTimestamp;
    private HybridTimestamp[] _remoteTimestamps = null!;

    [GlobalSetup]
    public void Setup()
    {
        _hlc = new HybridLogicalClock(nodeId: 1);

        // Warmup: Generate initial timestamp
        _ = _hlc.Now();

        // Setup remote timestamp for Update() benchmarks
        var remoteHlc = new HybridLogicalClock(nodeId: 2);
        _remoteTimestamp = remoteHlc.Now();

        // Setup array of remote timestamps for batch benchmarks
        _remoteTimestamps = new HybridTimestamp[1000];
        for (int i = 0; i < _remoteTimestamps.Length; i++)
        {
            _remoteTimestamps[i] = remoteHlc.Now();
        }
    }

    /// <summary>
    /// Baseline: Single timestamp generation.
    /// Target: <50ns (CPU baseline).
    /// </summary>
    [Benchmark(Baseline = true)]
    public HybridTimestamp Now()
    {
        return _hlc.Now();
    }

    /// <summary>
    /// Baseline: Timestamp update with remote timestamp.
    /// Measures merge operation performance.
    /// </summary>
    [Benchmark]
    public HybridTimestamp Update()
    {
        return _hlc.Update(_remoteTimestamp);
    }

    /// <summary>
    /// Baseline: Timestamp comparison operation.
    /// Critical for ordering verification.
    /// </summary>
    [Benchmark]
    public int CompareTo()
    {
        var ts1 = _hlc.Now();
        var ts2 = _hlc.Now();
        return ts1.CompareTo(ts2);
    }

    /// <summary>
    /// Throughput: 1000 sequential timestamp generations.
    /// Measures sustained timestamp generation rate.
    /// </summary>
    [Benchmark]
    public void Now_Batch1000()
    {
        for (int i = 0; i < 1000; i++)
        {
            _ = _hlc.Now();
        }
    }

    /// <summary>
    /// Throughput: 1000 sequential updates with remote timestamps.
    /// Measures merge operation throughput.
    /// </summary>
    [Benchmark]
    public void Update_Batch1000()
    {
        for (int i = 0; i < 1000; i++)
        {
            _ = _hlc.Update(_remoteTimestamps[i]);
        }
    }

    /// <summary>
    /// Memory allocation: Timestamp creation and disposal.
    /// Validates zero-allocation design (structs).
    /// </summary>
    [Benchmark]
    public HybridTimestamp Now_AllocationTest()
    {
        // HybridTimestamp is a struct - should not allocate
        return _hlc.Now();
    }

    /// <summary>
    /// Worst-case: Timestamp generation with logical counter increment.
    /// Simulates rapid timestamp generation at same physical time.
    /// </summary>
    [Benchmark]
    public void Now_LogicalCounterIncrement()
    {
        // Force logical counter increments by rapid generation
        for (int i = 0; i < 10; i++)
        {
            _ = _hlc.Now();
        }
    }
}

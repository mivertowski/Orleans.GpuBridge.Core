using BenchmarkDotNet.Running;
using Orleans.GpuBridge.Benchmarks;

namespace Orleans.GpuBridge.Benchmarks;

/// <summary>
/// Console application for running BenchmarkDotNet performance profiling.
/// Phase 7 Week 16: Comprehensive baseline performance benchmarks.
/// </summary>
public class Program
{
    public static void Main(string[] args)
    {
        Console.WriteLine("=== Orleans GpuBridge Performance Profiling ===");
        Console.WriteLine("Phase 7 Week 16: Comprehensive Baseline Benchmarks\n");

        // Run all benchmark suites using BenchmarkSwitcher
        var summary = BenchmarkSwitcher.FromAssembly(typeof(Program).Assembly).Run(args);

        Console.WriteLine("\n=== Week 16 Performance Targets ===");
        Console.WriteLine("HLC Benchmarks:");
        Console.WriteLine("  - Timestamp generation: <50ns (CPU baseline)");
        Console.WriteLine("  - Timestamp update: <70ns");
        Console.WriteLine("  - Comparison: <5ns (struct comparison)");
        Console.WriteLine("\nIntervalTree Benchmarks:");
        Console.WriteLine("  - Single insertion: <1μs");
        Console.WriteLine("  - Query (1K intervals): O(log 1000) ≈ 10 comparisons");
        Console.WriteLine("  - Query (1M intervals): O(log 1000000) ≈ 20 comparisons");
        Console.WriteLine("\nTemporalGraph Benchmarks:");
        Console.WriteLine("  - AddEdge: <5μs");
        Console.WriteLine("  - Time-range query: <10μs");
        Console.WriteLine("  - Path search (small graph): <10μs");
        Console.WriteLine("  - Path search (medium graph): <100μs");
        Console.WriteLine("\nMemory Benchmarks:");
        Console.WriteLine("  - HLC Now(): 0 bytes (stack-only structs)");
        Console.WriteLine("  - IntervalTree node: ~64 bytes per node");
        Console.WriteLine("  - GC Gen0: Minimize collections");
    }
}

using BenchmarkDotNet.Running;
using Orleans.GpuBridge.Benchmarks;

namespace Orleans.GpuBridge.Benchmarks;

/// <summary>
/// Console application for running BenchmarkDotNet performance profiling.
/// Phase 7 Week 13: Establish baseline performance metrics for optimization.
/// </summary>
public class Program
{
    public static void Main(string[] args)
    {
        Console.WriteLine("=== Orleans GpuBridge Performance Profiling ===");
        Console.WriteLine("Phase 7 Week 13: Baseline Performance Benchmarks\n");

        var summary = BenchmarkRunner.Run<TemporalProfilingHarness>(args: args);

        Console.WriteLine("\n=== Benchmark Summary ===");
        Console.WriteLine($"Total benchmarks run: {summary.Reports.Length}");
        Console.WriteLine($"Results directory: {summary.ResultsDirectoryPath}");
        Console.WriteLine($"Log file: {summary.LogFilePath}");
        Console.WriteLine("\nPerformance targets:");
        Console.WriteLine("  - HLC generation: <40ns");
        Console.WriteLine("  - HLC update: <70ns");
        Console.WriteLine("  - Graph AddEdge: <100μs");
        Console.WriteLine("  - Graph time-range query: <200ns");
        Console.WriteLine("  - Combined workflow: <500ns");
    }
}

using System.Diagnostics;
using FluentAssertions;
using Microsoft.Extensions.Logging.Abstractions;
using Orleans.GpuBridge.Runtime.Temporal.Clock;
using Xunit.Abstractions;

namespace Orleans.GpuBridge.Temporal.Tests.Benchmarks;

/// <summary>
/// Performance benchmarks for Phase 6 clock sources.
/// Measures time read latency, overhead, and memory usage.
/// </summary>
public sealed class ClockSourceBenchmarks
{
    private readonly ITestOutputHelper _output;

    public ClockSourceBenchmarks(ITestOutputHelper output)
    {
        _output = output;
    }

    [Fact]
    public async Task Benchmark_PtpClockSource_TimeReadLatency()
    {
        // Arrange
        var ptpClock = new PtpClockSource(NullLogger<PtpClockSource>.Instance);
        bool initialized = await ptpClock.InitializeAsync();

        if (!initialized)
        {
            _output.WriteLine("PTP hardware not available - skipping benchmark");
            return;
        }

        const int iterations = 10_000;
        var stopwatch = Stopwatch.StartNew();

        // Act - Measure time read performance
        for (int i = 0; i < iterations; i++)
        {
            _ = ptpClock.GetCurrentTimeNanos();
        }

        stopwatch.Stop();

        // Calculate metrics
        double avgLatencyNs = (stopwatch.ElapsedTicks * 1_000_000_000.0 / Stopwatch.Frequency) / iterations;
        double throughput = iterations / stopwatch.Elapsed.TotalSeconds;

        // Assert & Report
        _output.WriteLine($"=== PTP Clock Source Benchmark ===");
        _output.WriteLine($"Iterations: {iterations:N0}");
        _output.WriteLine($"Total Time: {stopwatch.ElapsedMilliseconds}ms");
        _output.WriteLine($"Avg Latency: {avgLatencyNs:F2}ns per read");
        _output.WriteLine($"Throughput: {throughput:N0} reads/second");
        _output.WriteLine($"Error Bound: ±{ptpClock.GetErrorBound()}ns");

        // PTP hardware should be sub-100ns per read
        avgLatencyNs.Should().BeLessThan(1000); // < 1μs per read

        ptpClock.Dispose();
    }

    [Fact]
    public void Benchmark_SystemClockSource_TimeReadLatency()
    {
        // Arrange
        var systemClock = new SystemClockSource(NullLogger<SystemClockSource>.Instance);

        const int iterations = 10_000;
        var stopwatch = Stopwatch.StartNew();

        // Act - Measure time read performance
        for (int i = 0; i < iterations; i++)
        {
            _ = systemClock.GetCurrentTimeNanos();
        }

        stopwatch.Stop();

        // Calculate metrics
        double avgLatencyNs = (stopwatch.ElapsedTicks * 1_000_000_000.0 / Stopwatch.Frequency) / iterations;
        double throughput = iterations / stopwatch.Elapsed.TotalSeconds;

        // Assert & Report
        _output.WriteLine($"=== System Clock Source Benchmark ===");
        _output.WriteLine($"Iterations: {iterations:N0}");
        _output.WriteLine($"Total Time: {stopwatch.ElapsedMilliseconds}ms");
        _output.WriteLine($"Avg Latency: {avgLatencyNs:F2}ns per read");
        _output.WriteLine($"Throughput: {throughput:N0} reads/second");
        _output.WriteLine($"Error Bound: ±{systemClock.GetErrorBound()}ns (±{systemClock.GetErrorBound() / 1_000_000}ms)");

        // System clock should be faster than 1μs per read
        avgLatencyNs.Should().BeLessThan(1000); // < 1μs per read
    }

    [Fact]
    public async Task Benchmark_SoftwarePtpClockSource_TimeReadLatency()
    {
        // Arrange
        var softwarePtp = new SoftwarePtpClockSource(NullLogger<SoftwarePtpClockSource>.Instance);
        bool initialized = await softwarePtp.InitializeAsync();

        if (!initialized)
        {
            _output.WriteLine("Software PTP initialization failed - skipping benchmark");
            return;
        }

        const int iterations = 10_000;
        var stopwatch = Stopwatch.StartNew();

        // Act - Measure time read performance
        for (int i = 0; i < iterations; i++)
        {
            _ = softwarePtp.GetCurrentTimeNanos();
        }

        stopwatch.Stop();

        // Calculate metrics
        double avgLatencyNs = (stopwatch.ElapsedTicks * 1_000_000_000.0 / Stopwatch.Frequency) / iterations;
        double throughput = iterations / stopwatch.Elapsed.TotalSeconds;

        // Assert & Report
        _output.WriteLine($"=== Software PTP Clock Source Benchmark ===");
        _output.WriteLine($"Iterations: {iterations:N0}");
        _output.WriteLine($"Total Time: {stopwatch.ElapsedMilliseconds}ms");
        _output.WriteLine($"Avg Latency: {avgLatencyNs:F2}ns per read");
        _output.WriteLine($"Throughput: {throughput:N0} reads/second");
        _output.WriteLine($"Error Bound: ±{softwarePtp.GetErrorBound()}ns (±{softwarePtp.GetErrorBound() / 1_000}μs)");

        // Software PTP should be fast (just adds offset to system time)
        avgLatencyNs.Should().BeLessThan(1000); // < 1μs per read

        softwarePtp.Dispose();
    }

    [Fact]
    public async Task Benchmark_ClockSourceSelector_InitializationTime()
    {
        // Arrange
        var stopwatch = Stopwatch.StartNew();

        // Act - Measure initialization performance
        var selector = new ClockSourceSelector(NullLogger<ClockSourceSelector>.Instance);
        await selector.InitializeAsync();

        stopwatch.Stop();

        // Assert & Report
        _output.WriteLine($"=== Clock Source Selector Initialization ===");
        _output.WriteLine($"Initialization Time: {stopwatch.ElapsedMilliseconds}ms");
        _output.WriteLine($"Available Sources: {selector.AvailableSources.Count}");
        _output.WriteLine($"Active Source: {selector.ActiveSource.GetType().Name}");
        _output.WriteLine($"Error Bound: ±{selector.ActiveSource.GetErrorBound()}ns");

        // Initialization should complete in < 1 second
        stopwatch.ElapsedMilliseconds.Should().BeLessThan(1000);
    }

    [Fact]
    public async Task Benchmark_ClockSourceSelector_SwitchingOverhead()
    {
        // Arrange
        var selector = new ClockSourceSelector(NullLogger<ClockSourceSelector>.Instance);
        await selector.InitializeAsync();

        var alternativeSource = selector.AvailableSources.FirstOrDefault(
            s => s != selector.ActiveSource);

        if (alternativeSource == null)
        {
            _output.WriteLine("Only one clock source available - skipping switch benchmark");
            return;
        }

        // Act - Measure clock source switching
        var stopwatch = Stopwatch.StartNew();
        selector.SwitchClockSource(alternativeSource);
        stopwatch.Stop();

        // Assert & Report
        _output.WriteLine($"=== Clock Source Switching ===");
        _output.WriteLine($"Switch Time: {stopwatch.Elapsed.TotalMicroseconds:F2}μs");
        _output.WriteLine($"New Active Source: {selector.ActiveSource.GetType().Name}");

        // Switching should be near-instant (< 1ms)
        stopwatch.ElapsedMilliseconds.Should().BeLessThan(1);
    }

    [Fact]
    public async Task Benchmark_ComparativePerformance_AllClockSources()
    {
        // Arrange
        var selector = new ClockSourceSelector(NullLogger<ClockSourceSelector>.Instance);
        await selector.InitializeAsync();

        const int iterations = 1_000;
        var results = new List<(string name, double latencyNs, long errorBound)>();

        // Act - Benchmark each available clock source
        foreach (var source in selector.AvailableSources)
        {
            if (!source.IsSynchronized)
                continue;

            var stopwatch = Stopwatch.StartNew();

            for (int i = 0; i < iterations; i++)
            {
                _ = source.GetCurrentTimeNanos();
            }

            stopwatch.Stop();

            double avgLatencyNs = (stopwatch.ElapsedTicks * 1_000_000_000.0 / Stopwatch.Frequency) / iterations;
            long errorBound = source.GetErrorBound();

            results.Add((source.GetType().Name, avgLatencyNs, errorBound));
        }

        // Assert & Report
        _output.WriteLine($"=== Comparative Clock Source Performance ===");
        _output.WriteLine($"{"Clock Source",-30} {"Avg Latency (ns)",20} {"Error Bound",20}");
        _output.WriteLine(new string('-', 72));

        foreach (var (name, latency, error) in results.OrderBy(r => r.latencyNs))
        {
            string errorStr = error < 1_000 ? $"±{error}ns" :
                             error < 1_000_000 ? $"±{error / 1_000}μs" :
                             $"±{error / 1_000_000}ms";

            _output.WriteLine($"{name,-30} {latency,20:F2} {errorStr,20}");
        }

        results.Should().NotBeEmpty();
    }

    [Fact]
    public async Task Benchmark_MemoryFootprint_ClockSources()
    {
        // Arrange
        var initialMemory = GC.GetTotalMemory(true);

        var ptpClock = new PtpClockSource(NullLogger<PtpClockSource>.Instance);
        await ptpClock.InitializeAsync();

        var softwarePtp = new SoftwarePtpClockSource(NullLogger<SoftwarePtpClockSource>.Instance);
        await softwarePtp.InitializeAsync();

        var selector = new ClockSourceSelector(NullLogger<ClockSourceSelector>.Instance);
        await selector.InitializeAsync();

        // Act - Force GC and measure memory
        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();

        var finalMemory = GC.GetTotalMemory(false);
        long memoryUsed = finalMemory - initialMemory;

        // Assert & Report
        _output.WriteLine($"=== Memory Footprint ===");
        _output.WriteLine($"Initial Memory: {initialMemory / 1024.0:F2} KB");
        _output.WriteLine($"Final Memory: {finalMemory / 1024.0:F2} KB");
        _output.WriteLine($"Memory Used: {memoryUsed / 1024.0:F2} KB");

        // Clock sources should use minimal memory (< 1MB)
        memoryUsed.Should().BeLessThan(1_000_000); // < 1MB

        ptpClock.Dispose();
        softwarePtp.Dispose();
    }
}

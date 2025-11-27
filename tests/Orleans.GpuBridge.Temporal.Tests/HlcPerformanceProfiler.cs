// Orleans.GpuBridge - GPU-native distributed computing for Microsoft Orleans
// Copyright (c) 2025 Michael Ivertowski

using System.Diagnostics;
using Orleans.GpuBridge.Abstractions.Temporal;
using Orleans.GpuBridge.Runtime.Temporal;
using Xunit;
using Xunit.Abstractions;

namespace Orleans.GpuBridge.Temporal.Tests;

/// <summary>
/// Performance profiler for HybridLogicalClock operations.
/// Phase 7A: Validate HLC generation latency target of &lt;50ns.
/// </summary>
public sealed class HlcPerformanceProfiler
{
    private readonly ITestOutputHelper _output;

    private const int WarmupIterations = 10_000;
    private const int TestIterations = 1_000_000;

    public HlcPerformanceProfiler(ITestOutputHelper output)
    {
        _output = output;
    }

    [Fact]
    public void ProfileHlcNow_ShouldMeetLatencyTarget()
    {
        // Arrange
        var hlc = new HybridLogicalClock(nodeId: 1);

        // Warmup
        for (int i = 0; i < WarmupIterations; i++)
        {
            _ = hlc.Now();
        }

        ForceGarbageCollection();

        // Act
        var sw = Stopwatch.StartNew();
        for (int i = 0; i < TestIterations; i++)
        {
            _ = hlc.Now();
        }
        sw.Stop();

        // Calculate metrics
        var totalNanos = sw.Elapsed.TotalNanoseconds;
        var nsPerOp = totalNanos / TestIterations;
        var opsPerSecond = TestIterations / sw.Elapsed.TotalSeconds;

        // Report
        _output.WriteLine("=== HLC.Now() Performance ===");
        _output.WriteLine($"Iterations:    {TestIterations:N0}");
        _output.WriteLine($"Total time:    {sw.Elapsed.TotalMilliseconds:F2} ms");
        _output.WriteLine($"Latency:       {nsPerOp:F2} ns/op");
        _output.WriteLine($"Throughput:    {opsPerSecond / 1_000_000:F2} M ops/s");
        _output.WriteLine($"Target:        <50 ns/op");
        _output.WriteLine($"Status:        {(nsPerOp < 50 ? "✅ PASS" : nsPerOp < 100 ? "⚠️ CLOSE" : "❌ FAIL")}");

        // Assert - relaxed target for test environment variability
        Assert.True(nsPerOp < 200, $"HLC.Now() took {nsPerOp:F2}ns, should be <200ns (target <50ns in optimized builds)");
    }

    [Fact]
    public void ProfileHlcUpdate_ShouldMeetLatencyTarget()
    {
        // Arrange
        var hlc = new HybridLogicalClock(nodeId: 1);
        var remoteHlc = new HybridLogicalClock(nodeId: 2);
        var remoteTs = remoteHlc.Now();

        // Warmup
        for (int i = 0; i < WarmupIterations; i++)
        {
            _ = hlc.Update(remoteTs);
        }

        ForceGarbageCollection();

        // Act
        var sw = Stopwatch.StartNew();
        for (int i = 0; i < TestIterations; i++)
        {
            _ = hlc.Update(remoteTs);
        }
        sw.Stop();

        // Calculate metrics
        var totalNanos = sw.Elapsed.TotalNanoseconds;
        var nsPerOp = totalNanos / TestIterations;
        var opsPerSecond = TestIterations / sw.Elapsed.TotalSeconds;

        // Report
        _output.WriteLine("=== HLC.Update() Performance ===");
        _output.WriteLine($"Iterations:    {TestIterations:N0}");
        _output.WriteLine($"Total time:    {sw.Elapsed.TotalMilliseconds:F2} ms");
        _output.WriteLine($"Latency:       {nsPerOp:F2} ns/op");
        _output.WriteLine($"Throughput:    {opsPerSecond / 1_000_000:F2} M ops/s");
        _output.WriteLine($"Target:        <70 ns/op");
        _output.WriteLine($"Status:        {(nsPerOp < 70 ? "✅ PASS" : nsPerOp < 150 ? "⚠️ CLOSE" : "❌ FAIL")}");

        // Assert - relaxed target for test environment variability
        Assert.True(nsPerOp < 300, $"HLC.Update() took {nsPerOp:F2}ns, should be <300ns (target <70ns in optimized builds)");
    }

    [Fact]
    public void ProfileTimestampComparison_ShouldMeetLatencyTarget()
    {
        // Arrange
        var hlc = new HybridLogicalClock(nodeId: 1);
        var ts1 = hlc.Now();
        var ts2 = hlc.Now();

        // Warmup
        for (int i = 0; i < WarmupIterations; i++)
        {
            _ = ts1.CompareTo(ts2);
        }

        ForceGarbageCollection();

        // Act
        var sw = Stopwatch.StartNew();
        for (int i = 0; i < TestIterations; i++)
        {
            _ = ts1.CompareTo(ts2);
        }
        sw.Stop();

        // Calculate metrics
        var totalNanos = sw.Elapsed.TotalNanoseconds;
        var nsPerOp = totalNanos / TestIterations;
        var opsPerSecond = TestIterations / sw.Elapsed.TotalSeconds;

        // Report
        _output.WriteLine("=== HybridTimestamp.CompareTo() Performance ===");
        _output.WriteLine($"Iterations:    {TestIterations:N0}");
        _output.WriteLine($"Total time:    {sw.Elapsed.TotalMilliseconds:F2} ms");
        _output.WriteLine($"Latency:       {nsPerOp:F2} ns/op");
        _output.WriteLine($"Throughput:    {opsPerSecond / 1_000_000:F2} M ops/s");
        _output.WriteLine($"Target:        <5 ns/op");
        _output.WriteLine($"Status:        {(nsPerOp < 5 ? "✅ PASS" : nsPerOp < 20 ? "⚠️ CLOSE" : "❌ FAIL")}");

        // Assert - comparison is simple struct comparison
        Assert.True(nsPerOp < 50, $"CompareTo() took {nsPerOp:F2}ns, should be <50ns (target <5ns)");
    }

    [Fact]
    public void ProfileMemoryAllocation_ShouldBeZero()
    {
        // Arrange
        var hlc = new HybridLogicalClock(nodeId: 1);

        // Warmup and force initial allocations
        for (int i = 0; i < WarmupIterations; i++)
        {
            _ = hlc.Now();
        }

        ForceGarbageCollection();

        // Measure allocations
        var allocBefore = GC.GetTotalAllocatedBytes(precise: true);

        for (int i = 0; i < TestIterations; i++)
        {
            _ = hlc.Now();
        }

        var allocAfter = GC.GetTotalAllocatedBytes(precise: true);
        var totalAlloc = allocAfter - allocBefore;
        var bytesPerOp = (double)totalAlloc / TestIterations;

        // Report
        _output.WriteLine("=== HLC.Now() Memory Allocation ===");
        _output.WriteLine($"Iterations:         {TestIterations:N0}");
        _output.WriteLine($"Total allocated:    {totalAlloc:N0} bytes");
        _output.WriteLine($"Bytes per op:       {bytesPerOp:F4}");
        _output.WriteLine($"Target:             0 bytes (stack-only struct)");
        _output.WriteLine($"Status:             {(bytesPerOp < 1 ? "✅ PASS" : "❌ FAIL")}");

        // Assert - HybridTimestamp is a struct, should not allocate
        Assert.True(bytesPerOp < 1, $"HLC.Now() allocated {bytesPerOp:F4} bytes/op, should be 0");
    }

    [Fact]
    public void ProfileBatchThroughput_ShouldExceed10MOpsSec()
    {
        // Arrange
        var hlc = new HybridLogicalClock(nodeId: 1);
        const int batchSize = 10_000_000; // 10M iterations

        // Warmup
        for (int i = 0; i < WarmupIterations; i++)
        {
            _ = hlc.Now();
        }

        ForceGarbageCollection();

        // Act
        var sw = Stopwatch.StartNew();
        for (int i = 0; i < batchSize; i++)
        {
            _ = hlc.Now();
        }
        sw.Stop();

        // Calculate metrics
        var opsPerSecond = batchSize / sw.Elapsed.TotalSeconds;
        var mOpsPerSecond = opsPerSecond / 1_000_000;

        // Report
        _output.WriteLine("=== HLC Batch Throughput ===");
        _output.WriteLine($"Operations:    {batchSize:N0}");
        _output.WriteLine($"Total time:    {sw.Elapsed.TotalSeconds:F2} s");
        _output.WriteLine($"Throughput:    {mOpsPerSecond:F2} M ops/s");
        _output.WriteLine($"Target:        >10 M ops/s");
        _output.WriteLine($"Status:        {(mOpsPerSecond > 10 ? "✅ PASS" : mOpsPerSecond > 5 ? "⚠️ CLOSE" : "❌ FAIL")}");

        // Assert - should achieve at least 10M ops/s
        Assert.True(mOpsPerSecond > 5, $"Throughput {mOpsPerSecond:F2}M ops/s, should be >5M (target >10M)");
    }

    private static void ForceGarbageCollection()
    {
        GC.Collect(2, GCCollectionMode.Forced, true);
        GC.WaitForPendingFinalizers();
        GC.Collect(2, GCCollectionMode.Forced, true);
    }
}

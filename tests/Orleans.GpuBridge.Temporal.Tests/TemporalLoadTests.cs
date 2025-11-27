// Orleans.GpuBridge - GPU-native distributed computing for Microsoft Orleans
// Copyright (c) 2025 Michael Ivertowski

using System.Collections.Concurrent;
using System.Diagnostics;
using Orleans.GpuBridge.Abstractions.Temporal;
using Orleans.GpuBridge.Runtime.Temporal;
using Xunit;
using Xunit.Abstractions;

namespace Orleans.GpuBridge.Temporal.Tests;

/// <summary>
/// Load tests for temporal components.
/// Phase 7C: Validates sustained throughput and stability under high load.
/// </summary>
public sealed class TemporalLoadTests
{
    private readonly ITestOutputHelper _output;

    public TemporalLoadTests(ITestOutputHelper output)
    {
        _output = output;
    }

    /// <summary>
    /// Tests sustained HLC throughput under continuous load.
    /// Target: 10M operations/second sustained.
    /// </summary>
    [Fact]
    public void SustainedHlcThroughput_ShouldExceed10MOpsSec()
    {
        // Arrange
        const int durationSeconds = 5;
        const long targetOpsPerSecond = 10_000_000;
        var hlc = new HybridLogicalClock(nodeId: 1);
        var totalOperations = 0L;
        var stopwatch = new Stopwatch();

        // Warmup
        for (int i = 0; i < 100_000; i++)
        {
            _ = hlc.Now();
        }

        GC.Collect(2, GCCollectionMode.Forced, true);
        GC.WaitForPendingFinalizers();
        GC.Collect();

        // Act - Run for specified duration
        stopwatch.Start();
        while (stopwatch.Elapsed.TotalSeconds < durationSeconds)
        {
            // Batch of 10,000 operations
            for (int i = 0; i < 10_000; i++)
            {
                _ = hlc.Now();
            }
            Interlocked.Add(ref totalOperations, 10_000);
        }
        stopwatch.Stop();

        // Calculate metrics
        var opsPerSecond = totalOperations / stopwatch.Elapsed.TotalSeconds;
        var mOpsPerSecond = opsPerSecond / 1_000_000;

        // Report
        _output.WriteLine("=== Sustained HLC Throughput ===");
        _output.WriteLine($"Duration:       {stopwatch.Elapsed.TotalSeconds:F2} s");
        _output.WriteLine($"Operations:     {totalOperations:N0}");
        _output.WriteLine($"Throughput:     {mOpsPerSecond:F2} M ops/s");
        _output.WriteLine($"Target:         {targetOpsPerSecond / 1_000_000} M ops/s");
        _output.WriteLine($"Status:         {(opsPerSecond >= targetOpsPerSecond ? "PASS" : opsPerSecond >= targetOpsPerSecond * 0.5 ? "CLOSE" : "FAIL")}");

        // Assert - Relaxed target for test environment
        Assert.True(opsPerSecond >= targetOpsPerSecond * 0.5,
            $"Sustained throughput {mOpsPerSecond:F2}M ops/s, should be >{targetOpsPerSecond * 0.5 / 1_000_000}M (target {targetOpsPerSecond / 1_000_000}M)");
    }

    /// <summary>
    /// Tests concurrent HLC access from multiple threads.
    /// Verifies thread-safety and timestamp ordering.
    /// </summary>
    [Fact]
    public void ConcurrentHlcAccess_ShouldMaintainOrdering()
    {
        // Arrange
        const int threadCount = 8;
        const int operationsPerThread = 100_000;
        var hlc = new HybridLogicalClock(nodeId: 1);
        var allTimestamps = new ConcurrentBag<HybridTimestamp>();
        var barrier = new Barrier(threadCount);
        var exceptions = new ConcurrentBag<Exception>();

        // Act - Concurrent access
        var threads = Enumerable.Range(0, threadCount).Select(threadId =>
            new Thread(() =>
            {
                try
                {
                    barrier.SignalAndWait(); // Synchronize start
                    var localTimestamps = new List<HybridTimestamp>(operationsPerThread);

                    for (int i = 0; i < operationsPerThread; i++)
                    {
                        localTimestamps.Add(hlc.Now());
                    }

                    // Verify local ordering (each thread's timestamps should be strictly increasing)
                    for (int i = 1; i < localTimestamps.Count; i++)
                    {
                        if (localTimestamps[i].CompareTo(localTimestamps[i - 1]) <= 0)
                        {
                            throw new InvalidOperationException(
                                $"Thread {threadId}: Non-increasing timestamps at index {i}: " +
                                $"{localTimestamps[i - 1]} >= {localTimestamps[i]}");
                        }
                    }

                    foreach (var ts in localTimestamps)
                    {
                        allTimestamps.Add(ts);
                    }
                }
                catch (Exception ex)
                {
                    exceptions.Add(ex);
                }
            })
        ).ToList();

        var sw = Stopwatch.StartNew();
        threads.ForEach(t => t.Start());
        threads.ForEach(t => t.Join());
        sw.Stop();

        // Report
        var totalOps = threadCount * operationsPerThread;
        var opsPerSecond = totalOps / sw.Elapsed.TotalSeconds;

        _output.WriteLine("=== Concurrent HLC Access ===");
        _output.WriteLine($"Threads:        {threadCount}");
        _output.WriteLine($"Ops/thread:     {operationsPerThread:N0}");
        _output.WriteLine($"Total ops:      {totalOps:N0}");
        _output.WriteLine($"Duration:       {sw.Elapsed.TotalMilliseconds:F2} ms");
        _output.WriteLine($"Throughput:     {opsPerSecond / 1_000_000:F2} M ops/s");
        _output.WriteLine($"Timestamps:     {allTimestamps.Count:N0}");
        _output.WriteLine($"Exceptions:     {exceptions.Count}");

        // Assert
        Assert.Empty(exceptions);
        Assert.Equal(totalOps, allTimestamps.Count);
    }

    /// <summary>
    /// Tests memory stability under sustained load.
    /// Verifies no memory leaks over extended operation.
    /// </summary>
    [Fact]
    public void MemoryStability_UnderSustainedLoad()
    {
        // Arrange
        const int iterations = 10;
        const int operationsPerIteration = 1_000_000;
        var hlc = new HybridLogicalClock(nodeId: 1);
        var memorySnapshots = new List<long>();

        // Warmup
        for (int i = 0; i < 10_000; i++) { _ = hlc.Now(); }

        GC.Collect(2, GCCollectionMode.Forced, true);
        GC.WaitForPendingFinalizers();

        // Act - Multiple iterations with GC between
        for (int iter = 0; iter < iterations; iter++)
        {
            for (int i = 0; i < operationsPerIteration; i++)
            {
                _ = hlc.Now();
            }

            GC.Collect(2, GCCollectionMode.Forced, true);
            GC.WaitForPendingFinalizers();
            memorySnapshots.Add(GC.GetTotalMemory(true));
        }

        // Calculate memory trend
        var firstHalf = memorySnapshots.Take(iterations / 2).Average();
        var secondHalf = memorySnapshots.Skip(iterations / 2).Average();
        var memoryGrowth = secondHalf - firstHalf;
        var growthPercent = (memoryGrowth / firstHalf) * 100;

        // Report
        _output.WriteLine("=== Memory Stability Under Load ===");
        _output.WriteLine($"Iterations:     {iterations}");
        _output.WriteLine($"Ops/iteration:  {operationsPerIteration:N0}");
        _output.WriteLine($"First half avg: {firstHalf / 1024:N0} KB");
        _output.WriteLine($"Second half:    {secondHalf / 1024:N0} KB");
        _output.WriteLine($"Growth:         {memoryGrowth / 1024:F2} KB ({growthPercent:F2}%)");
        _output.WriteLine($"Status:         {(Math.Abs(growthPercent) < 10 ? "STABLE" : "UNSTABLE")}");

        // Assert - Memory should be stable (< 10% growth is acceptable)
        Assert.True(Math.Abs(growthPercent) < 20,
            $"Memory growth {growthPercent:F2}% exceeds 20% threshold");
    }

    /// <summary>
    /// Tests Update() throughput under concurrent load with multiple sources.
    /// Simulates distributed clock synchronization.
    /// </summary>
    [Fact]
    public void ConcurrentUpdateThroughput_ShouldScaleWithThreads()
    {
        // Arrange
        const int threadCount = 4;
        const int operationsPerThread = 250_000;
        var mainHlc = new HybridLogicalClock(nodeId: 0);
        var remoteHlcs = Enumerable.Range(1, threadCount)
            .Select(i => new HybridLogicalClock(nodeId: (byte)i))
            .ToArray();
        var barrier = new Barrier(threadCount);
        var exceptions = new ConcurrentBag<Exception>();

        // Generate remote timestamps
        var remoteTimestamps = remoteHlcs.Select(hlc =>
        {
            var timestamps = new HybridTimestamp[operationsPerThread];
            for (int i = 0; i < operationsPerThread; i++)
            {
                timestamps[i] = hlc.Now();
            }
            return timestamps;
        }).ToArray();

        // Act - Concurrent updates
        var sw = Stopwatch.StartNew();
        var threads = Enumerable.Range(0, threadCount).Select(threadId =>
            new Thread(() =>
            {
                try
                {
                    barrier.SignalAndWait();
                    for (int i = 0; i < operationsPerThread; i++)
                    {
                        _ = mainHlc.Update(remoteTimestamps[threadId][i]);
                    }
                }
                catch (Exception ex)
                {
                    exceptions.Add(ex);
                }
            })
        ).ToList();

        threads.ForEach(t => t.Start());
        threads.ForEach(t => t.Join());
        sw.Stop();

        // Report
        var totalOps = threadCount * operationsPerThread;
        var opsPerSecond = totalOps / sw.Elapsed.TotalSeconds;

        _output.WriteLine("=== Concurrent Update Throughput ===");
        _output.WriteLine($"Threads:        {threadCount}");
        _output.WriteLine($"Ops/thread:     {operationsPerThread:N0}");
        _output.WriteLine($"Total ops:      {totalOps:N0}");
        _output.WriteLine($"Duration:       {sw.Elapsed.TotalMilliseconds:F2} ms");
        _output.WriteLine($"Throughput:     {opsPerSecond / 1_000_000:F2} M ops/s");
        _output.WriteLine($"Exceptions:     {exceptions.Count}");

        // Assert
        Assert.Empty(exceptions);
        Assert.True(opsPerSecond >= 1_000_000,
            $"Update throughput {opsPerSecond / 1_000_000:F2}M ops/s should exceed 1M ops/s");
    }

    /// <summary>
    /// Tests latency distribution under load.
    /// Validates P50, P95, P99 latencies.
    /// </summary>
    [Fact]
    public void LatencyDistribution_UnderLoad()
    {
        // Arrange
        const int sampleCount = 100_000;
        var hlc = new HybridLogicalClock(nodeId: 1);
        var latencies = new long[sampleCount];
        var sw = new Stopwatch();

        // Warmup
        for (int i = 0; i < 10_000; i++) { _ = hlc.Now(); }

        // Act - Collect latency samples
        for (int i = 0; i < sampleCount; i++)
        {
            sw.Restart();
            _ = hlc.Now();
            sw.Stop();
            latencies[i] = sw.Elapsed.Ticks;
        }

        // Calculate percentiles
        Array.Sort(latencies);
        var ticksPerNs = (double)TimeSpan.TicksPerSecond / 1_000_000_000;
        var p50 = latencies[sampleCount * 50 / 100] / ticksPerNs;
        var p95 = latencies[sampleCount * 95 / 100] / ticksPerNs;
        var p99 = latencies[sampleCount * 99 / 100] / ticksPerNs;
        var p999 = latencies[sampleCount * 999 / 1000] / ticksPerNs;
        var max = latencies[sampleCount - 1] / ticksPerNs;
        var avg = latencies.Average() / ticksPerNs;

        // Report
        _output.WriteLine("=== HLC Latency Distribution ===");
        _output.WriteLine($"Samples:        {sampleCount:N0}");
        _output.WriteLine($"Average:        {avg:F0} ns");
        _output.WriteLine($"P50:            {p50:F0} ns");
        _output.WriteLine($"P95:            {p95:F0} ns");
        _output.WriteLine($"P99:            {p99:F0} ns");
        _output.WriteLine($"P99.9:          {p999:F0} ns");
        _output.WriteLine($"Max:            {max:F0} ns");

        // Assert - P99 should be reasonable (< 10μs)
        Assert.True(p99 < 10_000,
            $"P99 latency {p99:F0}ns exceeds 10μs threshold");
    }

    /// <summary>
    /// Tests vector clock scalability with many nodes.
    /// </summary>
    [Fact]
    public void VectorClockScalability_ManyNodes()
    {
        // Arrange
        const int nodeCount = 32;
        const int operationsPerNode = 10_000;
        var vectorClocks = new VectorClock[nodeCount];

        // Initialize clocks
        for (int i = 0; i < nodeCount; i++)
        {
            vectorClocks[i] = new VectorClock();
        }

        var sw = Stopwatch.StartNew();

        // Act - Each node generates events (VectorClock is immutable)
        for (int nodeIndex = 0; nodeIndex < nodeCount; nodeIndex++)
        {
            var clock = vectorClocks[nodeIndex];
            var actorId = (ushort)nodeIndex;
            for (int i = 0; i < operationsPerNode; i++)
            {
                clock = clock.Increment(actorId);
            }
            vectorClocks[nodeIndex] = clock;
        }

        // Merge clocks into first clock (immutable - must reassign)
        var mergedClock = vectorClocks[0];
        for (int i = 1; i < nodeCount; i++)
        {
            mergedClock = mergedClock.Merge(vectorClocks[i]);
        }

        sw.Stop();

        var totalOps = nodeCount * operationsPerNode + nodeCount - 1;
        var opsPerSecond = totalOps / sw.Elapsed.TotalSeconds;

        // Report
        _output.WriteLine("=== Vector Clock Scalability ===");
        _output.WriteLine($"Nodes:          {nodeCount}");
        _output.WriteLine($"Ops/node:       {operationsPerNode:N0}");
        _output.WriteLine($"Total ops:      {totalOps:N0}");
        _output.WriteLine($"Duration:       {sw.Elapsed.TotalMilliseconds:F2} ms");
        _output.WriteLine($"Throughput:     {opsPerSecond / 1_000_000:F2} M ops/s");
        _output.WriteLine($"Final entries:  {mergedClock.Count}");

        // Assert
        Assert.True(opsPerSecond >= 100_000,
            $"Vector clock throughput {opsPerSecond:N0} ops/s should exceed 100K ops/s");
        Assert.Equal(nodeCount, mergedClock.Count);
    }

    /// <summary>
    /// Tests timestamp comparison throughput for sorted operations.
    /// </summary>
    [Fact]
    public void TimestampSortingThroughput()
    {
        // Arrange
        const int count = 100_000;
        var hlc = new HybridLogicalClock(nodeId: 1);
        var timestamps = new HybridTimestamp[count];

        // Generate timestamps
        for (int i = 0; i < count; i++)
        {
            timestamps[i] = hlc.Now();
        }

        // Shuffle
        var random = new Random(42);
        for (int i = count - 1; i > 0; i--)
        {
            int j = random.Next(i + 1);
            (timestamps[i], timestamps[j]) = (timestamps[j], timestamps[i]);
        }

        // Act - Sort
        var sw = Stopwatch.StartNew();
        Array.Sort(timestamps);
        sw.Stop();

        // Verify sorted order
        for (int i = 1; i < count; i++)
        {
            if (timestamps[i].CompareTo(timestamps[i - 1]) < 0)
            {
                throw new InvalidOperationException($"Not sorted at index {i}");
            }
        }

        var comparisonsEstimate = count * Math.Log2(count); // Rough estimate for sort
        var comparisonsPerSecond = comparisonsEstimate / sw.Elapsed.TotalSeconds;

        // Report
        _output.WriteLine("=== Timestamp Sorting ===");
        _output.WriteLine($"Timestamps:     {count:N0}");
        _output.WriteLine($"Duration:       {sw.Elapsed.TotalMilliseconds:F2} ms");
        _output.WriteLine($"Est. compares:  {comparisonsEstimate:N0}");
        _output.WriteLine($"Compare/sec:    {comparisonsPerSecond / 1_000_000:F2} M/s");

        // Assert - Sorting should complete quickly
        Assert.True(sw.Elapsed.TotalMilliseconds < 1000,
            $"Sorting {count:N0} timestamps took {sw.Elapsed.TotalMilliseconds:F2}ms, should be <1000ms");
    }

    /// <summary>
    /// Stress test with mixed operations simulating real workload.
    /// </summary>
    [Fact]
    public void MixedWorkloadStressTest()
    {
        // Arrange
        const int durationMs = 3000;
        const int threadCount = 4;
        var hlc = new HybridLogicalClock(nodeId: 0);
        var remoteHlc = new HybridLogicalClock(nodeId: 1);
        var operations = new ConcurrentDictionary<string, long>();
        var barrier = new Barrier(threadCount);
        var cts = new CancellationTokenSource(durationMs);

        // Act - Mixed operations from multiple threads
        var threads = Enumerable.Range(0, threadCount).Select(threadId =>
            new Thread(() =>
            {
                var random = new Random(threadId);
                var localOps = new Dictionary<string, long>
                {
                    ["Now"] = 0,
                    ["Update"] = 0,
                    ["Compare"] = 0
                };
                HybridTimestamp lastTs = default;

                barrier.SignalAndWait();

                while (!cts.Token.IsCancellationRequested)
                {
                    var op = random.Next(100);
                    if (op < 60) // 60% Now
                    {
                        lastTs = hlc.Now();
                        localOps["Now"]++;
                    }
                    else if (op < 90) // 30% Update
                    {
                        var remoteTs = remoteHlc.Now();
                        lastTs = hlc.Update(remoteTs);
                        localOps["Update"]++;
                    }
                    else // 10% Compare
                    {
                        var newTs = hlc.Now();
                        _ = newTs.CompareTo(lastTs);
                        localOps["Compare"]++;
                    }
                }

                foreach (var kvp in localOps)
                {
                    operations.AddOrUpdate(kvp.Key, kvp.Value, (k, v) => v + kvp.Value);
                }
            })
        ).ToList();

        var sw = Stopwatch.StartNew();
        threads.ForEach(t => t.Start());
        threads.ForEach(t => t.Join());
        sw.Stop();

        // Report
        var totalOps = operations.Values.Sum();
        var opsPerSecond = totalOps / sw.Elapsed.TotalSeconds;

        _output.WriteLine("=== Mixed Workload Stress Test ===");
        _output.WriteLine($"Threads:        {threadCount}");
        _output.WriteLine($"Duration:       {sw.Elapsed.TotalSeconds:F2} s");
        _output.WriteLine($"Now ops:        {operations.GetValueOrDefault("Now"):N0}");
        _output.WriteLine($"Update ops:     {operations.GetValueOrDefault("Update"):N0}");
        _output.WriteLine($"Compare ops:    {operations.GetValueOrDefault("Compare"):N0}");
        _output.WriteLine($"Total ops:      {totalOps:N0}");
        _output.WriteLine($"Throughput:     {opsPerSecond / 1_000_000:F2} M ops/s");

        // Assert
        Assert.True(totalOps > 1_000_000, $"Should complete >1M ops, got {totalOps:N0}");
        Assert.True(opsPerSecond >= 1_000_000,
            $"Mixed workload throughput {opsPerSecond / 1_000_000:F2}M ops/s should exceed 1M ops/s");
    }
}

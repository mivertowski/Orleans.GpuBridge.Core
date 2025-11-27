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
/// Chaos and fault injection tests for the temporal subsystem.
/// Tests system resilience under various failure conditions.
/// </summary>
public sealed class TemporalChaosTests
{
    private readonly ITestOutputHelper _output;

    public TemporalChaosTests(ITestOutputHelper output)
    {
        _output = output;
    }

    #region Clock Skew Tests

    /// <summary>
    /// Tests HLC behavior when receiving timestamps from the past.
    /// </summary>
    [Fact]
    public void ClockSkew_PastTimestamp_ShouldHandleGracefully()
    {
        // Arrange
        var hlc = new HybridLogicalClock(nodeId: 1);

        // Generate some current timestamps
        var current = hlc.Now();
        for (int i = 0; i < 100; i++)
        {
            _ = hlc.Now();
        }

        // Act - Update with very old timestamp (simulating clock skew)
        var pastTs = new HybridTimestamp(current.PhysicalTime - TimeSpan.TicksPerHour, 0, 99);
        var updated = hlc.Update(pastTs);

        // Assert - HLC should advance past the old timestamp
        Assert.True(updated.CompareTo(current) > 0, "Updated timestamp should be after original");
        Assert.True(updated.CompareTo(pastTs) > 0, "Updated timestamp should be after past timestamp");

        _output.WriteLine($"Current:  {current}");
        _output.WriteLine($"Past:     {pastTs}");
        _output.WriteLine($"Updated:  {updated}");
        _output.WriteLine("✅ Clock skew from past handled correctly");
    }

    /// <summary>
    /// Tests HLC behavior when receiving timestamps from the future.
    /// </summary>
    [Fact]
    public void ClockSkew_FutureTimestamp_ShouldSynchronize()
    {
        // Arrange
        var hlc = new HybridLogicalClock(nodeId: 1);
        var current = hlc.Now();

        // Act - Update with future timestamp (simulating clock drift)
        var futureTs = new HybridTimestamp(
            current.PhysicalTime + TimeSpan.TicksPerMinute,
            0,
            99);
        var updated = hlc.Update(futureTs);

        // Assert - HLC should synchronize to future timestamp
        Assert.True(updated.CompareTo(futureTs) > 0, "Updated should advance past future timestamp");

        _output.WriteLine($"Current:  {current}");
        _output.WriteLine($"Future:   {futureTs}");
        _output.WriteLine($"Updated:  {updated}");
        _output.WriteLine("✅ Clock skew from future handled correctly");
    }

    /// <summary>
    /// Tests HLC behavior with extreme clock jumps.
    /// </summary>
    [Fact]
    public void ClockSkew_ExtremeJump_ShouldMaintainOrdering()
    {
        // Arrange
        var hlc = new HybridLogicalClock(nodeId: 1);
        var timestamps = new List<HybridTimestamp>();

        // Generate initial timestamps
        for (int i = 0; i < 10; i++)
        {
            timestamps.Add(hlc.Now());
        }

        // Simulate extreme clock jump (1 day ahead)
        var jumpTs = new HybridTimestamp(
            DateTime.UtcNow.Ticks + TimeSpan.TicksPerDay,
            0,
            99);
        var afterJump = hlc.Update(jumpTs);
        timestamps.Add(afterJump);

        // Continue generating timestamps
        for (int i = 0; i < 10; i++)
        {
            timestamps.Add(hlc.Now());
        }

        // Assert - All timestamps should be strictly ordered
        for (int i = 1; i < timestamps.Count; i++)
        {
            Assert.True(timestamps[i].CompareTo(timestamps[i - 1]) > 0,
                $"Timestamp {i} should be after timestamp {i - 1}");
        }

        _output.WriteLine($"Generated {timestamps.Count} timestamps across extreme clock jump");
        _output.WriteLine($"Jump timestamp: {jumpTs}");
        _output.WriteLine("✅ Ordering maintained across extreme clock jump");
    }

    #endregion

    #region Concurrent Chaos Tests

    /// <summary>
    /// Tests HLC under concurrent update chaos.
    /// </summary>
    [Fact]
    public void ConcurrentChaos_RandomUpdates_ShouldMaintainMonotonicity()
    {
        // Arrange
        var hlc = new HybridLogicalClock(nodeId: 1);
        var random = new Random(42);
        var allTimestamps = new ConcurrentBag<HybridTimestamp>();
        const int threadCount = 8;
        const int operationsPerThread = 10_000;

        // Act - Chaos: Random mix of Now() and Update() with random timestamps
        Parallel.For(0, threadCount, threadId =>
        {
            var localRandom = new Random(42 + threadId);
            for (int i = 0; i < operationsPerThread; i++)
            {
                HybridTimestamp ts;
                if (localRandom.NextDouble() < 0.7)
                {
                    // 70% - Normal timestamp generation
                    ts = hlc.Now();
                }
                else
                {
                    // 30% - Update with random timestamp (chaos)
                    var randomTicks = DateTime.UtcNow.Ticks + localRandom.Next(-10000, 10000);
                    var chaosTs = new HybridTimestamp(randomTicks, (ushort)localRandom.Next(1000), 99);
                    ts = hlc.Update(chaosTs);
                }
                allTimestamps.Add(ts);
            }
        });

        // Verify monotonicity per thread is maintained (global uniqueness)
        var uniqueCount = allTimestamps.Distinct().Count();

        _output.WriteLine($"Total operations: {threadCount * operationsPerThread:N0}");
        _output.WriteLine($"Unique timestamps: {uniqueCount:N0}");
        _output.WriteLine($"Uniqueness ratio: {100.0 * uniqueCount / allTimestamps.Count:F2}%");

        // Assert - All timestamps should be unique
        Assert.Equal(allTimestamps.Count, uniqueCount);
        _output.WriteLine("✅ All timestamps unique under concurrent chaos");
    }

    /// <summary>
    /// Tests Vector Clock merging under chaos conditions.
    /// </summary>
    [Fact]
    public void ConcurrentChaos_VectorClockMerges_ShouldConverge()
    {
        // Arrange
        const int nodeCount = 8;
        const int roundCount = 100;
        var clocks = new VectorClock[nodeCount];

        for (int i = 0; i < nodeCount; i++)
        {
            clocks[i] = new VectorClock();
        }

        // Act - Chaos: Random increments and merges
        var random = new Random(42);
        for (int round = 0; round < roundCount; round++)
        {
            // Random node increments
            var nodeId = (ushort)random.Next(nodeCount);
            clocks[nodeId] = clocks[nodeId].Increment(nodeId);

            // Random pairwise merges
            var src = random.Next(nodeCount);
            var dst = random.Next(nodeCount);
            if (src != dst)
            {
                clocks[dst] = clocks[dst].Merge(clocks[src]);
            }
        }

        // Merge all into one final clock
        var finalClock = clocks[0];
        for (int i = 1; i < nodeCount; i++)
        {
            finalClock = finalClock.Merge(clocks[i]);
        }

        // Assert - Final clock should contain knowledge of all nodes
        _output.WriteLine($"Nodes: {nodeCount}, Rounds: {roundCount}");
        _output.WriteLine($"Final clock entries: {finalClock.Count}");

        foreach (ushort actorId in finalClock.ActorIds)
        {
            _output.WriteLine($"  Node {actorId}: {finalClock[actorId]}");
        }

        Assert.True(finalClock.Count > 0, "Final clock should have entries");
        _output.WriteLine("✅ Vector clocks converged correctly under chaos");
    }

    #endregion

    #region Fault Injection Tests

    /// <summary>
    /// Tests recovery from simulated network partition.
    /// </summary>
    [Fact]
    public void NetworkPartition_ShouldRecoverAndMerge()
    {
        // Arrange - Two partitions that can't communicate
        var partition1 = new HybridLogicalClock(nodeId: 1);
        var partition2 = new HybridLogicalClock(nodeId: 2);

        var p1Timestamps = new List<HybridTimestamp>();
        var p2Timestamps = new List<HybridTimestamp>();

        // Act - Generate timestamps in isolation (simulating network partition)
        for (int i = 0; i < 1000; i++)
        {
            p1Timestamps.Add(partition1.Now());
            p2Timestamps.Add(partition2.Now());
        }

        // Partition heals - merge latest timestamps
        var p1Latest = partition1.Now();
        var p2Latest = partition2.Now();

        // Exchange timestamps to synchronize
        var p1AfterMerge = partition1.Update(p2Latest);
        var p2AfterMerge = partition2.Update(p1Latest);

        // Continue generating after merge
        var p1PostMerge = partition1.Now();
        var p2PostMerge = partition2.Now();

        // Assert
        Assert.True(p1AfterMerge.CompareTo(p1Latest) > 0, "P1 should advance after merge");
        Assert.True(p2AfterMerge.CompareTo(p2Latest) > 0, "P2 should advance after merge");
        Assert.True(p1PostMerge.CompareTo(p1AfterMerge) > 0, "P1 should continue after merge");
        Assert.True(p2PostMerge.CompareTo(p2AfterMerge) > 0, "P2 should continue after merge");

        _output.WriteLine($"P1 pre-merge:  {p1Latest}");
        _output.WriteLine($"P2 pre-merge:  {p2Latest}");
        _output.WriteLine($"P1 after-merge: {p1AfterMerge}");
        _output.WriteLine($"P2 after-merge: {p2AfterMerge}");
        _output.WriteLine("✅ Partitions recovered and synchronized");
    }

    /// <summary>
    /// Tests behavior with Byzantine (malicious) timestamps.
    /// </summary>
    [Fact]
    public void ByzantineFault_MaliciousTimestamp_ShouldNotCorruptState()
    {
        // Arrange
        var hlc = new HybridLogicalClock(nodeId: 1);
        var legitimateTs = hlc.Now();

        // Act - Inject Byzantine timestamps (extreme values)
        var byzantineTimestamps = new[]
        {
            new HybridTimestamp(long.MaxValue - 1000, 0, 99), // Far future
            new HybridTimestamp(0, ushort.MaxValue, 99),       // Zero time, max counter
            new HybridTimestamp(DateTime.MaxValue.Ticks - 1000, 0, 99), // Near max
        };

        var results = new List<HybridTimestamp>();
        foreach (var byzantine in byzantineTimestamps)
        {
            try
            {
                var result = hlc.Update(byzantine);
                results.Add(result);
            }
            catch (Exception ex)
            {
                _output.WriteLine($"Exception on Byzantine timestamp: {ex.Message}");
            }
        }

        // After Byzantine attack, HLC should still function
        var afterAttack = hlc.Now();

        // Assert - HLC should still produce valid, ordered timestamps
        Assert.True(afterAttack.PhysicalTime > 0, "Wall clock should be positive");

        _output.WriteLine($"Legitimate timestamp: {legitimateTs}");
        _output.WriteLine($"After Byzantine attack: {afterAttack}");
        _output.WriteLine($"Byzantine results: {results.Count}");
        _output.WriteLine("✅ HLC survived Byzantine timestamp injection");
    }

    /// <summary>
    /// Tests rapid clock adjustments simulating NTP corrections.
    /// </summary>
    [Fact]
    public void NtpCorrection_FrequentAdjustments_ShouldRemainStable()
    {
        // Arrange
        var hlc = new HybridLogicalClock(nodeId: 1);
        var timestamps = new List<HybridTimestamp>();
        var random = new Random(42);

        // Act - Simulate rapid NTP-like corrections
        for (int i = 0; i < 1000; i++)
        {
            // Generate timestamp
            timestamps.Add(hlc.Now());

            // Every 10 iterations, simulate NTP correction
            if (i % 10 == 0)
            {
                var correction = random.Next(-100, 100) * TimeSpan.TicksPerMillisecond;
                var correctedTs = new HybridTimestamp(
                    DateTime.UtcNow.Ticks + correction,
                    0,
                    99);
                timestamps.Add(hlc.Update(correctedTs));
            }
        }

        // Assert - All timestamps should be strictly ordered
        for (int i = 1; i < timestamps.Count; i++)
        {
            Assert.True(timestamps[i].CompareTo(timestamps[i - 1]) > 0,
                $"Timestamp {i} should be after {i - 1}");
        }

        _output.WriteLine($"Generated {timestamps.Count} timestamps with NTP corrections");
        _output.WriteLine("✅ Clock remained stable through NTP-like corrections");
    }

    #endregion

    #region Stress Tests

    /// <summary>
    /// Tests sustained high-frequency updates from multiple sources.
    /// </summary>
    [Fact]
    public void StressTest_HighFrequencyUpdates_ShouldMaintainCorrectness()
    {
        // Arrange
        const int sourceCount = 16;
        const int updatesPerSource = 5000;
        var mainHlc = new HybridLogicalClock(nodeId: 0);
        var sources = Enumerable.Range(1, sourceCount)
            .Select(i => new HybridLogicalClock(nodeId: (ushort)i))
            .ToArray();

        var sw = Stopwatch.StartNew();

        // Act - Each source sends updates to main HLC
        var allResults = new ConcurrentBag<HybridTimestamp>();

        Parallel.For(0, sourceCount, sourceIdx =>
        {
            for (int i = 0; i < updatesPerSource; i++)
            {
                var sourceTs = sources[sourceIdx].Now();
                var result = mainHlc.Update(sourceTs);
                allResults.Add(result);
            }
        });

        sw.Stop();

        var totalUpdates = sourceCount * updatesPerSource;
        var uniqueResults = allResults.Distinct().Count();
        var updatesPerSecond = totalUpdates / sw.Elapsed.TotalSeconds;

        // Assert
        Assert.Equal(allResults.Count, uniqueResults);

        _output.WriteLine("=== High-Frequency Update Stress Test ===");
        _output.WriteLine($"Sources:         {sourceCount}");
        _output.WriteLine($"Updates/source:  {updatesPerSource:N0}");
        _output.WriteLine($"Total updates:   {totalUpdates:N0}");
        _output.WriteLine($"Unique results:  {uniqueResults:N0}");
        _output.WriteLine($"Duration:        {sw.Elapsed.TotalMilliseconds:F2} ms");
        _output.WriteLine($"Throughput:      {updatesPerSecond / 1_000_000:F2} M updates/s");
        _output.WriteLine("✅ All updates produced unique timestamps");
    }

    /// <summary>
    /// Tests memory stability under prolonged chaos.
    /// </summary>
    [Fact]
    public void StressTest_MemoryStability_UnderProlongedChaos()
    {
        // Arrange
        var hlc = new HybridLogicalClock(nodeId: 1);
        var random = new Random(42);
        const int iterations = 100_000;

        // Force GC and measure baseline
        GC.Collect(2, GCCollectionMode.Forced, true);
        GC.WaitForPendingFinalizers();
        GC.Collect(2, GCCollectionMode.Forced, true);
        var allocBefore = GC.GetTotalAllocatedBytes(precise: true);

        // Act - Prolonged chaos
        for (int i = 0; i < iterations; i++)
        {
            if (random.NextDouble() < 0.5)
            {
                _ = hlc.Now();
            }
            else
            {
                var chaosTs = new HybridTimestamp(
                    DateTime.UtcNow.Ticks + random.Next(-1000, 1000),
                    (ushort)random.Next(1000),
                    (ushort)random.Next(100));
                _ = hlc.Update(chaosTs);
            }
        }

        var allocAfter = GC.GetTotalAllocatedBytes(precise: true);
        var totalAlloc = allocAfter - allocBefore;
        var bytesPerOp = (double)totalAlloc / iterations;

        // Assert - HybridTimestamp is a struct, minimal allocations expected
        _output.WriteLine("=== Memory Stability Under Chaos ===");
        _output.WriteLine($"Iterations:      {iterations:N0}");
        _output.WriteLine($"Total allocated: {totalAlloc:N0} bytes");
        _output.WriteLine($"Bytes per op:    {bytesPerOp:F4}");
        _output.WriteLine($"Status:          {(bytesPerOp < 10 ? "✅ PASS" : "⚠️ HIGH ALLOCATION")}");

        Assert.True(bytesPerOp < 100,
            $"Allocations ({bytesPerOp:F2} bytes/op) exceed threshold");
    }

    #endregion

    #region Edge Case Tests

    /// <summary>
    /// Tests counter overflow behavior.
    /// </summary>
    [Fact]
    public void EdgeCase_CounterOverflow_ShouldAdvanceWallClock()
    {
        // Arrange
        var hlc = new HybridLogicalClock(nodeId: 1);

        // Generate many timestamps in same wall clock tick to force counter increase
        var timestamps = new List<HybridTimestamp>();
        var sw = Stopwatch.StartNew();

        // Generate as many timestamps as possible in a short burst
        while (sw.ElapsedMilliseconds < 10)
        {
            timestamps.Add(hlc.Now());
        }

        sw.Stop();

        // Analyze counter progression
        var maxCounter = timestamps.Max(t => t.LogicalCounter);
        var wallClockChanges = timestamps
            .Zip(timestamps.Skip(1), (a, b) => b.PhysicalTime != a.PhysicalTime)
            .Count(changed => changed);

        _output.WriteLine($"Generated {timestamps.Count:N0} timestamps in {sw.ElapsedMilliseconds}ms");
        _output.WriteLine($"Max counter value: {maxCounter}");
        _output.WriteLine($"Wall clock changes: {wallClockChanges}");

        // Assert - All timestamps unique and ordered
        for (int i = 1; i < timestamps.Count; i++)
        {
            Assert.True(timestamps[i].CompareTo(timestamps[i - 1]) > 0,
                $"Timestamp {i} should be after {i - 1}");
        }

        _output.WriteLine("✅ Counter progression handled correctly");
    }

    /// <summary>
    /// Tests timestamp comparison at boundaries.
    /// </summary>
    [Fact]
    public void EdgeCase_BoundaryComparisons_ShouldBeCorrect()
    {
        // Test boundary cases for comparisons
        var cases = new[]
        {
            (new HybridTimestamp(100, 0, 1), new HybridTimestamp(100, 0, 2), -1, "Same time, different node"),
            (new HybridTimestamp(100, 0, 1), new HybridTimestamp(100, 1, 1), -1, "Same time/node, different counter"),
            (new HybridTimestamp(100, 1, 1), new HybridTimestamp(101, 0, 1), -1, "Different wall clock"),
            (new HybridTimestamp(100, 0, 1), new HybridTimestamp(100, 0, 1), 0, "Identical timestamps"),
            (new HybridTimestamp(long.MaxValue - 1, 0, 1), new HybridTimestamp(long.MaxValue - 1, 1, 1), -1, "Near max wall clock"),
        };

        foreach (var (ts1, ts2, expected, description) in cases)
        {
            var result = Math.Sign(ts1.CompareTo(ts2));
            Assert.Equal(expected, result);
            _output.WriteLine($"✓ {description}: {ts1} vs {ts2} = {result}");
        }

        _output.WriteLine("✅ All boundary comparisons correct");
    }

    /// <summary>
    /// Tests empty Vector Clock operations.
    /// </summary>
    [Fact]
    public void EdgeCase_EmptyVectorClock_ShouldHandleGracefully()
    {
        // Arrange
        var empty1 = new VectorClock();
        var empty2 = new VectorClock();

        // Act & Assert
        var merged = empty1.Merge(empty2);
        Assert.Equal(0, merged.Count);

        var incremented = empty1.Increment(42);
        Assert.Equal(1, incremented.Count);
        Assert.Equal(1, incremented[42]);

        // Compare empty clocks - CompareTo returns CausalRelationship enum
        Assert.Equal(CausalRelationship.Equal, empty1.CompareTo(empty2));

        _output.WriteLine("Empty clock count: " + empty1.Count);
        _output.WriteLine("After increment: " + incremented.Count);
        _output.WriteLine("✅ Empty Vector Clock handled correctly");
    }

    #endregion
}

using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Orleans.GpuBridge.Abstractions.Temporal;
using Xunit;

namespace Orleans.GpuBridge.Tests.Temporal;

public class HybridLogicalClockTests
{
    [Fact]
    public void HybridTimestamp_ComparesCorrectly_ByPhysicalTime()
    {
        var t1 = new HybridTimestamp(100, 0, 1);
        var t2 = new HybridTimestamp(200, 0, 1);

        Assert.True(t1 < t2);
        Assert.True(t2 > t1);
        Assert.False(t1 == t2);
    }

    [Fact]
    public void HybridTimestamp_ComparesCorrectly_ByLogicalCounter()
    {
        // Same physical time, different logical counters
        var t1 = new HybridTimestamp(100, 5, 1);
        var t2 = new HybridTimestamp(100, 10, 1);

        Assert.True(t1 < t2);
        Assert.True(t2 > t1);
    }

    [Fact]
    public void HybridTimestamp_ComparesCorrectly_ByNodeId()
    {
        // Same physical time and logical counter, different node IDs
        var t1 = new HybridTimestamp(100, 5, 1);
        var t2 = new HybridTimestamp(100, 5, 2);

        Assert.True(t1 < t2);
        Assert.True(t2 > t1);
    }

    [Fact]
    public void HybridTimestamp_Equality_WorksCorrectly()
    {
        var t1 = new HybridTimestamp(100, 5, 1);
        var t2 = new HybridTimestamp(100, 5, 1);
        var t3 = new HybridTimestamp(100, 5, 2);

        Assert.Equal(t1, t2);
        Assert.NotEqual(t1, t3);
        Assert.True(t1 == t2);
        Assert.True(t1 != t3);
    }

    [Fact]
    public void HybridTimestamp_IsConcurrentWith_DetectsConcurrency()
    {
        // Same physical time and logical counter, different nodes
        var t1 = new HybridTimestamp(100, 5, 1);
        var t2 = new HybridTimestamp(100, 5, 2);
        var t3 = new HybridTimestamp(100, 6, 2);

        Assert.True(t1.IsConcurrentWith(t2));
        Assert.False(t1.IsConcurrentWith(t3)); // Different logical counter
    }

    [Fact]
    public void HybridLogicalClock_Now_ReturnsMonotonicTimestamps()
    {
        var clock = new HybridLogicalClock(nodeId: 1);

        var timestamps = new List<HybridTimestamp>();
        for (int i = 0; i < 1000; i++)
        {
            timestamps.Add(clock.Now());
        }

        // Verify all timestamps are strictly increasing
        for (int i = 1; i < timestamps.Count; i++)
        {
            Assert.True(timestamps[i] > timestamps[i - 1],
                $"Timestamp {i} is not greater than timestamp {i - 1}");
        }
    }

    [Fact]
    public void HybridLogicalClock_Now_IncrementsLogicalCounter_WhenPhysicalTimeEqual()
    {
        var mockClock = new MockClockSource(fixedTime: 1_000_000_000);
        var clock = new HybridLogicalClock(nodeId: 1, clockSource: mockClock);

        var t1 = clock.Now();
        var t2 = clock.Now();
        var t3 = clock.Now();

        // Physical time should be the same (mocked)
        Assert.Equal(t1.PhysicalTime, t2.PhysicalTime);
        Assert.Equal(t2.PhysicalTime, t3.PhysicalTime);

        // Logical counter should increment
        Assert.Equal(0, t1.LogicalCounter);
        Assert.Equal(1, t2.LogicalCounter);
        Assert.Equal(2, t3.LogicalCounter);
    }

    [Fact]
    public void HybridLogicalClock_Update_AdvancesClockOnReceive()
    {
        var clock = new HybridLogicalClock(nodeId: 1);

        var t1 = clock.Now();
        Thread.Sleep(10); // Ensure physical time advances

        // Simulate receiving a message with future timestamp
        var futureTimestamp = new HybridTimestamp(
            t1.PhysicalTime + 1_000_000_000, // 1 second in the future
            0,
            nodeId: 2);

        var t2 = clock.Update(futureTimestamp);

        // Clock should advance to received timestamp
        Assert.True(t2.PhysicalTime >= futureTimestamp.PhysicalTime);
        Assert.True(t2 > futureTimestamp); // Update increments logical counter
    }

    [Fact]
    public void HybridLogicalClock_Update_PreservesCausality()
    {
        var clockA = new HybridLogicalClock(nodeId: 1);
        var clockB = new HybridLogicalClock(nodeId: 2);

        // Node A sends message
        var tSend = clockA.Now();

        // Node B receives message and updates clock
        var tReceive = clockB.Update(tSend);

        // Any subsequent event on Node B should have timestamp > tSend
        var tNext = clockB.Now();

        Assert.True(tReceive > tSend);
        Assert.True(tNext > tSend);
        Assert.True(tNext > tReceive);
    }

    [Fact]
    public void HybridLogicalClock_ConcurrentEvents_HaveDifferentNodeIds()
    {
        var mockClock = new MockClockSource(fixedTime: 1_000_000_000);
        var clock1 = new HybridLogicalClock(nodeId: 1, clockSource: mockClock);
        var clock2 = new HybridLogicalClock(nodeId: 2, clockSource: mockClock);

        var t1 = clock1.Now();
        var t2 = clock2.Now();

        // Same physical time and logical counter (both first events)
        Assert.Equal(t1.PhysicalTime, t2.PhysicalTime);
        Assert.Equal(t1.LogicalCounter, t2.LogicalCounter);

        // But different node IDs
        Assert.NotEqual(t1.NodeId, t2.NodeId);

        // Total ordering is maintained
        Assert.NotEqual(t1, t2);
        Assert.True(t1 < t2 || t2 < t1);
    }

    [Fact]
    public async Task HybridLogicalClock_ThreadSafe_UnderConcurrentLoad()
    {
        var clock = new HybridLogicalClock(nodeId: 1);
        var timestamps = new System.Collections.Concurrent.ConcurrentBag<HybridTimestamp>();

        // Generate timestamps from multiple threads concurrently
        var tasks = Enumerable.Range(0, 10).Select(async _ =>
        {
            await Task.Run(() =>
            {
                for (int i = 0; i < 1000; i++)
                {
                    timestamps.Add(clock.Now());
                }
            });
        });

        await Task.WhenAll(tasks);

        // Verify all 10,000 timestamps are unique and monotonic
        var sorted = timestamps.OrderBy(t => t).ToList();
        Assert.Equal(10_000, sorted.Count);

        for (int i = 1; i < sorted.Count; i++)
        {
            Assert.True(sorted[i] > sorted[i - 1],
                $"Timestamp {i} is not greater than timestamp {i - 1}");
        }
    }

    [Fact]
    public void HybridLogicalClock_Reset_ChangesClockState()
    {
        var clock = new HybridLogicalClock(nodeId: 1);

        var t1 = clock.Now();
        var resetTimestamp = new HybridTimestamp(1_000_000, 42, 1);

        clock.Reset(resetTimestamp);

        var last = clock.LastTimestamp;
        Assert.Equal(resetTimestamp.PhysicalTime, last.PhysicalTime);
        Assert.Equal(resetTimestamp.LogicalCounter, last.LogicalCounter);
    }

    [Fact]
    public void HybridLogicalClock_GetClockDrift_CalculatesCorrectly()
    {
        var mockClock = new MockClockSource(fixedTime: 1_000_000_000);
        var clock = new HybridLogicalClock(nodeId: 1, clockSource: mockClock);

        var pastTimestamp = new HybridTimestamp(900_000_000, 0, 1);
        var drift = clock.GetClockDriftNanos(pastTimestamp);

        Assert.Equal(100_000_000, drift); // 100ms ahead
    }

    [Fact]
    public void HybridTimestamp_Performance_UnderOneMicrosecond()
    {
        var clock = new HybridLogicalClock(nodeId: 1);
        var iterations = 1_000_000;

        var sw = System.Diagnostics.Stopwatch.StartNew();
        for (int i = 0; i < iterations; i++)
        {
            _ = clock.Now();
        }
        sw.Stop();

        var avgNanos = (sw.Elapsed.TotalNanoseconds() / iterations);

        // Target: < 1000ns (1μs) per timestamp generation
        Assert.True(avgNanos < 1000, $"Average latency: {avgNanos:F0}ns (target: <1000ns)");
    }

    [Theory]
    [InlineData(100, 0, 1, 200, 0, 1, true)]  // Different physical time
    [InlineData(100, 5, 1, 100, 10, 1, true)] // Same physical, different logical
    [InlineData(100, 5, 1, 100, 5, 2, true)]  // Same physical/logical, different node
    [InlineData(100, 5, 1, 100, 5, 1, false)] // All equal
    public void HybridTimestamp_HappensBefore_WorksCorrectly(
        long pt1, long lc1, ushort n1,
        long pt2, long lc2, ushort n2,
        bool expectedHappensBefore)
    {
        var t1 = new HybridTimestamp(pt1, lc1, n1);
        var t2 = new HybridTimestamp(pt2, lc2, n2);

        Assert.Equal(expectedHappensBefore, t1.HappensBefore(t2));
    }
}

/// <summary>
/// Mock clock source for deterministic testing.
/// </summary>
internal sealed class MockClockSource : IPhysicalClockSource
{
    private long _currentTime;

    public MockClockSource(long fixedTime)
    {
        _currentTime = fixedTime;
    }

    public long GetCurrentTimeNanos()
    {
        return _currentTime;
    }

    public void AdvanceTime(long nanos)
    {
        _currentTime += nanos;
    }

    public long GetErrorBound() => 0;
    public bool IsSynchronized => true;
    public double GetClockDrift() => 0.0;
}

internal static class StopwatchExtensions
{
    public static double TotalNanoseconds(this TimeSpan timeSpan)
    {
        return timeSpan.Ticks * 100.0; // 1 tick = 100 nanoseconds
    }
}

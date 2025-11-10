using System;
using System.Collections.Immutable;
using System.Linq;
using Microsoft.Extensions.Logging.Abstractions;
using Orleans.GpuBridge.Abstractions.Temporal;
using Orleans.GpuBridge.Grains.Resident.Messages;
using Orleans.GpuBridge.Runtime.Temporal;
using Xunit;

namespace Orleans.GpuBridge.Tests.Temporal;

public class TemporalMessageQueueTests
{
    [Fact]
    public void TemporalMessageQueue_ProcessesMessages_InHLCOrder()
    {
        var queue = new TemporalMessageQueue(new MockClockSource(1_000_000_000));

        // Enqueue messages in reverse HLC order
        var msg1 = CreateMessage(hlc: (300, 0, 1));
        var msg2 = CreateMessage(hlc: (200, 0, 1));
        var msg3 = CreateMessage(hlc: (100, 0, 1));

        queue.Enqueue(msg1);
        queue.Enqueue(msg2);
        queue.Enqueue(msg3);

        // Should dequeue in HLC order (earliest first)
        Assert.True(queue.TryDequeue(out var first));
        Assert.True(queue.TryDequeue(out var second));
        Assert.True(queue.TryDequeue(out var third));

        Assert.Equal(msg3.RequestId, first!.RequestId); // HLC 100
        Assert.Equal(msg2.RequestId, second!.RequestId); // HLC 200
        Assert.Equal(msg1.RequestId, third!.RequestId); // HLC 300
    }

    [Fact]
    public void TemporalMessageQueue_ProcessesMessages_ByPriorityWithinSameHLC()
    {
        var queue = new TemporalMessageQueue();

        // Same HLC, different priorities
        var msgLow = CreateMessage(hlc: (100, 0, 1), priority: MessagePriority.Low);
        var msgHigh = CreateMessage(hlc: (100, 0, 1), priority: MessagePriority.High);
        var msgNormal = CreateMessage(hlc: (100, 0, 1), priority: MessagePriority.Normal);

        queue.Enqueue(msgLow);
        queue.Enqueue(msgHigh);
        queue.Enqueue(msgNormal);

        // Should process high priority first, then normal, then low
        Assert.True(queue.TryDequeue(out var first));
        Assert.True(queue.TryDequeue(out var second));
        Assert.True(queue.TryDequeue(out var third));

        Assert.Equal(msgHigh.RequestId, first!.RequestId);
        Assert.Equal(msgNormal.RequestId, second!.RequestId);
        Assert.Equal(msgLow.RequestId, third!.RequestId);
    }

    [Fact]
    public void TemporalMessageQueue_EnforcesCausalDependencies()
    {
        var queue = new TemporalMessageQueue();

        var msg1 = CreateMessage(hlc: (100, 0, 1));

        // msg2 depends on msg1
        var msg2 = CreateMessage(
            hlc: (200, 0, 1),
            dependencies: ImmutableArray.Create(msg1.RequestId));

        queue.Enqueue(msg2);
        queue.Enqueue(msg1);

        // Should process msg1 first (dependency of msg2)
        Assert.True(queue.TryDequeue(out var first));
        Assert.Equal(msg1.RequestId, first!.RequestId);

        // Mark msg1 as processed
        queue.MarkProcessed(msg1.RequestId);

        // Now msg2 should be available
        Assert.True(queue.TryDequeue(out var second));
        Assert.Equal(msg2.RequestId, second!.RequestId);
    }

    [Fact]
    public void TemporalMessageQueue_HandlesCausalChains()
    {
        var queue = new TemporalMessageQueue();

        // Create causal chain: msg1 → msg2 → msg3
        var msg1 = CreateMessage(hlc: (100, 0, 1));

        var msg2 = CreateMessage(
            hlc: (200, 0, 1),
            dependencies: ImmutableArray.Create(msg1.RequestId));

        var msg3 = CreateMessage(
            hlc: (300, 0, 1),
            dependencies: ImmutableArray.Create(msg2.RequestId));

        // Enqueue in reverse order
        queue.Enqueue(msg3);
        queue.Enqueue(msg2);
        queue.Enqueue(msg1);

        // Process chain
        Assert.True(queue.TryDequeue(out var first));
        Assert.Equal(msg1.RequestId, first!.RequestId);
        queue.MarkProcessed(msg1.RequestId);

        Assert.True(queue.TryDequeue(out var second));
        Assert.Equal(msg2.RequestId, second!.RequestId);
        queue.MarkProcessed(msg2.RequestId);

        Assert.True(queue.TryDequeue(out var third));
        Assert.Equal(msg3.RequestId, third!.RequestId);
    }

    [Fact]
    public void TemporalMessageQueue_HandlesDiamondDependency()
    {
        var queue = new TemporalMessageQueue();

        //     msg1
        //    /    \
        //  msg2  msg3
        //    \    /
        //     msg4

        var msg1 = CreateMessage(hlc: (100, 0, 1));

        var msg2 = CreateMessage(
            hlc: (200, 0, 1),
            dependencies: ImmutableArray.Create(msg1.RequestId));

        var msg3 = CreateMessage(
            hlc: (200, 1, 1),
            dependencies: ImmutableArray.Create(msg1.RequestId));

        var msg4 = CreateMessage(
            hlc: (300, 0, 1),
            dependencies: ImmutableArray.Create(msg2.RequestId, msg3.RequestId));

        queue.Enqueue(msg4);
        queue.Enqueue(msg3);
        queue.Enqueue(msg2);
        queue.Enqueue(msg1);

        // Process msg1
        Assert.True(queue.TryDequeue(out var dequeued));
        Assert.Equal(msg1.RequestId, dequeued!.RequestId);
        queue.MarkProcessed(msg1.RequestId);

        // Process msg2 and msg3 (both depend on msg1)
        Assert.True(queue.TryDequeue(out dequeued));
        Assert.Equal(msg2.RequestId, dequeued!.RequestId);
        queue.MarkProcessed(msg2.RequestId);

        Assert.True(queue.TryDequeue(out dequeued));
        Assert.Equal(msg3.RequestId, dequeued!.RequestId);
        queue.MarkProcessed(msg3.RequestId);

        // Now msg4 can be processed (both dependencies satisfied)
        Assert.True(queue.TryDequeue(out dequeued));
        Assert.Equal(msg4.RequestId, dequeued!.RequestId);
    }

    [Fact]
    public void TemporalMessageQueue_EvictsExpiredMessages()
    {
        var mockClock = new MockClockSource(fixedTime: 1_000_000_000);
        var queue = new TemporalMessageQueue(mockClock);

        // Create message with validity window that will expire
        var msg = CreateMessage(
            hlc: (100, 0, 1),
            validityWindow: new TimeRange
            {
                StartNanos = 0,
                EndNanos = 500_000_000 // Expires at 500ms
            });

        queue.Enqueue(msg);

        // Advance clock past deadline
        mockClock.AdvanceTime(1_000_000_000); // Advance 1 second

        // Evict expired messages
        queue.EvictExpiredMessages();

        // Message should be gone
        Assert.False(queue.TryDequeue(out _));
        Assert.Equal(1, queue.TotalExpired);
    }

    [Fact]
    public void TemporalMessageQueue_RejectsAlreadyExpiredMessages()
    {
        var mockClock = new MockClockSource(fixedTime: 2_000_000_000); // 2 seconds
        var queue = new TemporalMessageQueue(mockClock);

        // Create message that already expired
        var msg = CreateMessage(
            hlc: (100, 0, 1),
            validityWindow: new TimeRange
            {
                StartNanos = 0,
                EndNanos = 1_000_000_000 // Expired 1 second ago
            });

        queue.Enqueue(msg);

        // Message should be rejected immediately
        Assert.False(queue.TryDequeue(out _));
        Assert.Equal(1, queue.TotalExpired);
    }

    [Fact]
    public void TemporalMessageQueue_TracksStatistics()
    {
        var queue = new TemporalMessageQueue();

        var msg1 = CreateMessage(hlc: (100, 0, 1));
        var msg2 = CreateMessage(
            hlc: (200, 0, 1),
            dependencies: ImmutableArray.Create(msg1.RequestId));

        queue.Enqueue(msg1);
        queue.Enqueue(msg2);

        var stats = queue.GetStatistics();
        Assert.Equal(2, stats.TotalEnqueued);
        Assert.Equal(0, stats.TotalDequeued);

        queue.TryDequeue(out _);

        stats = queue.GetStatistics();
        Assert.Equal(1, stats.TotalDequeued);
        Assert.Equal(1, stats.TotalDependencyWaits); // msg2 waiting for msg1
    }

    [Fact]
    public void TemporalMessageQueue_HandlesEmptyQueue()
    {
        var queue = new TemporalMessageQueue();

        Assert.False(queue.TryDequeue(out var msg));
        Assert.Null(msg);
        Assert.Equal(0, queue.Count);
    }

    [Fact]
    public void TemporalMessageQueue_ClearResetsState()
    {
        var queue = new TemporalMessageQueue();

        var msg1 = CreateMessage(hlc: (100, 0, 1));
        var msg2 = CreateMessage(hlc: (200, 0, 1));

        queue.Enqueue(msg1);
        queue.Enqueue(msg2);
        queue.TryDequeue(out _);

        queue.Clear();

        Assert.Equal(0, queue.Count);
        Assert.Equal(0, queue.TotalEnqueued);
        Assert.Equal(0, queue.TotalDequeued);
    }

    [Theory]
    [InlineData(1000)]
    [InlineData(10000)]
    public void TemporalMessageQueue_Performance_UnderOneMillisecond(int messageCount)
    {
        var queue = new TemporalMessageQueue();
        var messages = Enumerable.Range(0, messageCount)
            .Select(i => CreateMessage(hlc: (i, 0, 1)))
            .ToList();

        // Measure enqueue performance
        var sw = System.Diagnostics.Stopwatch.StartNew();
        foreach (var msg in messages)
        {
            queue.Enqueue(msg);
        }
        sw.Stop();

        var enqueueAvgMicros = sw.Elapsed.TotalMicroseconds() / messageCount;

        // Measure dequeue performance
        sw.Restart();
        while (queue.TryDequeue(out var msg))
        {
            queue.MarkProcessed(msg!.RequestId);
        }
        sw.Stop();

        var dequeueAvgMicros = sw.Elapsed.TotalMicroseconds() / messageCount;

        // Target: < 10μs per operation (enqueue or dequeue)
        Assert.True(enqueueAvgMicros < 10, $"Enqueue avg: {enqueueAvgMicros:F2}μs (target: <10μs)");
        Assert.True(dequeueAvgMicros < 10, $"Dequeue avg: {dequeueAvgMicros:F2}μs (target: <10μs)");
    }

    [Fact]
    public void TimeRange_OverlapsDetection_WorksCorrectly()
    {
        var range1 = new TimeRange { StartNanos = 100, EndNanos = 200 };
        var range2 = new TimeRange { StartNanos = 150, EndNanos = 250 }; // Overlaps
        var range3 = new TimeRange { StartNanos = 300, EndNanos = 400 }; // No overlap

        Assert.True(range1.Overlaps(range2));
        Assert.True(range2.Overlaps(range1));
        Assert.False(range1.Overlaps(range3));
        Assert.False(range3.Overlaps(range1));
    }

    [Fact]
    public void TimeRange_ContainsTime_WorksCorrectly()
    {
        var range = new TimeRange { StartNanos = 100, EndNanos = 200 };

        Assert.True(range.Contains(100));  // Start (inclusive)
        Assert.True(range.Contains(150));  // Middle
        Assert.True(range.Contains(200));  // End (inclusive)
        Assert.False(range.Contains(99));  // Before
        Assert.False(range.Contains(201)); // After
    }

    // Helper methods

    private static TestTemporalMessage CreateMessage(
        (long physical, long logical, ushort node)? hlc = null,
        ImmutableArray<Guid>? dependencies = null,
        MessagePriority priority = MessagePriority.Normal,
        TimeRange? validityWindow = null)
    {
        var (physical, logical, node) = hlc ?? (HybridTimestamp.GetCurrentPhysicalTimeNanos(), 0, 1);

        return new TestTemporalMessage
        {
            HLC = new HybridTimestamp(physical, logical, node),
            PhysicalTimeNanos = physical,
            TimestampErrorBoundNanos = 1_000_000, // ±1ms
            CausalDependencies = dependencies ?? ImmutableArray<Guid>.Empty,
            Priority = priority,
            ValidityWindow = validityWindow
        };
    }
}

/// <summary>
/// Test message type for unit tests.
/// </summary>
internal sealed record TestTemporalMessage : TemporalResidentMessage
{
    public string? TestData { get; init; }
}

internal static class TimeSpanExtensions
{
    public static double TotalMicroseconds(this TimeSpan timeSpan)
    {
        return timeSpan.TotalMilliseconds * 1000.0;
    }
}

using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Orleans.GpuBridge.Abstractions.Temporal;
using Orleans.GpuBridge.Runtime.Temporal;
using Xunit;

namespace Orleans.GpuBridge.Tests.Temporal;

public class CausalOrderingTests
{
    #region HybridCausalClock Tests

    [Fact]
    public void HybridCausalClock_Now_IncrementsVectorClock()
    {
        var clock = new HybridCausalClock(actorId: 1);

        var t1 = clock.Now();
        var t2 = clock.Now();
        var t3 = clock.Now();

        Assert.Equal(1, t1.VectorClock[1]);
        Assert.Equal(2, t2.VectorClock[1]);
        Assert.Equal(3, t3.VectorClock[1]);
    }

    [Fact]
    public void HybridCausalClock_Now_IncrementsHLC()
    {
        var clock = new HybridCausalClock(actorId: 1);

        var t1 = clock.Now();
        var t2 = clock.Now();

        Assert.True(t2.HLC.CompareTo(t1.HLC) > 0);
    }

    [Fact]
    public void HybridCausalClock_Update_MergesVectorClocks()
    {
        var clock1 = new HybridCausalClock(actorId: 1);
        var clock2 = new HybridCausalClock(actorId: 2);

        // Actor 1 generates events
        var t1_1 = clock1.Now(); // [1:1]
        var t1_2 = clock1.Now(); // [1:2]

        // Actor 2 receives message from Actor 1
        var t2_1 = clock2.Update(t1_2); // [1:2, 2:1]

        Assert.Equal(2, t2_1.VectorClock[1]); // Knows Actor 1's events
        Assert.Equal(1, t2_1.VectorClock[2]); // Own event
    }

    [Fact]
    public void HybridCausalClock_CanDeliver_ChecksDependencies()
    {
        var clock1 = new HybridCausalClock(actorId: 1);
        var clock2 = new HybridCausalClock(actorId: 2);
        var clock3 = new HybridCausalClock(actorId: 3);

        // Actor 1 sends to Actor 2
        var t1 = clock1.Now();
        _ = clock2.Update(t1);

        // Actor 2 sends to Actor 3
        var t2 = clock2.Now();

        // Actor 3 hasn't seen t1 yet, so can't deliver t2
        Assert.False(clock3.CanDeliver(t2));

        // After Actor 3 receives t1
        _ = clock3.Update(t1);

        // Now Actor 3 can deliver t2
        Assert.True(clock3.CanDeliver(t2));
    }

    [Fact]
    public void HybridCausalTimestamp_GetCausalRelationship_Correct()
    {
        var clock = new HybridCausalClock(actorId: 1);

        var t1 = clock.Now();
        var t2 = clock.Now();

        Assert.Equal(CausalRelationship.HappensBefore, t1.GetCausalRelationship(t2));
        Assert.Equal(CausalRelationship.HappensAfter, t2.GetCausalRelationship(t1));
    }

    [Fact]
    public void HybridCausalTimestamp_Serialization_RoundTrip()
    {
        var clock = new HybridCausalClock(actorId: 5);
        var timestamp = clock.Now();

        var bytes = timestamp.ToBytes();
        var deserialized = HybridCausalTimestamp.FromBytes(bytes);

        Assert.Equal(timestamp.HLC, deserialized.HLC);
        Assert.Equal(timestamp.VectorClock, deserialized.VectorClock);
        Assert.Equal(timestamp.SenderId, deserialized.SenderId);
    }

    #endregion

    #region CausalOrderingQueue Tests

    [Fact]
    public async Task CausalOrderingQueue_InOrder_DeliverImmediately()
    {
        var clock = new HybridCausalClock(actorId: 1);
        var queue = new CausalOrderingQueue(clock);

        // Send messages in order
        var t1 = clock.Now();
        var msg1 = new CausalMessage
        {
            MessageId = Guid.NewGuid(),
            SenderId = 1,
            Timestamp = t1,
            Payload = "Message 1"
        };

        var delivered = await queue.EnqueueAsync(msg1);

        Assert.Single(delivered);
        Assert.Equal(msg1.MessageId, delivered[0].MessageId);
        Assert.Equal(0, queue.PendingCount);
    }

    [Fact]
    public async Task CausalOrderingQueue_OutOfOrder_BuffersUntilReady()
    {
        var clock1 = new HybridCausalClock(actorId: 1);
        var clock2 = new HybridCausalClock(actorId: 2);
        var receiverClock = new HybridCausalClock(actorId: 3);
        var queue = new CausalOrderingQueue(receiverClock);

        // Actor 1 sends message
        var t1 = clock1.Now();

        // Actor 2 receives t1 and sends message
        _ = clock2.Update(t1);
        var t2 = clock2.Now();

        // Receiver gets t2 BEFORE t1 (out of order)
        var msg2 = new CausalMessage
        {
            MessageId = Guid.NewGuid(),
            SenderId = 2,
            Timestamp = t2,
            Payload = "Message 2"
        };

        var delivered1 = await queue.EnqueueAsync(msg2);

        // msg2 should be buffered (depends on t1)
        Assert.Empty(delivered1);
        Assert.Equal(1, queue.PendingCount);

        // Now send t1
        var msg1 = new CausalMessage
        {
            MessageId = Guid.NewGuid(),
            SenderId = 1,
            Timestamp = t1,
            Payload = "Message 1"
        };

        var delivered2 = await queue.EnqueueAsync(msg1);

        // Both messages should be delivered now
        Assert.Equal(2, delivered2.Count);
        Assert.Equal(0, queue.PendingCount);

        // Verify order: msg1 then msg2
        Assert.Equal(msg1.MessageId, delivered2[0].MessageId);
        Assert.Equal(msg2.MessageId, delivered2[1].MessageId);
    }

    [Fact]
    public async Task CausalOrderingQueue_DiamondDependency_DeliversCorrectly()
    {
        // Setup: 4 actors
        var clock1 = new HybridCausalClock(actorId: 1);
        var clock2 = new HybridCausalClock(actorId: 2);
        var clock3 = new HybridCausalClock(actorId: 3);
        var clock4 = new HybridCausalClock(actorId: 4);
        var queue = new CausalOrderingQueue(clock4);

        // Actor 1 sends to both 2 and 3
        var t1 = clock1.Now();

        // Actor 2 processes
        _ = clock2.Update(t1);
        var t2 = clock2.Now();

        // Actor 3 processes
        _ = clock3.Update(t1);
        var t3 = clock3.Now();

        // Actor 4 receives t2, t3, t1 out of order
        var msg2 = new CausalMessage
        {
            MessageId = Guid.NewGuid(),
            SenderId = 2,
            Timestamp = t2,
            Payload = "From Actor 2"
        };

        var msg3 = new CausalMessage
        {
            MessageId = Guid.NewGuid(),
            SenderId = 3,
            Timestamp = t3,
            Payload = "From Actor 3"
        };

        var msg1 = new CausalMessage
        {
            MessageId = Guid.NewGuid(),
            SenderId = 1,
            Timestamp = t1,
            Payload = "From Actor 1"
        };

        // Enqueue out of order
        var d1 = await queue.EnqueueAsync(msg2); // Buffered
        var d2 = await queue.EnqueueAsync(msg3); // Buffered
        var d3 = await queue.EnqueueAsync(msg1); // Triggers delivery of all 3

        Assert.Empty(d1);
        Assert.Empty(d2);
        Assert.Equal(3, d3.Count);

        // msg1 must be first, msg2 and msg3 can be in any order
        Assert.Equal(msg1.MessageId, d3[0].MessageId);
    }

    [Fact]
    public async Task CausalOrderingQueue_ConcurrentMessages_BothDelivered()
    {
        var clock1 = new HybridCausalClock(actorId: 1);
        var clock2 = new HybridCausalClock(actorId: 2);
        var receiverClock = new HybridCausalClock(actorId: 3);
        var queue = new CausalOrderingQueue(receiverClock);

        // Two actors send concurrent messages (no causal relationship)
        var t1 = clock1.Now();
        var t2 = clock2.Now();

        var msg1 = new CausalMessage
        {
            MessageId = Guid.NewGuid(),
            SenderId = 1,
            Timestamp = t1,
            Payload = "Concurrent 1"
        };

        var msg2 = new CausalMessage
        {
            MessageId = Guid.NewGuid(),
            SenderId = 2,
            Timestamp = t2,
            Payload = "Concurrent 2"
        };

        var d1 = await queue.EnqueueAsync(msg1);
        var d2 = await queue.EnqueueAsync(msg2);

        // Both should be delivered immediately (no dependencies)
        Assert.Single(d1);
        Assert.Single(d2);
    }

    [Fact]
    public async Task CausalOrderingQueue_LongChain_DeliversInOrder()
    {
        // Create chain: 1 → 2 → 3 → 4 → 5
        var clocks = Enumerable.Range(1, 5)
            .Select(i => new HybridCausalClock((ushort)i))
            .ToList();

        var receiverClock = new HybridCausalClock(actorId: 99);
        var queue = new CausalOrderingQueue(receiverClock);

        // Generate chain of messages
        var timestamps = new List<HybridCausalTimestamp>();
        for (int i = 0; i < 5; i++)
        {
            if (i > 0)
            {
                _ = clocks[i].Update(timestamps[i - 1]);
            }
            timestamps.Add(clocks[i].Now());
        }

        // Create messages
        var messages = timestamps.Select((t, i) => new CausalMessage
        {
            MessageId = Guid.NewGuid(),
            SenderId = (ushort)(i + 1),
            Timestamp = t,
            Payload = $"Message {i + 1}"
        }).ToList();

        // Deliver in reverse order
        messages.Reverse();

        var allDelivered = new List<CausalMessage>();
        foreach (var msg in messages)
        {
            var delivered = await queue.EnqueueAsync(msg);
            allDelivered.AddRange(delivered);
        }

        // All messages should be delivered
        Assert.Equal(5, allDelivered.Count);

        // Verify correct order (1 → 2 → 3 → 4 → 5)
        for (int i = 0; i < 5; i++)
        {
            Assert.Equal((ushort)(i + 1), allDelivered[i].SenderId);
        }
    }

    [Fact]
    public async Task CausalOrderingQueue_ExplicitDependencies_Respected()
    {
        var clock = new HybridCausalClock(actorId: 1);
        var queue = new CausalOrderingQueue(clock);

        var t1 = clock.Now();
        var msg1 = new CausalMessage
        {
            MessageId = Guid.NewGuid(),
            SenderId = 1,
            Timestamp = t1,
            Payload = "Message 1"
        };

        var t2 = clock.Now();
        var msg2 = new CausalMessage
        {
            MessageId = Guid.NewGuid(),
            SenderId = 1,
            Timestamp = t2,
            Payload = "Message 2",
            Dependencies = new[] { msg1.MessageId } // Explicit dependency
        };

        // Enqueue msg2 first
        var d1 = await queue.EnqueueAsync(msg2);
        Assert.Empty(d1); // Buffered due to dependency

        // Enqueue msg1
        var d2 = await queue.EnqueueAsync(msg1);
        Assert.Equal(2, d2.Count); // Both delivered

        // Verify order
        Assert.Equal(msg1.MessageId, d2[0].MessageId);
        Assert.Equal(msg2.MessageId, d2[1].MessageId);
    }

    [Fact]
    public async Task CausalOrderingQueue_DeadlockDetection_WorksCorrectly()
    {
        var clock = new HybridCausalClock(actorId: 1);
        var queue = new CausalOrderingQueue(clock);

        // Create message with impossible dependency (references itself or non-existent message)
        var t1 = clock.Now();
        var msg = new CausalMessage
        {
            MessageId = Guid.NewGuid(),
            SenderId = 1,
            Timestamp = t1,
            Payload = "Deadlocked",
            Dependencies = new[] { Guid.NewGuid() } // Non-existent dependency
        };

        await queue.EnqueueAsync(msg);

        // Should detect deadlock
        Assert.True(queue.DetectDeadlock());
    }

    [Fact]
    public async Task CausalOrderingQueue_Statistics_AccurateTracking()
    {
        var clock = new HybridCausalClock(actorId: 1);
        var queue = new CausalOrderingQueue(clock);

        // Deliver some messages
        for (int i = 0; i < 10; i++)
        {
            var t = clock.Now();
            var msg = new CausalMessage
            {
                MessageId = Guid.NewGuid(),
                SenderId = 1,
                Timestamp = t,
                Payload = $"Message {i}"
            };

            await queue.EnqueueAsync(msg);
        }

        var stats = queue.GetStatistics();

        Assert.Equal(10, stats.TotalEnqueued);
        Assert.Equal(10, stats.TotalDelivered);
        Assert.Equal(0, stats.PendingCount);
        Assert.Equal(1.0, stats.DeliveryRate);
    }

    #endregion

    #region Performance Tests

    [Fact]
    public async Task CausalOrderingQueue_Performance_ThousandMessages()
    {
        var clock = new HybridCausalClock(actorId: 1);
        var queue = new CausalOrderingQueue(clock);

        var sw = System.Diagnostics.Stopwatch.StartNew();

        // Enqueue 1000 messages in order
        for (int i = 0; i < 1000; i++)
        {
            var t = clock.Now();
            var msg = new CausalMessage
            {
                MessageId = Guid.NewGuid(),
                SenderId = 1,
                Timestamp = t,
                Payload = $"Message {i}"
            };

            await queue.EnqueueAsync(msg);
        }

        sw.Stop();

        var avgMicros = sw.Elapsed.TotalMicroseconds / 1000.0;

        Assert.True(avgMicros < 100, $"Enqueue too slow: {avgMicros:F2}μs per message (target: <100μs)");
    }

    [Fact]
    public async Task HybridCausalClock_Performance_TimestampGeneration()
    {
        var clock = new HybridCausalClock(actorId: 1);

        var sw = System.Diagnostics.Stopwatch.StartNew();

        for (int i = 0; i < 10_000; i++)
        {
            _ = clock.Now();
        }

        sw.Stop();

        var avgNanos = sw.Elapsed.TotalNanoseconds / 10_000;

        Assert.True(avgNanos < 1000, $"Timestamp generation too slow: {avgNanos:F0}ns (target: <1000ns)");
    }

    #endregion

    #region Multi-Actor Scenarios

    [Fact]
    public async Task CausalOrderingQueue_FiveActors_ComplexInteractions()
    {
        // Scenario: 5 actors with complex message patterns
        var clocks = Enumerable.Range(1, 5)
            .ToDictionary(i => i, i => new HybridCausalClock((ushort)i));

        var actor3Queue = new CausalOrderingQueue(clocks[3]);

        // Actor 1 → Actor 2
        var t1 = clocks[1].Now();
        _ = clocks[2].Update(t1);

        // Actor 2 → Actor 3
        var t2 = clocks[2].Now();

        // Actor 4 → Actor 5
        var t4 = clocks[4].Now();
        _ = clocks[5].Update(t4);

        // Actor 5 → Actor 3
        var t5 = clocks[5].Now();

        // Actor 3 receives: t5, t2 (out of order)
        var msg5 = new CausalMessage
        {
            MessageId = Guid.NewGuid(),
            SenderId = 5,
            Timestamp = t5,
            Payload = "From 5"
        };

        var msg2 = new CausalMessage
        {
            MessageId = Guid.NewGuid(),
            SenderId = 2,
            Timestamp = t2,
            Payload = "From 2"
        };

        // msg5 depends on t4, msg2 depends on t1
        // Both should be buffered
        var d1 = await actor3Queue.EnqueueAsync(msg5);
        var d2 = await actor3Queue.EnqueueAsync(msg2);

        Assert.Empty(d1);
        Assert.Empty(d2);
        Assert.Equal(2, actor3Queue.PendingCount);

        // Deliver missing dependencies
        var msg1 = new CausalMessage
        {
            MessageId = Guid.NewGuid(),
            SenderId = 1,
            Timestamp = t1,
            Payload = "From 1"
        };

        var msg4 = new CausalMessage
        {
            MessageId = Guid.NewGuid(),
            SenderId = 4,
            Timestamp = t4,
            Payload = "From 4"
        };

        var d3 = await actor3Queue.EnqueueAsync(msg1);
        var d4 = await actor3Queue.EnqueueAsync(msg4);

        // All 4 messages should now be delivered
        var totalDelivered = d3.Count + d4.Count;
        Assert.Equal(4, totalDelivered);
    }

    #endregion
}

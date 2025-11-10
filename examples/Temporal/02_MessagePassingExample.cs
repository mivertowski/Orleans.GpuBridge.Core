using System;
using System.Collections.Immutable;
using Orleans.GpuBridge.Abstractions.Temporal;
using Orleans.GpuBridge.Grains.Resident.Messages;
using Orleans.GpuBridge.Runtime.Temporal;

namespace Orleans.GpuBridge.Examples.Temporal;

/// <summary>
/// Example 2: Message passing with causal dependencies.
/// </summary>
/// <remarks>
/// This example demonstrates:
/// - Creating temporal messages with HLC timestamps
/// - Causal dependency tracking
/// - Message queue with dependency enforcement
/// - Priority-based message processing
/// </remarks>
public static class MessagePassingExample
{
    public static void Run()
    {
        Console.WriteLine("=== Example 2: Message Passing with Causal Dependencies ===\n");

        var queue = new TemporalMessageQueue();
        var clock = new HybridLogicalClock(nodeId: 1);

        // Scenario: Three-node message chain
        // A → B → C (each message depends on the previous one)

        Console.WriteLine("Scenario: Message chain A → B → C");
        Console.WriteLine("Messages will be enqueued out of order to demonstrate dependency handling\n");

        // Message 1: A sends to B
        var msg1 = new ExampleMessage
        {
            Content = "A → B: Transfer $1000",
            HLC = clock.Now(),
            PhysicalTimeNanos = HybridTimestamp.GetCurrentPhysicalTimeNanos(),
            TimestampErrorBoundNanos = 1_000_000, // ±1ms
            SequenceNumber = 1
        };
        Console.WriteLine($"Created msg1: {msg1.Content}");
        Console.WriteLine($"  RequestId: {msg1.RequestId}");
        Console.WriteLine($"  HLC: {msg1.HLC}");

        // Message 2: B sends to C (depends on msg1)
        var msg2 = new ExampleMessage
        {
            Content = "B → C: Transfer $500",
            HLC = clock.Now(),
            PhysicalTimeNanos = HybridTimestamp.GetCurrentPhysicalTimeNanos(),
            TimestampErrorBoundNanos = 1_000_000,
            CausalDependencies = ImmutableArray.Create(msg1.RequestId),
            SequenceNumber = 2
        };
        Console.WriteLine($"\nCreated msg2: {msg2.Content}");
        Console.WriteLine($"  RequestId: {msg2.RequestId}");
        Console.WriteLine($"  Dependencies: [{msg1.RequestId}]");

        // Message 3: C sends to D (depends on msg2)
        var msg3 = new ExampleMessage
        {
            Content = "C → D: Transfer $250",
            HLC = clock.Now(),
            PhysicalTimeNanos = HybridTimestamp.GetCurrentPhysicalTimeNanos(),
            TimestampErrorBoundNanos = 1_000_000,
            CausalDependencies = ImmutableArray.Create(msg2.RequestId),
            SequenceNumber = 3
        };
        Console.WriteLine($"\nCreated msg3: {msg3.Content}");
        Console.WriteLine($"  RequestId: {msg3.RequestId}");
        Console.WriteLine($"  Dependencies: [{msg2.RequestId}]");

        // Enqueue messages in REVERSE order to demonstrate dependency handling
        Console.WriteLine("\n--- Enqueuing Messages (Reverse Order) ---");
        queue.Enqueue(msg3);
        Console.WriteLine($"Enqueued: {msg3.Content}");
        queue.Enqueue(msg2);
        Console.WriteLine($"Enqueued: {msg2.Content}");
        queue.Enqueue(msg1);
        Console.WriteLine($"Enqueued: {msg1.Content}");

        var stats = queue.GetStatistics();
        Console.WriteLine($"\nQueue stats: {stats.MessagesInQueue} in queue, {stats.MessagesWaitingDependencies} waiting");

        // Dequeue and process messages
        Console.WriteLine("\n--- Processing Messages (Enforcing Dependencies) ---");

        // First dequeue: Should get msg1 (no dependencies)
        if (queue.TryDequeue(out var dequeued))
        {
            Console.WriteLine($"\n✓ Dequeued: {dequeued!.Content}");
            Console.WriteLine($"  This message has no dependencies, so it's processed first");
            queue.MarkProcessed(dequeued.RequestId);
        }

        // Second dequeue: Should get msg2 (msg1 now satisfied)
        if (queue.TryDequeue(out dequeued))
        {
            Console.WriteLine($"\n✓ Dequeued: {dequeued!.Content}");
            Console.WriteLine($"  Dependency on msg1 is now satisfied");
            queue.MarkProcessed(dequeued.RequestId);
        }

        // Third dequeue: Should get msg3 (msg2 now satisfied)
        if (queue.TryDequeue(out dequeued))
        {
            Console.WriteLine($"\n✓ Dequeued: {dequeued!.Content}");
            Console.WriteLine($"  Dependency on msg2 is now satisfied");
            queue.MarkProcessed(dequeued.RequestId);
        }

        stats = queue.GetStatistics();
        Console.WriteLine($"\n✓ All messages processed!");
        Console.WriteLine($"  Total enqueued: {stats.TotalEnqueued}");
        Console.WriteLine($"  Total dequeued: {stats.TotalDequeued}");
        Console.WriteLine($"  Dependency waits: {stats.TotalDependencyWaits}");
    }

    /// <summary>
    /// Example showing diamond dependency pattern.
    /// </summary>
    public static void DemonstrateDiamondDependency()
    {
        Console.WriteLine("\n=== Diamond Dependency Pattern ===\n");

        var queue = new TemporalMessageQueue();
        var clock = new HybridLogicalClock(nodeId: 1);

        Console.WriteLine("Scenario: Diamond dependency");
        Console.WriteLine("     msg1");
        Console.WriteLine("    /    \\");
        Console.WriteLine("  msg2  msg3");
        Console.WriteLine("    \\    /");
        Console.WriteLine("     msg4\n");

        // Create messages
        var msg1 = new ExampleMessage
        {
            Content = "Root: Initiate workflow",
            HLC = clock.Now(),
            PhysicalTimeNanos = HybridTimestamp.GetCurrentPhysicalTimeNanos(),
            SequenceNumber = 1
        };

        var msg2 = new ExampleMessage
        {
            Content = "Branch A: Process left path",
            HLC = clock.Now(),
            PhysicalTimeNanos = HybridTimestamp.GetCurrentPhysicalTimeNanos(),
            CausalDependencies = ImmutableArray.Create(msg1.RequestId),
            SequenceNumber = 2
        };

        var msg3 = new ExampleMessage
        {
            Content = "Branch B: Process right path",
            HLC = clock.Now(),
            PhysicalTimeNanos = HybridTimestamp.GetCurrentPhysicalTimeNanos(),
            CausalDependencies = ImmutableArray.Create(msg1.RequestId),
            SequenceNumber = 3
        };

        var msg4 = new ExampleMessage
        {
            Content = "Join: Merge results",
            HLC = clock.Now(),
            PhysicalTimeNanos = HybridTimestamp.GetCurrentPhysicalTimeNanos(),
            CausalDependencies = ImmutableArray.Create(msg2.RequestId, msg3.RequestId),
            SequenceNumber = 4
        };

        // Enqueue all messages
        queue.Enqueue(msg4); // Join (depends on both branches)
        queue.Enqueue(msg3); // Branch B
        queue.Enqueue(msg2); // Branch A
        queue.Enqueue(msg1); // Root

        Console.WriteLine("Enqueued all messages\n");

        // Process messages
        Console.WriteLine("Processing order:");

        // Process msg1 (root)
        if (queue.TryDequeue(out var dequeued))
        {
            Console.WriteLine($"1. {dequeued!.Content}");
            queue.MarkProcessed(dequeued.RequestId);
        }

        // Process msg2 and msg3 (branches - can be in any order)
        if (queue.TryDequeue(out dequeued))
        {
            Console.WriteLine($"2. {dequeued!.Content}");
            queue.MarkProcessed(dequeued.RequestId);
        }

        if (queue.TryDequeue(out dequeued))
        {
            Console.WriteLine($"3. {dequeued!.Content}");
            queue.MarkProcessed(dequeued.RequestId);
        }

        // Process msg4 (join - waits for both branches)
        if (queue.TryDequeue(out dequeued))
        {
            Console.WriteLine($"4. {dequeued!.Content}");
            Console.WriteLine($"   ✓ Both dependencies satisfied before processing");
            queue.MarkProcessed(dequeued.RequestId);
        }
    }

    /// <summary>
    /// Example showing priority-based message processing.
    /// </summary>
    public static void DemonstratePriorityProcessing()
    {
        Console.WriteLine("\n=== Priority-Based Message Processing ===\n");

        var queue = new TemporalMessageQueue();
        var clock = new HybridLogicalClock(nodeId: 1);

        // Create messages with same HLC but different priorities
        var hlc = clock.Now();

        var msgLow = new ExampleMessage
        {
            Content = "Low priority: Analytics query",
            HLC = hlc,
            PhysicalTimeNanos = HybridTimestamp.GetCurrentPhysicalTimeNanos(),
            Priority = MessagePriority.Low,
            SequenceNumber = 1
        };

        var msgNormal = new ExampleMessage
        {
            Content = "Normal priority: User request",
            HLC = hlc,
            PhysicalTimeNanos = HybridTimestamp.GetCurrentPhysicalTimeNanos(),
            Priority = MessagePriority.Normal,
            SequenceNumber = 2
        };

        var msgHigh = new ExampleMessage
        {
            Content = "High priority: Real-time trading",
            HLC = hlc,
            PhysicalTimeNanos = HybridTimestamp.GetCurrentPhysicalTimeNanos(),
            Priority = MessagePriority.High,
            SequenceNumber = 3
        };

        var msgCritical = new ExampleMessage
        {
            Content = "Critical priority: System alert",
            HLC = hlc,
            PhysicalTimeNanos = HybridTimestamp.GetCurrentPhysicalTimeNanos(),
            Priority = MessagePriority.Critical,
            SequenceNumber = 4
        };

        // Enqueue in random order
        Console.WriteLine("Enqueuing messages with same HLC timestamp:");
        queue.Enqueue(msgLow);
        Console.WriteLine($"  {msgLow.Priority}: {msgLow.Content}");
        queue.Enqueue(msgHigh);
        Console.WriteLine($"  {msgHigh.Priority}: {msgHigh.Content}");
        queue.Enqueue(msgNormal);
        Console.WriteLine($"  {msgNormal.Priority}: {msgNormal.Content}");
        queue.Enqueue(msgCritical);
        Console.WriteLine($"  {msgCritical.Priority}: {msgCritical.Content}");

        // Process in priority order
        Console.WriteLine("\nProcessing order (highest priority first):");
        int order = 1;
        while (queue.TryDequeue(out var dequeued))
        {
            Console.WriteLine($"{order}. [{dequeued!.Priority}] {dequeued.Content}");
            queue.MarkProcessed(dequeued.RequestId);
            order++;
        }

        Console.WriteLine("\n✓ Critical and High priority messages processed first!");
    }

    /// <summary>
    /// Example showing deadline-based message eviction.
    /// </summary>
    public static void DemonstrateDeadlineEviction()
    {
        Console.WriteLine("\n=== Deadline-Based Message Eviction ===\n");

        var mockClock = new MockClockSource(fixedTime: 1_000_000_000); // Start at 1 second
        var queue = new TemporalMessageQueue(mockClock);
        var clock = new HybridLogicalClock(nodeId: 1, clockSource: mockClock);

        // Create message with 100ms deadline
        var msg = new ExampleMessage
        {
            Content = "Time-sensitive order",
            HLC = clock.Now(),
            PhysicalTimeNanos = mockClock.GetCurrentTimeNanos(),
            ValidityWindow = new TimeRange
            {
                StartNanos = 1_000_000_000,
                EndNanos = 1_100_000_000  // 100ms deadline
            }
        };

        Console.WriteLine($"Created message: {msg.Content}");
        Console.WriteLine($"  Validity window: {msg.ValidityWindow!.Value.StartNanos}ns - {msg.ValidityWindow.Value.EndNanos}ns");
        Console.WriteLine($"  Duration: {msg.ValidityWindow.Value.DurationNanos / 1_000_000.0}ms");

        queue.Enqueue(msg);
        Console.WriteLine($"\nEnqueued at time: {mockClock.GetCurrentTimeNanos()}ns");

        // Advance time past deadline
        mockClock.AdvanceTime(150_000_000); // Advance 150ms
        Console.WriteLine($"Advanced clock to: {mockClock.GetCurrentTimeNanos()}ns");
        Console.WriteLine($"Message expired by: {(mockClock.GetCurrentTimeNanos() - msg.ValidityWindow.Value.EndNanos) / 1_000_000.0}ms");

        // Try to dequeue (should evict expired message)
        var success = queue.TryDequeue(out var dequeued);
        Console.WriteLine($"\nTry dequeue: {(success ? "SUCCESS" : "FAILED (message expired)")}");

        var stats = queue.GetStatistics();
        Console.WriteLine($"\n✓ Expired messages: {stats.TotalExpired}");
        Console.WriteLine($"  Expiry ratio: {stats.ExpiredRatio * 100:F1}%");
    }
}

/// <summary>
/// Example message type for demonstrations.
/// </summary>
internal sealed record ExampleMessage : TemporalResidentMessage
{
    public string Content { get; init; } = string.Empty;
}

/// <summary>
/// Mock clock source that can be advanced for testing.
/// </summary>
internal sealed class MockClockSource : IPhysicalClockSource
{
    private long _currentTime;

    public MockClockSource(long fixedTime)
    {
        _currentTime = fixedTime;
    }

    public long GetCurrentTimeNanos() => _currentTime;
    public void AdvanceTime(long nanos) => _currentTime += nanos;
    public long GetErrorBound() => 0;
    public bool IsSynchronized => true;
    public double GetClockDrift() => 0.0;
}

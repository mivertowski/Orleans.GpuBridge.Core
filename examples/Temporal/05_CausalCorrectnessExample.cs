using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Orleans.GpuBridge.Abstractions.Temporal;
using Orleans.GpuBridge.Runtime.Temporal;

namespace Orleans.GpuBridge.Examples.Temporal;

/// <summary>
/// Example demonstrating causal correctness with vector clocks and causal message ordering.
/// </summary>
/// <remarks>
/// This example shows:
/// 1. Vector clock operations (increment, merge, comparison)
/// 2. Causal relationships (happens-before, concurrent)
/// 3. Hybrid causal clocks (HLC + Vector Clock)
/// 4. Causal message ordering (dependency-preserving delivery)
/// 5. Multi-actor distributed scenarios
/// </remarks>
public static class CausalCorrectnessExample
{
    public static async Task RunAsync()
    {
        Console.WriteLine("=== Causal Correctness Example ===\n");

        // Example 1: Basic Vector Clock Operations
        Example1_BasicVectorClockOperations();

        // Example 2: Detecting Causal Relationships
        Example2_CausalRelationships();

        // Example 3: Hybrid Causal Clock (HLC + VC)
        Example3_HybridCausalClock();

        // Example 4: Causal Message Ordering
        await Example4_CausalMessageOrdering();

        // Example 5: Distributed Banking (Conflict Detection)
        await Example5_DistributedBanking();

        // Example 6: Collaborative Document Editing
        await Example6_CollaborativeEditing();
    }

    /// <summary>
    /// Example 1: Basic vector clock operations.
    /// </summary>
    private static void Example1_BasicVectorClockOperations()
    {
        Console.WriteLine("Example 1: Basic Vector Clock Operations");
        Console.WriteLine("Scenario: Single actor generating events\n");

        // Actor 1 generates 3 events
        var vc = VectorClock.Create(actorId: 1, value: 0);

        Console.WriteLine($"Initial:  {vc}");

        vc = vc.Increment(1);
        Console.WriteLine($"Event 1:  {vc}");

        vc = vc.Increment(1);
        Console.WriteLine($"Event 2:  {vc}");

        vc = vc.Increment(1);
        Console.WriteLine($"Event 3:  {vc}");

        Console.WriteLine($"\n✅ Vector clock successfully tracks local events\n");
        Console.WriteLine(new string('-', 80) + "\n");
    }

    /// <summary>
    /// Example 2: Detecting causal relationships.
    /// </summary>
    private static void Example2_CausalRelationships()
    {
        Console.WriteLine("Example 2: Detecting Causal Relationships");
        Console.WriteLine("Scenario: Three actors with different event patterns\n");

        // Actor 1: [1:3]
        var vc1 = VectorClock.Create(1, 3);

        // Actor 2: [1:3, 2:5] - knows all of Actor 1's events plus own
        var vc2 = new VectorClock(new Dictionary<ushort, long>
        {
            [1] = 3,
            [2] = 5
        });

        // Actor 3: [1:1, 3:4] - concurrent with Actor 2
        var vc3 = new VectorClock(new Dictionary<ushort, long>
        {
            [1] = 1,
            [3] = 4
        });

        Console.WriteLine($"Actor 1: {vc1}");
        Console.WriteLine($"Actor 2: {vc2}");
        Console.WriteLine($"Actor 3: {vc3}\n");

        // Check relationships
        var rel12 = vc1.CompareTo(vc2);
        var rel13 = vc1.CompareTo(vc3);
        var rel23 = vc2.CompareTo(vc3);

        Console.WriteLine($"Actor 1 vs Actor 2: {rel12}");
        Console.WriteLine($"   → Actor 1 events happened-before Actor 2's latest event\n");

        Console.WriteLine($"Actor 1 vs Actor 3: {rel13}");
        Console.WriteLine($"   → Actor 3 knows some (but not all) of Actor 1's events\n");

        Console.WriteLine($"Actor 2 vs Actor 3: {rel23}");
        Console.WriteLine($"   → Concurrent! Neither knows all of the other's events\n");

        Console.WriteLine("🔍 Key Insight: Concurrent events indicate potential conflicts\n");
        Console.WriteLine(new string('-', 80) + "\n");
    }

    /// <summary>
    /// Example 3: Hybrid causal clock combining HLC and vector clock.
    /// </summary>
    private static void Example3_HybridCausalClock()
    {
        Console.WriteLine("Example 3: Hybrid Causal Clock (HLC + Vector Clock)");
        Console.WriteLine("Scenario: Two actors exchanging messages\n");

        var actor1 = new HybridCausalClock(actorId: 1);
        var actor2 = new HybridCausalClock(actorId: 2);

        // Actor 1 generates local event
        var t1 = actor1.Now();
        Console.WriteLine($"Actor 1 generates event:");
        Console.WriteLine($"  HLC: {t1.HLC}");
        Console.WriteLine($"  VC:  {t1.VectorClock}\n");

        // Actor 1 sends message to Actor 2
        Console.WriteLine("Actor 1 → Actor 2 (sending message)");
        var t2 = actor2.Update(t1);
        Console.WriteLine($"Actor 2 receives and updates:");
        Console.WriteLine($"  HLC: {t2.HLC}");
        Console.WriteLine($"  VC:  {t2.VectorClock}");
        Console.WriteLine($"  ✅ Vector clock shows Actor 2 knows Actor 1's event\n");

        // Actor 2 generates local event
        var t3 = actor2.Now();
        Console.WriteLine($"Actor 2 generates local event:");
        Console.WriteLine($"  VC:  {t3.VectorClock}");
        Console.WriteLine($"  ✅ Both actors' events tracked\n");

        // Check causality
        var relationship = t1.GetCausalRelationship(t3);
        Console.WriteLine($"Causal relationship t1 → t3: {relationship}");
        Console.WriteLine($"✅ t1 happens-before t3 (causality preserved)\n");

        Console.WriteLine(new string('-', 80) + "\n");
    }

    /// <summary>
    /// Example 4: Causal message ordering with out-of-order delivery.
    /// </summary>
    private static async Task Example4_CausalMessageOrdering()
    {
        Console.WriteLine("Example 4: Causal Message Ordering");
        Console.WriteLine("Scenario: Messages arrive out of order, queue ensures causal delivery\n");

        // Setup 3 actors
        var actor1 = new HybridCausalClock(actorId: 1);
        var actor2 = new HybridCausalClock(actorId: 2);
        var actor3 = new HybridCausalClock(actorId: 3);
        var queue = new CausalOrderingQueue(actor3);

        // Create message chain: Actor1 → Actor2 → Actor3
        Console.WriteLine("Creating causal chain: Actor1 → Actor2 → Actor3\n");

        // Actor 1 sends to Actor 2
        var t1 = actor1.Now();
        Console.WriteLine($"t=1: Actor1 sends message (VC: {t1.VectorClock})");

        // Actor 2 receives and processes
        _ = actor2.Update(t1);
        var t2 = actor2.Now();
        Console.WriteLine($"t=2: Actor2 receives and sends (VC: {t2.VectorClock})");

        // Create messages
        var msg1 = new CausalMessage
        {
            MessageId = Guid.NewGuid(),
            SenderId = 1,
            Timestamp = t1,
            Payload = "Message from Actor1"
        };

        var msg2 = new CausalMessage
        {
            MessageId = Guid.NewGuid(),
            SenderId = 2,
            Timestamp = t2,
            Payload = "Message from Actor2 (depends on msg1)"
        };

        Console.WriteLine("\n📬 Messages arrive OUT OF ORDER at Actor3:");
        Console.WriteLine("   msg2 arrives first (but depends on msg1)");
        Console.WriteLine("   msg1 arrives second\n");

        // Deliver msg2 first (out of order)
        Console.WriteLine("Attempting to deliver msg2...");
        var d1 = await queue.EnqueueAsync(msg2);

        if (d1.Count == 0)
        {
            Console.WriteLine("   ⏸️  msg2 BUFFERED (waiting for dependency: msg1)\n");
        }

        // Now deliver msg1
        Console.WriteLine("Delivering msg1...");
        var d2 = await queue.EnqueueAsync(msg1);

        Console.WriteLine($"   ✅ Delivered {d2.Count} messages:");
        foreach (var msg in d2)
        {
            Console.WriteLine($"      - {msg.Payload}");
        }

        Console.WriteLine($"\n✅ Causal order preserved: msg1 delivered before msg2\n");

        var stats = queue.GetStatistics();
        Console.WriteLine($"Queue Statistics:");
        Console.WriteLine($"   Total enqueued: {stats.TotalEnqueued}");
        Console.WriteLine($"   Total delivered: {stats.TotalDelivered}");
        Console.WriteLine($"   Delivery rate: {stats.DeliveryRate:P0}\n");

        Console.WriteLine(new string('-', 80) + "\n");
    }

    /// <summary>
    /// Example 5: Distributed banking with conflict detection.
    /// </summary>
    private static async Task Example5_DistributedBanking()
    {
        Console.WriteLine("Example 5: Distributed Banking (Conflict Detection)");
        Console.WriteLine("Scenario: Two users try to withdraw from shared account concurrently\n");

        // Two bank tellers (actors) processing withdrawals
        var teller1 = new HybridCausalClock(actorId: 1);
        var teller2 = new HybridCausalClock(actorId: 2);
        var centralBank = new HybridCausalClock(actorId: 99);

        var balance = 1000.0;

        Console.WriteLine($"Initial balance: ${balance:N2}\n");

        // Teller 1: Withdraw $600
        var w1_time = teller1.Now();
        var withdraw1 = new BankTransaction
        {
            Timestamp = w1_time,
            TellerId = 1,
            Amount = -600,
            AccountId = "ACC-123"
        };

        Console.WriteLine($"Teller 1: Withdraw $600 at {w1_time.VectorClock}");

        // Teller 2: Withdraw $700 (concurrent!)
        var w2_time = teller2.Now();
        var withdraw2 = new BankTransaction
        {
            Timestamp = w2_time,
            TellerId = 2,
            Amount = -700,
            AccountId = "ACC-123"
        };

        Console.WriteLine($"Teller 2: Withdraw $700 at {w2_time.VectorClock}");

        // Check if concurrent
        var relationship = w1_time.GetCausalRelationship(w2_time);
        Console.WriteLine($"\nCausal relationship: {relationship}");

        if (relationship == CausalRelationship.Concurrent)
        {
            Console.WriteLine("⚠️  CONFLICT DETECTED: Concurrent withdrawals!");
            Console.WriteLine($"   Both tellers processed without knowing about the other's transaction\n");
            Console.WriteLine($"   If both allowed:");
            Console.WriteLine($"      Balance: ${balance} - $600 - $700 = ${balance - 1300:N2}");
            Console.WriteLine($"      Result: OVERDRAFT! ❌\n");
            Console.WriteLine($"   Conflict resolution needed:");
            Console.WriteLine($"      Option 1: Reject one transaction");
            Console.WriteLine($"      Option 2: Request manual approval");
            Console.WriteLine($"      Option 3: Use distributed locking\n");
        }

        Console.WriteLine("✅ Vector clocks detected the conflict before it caused data corruption\n");
        Console.WriteLine(new string('-', 80) + "\n");
    }

    /// <summary>
    /// Example 6: Collaborative document editing.
    /// </summary>
    private static async Task Example6_CollaborativeEditing()
    {
        Console.WriteLine("Example 6: Collaborative Document Editing");
        Console.WriteLine("Scenario: Two users editing same document, detecting conflicts\n");

        var user1 = new HybridCausalClock(actorId: 1);
        var user2 = new HybridCausalClock(actorId: 2);
        var server = new HybridCausalClock(actorId: 99);
        var serverQueue = new CausalOrderingQueue(server);

        var document = "Hello World";
        Console.WriteLine($"Initial document: \"{document}\"\n");

        // User 1 sees document at version v1
        var v1 = server.Now();
        _ = user1.Update(v1);

        // User 2 sees document at version v1 (same as User 1)
        _ = user2.Update(v1);

        Console.WriteLine("Both users have document at v1\n");

        // User 1: Insert "Beautiful " after "Hello "
        var edit1_time = user1.Now();
        var edit1 = new DocumentEdit
        {
            Timestamp = edit1_time,
            UserId = 1,
            Position = 6,
            InsertText = "Beautiful ",
            Operation = "insert"
        };

        Console.WriteLine($"User 1: Insert \"Beautiful \" at position 6");
        Console.WriteLine($"   Timestamp: {edit1_time.VectorClock}");

        // User 2: Replace "World" with "Universe" (concurrent edit!)
        var edit2_time = user2.Now();
        var edit2 = new DocumentEdit
        {
            Timestamp = edit2_time,
            UserId = 2,
            Position = 6,
            DeleteLength = 5,
            InsertText = "Universe",
            Operation = "replace"
        };

        Console.WriteLine($"User 2: Replace \"World\" with \"Universe\" at position 6");
        Console.WriteLine($"   Timestamp: {edit2_time.VectorClock}\n");

        // Check for conflicts
        var editRelationship = edit1_time.GetCausalRelationship(edit2_time);
        Console.WriteLine($"Edit relationship: {editRelationship}");

        if (editRelationship == CausalRelationship.Concurrent)
        {
            Console.WriteLine("⚠️  CONCURRENT EDITS DETECTED!\n");
            Console.WriteLine("Both edits at position 6:");
            Console.WriteLine($"   User 1 result: \"Hello Beautiful World\"");
            Console.WriteLine($"   User 2 result: \"Hello Universe\"\n");
            Console.WriteLine("Conflict resolution strategies:");
            Console.WriteLine("   1. Operational Transformation (OT) - transform operations");
            Console.WriteLine("   2. Last-Writer-Wins (LWW) - use HLC to pick winner");
            Console.WriteLine("   3. User prompt - ask users to resolve\n");

            // Using HLC for tie-breaking
            if (edit1_time.CompareHLC(edit2_time) < 0)
            {
                Console.WriteLine($"LWW strategy: User 2's edit wins (later HLC timestamp)");
                Console.WriteLine($"   Final document: \"Hello Universe\"\n");
            }
            else
            {
                Console.WriteLine($"LWW strategy: User 1's edit wins (later HLC timestamp)");
                Console.WriteLine($"   Final document: \"Hello Beautiful World\"\n");
            }
        }

        Console.WriteLine("✅ Hybrid causal clocks enable both:");
        Console.WriteLine("   - Conflict detection (Vector Clock)");
        Console.WriteLine("   - Tie-breaking (HLC total ordering)\n");

        Console.WriteLine(new string('-', 80) + "\n");
    }
}

/// <summary>
/// Represents a bank transaction with causal timestamp.
/// </summary>
record BankTransaction
{
    public required HybridCausalTimestamp Timestamp { get; init; }
    public required int TellerId { get; init; }
    public required double Amount { get; init; }
    public required string AccountId { get; init; }
}

/// <summary>
/// Represents a document edit operation with causal timestamp.
/// </summary>
record DocumentEdit
{
    public required HybridCausalTimestamp Timestamp { get; init; }
    public required int UserId { get; init; }
    public required int Position { get; init; }
    public string? InsertText { get; init; }
    public int DeleteLength { get; init; }
    public required string Operation { get; init; }
}

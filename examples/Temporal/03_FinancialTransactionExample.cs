using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using Orleans.GpuBridge.Abstractions.Temporal;
using Orleans.GpuBridge.Grains.Resident.Messages;
using Orleans.GpuBridge.Runtime.Temporal;

namespace Orleans.GpuBridge.Examples.Temporal;

/// <summary>
/// Example 3: Financial transaction graph with temporal pattern detection.
/// </summary>
/// <remarks>
/// This example demonstrates a real-world use case:
/// - Money transfer between accounts
/// - Rapid transaction splitting detection (suspicious pattern)
/// - Temporal graph queries ("what happened in the last 5 seconds?")
/// - Causal transaction chains (A→B→C)
/// </remarks>
public static class FinancialTransactionExample
{
    public static void Run()
    {
        Console.WriteLine("=== Example 3: Financial Transaction Graph ===\n");

        var system = new TransactionSystem();

        // Scenario: Suspicious transaction pattern
        // Account A sends $1000 to Account B
        // Account B immediately splits to C ($500) and D ($500)
        // This happens within 2 seconds - suspicious pattern!

        Console.WriteLine("Scenario: Rapid transaction splitting (suspicious pattern)");
        Console.WriteLine("A → B ($1000) → C ($500) + D ($500) within 2 seconds\n");

        // Transaction 1: A → B
        var tx1 = system.Transfer(
            from: "Account-A",
            to: "Account-B",
            amount: 1000m,
            description: "Initial transfer");

        Console.WriteLine($"Transaction 1: {tx1.Description}");
        Console.WriteLine($"  From: {tx1.FromAccount} → To: {tx1.ToAccount}");
        Console.WriteLine($"  Amount: ${tx1.Amount}");
        Console.WriteLine($"  Timestamp: {tx1.HLC.ToDetailedString()}");
        Console.WriteLine($"  Transaction ID: {tx1.TransactionId}\n");

        // Simulate 1 second delay
        System.Threading.Thread.Sleep(1000);

        // Transaction 2: B → C (depends on receiving from A)
        var tx2 = system.Transfer(
            from: "Account-B",
            to: "Account-C",
            amount: 500m,
            description: "Split transfer to C",
            dependsOn: tx1.TransactionId);

        Console.WriteLine($"Transaction 2: {tx2.Description}");
        Console.WriteLine($"  From: {tx2.FromAccount} → To: {tx2.ToAccount}");
        Console.WriteLine($"  Amount: ${tx2.Amount}");
        Console.WriteLine($"  Depends on: Transaction {tx1.TransactionId}");
        Console.WriteLine($"  Time since tx1: {tx2.HLC.GetDifferenceNanos(tx1.HLC) / 1_000_000.0:F0}ms\n");

        // Transaction 3: B → D (also depends on receiving from A)
        var tx3 = system.Transfer(
            from: "Account-B",
            to: "Account-D",
            amount: 500m,
            description: "Split transfer to D",
            dependsOn: tx1.TransactionId);

        Console.WriteLine($"Transaction 3: {tx3.Description}");
        Console.WriteLine($"  From: {tx3.FromAccount} → To: {tx3.ToAccount}");
        Console.WriteLine($"  Amount: ${tx3.Amount}");
        Console.WriteLine($"  Depends on: Transaction {tx1.TransactionId}");
        Console.WriteLine($"  Time since tx1: {tx3.HLC.GetDifferenceNanos(tx1.HLC) / 1_000_000.0:F0}ms\n");

        // Process transactions in order
        Console.WriteLine("--- Processing Transactions (Enforcing Causality) ---\n");
        system.ProcessPendingTransactions();

        // Analyze patterns
        Console.WriteLine("\n--- Pattern Detection ---\n");
        system.DetectSuspiciousPatterns();

        // Query transaction history
        Console.WriteLine("\n--- Transaction History Queries ---\n");
        system.QueryTransactionHistory();
    }

    /// <summary>
    /// Example showing circular transaction detection.
    /// </summary>
    public static void DemonstrateCircularTransactionDetection()
    {
        Console.WriteLine("\n=== Circular Transaction Detection ===\n");

        var system = new TransactionSystem();

        Console.WriteLine("Scenario: Potential money laundering (circular flow)");
        Console.WriteLine("A → B → C → A (money returns to origin)\n");

        // Create circular transaction chain
        var tx1 = system.Transfer("Account-A", "Account-B", 1000m, "A to B");
        System.Threading.Thread.Sleep(500);

        var tx2 = system.Transfer("Account-B", "Account-C", 950m, "B to C", tx1.TransactionId);
        System.Threading.Thread.Sleep(500);

        var tx3 = system.Transfer("Account-C", "Account-A", 900m, "C to A", tx2.TransactionId);

        Console.WriteLine($"Transaction chain:");
        Console.WriteLine($"  {tx1.FromAccount} → {tx1.ToAccount}: ${tx1.Amount}");
        Console.WriteLine($"  {tx2.FromAccount} → {tx2.ToAccount}: ${tx2.Amount}");
        Console.WriteLine($"  {tx3.FromAccount} → {tx3.ToAccount}: ${tx3.Amount}");

        system.ProcessPendingTransactions();

        // Detect circular flow
        var hasCircularFlow = system.DetectCircularFlow("Account-A");
        Console.WriteLine($"\n{'⚠'} Circular flow detected: {hasCircularFlow}");

        if (hasCircularFlow)
        {
            Console.WriteLine("  This pattern is suspicious and should be flagged for review!");
        }
    }
}

/// <summary>
/// Simplified transaction processing system with temporal correctness.
/// </summary>
internal sealed class TransactionSystem
{
    private readonly TemporalMessageQueue _queue;
    private readonly HybridLogicalClock _clock;
    private readonly List<Transaction> _processedTransactions = new();
    private ulong _nextSequenceNumber = 1;

    public TransactionSystem()
    {
        _queue = new TemporalMessageQueue();
        _clock = new HybridLogicalClock(nodeId: 1);
    }

    /// <summary>
    /// Creates a new money transfer transaction.
    /// </summary>
    public Transaction Transfer(
        string from,
        string to,
        decimal amount,
        string description,
        Guid? dependsOn = null)
    {
        var dependencies = dependsOn.HasValue
            ? ImmutableArray.Create(dependsOn.Value)
            : ImmutableArray<Guid>.Empty;

        var transaction = new Transaction
        {
            FromAccount = from,
            ToAccount = to,
            Amount = amount,
            Description = description,
            HLC = _clock.Now(),
            PhysicalTimeNanos = HybridTimestamp.GetCurrentPhysicalTimeNanos(),
            TimestampErrorBoundNanos = 1_000_000, // ±1ms
            CausalDependencies = dependencies,
            SequenceNumber = _nextSequenceNumber++,
            Priority = amount >= 10000m ? MessagePriority.High : MessagePriority.Normal
        };

        _queue.Enqueue(transaction);
        return transaction;
    }

    /// <summary>
    /// Processes all pending transactions in causal order.
    /// </summary>
    public void ProcessPendingTransactions()
    {
        while (_queue.TryDequeue(out var transaction))
        {
            Console.WriteLine($"✓ Processed: {transaction!.FromAccount} → {transaction.ToAccount} (${transaction.Amount})");
            _processedTransactions.Add((Transaction)transaction);
            _queue.MarkProcessed(transaction.RequestId);
        }

        var stats = _queue.GetStatistics();
        Console.WriteLine($"\nProcessing complete:");
        Console.WriteLine($"  Transactions processed: {stats.TotalDequeued}");
        Console.WriteLine($"  Dependency waits: {stats.TotalDependencyWaits}");
    }

    /// <summary>
    /// Detects suspicious transaction patterns.
    /// </summary>
    public void DetectSuspiciousPatterns()
    {
        // Pattern: Rapid splitting (one account receives money and quickly splits to multiple recipients)
        var rapidSplitThresholdNanos = 5_000_000_000; // 5 seconds

        foreach (var inbound in _processedTransactions)
        {
            var targetAccount = inbound.ToAccount;
            var inboundTime = inbound.HLC.PhysicalTime;

            // Find outbound transactions from this account within 5 seconds
            var outbounds = _processedTransactions
                .Where(tx => tx.FromAccount == targetAccount)
                .Where(tx => tx.HLC.PhysicalTime > inboundTime)
                .Where(tx => tx.HLC.PhysicalTime - inboundTime <= rapidSplitThresholdNanos)
                .ToList();

            if (outbounds.Count >= 2)
            {
                Console.WriteLine($"⚠ SUSPICIOUS PATTERN: Rapid splitting detected!");
                Console.WriteLine($"  Account: {targetAccount}");
                Console.WriteLine($"  Received: ${inbound.Amount} from {inbound.FromAccount}");
                Console.WriteLine($"  Split into {outbounds.Count} transactions within {rapidSplitThresholdNanos / 1_000_000_000.0}s:");

                foreach (var outbound in outbounds)
                {
                    var delay = (outbound.HLC.PhysicalTime - inboundTime) / 1_000_000.0;
                    Console.WriteLine($"    → {outbound.ToAccount}: ${outbound.Amount} ({delay:F0}ms later)");
                }

                Console.WriteLine($"  Total outbound: ${outbounds.Sum(tx => tx.Amount)}");
                Console.WriteLine($"  Recommendation: Flag for manual review");
            }
        }

        if (!_processedTransactions.Any(tx => HasRapidSplit(tx)))
        {
            Console.WriteLine("✓ No suspicious patterns detected");
        }
    }

    private bool HasRapidSplit(Transaction inbound)
    {
        var targetAccount = inbound.ToAccount;
        var inboundTime = inbound.HLC.PhysicalTime;
        var rapidSplitThresholdNanos = 5_000_000_000;

        var outboundCount = _processedTransactions
            .Count(tx => tx.FromAccount == targetAccount &&
                         tx.HLC.PhysicalTime > inboundTime &&
                         tx.HLC.PhysicalTime - inboundTime <= rapidSplitThresholdNanos);

        return outboundCount >= 2;
    }

    /// <summary>
    /// Queries transaction history with temporal constraints.
    /// </summary>
    public void QueryTransactionHistory()
    {
        if (!_processedTransactions.Any())
        {
            Console.WriteLine("No transactions in history");
            return;
        }

        // Find all transactions in the last 5 seconds
        var currentTime = HybridTimestamp.GetCurrentPhysicalTimeNanos();
        var windowNanos = 5_000_000_000; // 5 seconds

        var recentTransactions = _processedTransactions
            .Where(tx => currentTime - tx.HLC.PhysicalTime <= windowNanos)
            .OrderBy(tx => tx.HLC)
            .ToList();

        Console.WriteLine($"Transactions in last 5 seconds: {recentTransactions.Count}");

        // Find transactions by account
        var accountTransactions = _processedTransactions
            .Where(tx => tx.FromAccount == "Account-B" || tx.ToAccount == "Account-B")
            .ToList();

        Console.WriteLine($"Transactions involving Account-B: {accountTransactions.Count}");
        foreach (var tx in accountTransactions)
        {
            var direction = tx.FromAccount == "Account-B" ? "OUT" : "IN";
            Console.WriteLine($"  [{direction}] {tx.FromAccount} → {tx.ToAccount}: ${tx.Amount}");
        }

        // Find causal chains
        Console.WriteLine("\nCausal chains:");
        foreach (var tx in _processedTransactions.Where(t => !t.CausalDependencies.IsEmpty))
        {
            var dependency = _processedTransactions
                .FirstOrDefault(t => t.RequestId == tx.CausalDependencies[0]);

            if (dependency != null)
            {
                Console.WriteLine($"  {dependency.TransactionId} → {tx.TransactionId}");
                Console.WriteLine($"    {dependency.FromAccount}→{dependency.ToAccount} caused {tx.FromAccount}→{tx.ToAccount}");
            }
        }
    }

    /// <summary>
    /// Detects circular flow of money.
    /// </summary>
    public bool DetectCircularFlow(string originAccount)
    {
        // Find chains where money leaves and returns to the same account
        var outbound = _processedTransactions
            .Where(tx => tx.FromAccount == originAccount)
            .ToList();

        foreach (var start in outbound)
        {
            if (HasPathBack(start.ToAccount, originAccount, new HashSet<string> { originAccount }))
            {
                return true;
            }
        }

        return false;
    }

    private bool HasPathBack(string currentAccount, string targetAccount, HashSet<string> visited)
    {
        if (currentAccount == targetAccount)
            return true;

        if (visited.Contains(currentAccount))
            return false;

        visited.Add(currentAccount);

        var outgoing = _processedTransactions
            .Where(tx => tx.FromAccount == currentAccount)
            .ToList();

        foreach (var tx in outgoing)
        {
            if (HasPathBack(tx.ToAccount, targetAccount, visited))
                return true;
        }

        return false;
    }
}

/// <summary>
/// Financial transaction message.
/// </summary>
internal sealed record Transaction : TemporalResidentMessage
{
    public Guid TransactionId => RequestId;
    public string FromAccount { get; init; } = string.Empty;
    public string ToAccount { get; init; } = string.Empty;
    public decimal Amount { get; init; }
    public string Description { get; init; } = string.Empty;
}

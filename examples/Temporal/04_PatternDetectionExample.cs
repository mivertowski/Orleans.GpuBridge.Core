using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Orleans.GpuBridge.Abstractions.Temporal;
using Orleans.GpuBridge.Runtime.Temporal.Graph;
using Orleans.GpuBridge.Runtime.Temporal.Patterns;

namespace Orleans.GpuBridge.Examples.Temporal;

/// <summary>
/// Example demonstrating temporal pattern detection for financial fraud detection.
/// </summary>
/// <remarks>
/// This example shows:
/// 1. Setting up pattern detection with multiple patterns
/// 2. Detecting rapid transaction splitting (money laundering)
/// 3. Detecting circular money flows (layering)
/// 4. Detecting high-frequency trading patterns
/// 5. Detecting velocity changes (sudden large transactions)
/// </remarks>
public static class PatternDetectionExample
{
    public static async Task RunAsync()
    {
        Console.WriteLine("=== Temporal Pattern Detection Example ===\n");

        // Example 1: Rapid Split Pattern (Money Laundering Detection)
        await Example1_RapidSplitDetection();

        // Example 2: Circular Flow Pattern (Layering Detection)
        await Example2_CircularFlowDetection();

        // Example 3: High-Frequency Pattern (Bot Detection)
        await Example3_HighFrequencyDetection();

        // Example 4: Velocity Change Pattern (Suspicious Activity)
        await Example4_VelocityChangeDetection();

        // Example 5: Multi-Pattern Detection
        await Example5_MultiPatternDetection();
    }

    /// <summary>
    /// Example 1: Detecting rapid transaction splitting.
    /// </summary>
    /// <remarks>
    /// Pattern: A→B ($10,000), then B→{C,D,E} within 5 seconds
    /// This is suspicious because legitimate transfers don't split that quickly.
    /// </remarks>
    private static async Task Example1_RapidSplitDetection()
    {
        Console.WriteLine("Example 1: Rapid Split Detection");
        Console.WriteLine("Scenario: Account B receives $10,000 and immediately splits it\n");

        var detector = new TemporalPatternDetector();
        detector.RegisterPattern(new RapidSplitPattern(
            windowSizeNanos: 5_000_000_000,  // 5 seconds
            minimumSplits: 2,
            minimumAmount: 1000.0));

        // Inbound: A → B ($10,000)
        Console.WriteLine("t=0s:  A → B ($10,000) [Inbound transaction]");
        await detector.ProcessEventAsync(CreateTransaction(
            timestampSec: 0,
            sourceId: 1, // A
            targetId: 2, // B
            amount: 10_000));

        // B splits money to C, D, E within 3 seconds
        Console.WriteLine("t=1s:  B → C ($3,500)  [Split 1]");
        await detector.ProcessEventAsync(CreateTransaction(
            timestampSec: 1,
            sourceId: 2, // B
            targetId: 3, // C
            amount: 3_500));

        Console.WriteLine("t=2s:  B → D ($3,500)  [Split 2]");
        await detector.ProcessEventAsync(CreateTransaction(
            timestampSec: 2,
            sourceId: 2, // B
            targetId: 4, // D
            amount: 3_500));

        Console.WriteLine("t=3s:  B → E ($3,000)  [Split 3]");
        var matches = await detector.ProcessEventAsync(CreateTransaction(
            timestampSec: 3,
            sourceId: 2, // B
            targetId: 5, // E
            amount: 3_000));

        // Display results
        if (matches.Any())
        {
            var match = matches.First();
            Console.WriteLine($"\n⚠️  PATTERN DETECTED: {match.PatternName}");
            Console.WriteLine($"   Severity: {match.Severity}");
            Console.WriteLine($"   Confidence: {match.Confidence:P0}");
            Console.WriteLine($"   Account: {match.Metadata["account"]}");
            Console.WriteLine($"   Inbound: ${match.Metadata["inbound_amount"]:N2}");
            Console.WriteLine($"   Splits: {match.Metadata["outbound_count"]} transactions");
            Console.WriteLine($"   Total outbound: ${match.Metadata["total_outbound"]:N2}");
            Console.WriteLine($"   Split ratio: {match.Metadata["split_ratio"]:P0}");
            Console.WriteLine($"   Max delay: {match.Metadata["max_delay_ms"]:F0}ms\n");
        }
        else
        {
            Console.WriteLine("\n✅ No suspicious patterns detected\n");
        }

        Console.WriteLine(new string('-', 80) + "\n");
    }

    /// <summary>
    /// Example 2: Detecting circular money flow.
    /// </summary>
    /// <remarks>
    /// Pattern: A→B→C→A (money returns to origin)
    /// This is a classic layering technique in money laundering.
    /// </remarks>
    private static async Task Example2_CircularFlowDetection()
    {
        Console.WriteLine("Example 2: Circular Flow Detection");
        Console.WriteLine("Scenario: Money travels in a circle A→B→C→A\n");

        var graph = new TemporalGraphStorage();
        var detector = new TemporalPatternDetector(graph: graph);
        detector.RegisterPattern(new CircularFlowPattern(
            windowSizeNanos: 60_000_000_000, // 60 seconds
            minimumHops: 3));

        // Create circular flow
        Console.WriteLine("t=0s:   A → B ($50,000)");
        await detector.ProcessEventAsync(CreateTransaction(0, 1, 2, 50_000));

        Console.WriteLine("t=10s:  B → C ($48,000)  [2% loss]");
        await detector.ProcessEventAsync(CreateTransaction(10, 2, 3, 48_000));

        Console.WriteLine("t=20s:  C → D ($46,000)  [Additional hop]");
        await detector.ProcessEventAsync(CreateTransaction(20, 3, 4, 46_000));

        Console.WriteLine("t=30s:  D → A ($44,000)  [Returns to origin]");
        var matches = await detector.ProcessEventAsync(CreateTransaction(30, 4, 1, 44_000));

        if (matches.Any())
        {
            var match = matches.First();
            Console.WriteLine($"\n🚨 CRITICAL PATTERN DETECTED: {match.PatternName}");
            Console.WriteLine($"   Severity: {match.Severity}");
            Console.WriteLine($"   Origin account: {match.Metadata["origin_account"]}");
            Console.WriteLine($"   Path length: {match.Metadata["path_length"]} hops");
            Console.WriteLine($"   Path: {match.Metadata["path_nodes"]}");
            Console.WriteLine($"   Total amount lost: ${50_000 - (double)match.Metadata["total_amount"]:N2}");
            Console.WriteLine($"   Money returned: ${match.Metadata["total_amount"]:N2}\n");
        }

        Console.WriteLine(new string('-', 80) + "\n");
    }

    /// <summary>
    /// Example 3: Detecting high-frequency transaction patterns.
    /// </summary>
    /// <remarks>
    /// Pattern: >10 transactions per second from same source
    /// May indicate bot activity or automated trading.
    /// </remarks>
    private static async Task Example3_HighFrequencyDetection()
    {
        Console.WriteLine("Example 3: High-Frequency Pattern Detection");
        Console.WriteLine("Scenario: Account A makes 20 transactions in 500ms\n");

        var detector = new TemporalPatternDetector();
        detector.RegisterPattern(new HighFrequencyPattern(
            windowSizeNanos: 1_000_000_000,  // 1 second
            minimumTransactions: 10,
            minimumTotalAmount: 10_000));

        Console.WriteLine("t=0.000s - t=0.500s: Burst of 20 transactions (40 tx/sec)");

        var matches = new List<PatternMatch>();
        for (int i = 0; i < 20; i++)
        {
            var newMatches = await detector.ProcessEventAsync(CreateTransaction(
                timestampSec: 0.0 + i * 0.025, // 25ms apart
                sourceId: 1,
                targetId: (ulong)(i + 2),
                amount: 1000));
            matches.AddRange(newMatches);
        }

        if (matches.Any())
        {
            var match = matches.First(m => m.PatternId == "high-frequency");
            Console.WriteLine($"\n⚠️  PATTERN DETECTED: {match.PatternName}");
            Console.WriteLine($"   Severity: {match.Severity}");
            Console.WriteLine($"   Source account: {match.Metadata["source_account"]}");
            Console.WriteLine($"   Transaction count: {match.Metadata["transaction_count"]}");
            Console.WriteLine($"   Total amount: ${match.Metadata["total_amount"]:N2}");
            Console.WriteLine($"   Rate: {match.Metadata["transactions_per_second"]:F1} tx/sec");
            Console.WriteLine($"   Average per tx: ${match.Metadata["average_amount"]:N2}\n");
        }

        Console.WriteLine(new string('-', 80) + "\n");
    }

    /// <summary>
    /// Example 4: Detecting velocity changes.
    /// </summary>
    /// <remarks>
    /// Pattern: Large transaction after period of inactivity
    /// May indicate account takeover or fraud.
    /// </remarks>
    private static async Task Example4_VelocityChangeDetection()
    {
        Console.WriteLine("Example 4: Velocity Change Detection");
        Console.WriteLine("Scenario: Account inactive for 40 minutes, then $100,000 transfer\n");

        var detector = new TemporalPatternDetector();
        detector.RegisterPattern(new VelocityChangePattern(
            windowSizeNanos: 3600_000_000_000, // 1 hour
            thresholdAmount: 50_000,
            inactivityPeriod: 1800_000_000_000)); // 30 minutes

        Console.WriteLine("t=0min:   A → B ($50)    [Normal small transaction]");
        await detector.ProcessEventAsync(CreateTransaction(0, 1, 2, 50));

        Console.WriteLine("t=40min:  A → C ($100,000) [LARGE transaction after inactivity]");
        var matches = await detector.ProcessEventAsync(CreateTransaction(
            timestampSec: 40 * 60, // 40 minutes
            sourceId: 1,
            targetId: 3,
            amount: 100_000));

        if (matches.Any())
        {
            var match = matches.First();
            Console.WriteLine($"\n⚠️  PATTERN DETECTED: {match.PatternName}");
            Console.WriteLine($"   Severity: {match.Severity}");
            Console.WriteLine($"   Confidence: {match.Confidence:P0}");
            Console.WriteLine($"   Source account: {match.Metadata["source_account"]}");
            Console.WriteLine($"   Inactivity period: {(double)match.Metadata["inactivity_seconds"] / 60:F1} minutes");
            Console.WriteLine($"   Transaction amount: ${match.Metadata["transaction_amount"]:N2}");
            Console.WriteLine($"   Previous amount: ${match.Metadata["previous_amount"]:N2}\n");
        }

        Console.WriteLine(new string('-', 80) + "\n");
    }

    /// <summary>
    /// Example 5: Multi-pattern detection on same data.
    /// </summary>
    /// <remarks>
    /// Demonstrates detecting multiple patterns simultaneously.
    /// </remarks>
    private static async Task Example5_MultiPatternDetection()
    {
        Console.WriteLine("Example 5: Multi-Pattern Detection");
        Console.WriteLine("Scenario: Complex suspicious activity triggering multiple patterns\n");

        var graph = new TemporalGraphStorage();
        var detector = new TemporalPatternDetector(graph: graph);

        // Register all patterns
        detector.RegisterPattern(new RapidSplitPattern());
        detector.RegisterPattern(new CircularFlowPattern());
        detector.RegisterPattern(new HighFrequencyPattern(minimumTransactions: 5));
        detector.RegisterPattern(new VelocityChangePattern());

        Console.WriteLine("Creating complex transaction network...");

        // Simulate complex suspicious activity
        var allMatches = new List<PatternMatch>();

        // Small transaction followed by large one (velocity change)
        allMatches.AddRange(await detector.ProcessEventAsync(
            CreateTransaction(0, 1, 2, 100)));

        allMatches.AddRange(await detector.ProcessEventAsync(
            CreateTransaction(2000, 1, 3, 75_000)));

        // High frequency burst from account 3
        for (int i = 0; i < 8; i++)
        {
            allMatches.AddRange(await detector.ProcessEventAsync(
                CreateTransaction(2001 + i * 0.1, 3, (ulong)(10 + i), 2000)));
        }

        // Rapid split: account 2 splits incoming money
        allMatches.AddRange(await detector.ProcessEventAsync(
            CreateTransaction(2010, 1, 2, 20_000)));

        allMatches.AddRange(await detector.ProcessEventAsync(
            CreateTransaction(2011, 2, 4, 10_000)));

        allMatches.AddRange(await detector.ProcessEventAsync(
            CreateTransaction(2012, 2, 5, 10_000)));

        // Circular flow: 1→2→3→1
        allMatches.AddRange(await detector.ProcessEventAsync(
            CreateTransaction(3000, 1, 2, 30_000)));

        allMatches.AddRange(await detector.ProcessEventAsync(
            CreateTransaction(3010, 2, 3, 28_000)));

        allMatches.AddRange(await detector.ProcessEventAsync(
            CreateTransaction(3020, 3, 1, 26_000)));

        // Display all detected patterns
        var detectedPatterns = allMatches
            .GroupBy(m => m.PatternId)
            .ToList();

        Console.WriteLine($"\n📊 Detection Summary:");
        Console.WriteLine($"   Total patterns detected: {allMatches.Count}");
        Console.WriteLine($"   Unique pattern types: {detectedPatterns.Count}\n");

        foreach (var group in detectedPatterns.OrderByDescending(g => g.First().Severity))
        {
            var first = group.First();
            Console.WriteLine($"   • {first.PatternName}");
            Console.WriteLine($"     Severity: {first.Severity}");
            Console.WriteLine($"     Occurrences: {group.Count()}");
            Console.WriteLine();
        }

        var stats = detector.GetStatistics();
        Console.WriteLine($"Performance Statistics:");
        Console.WriteLine($"   Events processed: {stats.TotalEventsProcessed}");
        Console.WriteLine($"   Pattern checks: {stats.TotalPatternChecks}");
        Console.WriteLine($"   Detection rate: {stats.DetectionRate:P2}\n");

        Console.WriteLine(new string('-', 80) + "\n");
    }

    /// <summary>
    /// Helper to create transaction events.
    /// </summary>
    private static TemporalEvent CreateTransaction(
        double timestampSec,
        ulong sourceId,
        ulong targetId,
        double amount)
    {
        return new TemporalEvent
        {
            EventId = Guid.NewGuid(),
            EventType = "transaction",
            TimestampNanos = (long)(timestampSec * 1_000_000_000),
            SourceId = sourceId,
            TargetId = targetId,
            Value = amount
        };
    }
}

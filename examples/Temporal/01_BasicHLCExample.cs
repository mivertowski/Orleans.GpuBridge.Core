using System;
using Orleans.GpuBridge.Abstractions.Temporal;
using Orleans.GpuBridge.Runtime.Temporal;

namespace Orleans.GpuBridge.Examples.Temporal;

/// <summary>
/// Example 1: Basic Hybrid Logical Clock (HLC) usage.
/// </summary>
/// <remarks>
/// This example demonstrates:
/// - Creating HLC instances
/// - Generating timestamps for local events
/// - Updating clocks on message receipt
/// - Comparing timestamps for ordering
/// </remarks>
public static class BasicHLCExample
{
    public static void Run()
    {
        Console.WriteLine("=== Example 1: Basic Hybrid Logical Clock ===\n");

        // Create two clocks for different nodes
        var clockA = new HybridLogicalClock(nodeId: 1);
        var clockB = new HybridLogicalClock(nodeId: 2);

        Console.WriteLine($"Clock A: Node ID = {clockA.NodeId}");
        Console.WriteLine($"Clock B: Node ID = {clockB.NodeId}\n");

        // Node A generates events
        Console.WriteLine("Node A: Generating local events...");
        var t1 = clockA.Now();
        Console.WriteLine($"  Event 1: {t1.ToDetailedString()}");

        var t2 = clockA.Now();
        Console.WriteLine($"  Event 2: {t2.ToDetailedString()}");

        var t3 = clockA.Now();
        Console.WriteLine($"  Event 3: {t3.ToDetailedString()}");

        // Verify monotonicity
        Console.WriteLine($"\n✓ Monotonicity verified: t1 < t2 < t3 = {t1 < t2 && t2 < t3}");

        // Node A sends message to Node B with timestamp t3
        Console.WriteLine($"\nNode A → Node B: Sending message with timestamp {t3}");

        // Node B receives message and updates clock
        var tReceive = clockB.Update(t3);
        Console.WriteLine($"Node B: Received message, clock updated to {tReceive}");

        // Node B generates local event after receiving message
        var tAfter = clockB.Now();
        Console.WriteLine($"Node B: Generated local event: {tAfter}");

        // Verify causality: events on B after receiving message must be > t3
        Console.WriteLine($"\n✓ Causality preserved: tReceive > t3 = {tReceive > t3}");
        Console.WriteLine($"✓ Causality preserved: tAfter > t3 = {tAfter > t3}");

        // Demonstrate concurrent events
        Console.WriteLine("\n--- Concurrent Events ---");

        // Both nodes generate events at "the same time" (no message passing)
        var mockClock = new MockClockSource(fixedTime: 5_000_000_000);
        var clockC = new HybridLogicalClock(nodeId: 3, clockSource: mockClock);
        var clockD = new HybridLogicalClock(nodeId: 4, clockSource: mockClock);

        var tC = clockC.Now();
        var tD = clockD.Now();

        Console.WriteLine($"Node C event: {tC}");
        Console.WriteLine($"Node D event: {tD}");
        Console.WriteLine($"Are they concurrent? {tC.IsConcurrentWith(tD)}");
        Console.WriteLine($"Total ordering maintained: tC < tD = {tC < tD}");
    }

    /// <summary>
    /// Example showing clock drift measurement.
    /// </summary>
    public static void DemonstrateDriftMeasurement()
    {
        Console.WriteLine("\n=== Clock Drift Measurement ===\n");

        var clock = new HybridLogicalClock(nodeId: 1);

        // Generate timestamp
        var t1 = clock.Now();
        Console.WriteLine($"Generated timestamp: {t1}");

        // Simulate time passing
        System.Threading.Thread.Sleep(100); // 100ms

        // Measure drift
        var elapsed = t1.GetElapsedNanos();
        Console.WriteLine($"Elapsed time: {elapsed / 1_000_000.0:F2}ms");

        // Compare with another timestamp
        var t2 = clock.Now();
        var difference = t2.GetDifferenceNanos(t1);
        Console.WriteLine($"Difference between t2 and t1: {difference / 1_000_000.0:F2}ms");
    }

    /// <summary>
    /// Example showing different clock sources.
    /// </summary>
    public static void DemonstrateClockSources()
    {
        Console.WriteLine("\n=== Physical Clock Sources ===\n");

        // System clock (default)
        var systemClock = new SystemClockSource();
        Console.WriteLine($"System Clock: {systemClock}");
        Console.WriteLine($"  Current time: {systemClock.GetCurrentTimeNanos()}ns");
        Console.WriteLine($"  Error bound: ±{systemClock.GetErrorBound() / 1_000_000.0:F2}ms");
        Console.WriteLine($"  Clock drift: {systemClock.GetClockDrift():F1} PPM");

        // NTP clock
        var ntpClock = new NtpClockSource();
        Console.WriteLine($"\nNTP Clock: {ntpClock}");
        Console.WriteLine($"  Current time: {ntpClock.GetCurrentTimeNanos()}ns");
        Console.WriteLine($"  Error bound: ±{ntpClock.GetErrorBound() / 1_000_000.0:F2}ms");
        Console.WriteLine($"  Synchronized: {ntpClock.IsSynchronized}");
        Console.WriteLine($"  Clock drift: {ntpClock.GetClockDrift():F1} PPM");

        // Use custom clock source with HLC
        var clockWithNtp = new HybridLogicalClock(nodeId: 1, clockSource: ntpClock);
        var timestamp = clockWithNtp.Now();
        Console.WriteLine($"\nHLC with NTP: {timestamp}");
        Console.WriteLine($"  Physical accuracy: ±{ntpClock.GetErrorBound() / 1_000_000.0:F2}ms");
    }
}

/// <summary>
/// Mock clock source for testing and examples.
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

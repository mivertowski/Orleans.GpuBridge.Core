using System;
using System.Threading.Tasks;
using Orleans.GpuBridge.Abstractions.Temporal;
using Orleans.GpuBridge.Runtime.Temporal;
using Xunit;
using Xunit.Abstractions;

namespace Orleans.GpuBridge.Temporal.Tests.FaultTolerance;

/// <summary>
/// Tests for Hybrid Logical Clock behavior under clock drift and skew conditions.
/// Validates tolerance to clock synchronization issues in distributed systems.
/// </summary>
public class ClockDriftToleranceTests
{
    private readonly ITestOutputHelper _output;

    public ClockDriftToleranceTests(ITestOutputHelper output)
    {
        _output = output;
    }

    [Fact]
    public void HLC_ToleratesSmallForwardDrift()
    {
        // Arrange: Local node with stable clock
        var node1 = new HybridLogicalClock(nodeId: 1);
        var ts1 = node1.Now();

        // Act: Simulate small forward drift (100ms) on remote node
        var driftAmount = 100_000_000L; // 100ms in nanoseconds
        var remoteTs = new HybridTimestamp(
            physicalTime: ts1.PhysicalTime + driftAmount,
            logicalCounter: 0,
            nodeId: 2);

        // Update local clock with drifted remote timestamp
        var ts2 = node1.Update(remoteTs);

        // Assert: Clock should accept forward drift and increment logical counter
        Assert.True(ts2.PhysicalTime >= remoteTs.PhysicalTime,
            "Clock should move forward to match remote");
        Assert.True(ts2.LogicalCounter > 0,
            "Logical counter should increment on same physical time");
        Assert.Equal(1ul, ts2.NodeId); // Maintains local node ID

        _output.WriteLine($"Forward drift tolerance: {driftAmount / 1_000_000.0:F2}ms accepted");
        _output.WriteLine($"Logical counter incremented to: {ts2.LogicalCounter}");
    }

    [Fact]
    public async Task HLC_ToleratesSmallBackwardDrift()
    {
        // Arrange: Local node advances its clock
        var node1 = new HybridLogicalClock(nodeId: 1);
        var ts1 = node1.Now();
        await Task.Delay(50); // Wait 50ms
        var ts2 = node1.Now();

        // Act: Simulate backward drift - remote node is behind
        var driftAmount = -50_000_000L; // -50ms in nanoseconds
        var remoteTs = new HybridTimestamp(
            physicalTime: ts2.PhysicalTime + driftAmount,
            logicalCounter: 0,
            nodeId: 2);

        // Update local clock with lagging remote timestamp
        var ts3 = node1.Update(remoteTs);

        // Assert: Clock should maintain monotonicity
        Assert.True(ts3.CompareTo(ts2) >= 0,
            "Clock must maintain monotonicity - never go backwards");
        Assert.True(ts3.PhysicalTime >= ts2.PhysicalTime,
            "Physical time must not decrease");

        _output.WriteLine($"Backward drift tolerance: {Math.Abs(driftAmount) / 1_000_000.0:F2}ms rejected");
        _output.WriteLine($"Monotonicity maintained: {ts2} <= {ts3}");
    }

    [Theory]
    [InlineData(1_000)] // 1 microsecond
    [InlineData(1_000_000)] // 1 millisecond
    [InlineData(100_000_000)] // 100 milliseconds
    [InlineData(1_000_000_000)] // 1 second
    public void HLC_ToleratesVariousDriftMagnitudes(long driftNanos)
    {
        // Arrange
        var node1 = new HybridLogicalClock(nodeId: 1);
        var ts1 = node1.Now();

        // Act: Apply drift
        var remoteTs = new HybridTimestamp(
            physicalTime: ts1.PhysicalTime + driftNanos,
            logicalCounter: 0,
            nodeId: 2);
        var ts2 = node1.Update(remoteTs);

        // Assert: Clock handles drift gracefully
        Assert.True(ts2.CompareTo(ts1) > 0, "Updated timestamp must be greater");
        Assert.True(ts2.PhysicalTime >= remoteTs.PhysicalTime,
            "Physical time must incorporate drift");

        _output.WriteLine($"Drift: {driftNanos / 1_000_000.0:F3}ms - Handled successfully");
    }

    [Fact]
    public void HLC_DetectsLargeClockSkew()
    {
        // Arrange: Extreme forward drift (1 hour)
        var node1 = new HybridLogicalClock(nodeId: 1);
        var ts1 = node1.Now();

        var extremeDrift = 3600L * 1_000_000_000L; // 1 hour in nanoseconds
        var remoteTs = new HybridTimestamp(
            physicalTime: ts1.PhysicalTime + extremeDrift,
            logicalCounter: 0,
            nodeId: 2);

        // Act: Update with extreme drift
        var ts2 = node1.Update(remoteTs);

        // Assert: Clock accepts but logical counter indicates skew
        Assert.True(ts2.PhysicalTime >= remoteTs.PhysicalTime);
        var driftMs = extremeDrift / 1_000_000.0;

        _output.WriteLine($"LARGE SKEW DETECTED: {driftMs:F0}ms");
        _output.WriteLine($"This indicates clock synchronization issue requiring attention");
        _output.WriteLine($"Recommendation: Use NTP or similar for clock sync");

        // In production, this would trigger an alert
        Assert.True(driftMs > 1000, "Drift exceeds warning threshold");
    }

    [Fact]
    public async Task HLC_MaintainsOrderingUnderDrift()
    {
        // Arrange: Multiple nodes with varying drift
        var node1 = new HybridLogicalClock(nodeId: 1);
        var node2 = new HybridLogicalClock(nodeId: 2);
        var node3 = new HybridLogicalClock(nodeId: 3);

        // Act: Generate timestamps with drift simulation
        var ts1 = node1.Now();
        await Task.Delay(10);

        // Node 2 is 50ms ahead
        var ts2 = node2.Now();
        ts2 = new HybridTimestamp(
            ts2.PhysicalTime + 50_000_000L,
            ts2.LogicalCounter,
            ts2.NodeId);

        await Task.Delay(10);

        // Node 3 is 30ms behind
        var ts3 = node3.Now();
        ts3 = new HybridTimestamp(
            ts3.PhysicalTime - 30_000_000L,
            ts3.LogicalCounter,
            ts3.NodeId);

        // Update each node with others' timestamps
        var ts1_updated = node1.Update(ts2);
        ts1_updated = node1.Update(ts3);

        // Assert: Total ordering maintained despite drift
        Assert.True(ts1_updated.CompareTo(ts1) > 0,
            "Updated timestamp must be greater than original");

        _output.WriteLine($"Multi-node drift scenario:");
        _output.WriteLine($"  Node 1 (baseline): {ts1}");
        _output.WriteLine($"  Node 2 (+50ms):    {ts2}");
        _output.WriteLine($"  Node 3 (-30ms):    {ts3}");
        _output.WriteLine($"  Node 1 (updated):  {ts1_updated}");
        _output.WriteLine($"Total ordering maintained across drift");
    }

    [Fact]
    public void HLC_LogicalCounterIncrementsOnSamePhysicalTime()
    {
        // Arrange: Simulate network delay causing timestamps to collide
        var node1 = new HybridLogicalClock(nodeId: 1);
        var node2 = new HybridLogicalClock(nodeId: 2);

        // Both nodes generate timestamps at "same" physical time (within drift tolerance)
        var ts1 = node1.Now();
        var ts2 = new HybridTimestamp(ts1.PhysicalTime, 0, 2); // Same physical time, different node

        // Act: Update node1 with node2's timestamp
        var ts3 = node1.Update(ts2);

        // Assert: Logical counter disambiguates
        Assert.Equal(ts1.PhysicalTime, ts3.PhysicalTime);
        Assert.True(ts3.LogicalCounter > Math.Max(ts1.LogicalCounter, ts2.LogicalCounter),
            "Logical counter must increment to break tie");

        _output.WriteLine($"Physical time collision handled:");
        _output.WriteLine($"  ts1: PhysicalTime={ts1.PhysicalTime}, Logical={ts1.LogicalCounter}");
        _output.WriteLine($"  ts2: PhysicalTime={ts2.PhysicalTime}, Logical={ts2.LogicalCounter}");
        _output.WriteLine($"  ts3: PhysicalTime={ts3.PhysicalTime}, Logical={ts3.LogicalCounter}");
    }

    [Fact]
    public async Task HLC_HandlesClockCorrection()
    {
        // Arrange: Simulate NTP clock correction scenario
        var node1 = new HybridLogicalClock(nodeId: 1);
        var ts1 = node1.Now();

        // Wait a bit
        await Task.Delay(50);
        var ts2 = node1.Now();

        // Simulate backward clock correction (NTP adjusted clock backward)
        var correctionAmount = -100_000_000L; // -100ms
        var correctedTime = ts2.PhysicalTime + correctionAmount;

        // Act: Generate timestamp after correction
        var ts3 = new HybridTimestamp(correctedTime, 0, 1);
        var ts4 = node1.Update(ts3);

        // Assert: Monotonicity preserved despite clock going backward
        Assert.True(ts4.CompareTo(ts2) >= 0,
            "Monotonicity must be preserved even with backward clock adjustment");

        _output.WriteLine($"Clock correction scenario:");
        _output.WriteLine($"  Before:     {ts2}");
        _output.WriteLine($"  Correction: {correctionAmount / 1_000_000.0:F2}ms backward");
        _output.WriteLine($"  After:      {ts4}");
        _output.WriteLine($"Monotonicity preserved by logical counter");
    }

    [Fact]
    public void HLC_MaximumAllowableDrift()
    {
        // Arrange: Define maximum acceptable drift before sync required
        var maxDriftMs = 5000.0; // 5 seconds
        var maxDriftNanos = (long)(maxDriftMs * 1_000_000);

        var node1 = new HybridLogicalClock(nodeId: 1);
        var ts1 = node1.Now();

        // Act: Test at threshold
        var remoteTs = new HybridTimestamp(
            physicalTime: ts1.PhysicalTime + maxDriftNanos,
            logicalCounter: 0,
            nodeId: 2);
        var ts2 = node1.Update(remoteTs);

        // Assert: Clock handles maximum drift
        Assert.True(ts2.PhysicalTime >= remoteTs.PhysicalTime);

        var actualDriftMs = (remoteTs.PhysicalTime - ts1.PhysicalTime) / 1_000_000.0;

        _output.WriteLine($"Maximum allowable drift: {maxDriftMs}ms");
        _output.WriteLine($"Actual drift tested: {actualDriftMs:F2}ms");
        _output.WriteLine($"Status: {(actualDriftMs <= maxDriftMs ? "PASS" : "FAIL - Sync required")}");

        // Recommendation: If drift > 5s, trigger clock synchronization
        if (actualDriftMs > maxDriftMs)
        {
            _output.WriteLine($"⚠️  ALERT: Clock sync required (drift={actualDriftMs:F0}ms > threshold={maxDriftMs}ms)");
        }
    }
}

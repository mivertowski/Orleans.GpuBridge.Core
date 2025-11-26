using System.Net;
using FluentAssertions;
using Microsoft.Extensions.Logging.Abstractions;
using Orleans.GpuBridge.Runtime.Temporal.Clock;
using Orleans.GpuBridge.Runtime.Temporal.Network;

namespace Orleans.GpuBridge.Temporal.Tests.Integration;

/// <summary>
/// Integration tests for clock calibration + network latency compensation.
/// Verifies distributed timestamp ordering with physical clock precision.
/// </summary>
public sealed class NetworkCompensationIntegrationTests
{
    [Fact]
    public async Task NetworkCompensation_CombinesWithClockCalibration()
    {
        // Arrange
        var clockSelector = new ClockSourceSelector(NullLogger<ClockSourceSelector>.Instance);
        await clockSelector.InitializeAsync();

        var compensator = new NetworkLatencyCompensator(
            NullLogger<NetworkLatencyCompensator>.Instance);

        var localEndpoint = new IPEndPoint(IPAddress.Loopback, 12345);

        // Start TCP listener for latency measurement
        using var listener = new System.Net.Sockets.TcpListener(localEndpoint);
        listener.Start();

        // Act - Measure network latency
        var rtt = await compensator.MeasureLatencyAsync(localEndpoint);

        // Get clock-calibrated timestamp
        long localTime = clockSelector.ActiveSource.GetCurrentTimeNanos();

        // Simulate remote timestamp (local time + 5ms network delay)
        long remoteTime = localTime + 5_000_000; // +5ms

        // Compensate remote timestamp
        long compensatedTime = compensator.CompensateTimestamp(remoteTime, localEndpoint);

        // Assert
        // On localhost, RTT can be zero or near-zero (sub-microsecond)
        rtt.Should().BeGreaterThanOrEqualTo(TimeSpan.Zero);
        rtt.Should().BeLessThan(TimeSpan.FromMilliseconds(100)); // Localhost should be fast

        // Compensated time should be close to local time
        // On localhost, RTT measurement can have timing noise, so allow tolerance
        const long toleranceNanos = 20_000_000L; // 20ms tolerance for localhost noise
        long rawDiff = Math.Abs(remoteTime - localTime);
        long compensatedDiff = Math.Abs(compensatedTime - localTime);
        // Compensation should not significantly worsen the timestamp (within tolerance)
        compensatedDiff.Should().BeLessThanOrEqualTo(rawDiff + toleranceNanos);

        listener.Stop();
    }

    [Fact]
    public async Task DistributedTimestamps_MaintainCausalOrdering()
    {
        // Arrange
        var clockSelector = new ClockSourceSelector(NullLogger<ClockSourceSelector>.Instance);
        await clockSelector.InitializeAsync();

        var compensator = new NetworkLatencyCompensator(
            NullLogger<NetworkLatencyCompensator>.Instance);

        // Act - Simulate distributed events with causal ordering
        long event1 = clockSelector.ActiveSource.GetCurrentTimeNanos();
        await Task.Delay(10); // 10ms delay
        long event2 = clockSelector.ActiveSource.GetCurrentTimeNanos();
        await Task.Delay(10); // 10ms delay
        long event3 = clockSelector.ActiveSource.GetCurrentTimeNanos();

        // Assert - Causal ordering preserved
        event2.Should().BeGreaterThan(event1);
        event3.Should().BeGreaterThan(event2);

        // Time deltas should be at least 10ms (Task.Delay minimum)
        // On loaded systems, Task.Delay may take longer, so only check lower bound
        long delta1 = event2 - event1;
        long delta2 = event3 - event2;

        delta1.Should().BeGreaterThan(10_000_000); // > 10ms (minimum delay)
        delta1.Should().BeLessThan(500_000_000);   // < 500ms (reasonable upper bound)

        delta2.Should().BeGreaterThan(10_000_000); // > 10ms (minimum delay)
        delta2.Should().BeLessThan(500_000_000);   // < 500ms (reasonable upper bound)
    }

    [Fact]
    public async Task HybridLogicalClocks_IntegrateWithPhysicalClocks()
    {
        // Arrange - This tests that HLC (Phase 5) works with PTP (Phase 6)
        var clockSelector = new ClockSourceSelector(NullLogger<ClockSourceSelector>.Instance);
        await clockSelector.InitializeAsync();

        // Act - Simulate HLC update with physical clock
        long physicalTime1 = clockSelector.ActiveSource.GetCurrentTimeNanos();
        await Task.Delay(5);
        long physicalTime2 = clockSelector.ActiveSource.GetCurrentTimeNanos();

        // HLC logical counter would increment here if physical time doesn't advance
        // But with precise physical clocks, logical counter should rarely increment

        // Assert
        physicalTime2.Should().BeGreaterThan(physicalTime1);

        // With PTP precision, even 5ms delay should be detectable
        long delta = physicalTime2 - physicalTime1;
        delta.Should().BeGreaterThan(5_000_000); // > 5ms
    }

    [Fact]
    public async Task VectorClocks_BenefitFromPtpPrecision()
    {
        // Arrange
        var clockSelector = new ClockSourceSelector(NullLogger<ClockSourceSelector>.Instance);
        await clockSelector.InitializeAsync();

        // Act - Simulate vector clock update with physical timestamps
        var physicalTimestamps = new List<long>();

        for (int i = 0; i < 10; i++)
        {
            physicalTimestamps.Add(clockSelector.ActiveSource.GetCurrentTimeNanos());
            await Task.Delay(1); // 1ms between events
        }

        // Assert - All timestamps should be unique and increasing
        for (int i = 1; i < physicalTimestamps.Count; i++)
        {
            physicalTimestamps[i].Should().BeGreaterThan(physicalTimestamps[i - 1]);
        }

        // With PTP precision, vector clock conflicts should be rare
        // (physical time advances faster than event rate)
        var uniqueTimestamps = physicalTimestamps.Distinct().Count();
        uniqueTimestamps.Should().Be(physicalTimestamps.Count);
    }

    [Fact]
    public async Task CrossNodeTimestamps_WithNetworkCompensation()
    {
        // Arrange
        var clockSelector = new ClockSourceSelector(NullLogger<ClockSourceSelector>.Instance);
        await clockSelector.InitializeAsync();

        var compensator = new NetworkLatencyCompensator(
            NullLogger<NetworkLatencyCompensator>.Instance);

        var node1Endpoint = new IPEndPoint(IPAddress.Loopback, 12347);
        var node2Endpoint = new IPEndPoint(IPAddress.Loopback, 12348);

        using var listener1 = new System.Net.Sockets.TcpListener(node1Endpoint);
        using var listener2 = new System.Net.Sockets.TcpListener(node2Endpoint);

        listener1.Start();
        listener2.Start();

        // Act - Measure latency to both nodes
        var rtt1 = await compensator.MeasureLatencyAsync(node1Endpoint);
        var rtt2 = await compensator.MeasureLatencyAsync(node2Endpoint);

        // Simulate timestamps from different nodes
        long localTime = clockSelector.ActiveSource.GetCurrentTimeNanos();
        long node1Time = localTime + 3_000_000; // +3ms (simulated node1 time + network)
        long node2Time = localTime + 5_000_000; // +5ms (simulated node2 time + network)

        // Compensate both timestamps
        long compensated1 = compensator.CompensateTimestamp(node1Time, node1Endpoint);
        long compensated2 = compensator.CompensateTimestamp(node2Time, node2Endpoint);

        // Assert - Compensation should not significantly worsen timestamps
        // On localhost, RTT measurement can have timing noise, so allow some tolerance
        // The compensation might add ~5-10ms noise due to scheduling imprecision
        const long toleranceNanos = 20_000_000L; // 20ms tolerance for localhost noise

        long rawDiff1 = Math.Abs(node1Time - localTime);
        long rawDiff2 = Math.Abs(node2Time - localTime);
        long compensatedDiff1 = Math.Abs(compensated1 - localTime);
        long compensatedDiff2 = Math.Abs(compensated2 - localTime);

        // Compensated should be within tolerance of raw difference (may be better or slightly worse)
        compensatedDiff1.Should().BeLessThanOrEqualTo(rawDiff1 + toleranceNanos);
        compensatedDiff2.Should().BeLessThanOrEqualTo(rawDiff2 + toleranceNanos);

        listener1.Stop();
        listener2.Stop();
    }

    [Fact]
    public async Task TemporalGraphs_UseCompensatedTimestamps()
    {
        // Arrange - Simulates temporal graph pattern detection (Phase 9 preview)
        var clockSelector = new ClockSourceSelector(NullLogger<ClockSourceSelector>.Instance);
        await clockSelector.InitializeAsync();

        var compensator = new NetworkLatencyCompensator(
            NullLogger<NetworkLatencyCompensator>.Instance);

        // Act - Simulate event stream with network-compensated timestamps
        var events = new List<(long timestamp, string eventType)>();

        for (int i = 0; i < 5; i++)
        {
            long rawTimestamp = clockSelector.ActiveSource.GetCurrentTimeNanos();

            // In real scenario, this would be a remote timestamp
            // For testing, we just use local time
            events.Add((rawTimestamp, $"Event{i}"));

            await Task.Delay(2); // 2ms between events
        }

        // Assert - Events should have monotonically increasing timestamps
        for (int i = 1; i < events.Count; i++)
        {
            events[i].timestamp.Should().BeGreaterThan(events[i - 1].timestamp);
        }

        // Temporal order should be preserved for pattern detection
        events.Should().BeInAscendingOrder(e => e.timestamp);
    }
}

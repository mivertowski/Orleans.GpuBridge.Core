using System.Net;
using FluentAssertions;
using Microsoft.Extensions.Logging.Abstractions;
using Orleans.GpuBridge.Runtime.Temporal.Network;

namespace Orleans.GpuBridge.Temporal.Tests.Unit.Network;

/// <summary>
/// Unit tests for network latency measurement and timestamp compensation.
/// </summary>
public sealed class NetworkLatencyCompensatorTests
{
    [Fact]
    public async Task NetworkLatencyCompensator_MeasuresRtt()
    {
        // Arrange
        var compensator = new NetworkLatencyCompensator(
            NullLogger<NetworkLatencyCompensator>.Instance,
            TimeSpan.FromMinutes(5));

        var localEndpoint = new IPEndPoint(IPAddress.Loopback, 12345);

        // Start a simple TCP listener for testing
        using var listener = new System.Net.Sockets.TcpListener(localEndpoint);
        listener.Start();

        // Act
        var rtt = await compensator.MeasureLatencyAsync(localEndpoint);

        // Assert
        rtt.Should().BeGreaterThan(TimeSpan.Zero);
        rtt.Should().BeLessThan(TimeSpan.FromMilliseconds(100)); // Localhost should be very fast

        listener.Stop();
    }

    [Fact]
    public void NetworkLatencyCompensator_CompensatesTimestamp()
    {
        // Arrange
        var compensator = new NetworkLatencyCompensator(
            NullLogger<NetworkLatencyCompensator>.Instance);

        var endpoint = new IPEndPoint(IPAddress.Loopback, 12345);
        var remoteTimestampNanos = 1_000_000_000L; // 1 second after epoch

        // Simulate measured latency by adding to cache
        var stats = new LatencyStatistics(endpoint, NullLogger.Instance);
        stats.AddSample(TimeSpan.FromMilliseconds(10)); // 10ms RTT
        typeof(NetworkLatencyCompensator)
            .GetField("_latencyCache", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance)!
            .GetValue(compensator)
            .GetType()
            .GetMethod("TryAdd")!
            .Invoke(
                typeof(NetworkLatencyCompensator)
                    .GetField("_latencyCache", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance)!
                    .GetValue(compensator),
                new object[] { endpoint, stats });

        // Act
        long compensated = compensator.CompensateTimestamp(remoteTimestampNanos, endpoint);

        // Assert
        // Compensation should subtract one-way latency (RTT/2 = 5ms = 5,000,000ns)
        long expectedCompensation = 5_000_000; // 5ms in nanoseconds
        compensated.Should().BeLessThan(remoteTimestampNanos);
        (remoteTimestampNanos - compensated).Should().BeCloseTo(expectedCompensation, 1_000_000); // ±1ms tolerance
    }

    [Fact]
    public void NetworkLatencyCompensator_ReturnsUncompensatedWhenNoData()
    {
        // Arrange
        var compensator = new NetworkLatencyCompensator(
            NullLogger<NetworkLatencyCompensator>.Instance);

        var endpoint = new IPEndPoint(IPAddress.Loopback, 12345);
        var remoteTimestampNanos = 1_000_000_000L;

        // Act
        long compensated = compensator.CompensateTimestamp(remoteTimestampNanos, endpoint);

        // Assert
        // Should return uncompensated timestamp when no latency data available
        compensated.Should().Be(remoteTimestampNanos);
    }

    [Fact]
    public async Task NetworkLatencyCompensator_ThrowsOnConnectionFailure()
    {
        // Arrange
        var compensator = new NetworkLatencyCompensator(
            NullLogger<NetworkLatencyCompensator>.Instance);

        var unreachableEndpoint = new IPEndPoint(IPAddress.Parse("192.0.2.1"), 9999); // TEST-NET-1 (RFC 5737)

        // Act
        var act = async () => await compensator.MeasureLatencyAsync(unreachableEndpoint);

        // Assert
        await act.Should().ThrowAsync<InvalidOperationException>()
            .WithMessage("*Cannot measure RTT*");
    }

    [Fact]
    public async Task NetworkLatencyCompensator_CachesResults()
    {
        // Arrange
        var compensator = new NetworkLatencyCompensator(
            NullLogger<NetworkLatencyCompensator>.Instance);

        var localEndpoint = new IPEndPoint(IPAddress.Loopback, 12346);

        using var listener = new System.Net.Sockets.TcpListener(localEndpoint);
        listener.Start();

        // Act
        var rtt1 = await compensator.MeasureLatencyAsync(localEndpoint);
        var stats = compensator.GetStatistics(localEndpoint);

        // Assert
        stats.Should().NotBeNull();
        stats!.MedianRtt.Should().Be(rtt1);
        stats.SampleCount.Should().BeGreaterThan(0);

        listener.Stop();
    }

    [Fact]
    public void LatencyStatistics_TracksMultipleSamples()
    {
        // Arrange
        var endpoint = new IPEndPoint(IPAddress.Loopback, 12345);
        var stats = new LatencyStatistics(endpoint, NullLogger.Instance);

        // Act
        stats.AddSample(TimeSpan.FromMilliseconds(10));
        stats.AddSample(TimeSpan.FromMilliseconds(15));
        stats.AddSample(TimeSpan.FromMilliseconds(12));
        stats.AddSample(TimeSpan.FromMilliseconds(11));
        stats.AddSample(TimeSpan.FromMilliseconds(13));

        // Assert
        stats.SampleCount.Should().Be(5);
        stats.MedianRtt.Should().Be(TimeSpan.FromMilliseconds(12));
        stats.MinRtt.Should().Be(TimeSpan.FromMilliseconds(10));
        stats.MaxRtt.Should().Be(TimeSpan.FromMilliseconds(15));
    }

    [Fact]
    public void LatencyStatistics_CalculatesP99Correctly()
    {
        // Arrange
        var endpoint = new IPEndPoint(IPAddress.Loopback, 12345);
        var stats = new LatencyStatistics(endpoint, NullLogger.Instance);

        // Add 100 samples
        for (int i = 1; i <= 100; i++)
        {
            stats.AddSample(TimeSpan.FromMilliseconds(i));
        }

        // Act & Assert
        stats.SampleCount.Should().Be(100);
        stats.P99Rtt.Should().Be(TimeSpan.FromMilliseconds(99));
    }

    [Fact]
    public void LatencyStatistics_DetectsHighVariance()
    {
        // Arrange
        var endpoint = new IPEndPoint(IPAddress.Loopback, 12345);
        var stats = new LatencyStatistics(endpoint, NullLogger.Instance);

        // Add samples with low variance
        for (int i = 0; i < 10; i++)
        {
            stats.AddSample(TimeSpan.FromMilliseconds(10));
        }

        // Act & Assert
        stats.HasHighVariance().Should().BeFalse();

        // Add outlier to create high variance
        stats.AddSample(TimeSpan.FromMilliseconds(100));

        // Should now detect high variance (p99 > 2× median)
        stats.HasHighVariance().Should().BeTrue();
    }

    [Fact]
    public void LatencyStatistics_MaintainsSlidingWindow()
    {
        // Arrange
        var endpoint = new IPEndPoint(IPAddress.Loopback, 12345);
        var stats = new LatencyStatistics(endpoint, NullLogger.Instance);

        // Act - Add 150 samples (window size is 100)
        for (int i = 1; i <= 150; i++)
        {
            stats.AddSample(TimeSpan.FromMilliseconds(i));
        }

        // Assert
        stats.SampleCount.Should().Be(150);
        // Window should only keep last 100 samples
        // So MinRtt should be from sample 51 (oldest in window)
        stats.MinRtt.Should().Be(TimeSpan.FromMilliseconds(51));
        stats.MaxRtt.Should().Be(TimeSpan.FromMilliseconds(150));
    }

    [Fact]
    public void NetworkLatencyCompensator_GetStatisticsReturnsNull()
    {
        // Arrange
        var compensator = new NetworkLatencyCompensator(
            NullLogger<NetworkLatencyCompensator>.Instance);

        var endpoint = new IPEndPoint(IPAddress.Loopback, 12345);

        // Act
        var stats = compensator.GetStatistics(endpoint);

        // Assert
        stats.Should().BeNull();
    }
}

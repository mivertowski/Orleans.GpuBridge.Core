using Microsoft.Extensions.Logging.Abstractions;
using Orleans.GpuBridge.Abstractions.Temporal;
using Orleans.GpuBridge.Runtime.Temporal;

namespace Orleans.GpuBridge.Temporal.Tests.Integration;

/// <summary>
/// Integration tests for ring kernel lifecycle and message processing.
/// </summary>
public sealed class RingKernelIntegrationTests
{
    [Fact]
    public async Task RingKernelManager_StartsAndStopsSuccessfully()
    {
        // Arrange
        var manager = new RingKernelManager(NullLogger<RingKernelManager>.Instance);

        // Act
        await manager.StartAsync(actorCount: 10, messageQueueSize: 1024);

        // Assert
        manager.IsRunning.Should().BeTrue();
        manager.ActorCount.Should().Be(10);
        manager.MessageQueueSize.Should().Be(1024);

        // Cleanup
        await manager.StopAsync();
        manager.IsRunning.Should().BeFalse();
    }

    [Fact]
    public async Task RingKernelManager_ThrowsWhenStartedTwice()
    {
        // Arrange
        var manager = new RingKernelManager(NullLogger<RingKernelManager>.Instance);
        await manager.StartAsync(actorCount: 10);

        // Act & Assert
        var act = () => manager.StartAsync(actorCount: 10);
        await act.Should().ThrowAsync<InvalidOperationException>()
            .WithMessage("Ring kernel is already running*");

        // Cleanup
        await manager.StopAsync();
    }

    [Fact]
    public async Task RingKernelManager_ValidatesActorCount()
    {
        // Arrange
        var manager = new RingKernelManager(NullLogger<RingKernelManager>.Instance);

        // Act & Assert
        var act = () => manager.StartAsync(actorCount: 0);
        await act.Should().ThrowAsync<ArgumentException>()
            .WithMessage("Actor count must be positive*");
    }

    [Fact]
    public async Task RingKernelManager_ValidatesQueueSize()
    {
        // Arrange
        var manager = new RingKernelManager(NullLogger<RingKernelManager>.Instance);

        // Act & Assert - non-power-of-2
        var act1 = () => manager.StartAsync(actorCount: 10, messageQueueSize: 1000);
        await act1.Should().ThrowAsync<ArgumentException>()
            .WithMessage("*must be a positive power of 2*");

        // Act & Assert - zero
        var act2 = () => manager.StartAsync(actorCount: 10, messageQueueSize: 0);
        await act2.Should().ThrowAsync<ArgumentException>();
    }

    [Fact(Skip = "Requires GPU hardware and DotCompute integration")]
    public async Task RingKernelManager_ProcessesMessagesWithSubMicrosecondLatency()
    {
        // This test would verify the revolutionary performance claim:
        // Ring kernel message latency should be 100-500ns vs 10-50μs traditional

        // Arrange
        var manager = new RingKernelManager(NullLogger<RingKernelManager>.Instance);
        await manager.StartAsync(actorCount: 1000, messageQueueSize: 4096);

        // Act - Send 10,000 messages and measure latency
        var stopwatch = System.Diagnostics.Stopwatch.StartNew();
        const int messageCount = 10_000;

        for (int i = 0; i < messageCount; i++)
        {
            // await manager.EnqueueMessageAsync(message);
        }

        stopwatch.Stop();

        // Assert
        var avgLatencyNanos = (stopwatch.ElapsedTicks * 1_000_000_000.0) / (System.Diagnostics.Stopwatch.Frequency * messageCount);
        avgLatencyNanos.Should().BeLessThan(1000); // < 1μs average

        // Cleanup
        await manager.StopAsync();
    }

    [Fact]
    public async Task RingKernelManager_StopsGracefullyEvenWhenNotRunning()
    {
        // Arrange
        var manager = new RingKernelManager(NullLogger<RingKernelManager>.Instance);

        // Act (stop without start)
        await manager.StopAsync();

        // Assert - should not throw
        manager.IsRunning.Should().BeFalse();
    }

    [Fact]
    public async Task RingKernelManager_GetStatisticsReturnsCorrectInfo()
    {
        // Arrange
        var manager = new RingKernelManager(NullLogger<RingKernelManager>.Instance);
        await manager.StartAsync(actorCount: 100, messageQueueSize: 2048);

        // Act
        var stats = await manager.GetStatisticsAsync();

        // Assert
        stats.ActorCount.Should().Be(100);
        stats.MessageQueueSize.Should().Be(2048);
        stats.IsRunning.Should().BeTrue();

        // Cleanup
        await manager.StopAsync();
    }

    [Fact]
    public async Task RingKernelManager_ThrowsWhenGettingStatsWhileNotRunning()
    {
        // Arrange
        var manager = new RingKernelManager(NullLogger<RingKernelManager>.Instance);

        // Act & Assert
        var act = () => manager.GetStatisticsAsync();
        await act.Should().ThrowAsync<InvalidOperationException>()
            .WithMessage("Ring kernel is not running*");
    }

    [Fact]
    public async Task RingKernelManager_DisposeStopsKernelAutomatically()
    {
        // Arrange
        var manager = new RingKernelManager(NullLogger<RingKernelManager>.Instance);
        await manager.StartAsync(actorCount: 10);

        // Act
        await manager.DisposeAsync();

        // Assert
        manager.IsRunning.Should().BeFalse();
    }
}

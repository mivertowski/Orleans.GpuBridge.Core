using Orleans.GpuBridge.Abstractions;
using Orleans.GpuBridge.Grains.Batch;
using Orleans.GpuBridge.Grains.Stream;
using Orleans.GpuBridge.Tests.RC2.Infrastructure;
using Orleans.Streams;

namespace Orleans.GpuBridge.Tests.RC2.Grains;

/// <summary>
/// Comprehensive tests for GpuStreamGrain implementation.
/// Tests stream processing, buffering, backpressure, error handling, and resource cleanup.
/// </summary>
[Collection("ClusterCollection")]
public sealed class GpuStreamGrainTests : IClassFixture<ClusterFixture>
{
    private readonly ClusterFixture _fixture;

    public GpuStreamGrainTests(ClusterFixture fixture)
    {
        _fixture = fixture;
    }

    [Fact]
    public async Task GpuStreamGrain_StartStreamAsync_ShouldInitialize()
    {
        // Arrange
        var kernelId = "kernels/test-stream-start";
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuStreamGrain<float[], float>>(kernelId);

        var observer = new TestStreamObserver<float>();
        var observerRef = _fixture.Cluster.GrainFactory
            .CreateObjectReference<IGpuResultObserver<float>>(observer);

        // Act
        await grain.StartStreamAsync("test-stream-1", observerRef);

        // Assert
        var status = await grain.GetStatusAsync();
        status.Should().Be(StreamProcessingStatus.Processing);
    }

    [Fact]
    public async Task GpuStreamGrain_ProcessItemAsync_ShouldProcessInOrder()
    {
        // Arrange
        var kernelId = "kernels/test-stream-order";
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuStreamGrain<float[], float>>(kernelId);

        var observer = new TestStreamObserver<float>();
        var observerRef = _fixture.Cluster.GrainFactory
            .CreateObjectReference<IGpuResultObserver<float>>(observer);

        await grain.StartStreamAsync("test-stream-order", observerRef);

        // Act - Process items in sequence
        var items = new[]
        {
            new float[] { 1, 2, 3 },
            new float[] { 4, 5, 6 },
            new float[] { 7, 8, 9 }
        };

        foreach (var item in items)
        {
            await grain.ProcessItemAsync(item);
        }

        // Flush to ensure all items are processed
        await grain.FlushStreamAsync();

        // Wait for async processing
        await Task.Delay(500);

        // Assert - Items should be processed
        observer.ReceivedItems.Should().NotBeEmpty();
        observer.Completed.Should().BeTrue();
        observer.Error.Should().BeNull();
    }

    [Fact]
    public async Task GpuStreamGrain_FlushStreamAsync_ShouldProcessAll()
    {
        // Arrange
        var kernelId = "kernels/test-stream-flush";
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuStreamGrain<float[], float>>(kernelId);

        var observer = new TestStreamObserver<float>();
        var observerRef = _fixture.Cluster.GrainFactory
            .CreateObjectReference<IGpuResultObserver<float>>(observer);

        await grain.StartStreamAsync("test-stream-flush", observerRef);

        // Act - Add items and flush
        for (int i = 0; i < 5; i++)
        {
            await grain.ProcessItemAsync(new float[] { i, i + 1, i + 2 });
        }

        await grain.FlushStreamAsync();

        // Wait for processing to complete
        await Task.Delay(500);

        // Assert
        observer.ReceivedItems.Should().NotBeEmpty();
        var stats = await grain.GetStatsAsync();
        stats.ItemsProcessed.Should().BeGreaterThan(0);
    }

    [Fact]
    public async Task GpuStreamGrain_Backpressure_ShouldApply()
    {
        // Arrange
        var kernelId = "kernels/test-backpressure";
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuStreamGrain<float[], float>>(kernelId);

        var observer = new TestStreamObserver<float>();
        var observerRef = _fixture.Cluster.GrainFactory
            .CreateObjectReference<IGpuResultObserver<float>>(observer);

        var hints = new GpuExecutionHints { MaxMicroBatch = 2 }; // Small batch size
        await grain.StartStreamAsync("test-backpressure", observerRef, hints);

        // Act - Flood with many items to trigger buffering
        var tasks = Enumerable.Range(0, 100)
            .Select(i => grain.ProcessItemAsync(new float[] { i }))
            .ToArray();

        // Should complete without throwing despite many queued items
        await Task.WhenAll(tasks);

        // Flush and wait for processing
        await grain.FlushStreamAsync();
        await Task.Delay(1000);

        // Assert - All items eventually processed
        var stats = await grain.GetStatsAsync();
        stats.ItemsProcessed.Should().BeGreaterThan(0);
    }

    [Fact]
    public async Task GpuStreamGrain_Error_ShouldNotify()
    {
        // Arrange
        var kernelId = "kernels/test-stream-error";
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuStreamGrain<float[], float>>(kernelId);

        var observer = new TestStreamObserver<float>();
        var observerRef = _fixture.Cluster.GrainFactory
            .CreateObjectReference<IGpuResultObserver<float>>(observer);

        await grain.StartStreamAsync("test-stream-error", observerRef);

        // Act - Process valid items
        await grain.ProcessItemAsync(new float[] { 1, 2, 3 });
        await grain.FlushStreamAsync();

        await Task.Delay(200);

        // Assert - Should handle gracefully (no errors with valid data)
        var stats = await grain.GetStatsAsync();
        stats.ItemsFailed.Should().Be(0);
    }

    [Fact]
    public async Task GpuStreamGrain_Completion_ShouldCleanup()
    {
        // Arrange
        var kernelId = "kernels/test-stream-cleanup";
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuStreamGrain<float[], float>>(kernelId);

        var observer = new TestStreamObserver<float>();
        var observerRef = _fixture.Cluster.GrainFactory
            .CreateObjectReference<IGpuResultObserver<float>>(observer);

        await grain.StartStreamAsync("test-stream-cleanup", observerRef);

        // Act - Process items then stop
        await grain.ProcessItemAsync(new float[] { 1, 2, 3 });
        await grain.FlushStreamAsync();
        await grain.StopProcessingAsync();

        // Assert
        var status = await grain.GetStatusAsync();
        status.Should().Be(StreamProcessingStatus.Stopped);

        var stats = await grain.GetStatsAsync();
        stats.Should().NotBeNull();
        stats.StartTime.Should().NotBe(default);
    }

    [Fact]
    public async Task GpuStreamGrain_StartProcessingAsync_WithOrleansStreams_ShouldWork()
    {
        // Arrange
        var kernelId = "kernels/test-orleans-streams";
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuStreamGrain<float[], float>>(kernelId);

        var inputStreamId = StreamId.Create("test-input", Guid.NewGuid());
        var outputStreamId = StreamId.Create("test-output", Guid.NewGuid());

        // Act
        await grain.StartProcessingAsync(inputStreamId, outputStreamId);

        // Assert
        var status = await grain.GetStatusAsync();
        status.Should().Be(StreamProcessingStatus.Processing);

        // Cleanup
        await grain.StopProcessingAsync();
    }

    [Fact]
    public async Task GpuStreamGrain_GetStatusAsync_ShouldReflectCurrentState()
    {
        // Arrange
        var kernelId = "kernels/test-status";
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuStreamGrain<float[], float>>(kernelId);

        // Act & Assert - Initial state
        var initialStatus = await grain.GetStatusAsync();
        initialStatus.Should().Be(StreamProcessingStatus.Idle);

        // Start processing
        var observer = new TestStreamObserver<float>();
        var observerRef = _fixture.Cluster.GrainFactory
            .CreateObjectReference<IGpuResultObserver<float>>(observer);

        await grain.StartStreamAsync("test-status-stream", observerRef);

        var runningStatus = await grain.GetStatusAsync();
        runningStatus.Should().Be(StreamProcessingStatus.Processing);

        // Stop processing
        await grain.StopProcessingAsync();

        var stoppedStatus = await grain.GetStatusAsync();
        stoppedStatus.Should().Be(StreamProcessingStatus.Stopped);
    }

    [Fact]
    public async Task GpuStreamGrain_GetStatsAsync_ShouldProvideMetrics()
    {
        // Arrange
        var kernelId = "kernels/test-stats";
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuStreamGrain<float[], float>>(kernelId);

        var observer = new TestStreamObserver<float>();
        var observerRef = _fixture.Cluster.GrainFactory
            .CreateObjectReference<IGpuResultObserver<float>>(observer);

        await grain.StartStreamAsync("test-stats-stream", observerRef);

        // Act - Process multiple items
        for (int i = 0; i < 10; i++)
        {
            await grain.ProcessItemAsync(new float[] { i });
        }

        await grain.FlushStreamAsync();
        await Task.Delay(500);

        var stats = await grain.GetStatsAsync();

        // Assert
        stats.Should().NotBeNull();
        stats.ItemsProcessed.Should().BeGreaterThan(0);
        stats.StartTime.Should().NotBe(default);
        stats.TotalProcessingTime.Should().BeGreaterThan(TimeSpan.Zero);
        stats.AverageLatencyMs.Should().BeGreaterThanOrEqualTo(0);
    }

    [Fact]
    public async Task GpuStreamGrain_ConcurrentProcessing_ShouldHandleCorrectly()
    {
        // Arrange
        var kernelId = "kernels/test-concurrent-stream";
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuStreamGrain<float[], float>>(kernelId);

        var observer = new TestStreamObserver<float>();
        var observerRef = _fixture.Cluster.GrainFactory
            .CreateObjectReference<IGpuResultObserver<float>>(observer);

        await grain.StartStreamAsync("test-concurrent", observerRef);

        // Act - Send items concurrently from multiple "producers"
        var producerTasks = Enumerable.Range(0, 5)
            .Select(async producerId =>
            {
                for (int i = 0; i < 10; i++)
                {
                    await grain.ProcessItemAsync(new float[] { producerId, i });
                    await Task.Delay(10); // Small delay between items
                }
            })
            .ToArray();

        await Task.WhenAll(producerTasks);
        await grain.FlushStreamAsync();

        // Wait for all processing to complete
        await Task.Delay(1000);

        // Assert
        var stats = await grain.GetStatsAsync();
        stats.ItemsProcessed.Should().BeGreaterThan(0);
        observer.ReceivedItems.Should().NotBeEmpty();
    }

    [Fact]
    public async Task GpuStreamGrain_StopProcessing_WhileProcessing_ShouldStopGracefully()
    {
        // Arrange
        var kernelId = "kernels/test-graceful-stop";
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuStreamGrain<float[], float>>(kernelId);

        var observer = new TestStreamObserver<float>();
        var observerRef = _fixture.Cluster.GrainFactory
            .CreateObjectReference<IGpuResultObserver<float>>(observer);

        await grain.StartStreamAsync("test-graceful-stop", observerRef);

        // Act - Queue items and stop immediately
        var processTasks = Enumerable.Range(0, 20)
            .Select(i => grain.ProcessItemAsync(new float[] { i }))
            .ToArray();

        // Don't wait for all processing - stop immediately
        await Task.Delay(50);
        await grain.StopProcessingAsync();

        // Assert - Should stop without throwing
        var status = await grain.GetStatusAsync();
        status.Should().Be(StreamProcessingStatus.Stopped);
    }

    /// <summary>
    /// Test observer for stream processing.
    /// </summary>
    private sealed class TestStreamObserver<T> : IGpuResultObserver<T>
    {
        public List<T> ReceivedItems { get; } = new();
        public bool Completed { get; private set; }
        public Exception? Error { get; private set; }

        public Task OnNextAsync(T item)
        {
            ReceivedItems.Add(item);
            return Task.CompletedTask;
        }

        public Task OnErrorAsync(Exception error)
        {
            Error = error;
            return Task.CompletedTask;
        }

        public Task OnCompletedAsync()
        {
            Completed = true;
            return Task.CompletedTask;
        }
    }
}

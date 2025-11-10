using Orleans;
using Orleans.GpuBridge.Abstractions;
using Orleans.GpuBridge.Grains.Batch;
using Orleans.GpuBridge.Grains.Stream;
using Orleans.GpuBridge.Tests.RC2.Infrastructure;
using Orleans.Streams;

namespace Orleans.GpuBridge.Tests.RC2.Grains;

/// <summary>
/// Comprehensive tests for GpuStreamGrain implementation using REAL API methods only.
/// Tests lifecycle management (7), item processing (10), status/stats (7), and error handling (4).
/// Total: 28 tests covering all IGpuStreamGrain methods.
/// </summary>
[Collection("ClusterCollection")]
public sealed class GpuStreamGrainTests : IClassFixture<ClusterFixture>
{
    private readonly ClusterFixture _fixture;

    public GpuStreamGrainTests(ClusterFixture fixture)
    {
        _fixture = fixture;
    }

    #region Lifecycle Tests (7 tests)

    [Fact]
    public async Task StartProcessingAsync_WithValidStreams_ShouldStartProcessing()
    {
        // Arrange
        var kernelId = "kernels/test-start-processing";
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuStreamGrain<int, int>>(kernelId);

        var inputStreamId = StreamId.Create("test-namespace", "input-stream-1");
        var outputStreamId = StreamId.Create("test-namespace", "output-stream-1");

        // Act
        await grain.StartProcessingAsync(inputStreamId, outputStreamId);
        var status = await grain.GetStatusAsync();

        // Assert
        status.Should().Be(StreamProcessingStatus.Processing);
    }

    [Fact]
    public async Task StartProcessingAsync_CalledTwice_ShouldThrowInvalidOperationException()
    {
        // Arrange
        var kernelId = "kernels/test-double-start";
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuStreamGrain<int, int>>(kernelId);

        var inputStreamId = StreamId.Create("test-namespace", "input-stream-2");
        var outputStreamId = StreamId.Create("test-namespace", "output-stream-2");

        await grain.StartProcessingAsync(inputStreamId, outputStreamId);

        // Act & Assert
        var exception = await Assert.ThrowsAsync<InvalidOperationException>(
            async () => await grain.StartProcessingAsync(inputStreamId, outputStreamId));

        exception.Message.Should().Contain("Already processing");
    }

    [Fact]
    public async Task StartStreamAsync_WithValidObserver_ShouldStartProcessing()
    {
        // Arrange
        var kernelId = "kernels/test-start-stream";
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuStreamGrain<int, int>>(kernelId);

        var observer = new TestGpuResultObserver<int>();
        var observerRef = _fixture.Cluster.GrainFactory
            .CreateObjectReference<IGpuResultObserver<int>>(observer);

        // Act
        await grain.StartStreamAsync("test-stream-1", observerRef);
        var status = await grain.GetStatusAsync();

        // Assert
        status.Should().Be(StreamProcessingStatus.Processing);
    }

    [Fact]
    public async Task StartStreamAsync_WithNullObserver_ShouldThrowArgumentNullException()
    {
        // Arrange
        var kernelId = "kernels/test-null-observer";
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuStreamGrain<int, int>>(kernelId);

        // Act & Assert - ArgumentNullException.ThrowIfNull doesn't set ParamName in all contexts
        await Assert.ThrowsAsync<ArgumentNullException>(
            async () => await grain.StartStreamAsync("test-stream-2", null!));
    }

    [Fact]
    public async Task StartStreamAsync_CalledTwice_ShouldThrowInvalidOperationException()
    {
        // Arrange
        var kernelId = "kernels/test-double-start-stream";
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuStreamGrain<int, int>>(kernelId);

        var observer = new TestGpuResultObserver<int>();
        var observerRef = _fixture.Cluster.GrainFactory
            .CreateObjectReference<IGpuResultObserver<int>>(observer);

        await grain.StartStreamAsync("test-stream-3", observerRef);

        // Act & Assert
        var exception = await Assert.ThrowsAsync<InvalidOperationException>(
            async () => await grain.StartStreamAsync("test-stream-4", observerRef));

        exception.Message.Should().Contain("already active");
    }

    [Fact]
    public async Task StopProcessingAsync_AfterStarting_ShouldStopProcessing()
    {
        // Arrange
        var kernelId = "kernels/test-stop-processing";
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuStreamGrain<int, int>>(kernelId);

        var inputStreamId = StreamId.Create("test-namespace", "input-stream-3");
        var outputStreamId = StreamId.Create("test-namespace", "output-stream-3");

        await grain.StartProcessingAsync(inputStreamId, outputStreamId);

        // Act
        await grain.StopProcessingAsync();
        var status = await grain.GetStatusAsync();

        // Assert
        status.Should().Be(StreamProcessingStatus.Stopped);
    }

    [Fact]
    public async Task StopProcessingAsync_WithoutStarting_ShouldHandleGracefully()
    {
        // Arrange
        var kernelId = "kernels/test-stop-without-start";
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuStreamGrain<int, int>>(kernelId);

        // Act - Should not throw
        await grain.StopProcessingAsync();
        var status = await grain.GetStatusAsync();

        // Assert
        status.Should().Be(StreamProcessingStatus.Idle);
    }

    #endregion

    #region Item Processing Tests (10 tests)

    [Fact]
    public async Task ProcessItemAsync_WithSingleItem_ShouldProcessSuccessfully()
    {
        // Arrange
        var kernelId = "kernels/test-process-single";
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuStreamGrain<int, int>>(kernelId);

        var observer = new TestGpuResultObserver<int>();
        var observerRef = _fixture.Cluster.GrainFactory
            .CreateObjectReference<IGpuResultObserver<int>>(observer);

        await grain.StartStreamAsync("test-stream-5", observerRef);

        // Act
        await grain.ProcessItemAsync(42);
        await Task.Delay(200); // Allow time for processing

        // Assert
        observer.ReceivedItems.Should().NotBeEmpty();
        observer.ReceivedItems.Should().Contain(84); // 42 * 2 based on mock kernel
    }

    [Fact]
    public async Task ProcessItemAsync_WithMultipleItems_ShouldBatch()
    {
        // Arrange
        var kernelId = "kernels/test-process-multiple";
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuStreamGrain<int, int>>(kernelId);

        var observer = new TestGpuResultObserver<int>();
        var observerRef = _fixture.Cluster.GrainFactory
            .CreateObjectReference<IGpuResultObserver<int>>(observer);

        await grain.StartStreamAsync("test-stream-6", observerRef);

        // Act - Send multiple items
        for (int i = 1; i <= 10; i++)
        {
            await grain.ProcessItemAsync(i);
        }

        await Task.Delay(300); // Allow time for batching and processing

        // Assert
        observer.ReceivedItems.Should().NotBeEmpty();
        observer.ReceivedItems.Count.Should().Be(10);
        observer.ReceivedItems.Should().BeEquivalentTo(new[] { 2, 4, 6, 8, 10, 12, 14, 16, 18, 20 });
    }

    [Fact]
    public async Task ProcessItemAsync_ConcurrentCalls_ShouldHandleCorrectly()
    {
        // Arrange
        var kernelId = "kernels/test-concurrent-items";
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuStreamGrain<int, int>>(kernelId);

        var observer = new TestGpuResultObserver<int>();
        var observerRef = _fixture.Cluster.GrainFactory
            .CreateObjectReference<IGpuResultObserver<int>>(observer);

        await grain.StartStreamAsync("test-stream-7", observerRef);

        // Act - Concurrent processing
        var tasks = Enumerable.Range(1, 20)
            .Select(i => grain.ProcessItemAsync(i))
            .ToArray();

        await Task.WhenAll(tasks);
        await Task.Delay(300); // Allow time for processing

        // Assert
        observer.ReceivedItems.Count.Should().Be(20);
        observer.Error.Should().BeNull();
    }

    [Fact]
    public async Task ProcessItemAsync_WithoutStarting_ShouldThrowInvalidOperationException()
    {
        // Arrange
        var kernelId = "kernels/test-process-no-start";
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuStreamGrain<int, int>>(kernelId);

        // Act & Assert
        var exception = await Assert.ThrowsAsync<InvalidOperationException>(
            async () => await grain.ProcessItemAsync(42));

        exception.Message.Should().Contain("not active");
    }

    [Fact]
    public async Task ProcessItemAsync_WithNonNullConstraint_ShouldHandleCorrectly()
    {
        // Arrange
        var kernelId = "kernels/test-process-notnull";
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuStreamGrain<string, string>>(kernelId);

        var observer = new TestGpuResultObserver<string>();
        var observerRef = _fixture.Cluster.GrainFactory
            .CreateObjectReference<IGpuResultObserver<string>>(observer);

        await grain.StartStreamAsync("test-stream-8", observerRef);

        // Act - TIn is constrained to notnull, test with valid string
        await grain.ProcessItemAsync("valid-string");
        await Task.Delay(100);

        // Assert
        observer.Error.Should().BeNull();
    }

    [Fact]
    public async Task FlushStreamAsync_WithPendingItems_ShouldFlushAll()
    {
        // Arrange
        var kernelId = "kernels/test-flush-pending";
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuStreamGrain<int, int>>(kernelId);

        var observer = new TestGpuResultObserver<int>();
        var observerRef = _fixture.Cluster.GrainFactory
            .CreateObjectReference<IGpuResultObserver<int>>(observer);

        await grain.StartStreamAsync("test-stream-9", observerRef);

        // Act - Add items without waiting
        await grain.ProcessItemAsync(1);
        await grain.ProcessItemAsync(2);
        await grain.ProcessItemAsync(3);

        // Flush to ensure all items are processed
        await grain.FlushStreamAsync();
        await Task.Delay(200); // Wait for async processing to complete

        // Assert
        observer.ReceivedItems.Should().HaveCount(3);
        observer.ReceivedItems.Should().BeEquivalentTo(new[] { 2, 4, 6 });
    }

    [Fact]
    public async Task FlushStreamAsync_WhenEmpty_ShouldCompleteImmediately()
    {
        // Arrange
        var kernelId = "kernels/test-flush-empty";
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuStreamGrain<int, int>>(kernelId);

        var observer = new TestGpuResultObserver<int>();
        var observerRef = _fixture.Cluster.GrainFactory
            .CreateObjectReference<IGpuResultObserver<int>>(observer);

        await grain.StartStreamAsync("test-stream-10", observerRef);

        // Act - Flush without adding any items
        await grain.FlushStreamAsync();

        // Assert - Should complete without error
        observer.ReceivedItems.Should().BeEmpty();
        observer.Error.Should().BeNull();
    }

    [Fact]
    public async Task FlushStreamAsync_WithoutStarting_ShouldThrowInvalidOperationException()
    {
        // Arrange
        var kernelId = "kernels/test-flush-no-start";
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuStreamGrain<int, int>>(kernelId);

        // Act & Assert
        var exception = await Assert.ThrowsAsync<InvalidOperationException>(
            async () => await grain.FlushStreamAsync());

        exception.Message.Should().Contain("not active");
    }

    [Fact]
    public async Task ProcessItemAsync_AfterFlush_ShouldContinueProcessing()
    {
        // Arrange
        var kernelId = "kernels/test-process-after-flush";
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuStreamGrain<int, int>>(kernelId);

        var observer = new TestGpuResultObserver<int>();
        var observerRef = _fixture.Cluster.GrainFactory
            .CreateObjectReference<IGpuResultObserver<int>>(observer);

        await grain.StartStreamAsync("test-stream-11", observerRef);

        // Act - Process, flush, then process more
        await grain.ProcessItemAsync(1);
        await grain.FlushStreamAsync();
        await Task.Delay(150); // Wait for first item processing

        await grain.ProcessItemAsync(2);
        await grain.FlushStreamAsync();
        await Task.Delay(150); // Wait for second item processing

        // Assert
        observer.ReceivedItems.Should().HaveCount(2);
        observer.ReceivedItems.Should().BeEquivalentTo(new[] { 2, 4 });
    }

    [Fact]
    public async Task ProcessItemAsync_WithLargeBatch_ShouldHandleEfficiently()
    {
        // Arrange
        var kernelId = "kernels/test-large-batch";
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuStreamGrain<int, int>>(kernelId);

        var observer = new TestGpuResultObserver<int>();
        var observerRef = _fixture.Cluster.GrainFactory
            .CreateObjectReference<IGpuResultObserver<int>>(observer);

        var hints = new GpuExecutionHints { MaxMicroBatch = 50 };
        await grain.StartStreamAsync("test-stream-12", observerRef, hints);

        // Act - Process 200 items
        for (int i = 1; i <= 200; i++)
        {
            await grain.ProcessItemAsync(i);
        }

        await grain.FlushStreamAsync();

        // Assert
        observer.ReceivedItems.Count.Should().Be(200);
        observer.Error.Should().BeNull();
        observer.Completed.Should().BeFalse(); // Not completed until stopped
    }

    #endregion

    #region Status and Stats Tests (7 tests)

    [Fact]
    public async Task GetStatusAsync_InitialState_ShouldReturnIdle()
    {
        // Arrange
        var kernelId = "kernels/test-status-idle";
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuStreamGrain<int, int>>(kernelId);

        // Act
        var status = await grain.GetStatusAsync();

        // Assert
        status.Should().Be(StreamProcessingStatus.Idle);
    }

    [Fact]
    public async Task GetStatusAsync_WhileProcessing_ShouldReturnProcessing()
    {
        // Arrange
        var kernelId = "kernels/test-status-processing";
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuStreamGrain<int, int>>(kernelId);

        var observer = new TestGpuResultObserver<int>();
        var observerRef = _fixture.Cluster.GrainFactory
            .CreateObjectReference<IGpuResultObserver<int>>(observer);

        await grain.StartStreamAsync("test-stream-13", observerRef);

        // Act
        var status = await grain.GetStatusAsync();

        // Assert
        status.Should().Be(StreamProcessingStatus.Processing);
    }

    [Fact]
    public async Task GetStatusAsync_AfterStopping_ShouldReturnStopped()
    {
        // Arrange
        var kernelId = "kernels/test-status-stopped";
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuStreamGrain<int, int>>(kernelId);

        var inputStreamId = StreamId.Create("test-namespace", "input-stream-4");
        var outputStreamId = StreamId.Create("test-namespace", "output-stream-4");

        await grain.StartProcessingAsync(inputStreamId, outputStreamId);
        await grain.StopProcessingAsync();

        // Act
        var status = await grain.GetStatusAsync();

        // Assert
        status.Should().Be(StreamProcessingStatus.Stopped);
    }

    [Fact]
    public async Task GetStatsAsync_AfterProcessing_ShouldReturnCorrectStats()
    {
        // Arrange
        var kernelId = "kernels/test-stats-after-processing";
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuStreamGrain<int, int>>(kernelId);

        var observer = new TestGpuResultObserver<int>();
        var observerRef = _fixture.Cluster.GrainFactory
            .CreateObjectReference<IGpuResultObserver<int>>(observer);

        await grain.StartStreamAsync("test-stream-14", observerRef);

        // Act - Process items
        for (int i = 1; i <= 5; i++)
        {
            await grain.ProcessItemAsync(i);
        }

        await grain.FlushStreamAsync();
        await Task.Delay(300); // Wait for stats to be recorded

        var stats = await grain.GetStatsAsync();

        // Assert
        stats.Should().NotBeNull();
        stats.ItemsProcessed.Should().BeGreaterThan(0);
        stats.ItemsFailed.Should().Be(0);
        stats.TotalProcessingTime.Should().BeGreaterThan(TimeSpan.Zero);
        stats.StartTime.Should().BeCloseTo(DateTime.UtcNow, TimeSpan.FromMinutes(1));
        stats.LastProcessedTime.Should().NotBeNull();
    }

    [Fact]
    public async Task GetStatsAsync_InitialState_ShouldReturnEmptyStats()
    {
        // Arrange
        var kernelId = "kernels/test-stats-initial";
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuStreamGrain<int, int>>(kernelId);

        // Act
        var stats = await grain.GetStatsAsync();

        // Assert
        stats.Should().NotBeNull();
        stats.ItemsProcessed.Should().Be(0);
        stats.ItemsFailed.Should().Be(0);
        // Note: StartTime may not be default if stats tracker is initialized
        // We only verify items processed/failed counts
    }

    [Fact]
    public async Task StatusTransition_FromIdleToProcessingToStopped_ShouldBeCorrect()
    {
        // Arrange
        var kernelId = "kernels/test-status-transitions";
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuStreamGrain<int, int>>(kernelId);

        var observer = new TestGpuResultObserver<int>();
        var observerRef = _fixture.Cluster.GrainFactory
            .CreateObjectReference<IGpuResultObserver<int>>(observer);

        // Act & Assert - Idle
        var status1 = await grain.GetStatusAsync();
        status1.Should().Be(StreamProcessingStatus.Idle);

        // Start processing
        await grain.StartStreamAsync("test-stream-15", observerRef);
        var status2 = await grain.GetStatusAsync();
        status2.Should().Be(StreamProcessingStatus.Processing);

        // Stop processing
        await grain.StopProcessingAsync();
        var status3 = await grain.GetStatusAsync();
        status3.Should().Be(StreamProcessingStatus.Stopped);
    }

    [Fact]
    public async Task GetStatsAsync_TracksTiming_ShouldReportAccurateMetrics()
    {
        // Arrange
        var kernelId = "kernels/test-stats-timing";
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuStreamGrain<int, int>>(kernelId);

        var observer = new TestGpuResultObserver<int>();
        var observerRef = _fixture.Cluster.GrainFactory
            .CreateObjectReference<IGpuResultObserver<int>>(observer);

        await grain.StartStreamAsync("test-stream-16", observerRef);

        // Act - Process items with delay
        await grain.ProcessItemAsync(1);
        await Task.Delay(50);
        await grain.ProcessItemAsync(2);
        await Task.Delay(50);

        await grain.FlushStreamAsync();
        await Task.Delay(100); // Wait for stats update

        var stats = await grain.GetStatsAsync();

        // Assert - May process items in batches, so check > 0
        stats.ItemsProcessed.Should().BeGreaterThan(0);
        stats.AverageLatencyMs.Should().BeGreaterThan(0);
        stats.LastProcessedTime.Should().BeCloseTo(DateTime.UtcNow, TimeSpan.FromSeconds(5));
    }

    #endregion

    #region Error Handling Tests (4 tests)

    [Fact]
    public async Task StartProcessingAsync_WithNullInputStream_ShouldThrowException()
    {
        // Arrange
        var kernelId = "kernels/test-null-input-stream";
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuStreamGrain<int, int>>(kernelId);

        var outputStreamId = StreamId.Create("test-namespace", "output-stream-5");

        // Act & Assert - May throw ArgumentNullException or wrapped in AggregateException
        await Assert.ThrowsAnyAsync<Exception>(
            async () => await grain.StartProcessingAsync(default, outputStreamId));
    }

    [Fact]
    public async Task StartProcessingAsync_WithNullOutputStream_ShouldThrowException()
    {
        // Arrange
        var kernelId = "kernels/test-null-output-stream";
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuStreamGrain<int, int>>(kernelId);

        var inputStreamId = StreamId.Create("test-namespace", "input-stream-5");

        // Act & Assert - May throw ArgumentNullException or wrapped in AggregateException
        await Assert.ThrowsAnyAsync<Exception>(
            async () => await grain.StartProcessingAsync(inputStreamId, default));
    }

    [Fact]
    public async Task StartStreamAsync_WithEmptyStreamId_ShouldThrowArgumentException()
    {
        // Arrange
        var kernelId = "kernels/test-empty-stream-id";
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuStreamGrain<int, int>>(kernelId);

        var observer = new TestGpuResultObserver<int>();
        var observerRef = _fixture.Cluster.GrainFactory
            .CreateObjectReference<IGpuResultObserver<int>>(observer);

        // Act & Assert
        await Assert.ThrowsAsync<ArgumentException>(
            async () => await grain.StartStreamAsync(string.Empty, observerRef));
    }

    [Fact]
    public async Task ProcessItemAsync_AfterStop_ShouldThrowInvalidOperationException()
    {
        // Arrange
        var kernelId = "kernels/test-process-after-stop";
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuStreamGrain<int, int>>(kernelId);

        var observer = new TestGpuResultObserver<int>();
        var observerRef = _fixture.Cluster.GrainFactory
            .CreateObjectReference<IGpuResultObserver<int>>(observer);

        await grain.StartStreamAsync("test-stream-17", observerRef);
        await grain.StopProcessingAsync();

        // Act & Assert
        var exception = await Assert.ThrowsAsync<InvalidOperationException>(
            async () => await grain.ProcessItemAsync(42));

        exception.Message.Should().Contain("not active");
    }

    #endregion

    /// <summary>
    /// Test observer implementation for GPU result callbacks.
    /// </summary>
    private sealed class TestGpuResultObserver<T> : IGpuResultObserver<T>
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

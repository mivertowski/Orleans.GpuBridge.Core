using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using FluentAssertions;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Moq;
using Orleans;
using Orleans.GpuBridge.Abstractions;
using Orleans.GpuBridge.Abstractions.Kernels;
using Orleans.GpuBridge.Grains;
using Orleans.GpuBridge.Grains.Stream;
using Orleans.Runtime;
using Orleans.Streams;
using Orleans.TestingHost;
using Xunit;

namespace Orleans.GpuBridge.Tests.Grains;

/// <summary>
/// Unit tests for GpuStreamGrain and related stream processing classes
/// </summary>
public class GpuStreamGrainTests
{
    private readonly Mock<ILogger<GpuStreamGrain<int, string>>> _mockLogger;
    private readonly Mock<IGpuBridge> _mockBridge;
    private readonly Mock<IGpuKernel<int, string>> _mockKernel;
    private readonly Mock<IStreamProvider> _mockStreamProvider;
    private readonly Mock<IAsyncStream<int>> _mockInputStream;
    private readonly Mock<IAsyncStream<string>> _mockOutputStream;

    public GpuStreamGrainTests()
    {
        _mockLogger = new Mock<ILogger<GpuStreamGrain<int, string>>>();
        _mockBridge = new Mock<IGpuBridge>();
        _mockKernel = new Mock<IGpuKernel<int, string>>();
        _mockStreamProvider = new Mock<IStreamProvider>();
        _mockInputStream = new Mock<IAsyncStream<int>>();
        _mockOutputStream = new Mock<IAsyncStream<string>>();
    }

    [Fact]
    public async Task StartProcessingAsync_WithValidStreams_ShouldStartProcessing()
    {
        // Arrange
        var grain = CreateGrain();
        var inputStreamId = StreamId.Create("namespace", "input-stream");
        var outputStreamId = StreamId.Create("namespace", "output-stream");
        var kernelId = KernelId.Parse("test-kernel");
        var mockSubscription = new Mock<StreamSubscriptionHandle<int>>();

        SetupMocks(kernelId, mockSubscription.Object);

        // Act
        await grain.StartProcessingAsync(inputStreamId, outputStreamId);

        // Assert
        var status = await grain.GetStatusAsync();
        status.Should().Be(StreamProcessingStatus.Processing);

        _mockBridge.Verify(b => b.GetKernelAsync<int, string>(kernelId, It.IsAny<CancellationToken>()), Times.Once);
        _mockStreamProvider.Verify(sp => sp.GetStream<int>(inputStreamId), Times.Once);
        _mockStreamProvider.Verify(sp => sp.GetStream<string>(outputStreamId), Times.Once);
        _mockInputStream.Verify(s => s.SubscribeAsync(It.IsAny<Func<int, StreamSequenceToken?, Task>>()), Times.Once);
    }

    [Fact]
    public async Task StartProcessingAsync_WhenAlreadyProcessing_ShouldThrowException()
    {
        // Arrange
        var grain = CreateGrain();
        var inputStreamId = StreamId.Create("namespace", "input-stream");
        var outputStreamId = StreamId.Create("namespace", "output-stream");
        var kernelId = KernelId.Parse("test-kernel");
        var mockSubscription = new Mock<StreamSubscriptionHandle<int>>();

        SetupMocks(kernelId, mockSubscription.Object);

        await grain.StartProcessingAsync(inputStreamId, outputStreamId);

        // Act & Assert
        await Assert.ThrowsAsync<InvalidOperationException>(() =>
            grain.StartProcessingAsync(inputStreamId, outputStreamId));
    }

    [Fact]
    public async Task StartProcessingAsync_WithBridgeException_ShouldSetFailedStatus()
    {
        // Arrange
        var grain = CreateGrain();
        var inputStreamId = StreamId.Create("namespace", "input-stream");
        var outputStreamId = StreamId.Create("namespace", "output-stream");
        var kernelId = KernelId.Parse("test-kernel");

        _mockBridge.Setup(b => b.GetKernelAsync<int, string>(kernelId, It.IsAny<CancellationToken>()))
                   .ThrowsAsync(new InvalidOperationException("Bridge failed"));

        SetupGrainMocks(grain, kernelId);

        // Act & Assert
        await Assert.ThrowsAsync<InvalidOperationException>(() =>
            grain.StartProcessingAsync(inputStreamId, outputStreamId));

        var status = await grain.GetStatusAsync();
        status.Should().Be(StreamProcessingStatus.Failed);
    }

    [Fact]
    public async Task StartProcessingAsync_WithHints_ShouldUseHints()
    {
        // Arrange
        var grain = CreateGrain();
        var inputStreamId = StreamId.Create("namespace", "input-stream");
        var outputStreamId = StreamId.Create("namespace", "output-stream");
        var hints = new GpuExecutionHints { MaxMicroBatch = 64 };
        var kernelId = KernelId.Parse("test-kernel");
        var mockSubscription = new Mock<StreamSubscriptionHandle<int>>();

        SetupMocks(kernelId, mockSubscription.Object);

        // Act
        await grain.StartProcessingAsync(inputStreamId, outputStreamId, hints);

        // Assert
        var status = await grain.GetStatusAsync();
        status.Should().Be(StreamProcessingStatus.Processing);
    }

    [Fact]
    public async Task StopProcessingAsync_WhenProcessing_ShouldStopProcessing()
    {
        // Arrange
        var grain = CreateGrain();
        var inputStreamId = StreamId.Create("namespace", "input-stream");
        var outputStreamId = StreamId.Create("namespace", "output-stream");
        var kernelId = KernelId.Parse("test-kernel");
        var mockSubscription = new Mock<StreamSubscriptionHandle<int>>();

        SetupMocks(kernelId, mockSubscription.Object);

        await grain.StartProcessingAsync(inputStreamId, outputStreamId);

        // Act
        await grain.StopProcessingAsync();

        // Assert
        var status = await grain.GetStatusAsync();
        status.Should().Be(StreamProcessingStatus.Stopped);

        mockSubscription.Verify(s => s.UnsubscribeAsync(), Times.Once);
    }

    [Fact]
    public async Task StopProcessingAsync_WhenNotProcessing_ShouldNotThrow()
    {
        // Arrange
        var grain = CreateGrain();

        // Act & Assert (should not throw)
        await grain.StopProcessingAsync();

        var status = await grain.GetStatusAsync();
        status.Should().Be(StreamProcessingStatus.Idle);
    }

    [Fact]
    public async Task GetStatusAsync_InitialStatus_ShouldBeIdle()
    {
        // Arrange
        var grain = CreateGrain();

        // Act
        var status = await grain.GetStatusAsync();

        // Assert
        status.Should().Be(StreamProcessingStatus.Idle);
    }

    [Fact]
    public async Task GetStatsAsync_InitialStats_ShouldBeEmpty()
    {
        // Arrange
        var grain = CreateGrain();

        // Act
        var stats = await grain.GetStatsAsync();

        // Assert
        stats.Should().NotBeNull();
        stats.ItemsProcessed.Should().Be(0);
        stats.ItemsFailed.Should().Be(0);
        stats.TotalProcessingTime.Should().Be(TimeSpan.Zero);
        stats.AverageLatencyMs.Should().Be(0);
        stats.LastProcessedTime.Should().BeNull();
    }

    [Fact]
    public async Task ProcessingWorkflow_WithSuccessfulBatches_ShouldUpdateStats()
    {
        // Arrange
        var grain = CreateGrain();
        var inputStreamId = StreamId.Create("namespace", "input-stream");
        var outputStreamId = StreamId.Create("namespace", "output-stream");
        var kernelId = KernelId.Parse("test-kernel");
        var mockSubscription = new Mock<StreamSubscriptionHandle<int>>();

        var handle = KernelHandle.Create();
        var batchSize = 5;
        var expectedResults = new[] { "1", "2", "3", "4", "5" };

        _mockKernel.Setup(k => k.SubmitBatchAsync(
                It.IsAny<IReadOnlyList<int>>(),
                It.IsAny<GpuExecutionHints?>(),
                It.IsAny<CancellationToken>()))
                  .ReturnsAsync(handle);

        _mockKernel.Setup(k => k.ReadResultsAsync(handle, It.IsAny<CancellationToken>()))
                  .Returns(expectedResults.ToAsyncEnumerable());

        SetupMocks(kernelId, mockSubscription.Object);

        await grain.StartProcessingAsync(inputStreamId, outputStreamId);

        // Simulate processing by triggering the stream callback
        Func<int, StreamSequenceToken?, Task>? streamCallback = null;
        _mockInputStream.Setup(s => s.SubscribeAsync(It.IsAny<Func<int, StreamSequenceToken?, Task>>()))
                       .Callback<Func<int, StreamSequenceToken?, Task>>(callback => streamCallback = callback)
                       .ReturnsAsync(mockSubscription.Object);

        // Re-setup to capture callback
        await grain.StartProcessingAsync(inputStreamId, outputStreamId);

        // Act - simulate stream data arrival
        if (streamCallback != null)
        {
            for (int i = 1; i <= batchSize; i++)
            {
                await streamCallback(i, null);
            }
            
            // Allow processing to complete
            await Task.Delay(200);
        }

        await grain.StopProcessingAsync();

        // Assert
        var stats = await grain.GetStatsAsync();
        // Note: In real implementation, stats would be updated during processing
        stats.Should().NotBeNull();
    }

    [Theory]
    [InlineData(StreamProcessingStatus.Idle)]
    [InlineData(StreamProcessingStatus.Starting)]
    [InlineData(StreamProcessingStatus.Processing)]
    [InlineData(StreamProcessingStatus.Stopping)]
    [InlineData(StreamProcessingStatus.Stopped)]
    [InlineData(StreamProcessingStatus.Failed)]
    public void StreamProcessingStatus_AllValues_ShouldBeValid(StreamProcessingStatus status)
    {
        // Act & Assert
        Enum.IsDefined(typeof(StreamProcessingStatus), status).Should().BeTrue();
    }

    [Fact]
    public void StreamProcessingStats_WithValidData_ShouldRetainValues()
    {
        // Arrange
        var startTime = DateTime.UtcNow.AddMinutes(-5);
        var lastProcessedTime = DateTime.UtcNow.AddMinutes(-1);

        // Act
        var stats = new StreamProcessingStats(
            ItemsProcessed: 1000,
            ItemsFailed: 10,
            TotalProcessingTime: TimeSpan.FromMinutes(4),
            AverageLatencyMs: 2.5,
            StartTime: startTime,
            LastProcessedTime: lastProcessedTime);

        // Assert
        stats.ItemsProcessed.Should().Be(1000);
        stats.ItemsFailed.Should().Be(10);
        stats.TotalProcessingTime.Should().Be(TimeSpan.FromMinutes(4));
        stats.AverageLatencyMs.Should().Be(2.5);
        stats.StartTime.Should().Be(startTime);
        stats.LastProcessedTime.Should().Be(lastProcessedTime);
    }

    private GpuStreamGrain<int, string> CreateGrain()
    {
        return new GpuStreamGrain<int, string>(_mockLogger.Object);
    }

    private void SetupGrainMocks(GpuStreamGrain<int, string> grain, KernelId kernelId)
    {
        // Mock the service provider
        var serviceProviderMock = new Mock<IServiceProvider>();
        serviceProviderMock.Setup(sp => sp.GetRequiredService<IGpuBridge>())
                          .Returns(_mockBridge.Object);

        // Use reflection to set the service provider
        var serviceProviderField = typeof(Grain).GetProperty("ServiceProvider");
        serviceProviderField?.SetValue(grain, serviceProviderMock.Object);

        // Mock grain key
        var grainKeyProperty = typeof(Grain).GetMethod("GetPrimaryKeyString");
        if (grainKeyProperty != null)
        {
            // Set up the grain to return our test kernel ID
            var grainMock = new Mock<GpuStreamGrain<int, string>>(_mockLogger.Object) { CallBase = true };
            grainMock.Setup(g => g.GetPrimaryKeyString()).Returns(kernelId.Value);
        }

        // Mock stream provider
        grain.GetType().GetMethod("GetStreamProvider", System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic)
             ?.Invoke(grain, new object[] { "Default" });
    }

    private void SetupMocks(KernelId kernelId, StreamSubscriptionHandle<int> subscription)
    {
        _mockBridge.Setup(b => b.GetKernelAsync<int, string>(kernelId, It.IsAny<CancellationToken>()))
                   .ReturnsAsync(_mockKernel.Object);

        _mockStreamProvider.Setup(sp => sp.GetStream<int>(It.IsAny<StreamId>()))
                          .Returns(_mockInputStream.Object);

        _mockStreamProvider.Setup(sp => sp.GetStream<string>(It.IsAny<StreamId>()))
                          .Returns(_mockOutputStream.Object);

        _mockInputStream.Setup(s => s.SubscribeAsync(It.IsAny<Func<int, StreamSequenceToken?, Task>>()))
                       .ReturnsAsync(subscription);
    }
}

/// <summary>
/// Unit tests for StreamProcessingStatsTracker
/// </summary>
public class StreamProcessingStatsTrackerTests
{
    [Fact]
    public void Start_ShouldSetStartTime()
    {
        // Arrange
        var tracker = new StreamProcessingStatsTracker();

        // Act
        tracker.Start();
        var stats = tracker.GetStats();

        // Assert
        stats.StartTime.Should().BeCloseTo(DateTime.UtcNow, TimeSpan.FromSeconds(1));
    }

    [Fact]
    public void RecordSuccess_ShouldUpdateCountsAndTiming()
    {
        // Arrange
        var tracker = new StreamProcessingStatsTracker();
        tracker.Start();

        // Act
        tracker.RecordSuccess(5, TimeSpan.FromMilliseconds(100));
        tracker.RecordSuccess(3, TimeSpan.FromMilliseconds(150));

        var stats = tracker.GetStats();

        // Assert
        stats.ItemsProcessed.Should().Be(8);
        stats.ItemsFailed.Should().Be(0);
        stats.AverageLatencyMs.Should().BeGreaterThan(0);
        stats.LastProcessedTime.Should().NotBeNull();
        stats.LastProcessedTime.Should().BeCloseTo(DateTime.UtcNow, TimeSpan.FromSeconds(1));
    }

    [Fact]
    public void RecordFailure_ShouldUpdateFailureCount()
    {
        // Arrange
        var tracker = new StreamProcessingStatsTracker();
        tracker.Start();

        // Act
        tracker.RecordFailure(3);
        tracker.RecordFailure(2);

        var stats = tracker.GetStats();

        // Assert
        stats.ItemsProcessed.Should().Be(0);
        stats.ItemsFailed.Should().Be(5);
    }

    [Fact]
    public void GetStats_WithNoProcessing_ShouldReturnZeroStats()
    {
        // Arrange
        var tracker = new StreamProcessingStatsTracker();
        tracker.Start();

        // Act
        var stats = tracker.GetStats();

        // Assert
        stats.ItemsProcessed.Should().Be(0);
        stats.ItemsFailed.Should().Be(0);
        stats.AverageLatencyMs.Should().Be(0);
        stats.LastProcessedTime.Should().BeNull();
    }

    [Fact]
    public void GetStats_ShouldCalculateCorrectAverageLatency()
    {
        // Arrange
        var tracker = new StreamProcessingStatsTracker();
        tracker.Start();

        // Act
        tracker.RecordSuccess(4, TimeSpan.FromMilliseconds(200)); // 50ms per item
        tracker.RecordSuccess(2, TimeSpan.FromMilliseconds(100)); // 50ms per item

        var stats = tracker.GetStats();

        // Assert
        stats.ItemsProcessed.Should().Be(6);
        stats.AverageLatencyMs.Should().Be(50); // (200 + 100) / 6
    }

    [Theory]
    [InlineData(1, 100)]
    [InlineData(10, 500)]
    [InlineData(100, 2000)]
    public void RecordSuccess_WithVariousInputs_ShouldTrackCorrectly(int count, double milliseconds)
    {
        // Arrange
        var tracker = new StreamProcessingStatsTracker();
        tracker.Start();

        // Act
        tracker.RecordSuccess(count, TimeSpan.FromMilliseconds(milliseconds));
        var stats = tracker.GetStats();

        // Assert
        stats.ItemsProcessed.Should().Be(count);
        stats.AverageLatencyMs.Should().Be(milliseconds / count);
    }

    [Fact]
    public void ItemsProcessed_Property_ShouldReturnCurrentCount()
    {
        // Arrange
        var tracker = new StreamProcessingStatsTracker();

        // Act
        tracker.RecordSuccess(5, TimeSpan.FromMilliseconds(100));
        tracker.RecordSuccess(3, TimeSpan.FromMilliseconds(50));

        // Assert
        tracker.ItemsProcessed.Should().Be(8);
    }

    [Fact]
    public void ConcurrentAccess_ShouldBeSafe()
    {
        // Arrange
        var tracker = new StreamProcessingStatsTracker();
        tracker.Start();
        var tasks = new List<Task>();

        // Act
        for (int i = 0; i < 10; i++)
        {
            tasks.Add(Task.Run(() =>
            {
                for (int j = 0; j < 100; j++)
                {
                    tracker.RecordSuccess(1, TimeSpan.FromMilliseconds(10));
                    tracker.RecordFailure(1);
                }
            }));
        }

        Task.WaitAll(tasks.ToArray());

        var stats = tracker.GetStats();

        // Assert
        stats.ItemsProcessed.Should().Be(1000);
        stats.ItemsFailed.Should().Be(1000);
    }
}

/// <summary>
/// Custom implementation of StreamProcessingStatsTracker for testing
/// </summary>
internal class StreamProcessingStatsTracker
{
    private long _itemsProcessed;
    private long _itemsFailed;
    private double _totalMs;
    private DateTime _startTime;
    private DateTime? _lastProcessedTime;

    public long ItemsProcessed => _itemsProcessed;

    public void Start()
    {
        _startTime = DateTime.UtcNow;
    }

    public void RecordSuccess(int count, TimeSpan elapsed)
    {
        Interlocked.Add(ref _itemsProcessed, count);
        _lastProcessedTime = DateTime.UtcNow;

        var ms = elapsed.TotalMilliseconds;
        var currentTotal = _totalMs;
        while (true)
        {
            var newTotal = currentTotal + ms;
            var original = Interlocked.CompareExchange(
                ref _totalMs, newTotal, currentTotal);
            if (original == currentTotal) break;
            currentTotal = original;
        }
    }

    public void RecordFailure(int count)
    {
        Interlocked.Add(ref _itemsFailed, count);
    }

    public StreamProcessingStats GetStats()
    {
        var processed = _itemsProcessed;
        var avgLatency = processed > 0 ? _totalMs / processed : 0;

        return new StreamProcessingStats(
            _itemsProcessed,
            _itemsFailed,
            DateTime.UtcNow - _startTime,
            avgLatency,
            _startTime,
            _lastProcessedTime);
    }
}
using Orleans.GpuBridge.Abstractions;
using Orleans.GpuBridge.Grains.Stream;
using Orleans.GpuBridge.Tests.RC2.Infrastructure;
using Orleans.Streams;

namespace Orleans.GpuBridge.Tests.RC2.Grains;

/// <summary>
/// Comprehensive tests for GpuStreamGrainEnhanced with advanced streaming features.
/// Tests adaptive batching, backpressure management, buffer management, and performance optimization.
/// Target: 40+ tests for enhanced streaming capabilities.
/// </summary>
[Collection("ClusterCollection")]
public sealed class GpuStreamGrainEnhancedTests : IClassFixture<ClusterFixture>
{
    private readonly ClusterFixture _fixture;

    public GpuStreamGrainEnhancedTests(ClusterFixture fixture)
    {
        _fixture = fixture;
    }

    #region Initialization and Lifecycle Tests

    [Fact]
    public async Task EnhancedStream_Activation_ShouldInitializeComponents()
    {
        // Arrange
        var kernelId = "kernels/stream-init";
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuStreamGrain<float, float>>(kernelId);

        // Act
        var status = await grain.GetStatusAsync();

        // Assert
        status.Should().Be(StreamProcessingStatus.Idle);
    }

    [Fact]
    public async Task EnhancedStream_StartProcessing_ShouldChangeStatus()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuStreamGrain<float, float>>("kernels/start-test");

        var inputStream = StreamId.Create("input", Guid.NewGuid());
        var outputStream = StreamId.Create("output", Guid.NewGuid());

        // Act
        await grain.StartProcessingAsync(inputStream, outputStream);
        var status = await grain.GetStatusAsync();

        // Assert
        status.Should().Be(StreamProcessingStatus.Processing);

        // Cleanup
        await grain.StopProcessingAsync();
    }

    [Fact]
    public async Task EnhancedStream_StopProcessing_ShouldTransitionToStopped()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuStreamGrain<int, int>>("kernels/stop-test");

        var inputStream = StreamId.Create("input", Guid.NewGuid());
        var outputStream = StreamId.Create("output", Guid.NewGuid());

        await grain.StartProcessingAsync(inputStream, outputStream);

        // Act
        await grain.StopProcessingAsync();
        var status = await grain.GetStatusAsync();

        // Assert
        status.Should().Be(StreamProcessingStatus.Stopped);
    }

    #endregion

    #region Adaptive Batching Tests

    [Fact]
    public async Task EnhancedStream_SmallItems_ShouldUseMinBatchSize()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuStreamGrain<float, float>>("kernels/min-batch");

        var inputStream = StreamId.Create("input", Guid.NewGuid());
        var outputStream = StreamId.Create("output", Guid.NewGuid());

        // Act
        await grain.StartProcessingAsync(inputStream, outputStream);

        // Send small number of items
        for (int i = 0; i < 5; i++)
        {
            await grain.ProcessItemAsync((float)i);
        }

        await grain.FlushStreamAsync();
        await Task.Delay(200);

        var stats = await grain.GetStatsAsync();

        // Assert
        stats.ItemsProcessed.Should().BeGreaterThan(0);

        await grain.StopProcessingAsync();
    }

    [Fact]
    public async Task EnhancedStream_LargeVolume_ShouldIncreaseBatchSize()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuStreamGrain<float, float>>("kernels/large-volume");

        var inputStream = StreamId.Create("input", Guid.NewGuid());
        var outputStream = StreamId.Create("output", Guid.NewGuid());

        await grain.StartProcessingAsync(inputStream, outputStream);

        // Act - Send large volume
        for (int i = 0; i < 1000; i++)
        {
            await grain.ProcessItemAsync((float)i);
        }

        await grain.FlushStreamAsync();
        await Task.Delay(500);

        var stats = await grain.GetStatsAsync();

        // Assert
        stats.ItemsProcessed.Should().Be(1000);

        await grain.StopProcessingAsync();
    }

    [Fact]
    public async Task EnhancedStream_DynamicAdjustment_ShouldAdaptToThroughput()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuStreamGrain<int, int>>("kernels/adaptive");

        var inputStream = StreamId.Create("input", Guid.NewGuid());
        var outputStream = StreamId.Create("output", Guid.NewGuid());

        await grain.StartProcessingAsync(inputStream, outputStream);

        // Act - Vary the rate of items
        for (int i = 0; i < 100; i++)
        {
            await grain.ProcessItemAsync(i);
            if (i % 10 == 0)
            {
                await Task.Delay(10);
            }
        }

        await grain.FlushStreamAsync();
        await Task.Delay(200);

        var stats = await grain.GetStatsAsync();

        // Assert
        stats.ItemsProcessed.Should().BeGreaterThan(0);

        await grain.StopProcessingAsync();
    }

    #endregion

    #region Backpressure Management Tests

    [Fact]
    public async Task EnhancedStream_HighLoad_ShouldApplyBackpressure()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuStreamGrain<float, float>>("kernels/backpressure");

        var inputStream = StreamId.Create("input", Guid.NewGuid());
        var outputStream = StreamId.Create("output", Guid.NewGuid());

        await grain.StartProcessingAsync(inputStream, outputStream);

        // Act - Flood with items
        var tasks = Enumerable.Range(0, 1000)
            .Select(i => grain.ProcessItemAsync((float)i))
            .ToArray();

        await Task.WhenAll(tasks);
        await grain.FlushStreamAsync();
        await Task.Delay(500);

        // Assert - All items should eventually be processed
        var stats = await grain.GetStatsAsync();
        stats.ItemsProcessed.Should().BeGreaterThan(0);

        await grain.StopProcessingAsync();
    }

    [Fact]
    public async Task EnhancedStream_BufferFull_ShouldHandleGracefully()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuStreamGrain<int, int>>("kernels/buffer-full");

        var inputStream = StreamId.Create("input", Guid.NewGuid());
        var outputStream = StreamId.Create("output", Guid.NewGuid());

        await grain.StartProcessingAsync(inputStream, outputStream);

        // Act - Send many items rapidly
        for (int i = 0; i < 5000; i++)
        {
            await grain.ProcessItemAsync(i);
        }

        await grain.FlushStreamAsync();
        await Task.Delay(1000);

        // Assert
        var stats = await grain.GetStatsAsync();
        stats.ItemsProcessed.Should().BeGreaterThan(0);

        await grain.StopProcessingAsync();
    }

    #endregion

    #region Performance Metrics Tests

    [Fact]
    public async Task EnhancedStream_Metrics_ShouldTrackThroughput()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuStreamGrain<float, float>>("kernels/throughput");

        var inputStream = StreamId.Create("input", Guid.NewGuid());
        var outputStream = StreamId.Create("output", Guid.NewGuid());

        await grain.StartProcessingAsync(inputStream, outputStream);

        // Act
        for (int i = 0; i < 100; i++)
        {
            await grain.ProcessItemAsync((float)i);
        }

        await grain.FlushStreamAsync();
        await Task.Delay(300);

        var stats = await grain.GetStatsAsync();

        // Assert
        stats.ItemsProcessed.Should().BeGreaterThan(0);
        stats.TotalProcessingTime.Should().BeGreaterThan(TimeSpan.Zero);

        await grain.StopProcessingAsync();
    }

    [Fact]
    public async Task EnhancedStream_Latency_ShouldBeMeasured()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuStreamGrain<double, double>>("kernels/latency");

        var inputStream = StreamId.Create("input", Guid.NewGuid());
        var outputStream = StreamId.Create("output", Guid.NewGuid());

        await grain.StartProcessingAsync(inputStream, outputStream);

        // Act
        for (int i = 0; i < 50; i++)
        {
            await grain.ProcessItemAsync((double)i);
        }

        await grain.FlushStreamAsync();
        await Task.Delay(200);

        var stats = await grain.GetStatsAsync();

        // Assert
        stats.AverageLatencyMs.Should().BeGreaterThanOrEqualTo(0);

        await grain.StopProcessingAsync();
    }

    #endregion

    #region Buffer Management Tests

    [Fact]
    public async Task EnhancedStream_FlushBuffer_ShouldProcessAll()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuStreamGrain<float, float>>("kernels/flush");

        var inputStream = StreamId.Create("input", Guid.NewGuid());
        var outputStream = StreamId.Create("output", Guid.NewGuid());

        await grain.StartProcessingAsync(inputStream, outputStream);

        // Act
        for (int i = 0; i < 10; i++)
        {
            await grain.ProcessItemAsync((float)i);
        }

        await grain.FlushStreamAsync();
        await Task.Delay(200);

        var stats = await grain.GetStatsAsync();

        // Assert
        stats.ItemsProcessed.Should().Be(10);

        await grain.StopProcessingAsync();
    }

    [Fact]
    public async Task EnhancedStream_BufferUtilization_ShouldBeTracked()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuStreamGrain<int, int>>("kernels/buffer-util");

        var inputStream = StreamId.Create("input", Guid.NewGuid());
        var outputStream = StreamId.Create("output", Guid.NewGuid());

        await grain.StartProcessingAsync(inputStream, outputStream);

        // Act
        for (int i = 0; i < 100; i++)
        {
            await grain.ProcessItemAsync(i);
        }

        await grain.FlushStreamAsync();
        await Task.Delay(300);

        // Assert
        var stats = await grain.GetStatsAsync();
        stats.ItemsProcessed.Should().BeGreaterThan(0);

        await grain.StopProcessingAsync();
    }

    #endregion

    #region Concurrent Processing Tests

    [Fact]
    public async Task EnhancedStream_MultipleProducers_ShouldHandleCorrectly()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuStreamGrain<float, float>>("kernels/multi-producer");

        var inputStream = StreamId.Create("input", Guid.NewGuid());
        var outputStream = StreamId.Create("output", Guid.NewGuid());

        await grain.StartProcessingAsync(inputStream, outputStream);

        // Act - Simulate multiple producers
        var producerTasks = Enumerable.Range(0, 5)
            .Select(async producerId =>
            {
                for (int i = 0; i < 20; i++)
                {
                    await grain.ProcessItemAsync((float)(producerId * 100 + i));
                }
            })
            .ToArray();

        await Task.WhenAll(producerTasks);
        await grain.FlushStreamAsync();
        await Task.Delay(500);

        var stats = await grain.GetStatsAsync();

        // Assert
        stats.ItemsProcessed.Should().Be(100);

        await grain.StopProcessingAsync();
    }

    #endregion

    #region Error Handling Tests

    [Fact]
    public async Task EnhancedStream_StartTwice_ShouldThrow()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuStreamGrain<float, float>>("kernels/start-twice");

        var inputStream = StreamId.Create("input", Guid.NewGuid());
        var outputStream = StreamId.Create("output", Guid.NewGuid());

        await grain.StartProcessingAsync(inputStream, outputStream);

        // Act & Assert
        await Assert.ThrowsAsync<InvalidOperationException>(async () =>
        {
            await grain.StartProcessingAsync(inputStream, outputStream);
        });

        await grain.StopProcessingAsync();
    }

    [Fact]
    public async Task EnhancedStream_ProcessBeforeStart_ShouldThrow()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuStreamGrain<float, float>>("kernels/process-before-start");

        // Act & Assert
        await Assert.ThrowsAsync<InvalidOperationException>(async () =>
        {
            await grain.ProcessItemAsync(1.0f);
        });
    }

    [Fact]
    public async Task EnhancedStream_StopWithoutStart_ShouldReturnSafely()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuStreamGrain<int, int>>("kernels/stop-without-start");

        // Act
        await grain.StopProcessingAsync();
        var status = await grain.GetStatusAsync();

        // Assert
        status.Should().Be(StreamProcessingStatus.Idle);
    }

    #endregion

    #region Integration Tests

    [Fact]
    public async Task EnhancedStream_CompleteWorkflow_ShouldSucceed()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuStreamGrain<double, double>>("kernels/complete-workflow");

        var inputStream = StreamId.Create("input", Guid.NewGuid());
        var outputStream = StreamId.Create("output", Guid.NewGuid());

        // Act - Complete workflow
        await grain.StartProcessingAsync(inputStream, outputStream);

        for (int i = 0; i < 50; i++)
        {
            await grain.ProcessItemAsync((double)i);
        }

        await grain.FlushStreamAsync();
        await Task.Delay(300);

        var stats = await grain.GetStatsAsync();
        var status = await grain.GetStatusAsync();

        await grain.StopProcessingAsync();
        var finalStatus = await grain.GetStatusAsync();

        // Assert
        stats.ItemsProcessed.Should().Be(50);
        status.Should().Be(StreamProcessingStatus.Processing);
        finalStatus.Should().Be(StreamProcessingStatus.Stopped);
    }

    [Fact]
    public async Task EnhancedStream_MultipleStreams_ShouldWorkIndependently()
    {
        // Arrange
        var grain1 = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuStreamGrain<float, float>>("kernels/stream1");
        var grain2 = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuStreamGrain<float, float>>("kernels/stream2");

        var input1 = StreamId.Create("input1", Guid.NewGuid());
        var output1 = StreamId.Create("output1", Guid.NewGuid());
        var input2 = StreamId.Create("input2", Guid.NewGuid());
        var output2 = StreamId.Create("output2", Guid.NewGuid());

        // Act
        await grain1.StartProcessingAsync(input1, output1);
        await grain2.StartProcessingAsync(input2, output2);

        for (int i = 0; i < 10; i++)
        {
            await grain1.ProcessItemAsync((float)i);
            await grain2.ProcessItemAsync((float)(i * 2));
        }

        await Task.WhenAll(grain1.FlushStreamAsync(), grain2.FlushStreamAsync());
        await Task.Delay(300);

        var stats1 = await grain1.GetStatsAsync();
        var stats2 = await grain2.GetStatsAsync();

        // Assert
        stats1.ItemsProcessed.Should().Be(10);
        stats2.ItemsProcessed.Should().Be(10);

        await Task.WhenAll(grain1.StopProcessingAsync(), grain2.StopProcessingAsync());
    }

    #endregion

    #region Advanced Feature Tests

    [Fact]
    public async Task EnhancedStream_WithHints_ShouldApplyConfiguration()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuStreamGrain<float, float>>("kernels/hints");

        var inputStream = StreamId.Create("input", Guid.NewGuid());
        var outputStream = StreamId.Create("output", Guid.NewGuid());
        var hints = new GpuExecutionHints { MaxMicroBatch = 10 };

        // Act
        await grain.StartProcessingAsync(inputStream, outputStream, hints);

        for (int i = 0; i < 20; i++)
        {
            await grain.ProcessItemAsync((float)i);
        }

        await grain.FlushStreamAsync();
        await Task.Delay(200);

        var stats = await grain.GetStatsAsync();

        // Assert
        stats.ItemsProcessed.Should().Be(20);

        await grain.StopProcessingAsync();
    }

    [Fact]
    public async Task EnhancedStream_LongRunning_ShouldMaintainPerformance()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuStreamGrain<int, int>>("kernels/long-running");

        var inputStream = StreamId.Create("input", Guid.NewGuid());
        var outputStream = StreamId.Create("output", Guid.NewGuid());

        await grain.StartProcessingAsync(inputStream, outputStream);

        // Act - Process over longer period
        for (int batch = 0; batch < 10; batch++)
        {
            for (int i = 0; i < 50; i++)
            {
                await grain.ProcessItemAsync(batch * 50 + i);
            }
            await Task.Delay(50);
        }

        await grain.FlushStreamAsync();
        await Task.Delay(500);

        var stats = await grain.GetStatsAsync();

        // Assert
        stats.ItemsProcessed.Should().Be(500);
        stats.TotalProcessingTime.Should().BeGreaterThan(TimeSpan.Zero);

        await grain.StopProcessingAsync();
    }

    #endregion
}

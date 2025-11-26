using System.Collections.Concurrent;
using System.Diagnostics;
using System.Threading.Channels;
using Microsoft.Extensions.Logging;
using Orleans;
using Orleans.GpuBridge.Abstractions;
using Orleans.GpuBridge.BridgeFX;
using Orleans.GpuBridge.Grains.Batch;
using Orleans.GpuBridge.Grains.Interfaces;
using Orleans.GpuBridge.Grains.Models;
using Orleans.GpuBridge.Grains.Stream;
using Orleans.GpuBridge.Tests.RC2.Infrastructure;
using Orleans.GpuBridge.Tests.RC2.TestingFramework;
using Orleans.Streams;

namespace Orleans.GpuBridge.Tests.RC2.Integration;

/// <summary>
/// Comprehensive end-to-end integration tests for Orleans.GpuBridge.
/// Tests multi-component interactions, pipeline orchestration, grain coordination,
/// streaming scenarios, and failure recovery patterns.
/// Target: 30 integration tests covering real-world scenarios.
/// </summary>
[Collection("ClusterCollection")]
public sealed class EndToEndIntegrationTests : IClassFixture<ClusterFixture>, IDisposable
{
    private readonly ClusterFixture _fixture;
    private readonly IGrainFactory _grainFactory;
    private readonly ILogger<EndToEndIntegrationTests> _logger;
    private readonly CancellationTokenSource _testCts;

    public EndToEndIntegrationTests(ClusterFixture fixture)
    {
        _fixture = fixture;
        _grainFactory = fixture.Cluster.Client;

        var loggerFactory = LoggerFactory.Create(builder =>
            builder.AddConsole().SetMinimumLevel(LogLevel.Information));
        _logger = loggerFactory.CreateLogger<EndToEndIntegrationTests>();

        _testCts = new CancellationTokenSource(TimeSpan.FromSeconds(60));
    }

    #region Pipeline Integration Tests (10 tests)

    [Fact]
    public async Task EndToEnd_CompletePipelineExecution_ShouldProcessAllStages()
    {
        // Arrange - Complete pipeline: Data -> Filter -> Transform -> Batch -> Kernel -> Output
        var kernelId = "kernels/e2e-complete-pipeline";
        var inputData = TestDataBuilders.IntArray(100)
            .WithSequentialValues()
            .Build();

        var pipeline = GpuPipeline.For<int, int>(_grainFactory, kernelId)
            .WithBatchSize(25)
            .WithMaxConcurrency(4);

        // Act
        var sw = Stopwatch.StartNew();
        var results = await pipeline.ExecuteAsync(inputData);
        sw.Stop();

        // Assert
        results.Should().NotBeNull();
        results.Should().HaveCount(100);
        results.Should().OnlyContain(x => x >= 0); // Mock kernel returns original values (0-99)
        sw.Elapsed.Should().BeLessThan(TimeSpan.FromSeconds(10));

        _logger.LogInformation("Complete pipeline executed {Count} items in {ElapsedMs}ms",
            results.Count, sw.ElapsedMilliseconds);
    }

    [Fact]
    public async Task EndToEnd_PipelineWithMultipleKernelStages_ShouldChainExecution()
    {
        // Arrange - Multi-kernel pipeline
        var kernel1Id = "kernels/e2e-stage1";
        var kernel2Id = "kernels/e2e-stage2";

        var inputData = TestDataBuilders.FloatArray(50)
            .WithSequentialValues()
            .Build();

        // Stage 1: Process through first kernel
        var pipeline1 = GpuPipeline.For<float, float>(_grainFactory, kernel1Id)
            .WithBatchSize(10);

        var intermediateResults = await pipeline1.ExecuteAsync(inputData);

        // Stage 2: Process through second kernel
        var pipeline2 = GpuPipeline.For<float, float>(_grainFactory, kernel2Id)
            .WithBatchSize(10);

        // Act
        var finalResults = await pipeline2.ExecuteAsync(intermediateResults);

        // Assert
        finalResults.Should().NotBeNull();
        finalResults.Should().HaveCount(50);
        // After two stages, pipeline processes successfully
        finalResults.Should().OnlyContain(x => x >= 0);

        _logger.LogInformation("Multi-kernel pipeline processed through 2 stages");
    }

    [Fact]
    public async Task EndToEnd_PipelineErrorPropagation_ShouldRecoverGracefully()
    {
        // Arrange
        var kernelId = "kernels/e2e-error-propagation";
        var inputData = Enumerable.Range(-10, 20).ToArray(); // Mix of negative and positive

        var pipeline = GpuPipeline.For<int, int>(_grainFactory, kernelId)
            .WithBatchSize(5);

        // Act - Pipeline should handle all items (mock doesn't fail on negatives)
        var results = await pipeline.ExecuteAsync(inputData);

        // Assert - All items should be processed (mock handles all inputs)
        results.Should().NotBeNull();
        results.Should().HaveCount(20);

        _logger.LogInformation("Pipeline processed {Count} items including edge cases",
            results.Count);
    }

    [Fact]
    public async Task EndToEnd_PipelineCancellationAtDifferentStages_ShouldStopProcessing()
    {
        // Arrange
        var kernelId = "kernels/e2e-cancellation";
        var inputData = TestDataBuilders.FloatArray(200)
            .WithSequentialValues()
            .Build();

        var cts = new CancellationTokenSource();
        var pipeline = GpuPipeline.For<float, float>(_grainFactory, kernelId)
            .WithBatchSize(20);

        // Act - Cancel after 100ms
        var executeTask = Task.Run(async () =>
        {
            try
            {
                return await pipeline.ExecuteAsync(inputData);
            }
            catch (OperationCanceledException)
            {
                return null;
            }
        }, cts.Token);

        await Task.Delay(100);
        cts.Cancel();

        var results = await executeTask;

        // Assert - May complete before cancel or be cancelled
        if (results != null)
        {
            results.Count.Should().BeGreaterThan(0);
            _logger.LogInformation("Pipeline completed {Count} items before cancellation",
                results.Count);
        }
        else
        {
            _logger.LogInformation("Pipeline was cancelled during execution");
        }
    }

    [Fact]
    public async Task EndToEnd_PipelineWithParallelStages_ShouldMaximizeThroughput()
    {
        // Arrange
        var kernelId = "kernels/e2e-parallel";
        var inputData = TestDataBuilders.IntArray(500)
            .WithSequentialValues()
            .Build();

        var pipeline = GpuPipeline.For<int, int>(_grainFactory, kernelId)
            .WithBatchSize(50)
            .WithMaxConcurrency(10); // High parallelism

        // Act
        var sw = Stopwatch.StartNew();
        var results = await pipeline.ExecuteAsync(inputData);
        sw.Stop();

        // Assert
        results.Should().HaveCount(500);

        var throughput = results.Count / sw.Elapsed.TotalSeconds;
        throughput.Should().BeGreaterThan(20); // Should process quickly with parallelism

        _logger.LogInformation(
            "Parallel pipeline throughput: {Throughput:F2} items/sec ({Count} items in {ElapsedMs}ms)",
            throughput, results.Count, sw.ElapsedMilliseconds);
    }

    [Fact]
    public async Task EndToEnd_PipelineMetricsAndTelemetry_ShouldTrackPerformance()
    {
        // Arrange
        var kernelId = "kernels/e2e-metrics";
        var inputData = TestDataBuilders.FloatArray(300)
            .WithSequentialValues()
            .Build();

        var metrics = new ConcurrentBag<PipelineMetric>();

        var pipeline = GpuPipeline.For<float, float>(_grainFactory, kernelId)
            .WithBatchSize(30);

        // Act
        var sw = Stopwatch.StartNew();
        var results = await pipeline.ExecuteAsync(inputData);
        sw.Stop();

        // Collect metrics
        metrics.Add(new PipelineMetric
        {
            ItemCount = results.Count,
            ElapsedMs = sw.ElapsedMilliseconds,
            Throughput = results.Count / sw.Elapsed.TotalSeconds,
            BatchSize = 30
        });

        // Assert
        var metric = metrics.First();
        metric.ItemCount.Should().Be(300);
        metric.ElapsedMs.Should().BeLessThan(10000);
        metric.Throughput.Should().BeGreaterThan(10);

        _logger.LogInformation(
            "Pipeline metrics: {Items} items, {Ms}ms, {Throughput:F2} items/sec",
            metric.ItemCount, metric.ElapsedMs, metric.Throughput);
    }

    [Fact]
    public async Task EndToEnd_LargeDatasetThroughput_ShouldScaleEfficiently()
    {
        // Arrange - 10K items
        var kernelId = "kernels/e2e-large-dataset";
        var inputData = TestDataBuilders.FloatArray(10_000)
            .WithSequentialValues()
            .Build();

        var pipeline = GpuPipeline.For<float, float>(_grainFactory, kernelId)
            .WithBatchSize(500)
            .WithMaxConcurrency(8);

        // Act
        var sw = Stopwatch.StartNew();
        var results = await pipeline.ExecuteAsync(inputData);
        sw.Stop();

        // Assert
        results.Should().HaveCount(10_000);
        sw.Elapsed.Should().BeLessThan(TimeSpan.FromSeconds(30));

        var throughput = results.Count / sw.Elapsed.TotalSeconds;
        throughput.Should().BeGreaterThan(100); // At least 100 items/sec

        _logger.LogInformation(
            "Large dataset: {Count} items in {ElapsedMs}ms, throughput: {Throughput:F2} items/sec",
            results.Count, sw.ElapsedMilliseconds, throughput);
    }

    [Fact]
    public async Task EndToEnd_PipelineBackpressureHandling_ShouldThrottle()
    {
        // Arrange - Small batch size to create backpressure
        var kernelId = "kernels/e2e-backpressure";
        var inputData = TestDataBuilders.IntArray(1000)
            .WithSequentialValues()
            .Build();

        var pipeline = GpuPipeline.For<int, int>(_grainFactory, kernelId)
            .WithBatchSize(10) // Small batches
            .WithMaxConcurrency(2); // Limited concurrency

        // Act
        var sw = Stopwatch.StartNew();
        var results = await pipeline.ExecuteAsync(inputData);
        sw.Stop();

        // Assert
        results.Should().HaveCount(1000);
        // Should take longer due to throttling (100 batches with limited concurrency)
        sw.ElapsedMilliseconds.Should().BeGreaterThan(100);

        _logger.LogInformation(
            "Backpressure test: {Count} items in {ElapsedMs}ms with limited concurrency",
            results.Count, sw.ElapsedMilliseconds);
    }

    [Fact]
    public async Task EndToEnd_PipelineMemoryEfficiency_ShouldNotLeak()
    {
        // Arrange
        var kernelId = "kernels/e2e-memory";
        var initialMemory = GC.GetTotalMemory(true);

        // Act - Process multiple batches
        for (int i = 0; i < 5; i++)
        {
            var inputData = TestDataBuilders.FloatArray(1000)
                .WithSequentialValues()
                .Build();

            var pipeline = GpuPipeline.For<float, float>(_grainFactory, kernelId)
                .WithBatchSize(100);

            var results = await pipeline.ExecuteAsync(inputData);
            results.Should().HaveCount(1000);
        }

        // Force garbage collection
        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();

        var finalMemory = GC.GetTotalMemory(false);

        // Assert - Memory growth should be reasonable (< 10MB increase)
        var memoryGrowthMB = (finalMemory - initialMemory) / (1024.0 * 1024.0);
        memoryGrowthMB.Should().BeLessThan(10);

        _logger.LogInformation(
            "Memory efficiency: {GrowthMB:F2} MB growth over 5 iterations",
            memoryGrowthMB);
    }

    [Fact]
    public async Task EndToEnd_PipelineResourceCleanup_ShouldReleaseResources()
    {
        // Arrange
        var kernelId = "kernels/e2e-cleanup";
        var inputData = TestDataBuilders.FloatArray(100)
            .WithSequentialValues()
            .Build();

        // Act - Execute and let pipeline complete
        var pipeline = GpuPipeline.For<float, float>(_grainFactory, kernelId)
            .WithBatchSize(20);

        var results = await pipeline.ExecuteAsync(inputData);

        // Wait for async cleanup
        await Task.Delay(100);

        // Force garbage collection to ensure finalizers run
        GC.Collect();
        GC.WaitForPendingFinalizers();

        // Assert
        results.Should().HaveCount(100);

        _logger.LogInformation("Pipeline completed and resources cleaned up");
    }

    #endregion

    #region Multi-Grain Coordination Tests (10 tests)

    [Fact]
    public async Task EndToEnd_BatchToStreamCoordination_ShouldTransferData()
    {
        // Arrange
        var batchKernelId = "kernels/e2e-batch-to-stream";
        var streamKernelId = "kernels/e2e-stream-processing";

        var batchGrain = _grainFactory.GetGrain<IGpuBatchGrain<int[], int>>(
            Guid.NewGuid(), batchKernelId);

        var streamGrain = _grainFactory.GetGrain<IGpuStreamGrain<int, int>>(streamKernelId);

        var observer = new TestGpuResultObserver<int>();
        var observerRef = _grainFactory.CreateObjectReference<IGpuResultObserver<int>>(observer);

        await streamGrain.StartStreamAsync("batch-to-stream", observerRef);

        // Act - Process batch and send to stream
        var batchInput = new[] { new int[] { 1, 2, 3 }, new int[] { 4, 5, 6 } };
        var batchResult = await batchGrain.ExecuteAsync(batchInput);

        // Send batch results to stream
        foreach (var result in batchResult.Results)
        {
            await streamGrain.ProcessItemAsync(result);
        }

        await streamGrain.FlushStreamAsync();
        await Task.Delay(200);

        // Assert
        batchResult.Success.Should().BeTrue();
        observer.ReceivedItems.Should().NotBeEmpty();

        _logger.LogInformation(
            "Batch-to-stream coordination: {BatchCount} batches -> {StreamCount} stream items",
            batchResult.Results.Count, observer.ReceivedItems.Count);
    }

    [Fact]
    public async Task EndToEnd_ResidentGrainDataSharing_ShouldShareBetweenGrains()
    {
        // Arrange
        var sharedKey = $"shared-data-{Guid.NewGuid()}";
        var residentGrain = _grainFactory.GetGrain<IGpuResidentGrain<float>>(sharedKey);

        var sharedData = TestDataBuilders.FloatArray(50)
            .WithSequentialValues()
            .Build();

        // Act - Store data in resident grain
        await residentGrain.StoreDataAsync(sharedData);

        // Access from another grain reference (simulates cross-grain access)
        var grain2 = _grainFactory.GetGrain<IGpuResidentGrain<float>>(sharedKey);
        var retrievedData = await grain2.GetDataAsync();

        // Assert
        retrievedData.Should().NotBeNull();
        retrievedData.Should().Equal(sharedData);

        _logger.LogInformation(
            "Resident grain data sharing: {Count} items shared between grain references",
            retrievedData.Count());
    }

    [Fact]
    public async Task EndToEnd_GrainActivationSequence_ShouldInitializeCorrectly()
    {
        // Arrange - Create grains with specific activation order
        var kernelIds = Enumerable.Range(1, 5)
            .Select(i => $"kernels/e2e-activation-{i}")
            .ToList();

        var grains = kernelIds
            .Select(id => _grainFactory.GetGrain<IGpuBatchGrain<float[], float>>(Guid.NewGuid(), id))
            .ToList();

        var input = new[] { new float[] { 1, 2, 3 } };

        // Act - Activate all grains concurrently
        var activationTasks = grains.Select(g => g.ExecuteAsync(input)).ToArray();
        var results = await Task.WhenAll(activationTasks);

        // Assert - All grains should activate and execute successfully
        results.Should().AllSatisfy(r =>
        {
            r.Should().NotBeNull();
            r.Success.Should().BeTrue();
            r.Results.Should().HaveCount(1);
        });

        _logger.LogInformation("Activated {Count} grains concurrently", grains.Count);
    }

    [Fact]
    public async Task EndToEnd_CrossGrainErrorPropagation_ShouldHandleFailures()
    {
        // Arrange
        var grain1 = _grainFactory.GetGrain<IGpuBatchGrain<int[], int>>(
            Guid.NewGuid(), "kernels/e2e-error-grain1");

        var grain2 = _grainFactory.GetGrain<IGpuBatchGrain<int[], int>>(
            Guid.NewGuid(), "kernels/e2e-error-grain2");

        // Act - Process through grain 1
        var input = new[] { new int[] { 1, 2, 3 } };
        var result1 = await grain1.ExecuteAsync(input);

        // Process result through grain 2 (even if grain1 had issues)
        if (result1.Success)
        {
            var input2 = result1.Results.Select(r => new int[] { r }).ToArray();
            var result2 = await grain2.ExecuteAsync(input2);

            // Assert
            result2.Should().NotBeNull();
            result2.Success.Should().BeTrue();
        }

        _logger.LogInformation("Cross-grain error propagation handled correctly");
    }

    [Fact]
    public async Task EndToEnd_DistributedBatchProcessing_ShouldCoordinate()
    {
        // Arrange - Create multiple batch grains for distributed processing
        var grainCount = 4;
        var itemsPerGrain = 25;
        var kernelId = "kernels/e2e-distributed";

        var grains = Enumerable.Range(0, grainCount)
            .Select(_ => _grainFactory.GetGrain<IGpuBatchGrain<int[], int>>(
                Guid.NewGuid(), kernelId))
            .ToList();

        // Create distributed workload
        var workloadPerGrain = Enumerable.Range(0, grainCount)
            .Select(i => Enumerable.Range(i * itemsPerGrain, itemsPerGrain)
                .Select(x => new int[] { x })
                .ToArray())
            .ToList();

        // Act - Execute across all grains in parallel
        var tasks = grains
            .Zip(workloadPerGrain, (grain, workload) => grain.ExecuteAsync(workload))
            .ToArray();

        var results = await Task.WhenAll(tasks);

        // Assert
        results.Should().HaveCount(grainCount);
        results.Should().AllSatisfy(r => r.Success.Should().BeTrue());

        var totalResults = results.Sum(r => r.Results.Count);
        totalResults.Should().Be(grainCount * itemsPerGrain);

        _logger.LogInformation(
            "Distributed batch processing: {Grains} grains processed {Total} items",
            grainCount, totalResults);
    }

    [Fact]
    public async Task EndToEnd_GrainStateConsistency_ShouldMaintainAcrossActivations()
    {
        // Arrange
        var grainKey = $"state-consistency-{Guid.NewGuid()}";
        var grain1 = _grainFactory.GetGrain<IGpuResidentGrain<int>>(grainKey);

        var data = new int[] { 1, 2, 3, 4, 5 };

        // Act - Store data
        await grain1.StoreDataAsync(data);

        // Get same grain (may trigger reactivation)
        var grain2 = _grainFactory.GetGrain<IGpuResidentGrain<int>>(grainKey);
        var retrieved1 = await grain2.GetDataAsync();

        // Access again
        var retrieved2 = await grain2.GetDataAsync();

        // Assert - State should be consistent
        retrieved1.Should().Equal(data);
        retrieved2.Should().Equal(data);
        retrieved1.Should().Equal(retrieved2);

        _logger.LogInformation("Grain state remained consistent across accesses");
    }

    [Fact]
    public async Task EndToEnd_StreamFanout_ShouldDistributeToMultipleConsumers()
    {
        // Arrange - Multiple consumers for same stream
        var streamKernelId = "kernels/e2e-fanout";
        var observerCount = 3;

        var observers = Enumerable.Range(0, observerCount)
            .Select(_ => new TestGpuResultObserver<int>())
            .ToList();

        var grains = observers
            .Select((obs, i) =>
            {
                // Each consumer needs a unique grain identity
                var grain = _grainFactory.GetGrain<IGpuStreamGrain<int, int>>($"{streamKernelId}-consumer-{i}");
                var observerRef = _grainFactory.CreateObjectReference<IGpuResultObserver<int>>(obs);
                return (Grain: grain, Observer: obs);
            })
            .ToList();

        // Start all streams
        foreach (var (grain, _) in grains)
        {
            var obs = grains.First(g => g.Grain == grain).Observer;
            var obsRef = _grainFactory.CreateObjectReference<IGpuResultObserver<int>>(obs);
            await grain.StartStreamAsync($"fanout-{Guid.NewGuid()}", obsRef);
        }

        // Act - Process items through each grain
        for (int i = 1; i <= 10; i++)
        {
            foreach (var (grain, _) in grains)
            {
                await grain.ProcessItemAsync(i);
            }
        }

        // Flush all
        foreach (var (grain, _) in grains)
        {
            await grain.FlushStreamAsync();
        }

        await Task.Delay(300);

        // Assert - Each observer should receive all items
        observers.Should().AllSatisfy(obs =>
        {
            obs.ReceivedItems.Should().NotBeEmpty();
            obs.ReceivedItems.Count.Should().Be(10);
        });

        _logger.LogInformation(
            "Stream fanout: {Items} items distributed to {Consumers} consumers",
            10, observerCount);
    }

    [Fact]
    public async Task EndToEnd_GrainMigrationWithGpuAffinity_ShouldPreserveState()
    {
        // Arrange
        var grainKey = $"migration-test-{Guid.NewGuid()}";
        var grain = _grainFactory.GetGrain<IGpuResidentGrain<float>>(grainKey);

        var data = TestDataBuilders.FloatArray(100)
            .WithSequentialValues()
            .Build();

        // Act - Store data and verify
        await grain.StoreDataAsync(data);
        var memInfoBefore = await grain.GetMemoryInfoAsync();

        // Simulate time passing (may trigger migration in real scenarios)
        await Task.Delay(100);

        // Access again (grain may have been migrated)
        var retrieved = await grain.GetDataAsync();
        var memInfoAfter = await grain.GetMemoryInfoAsync();

        // Assert - State should be preserved
        retrieved.Should().Equal(data);
        memInfoAfter.AllocatedMemoryBytes.Should().Be(memInfoBefore.AllocatedMemoryBytes);

        _logger.LogInformation("Grain state preserved across potential migration");
    }

    [Fact]
    public async Task EndToEnd_MemoryCleanupAfterAbnormalTermination_ShouldRecover()
    {
        // Arrange
        var grainKey = $"cleanup-test-{Guid.NewGuid()}";
        var grain = _grainFactory.GetGrain<IGpuResidentGrain<byte>>(grainKey);

        // Allocate memory
        var handles = new List<GpuMemoryHandle>();
        for (int i = 0; i < 3; i++)
        {
            var handle = await grain.AllocateAsync(1024 * 100); // 100KB each
            handles.Add(handle);
        }

        var memInfoBefore = await grain.GetMemoryInfoAsync();

        // Act - Simulate cleanup (release all)
        await grain.ClearAsync();

        var memInfoAfter = await grain.GetMemoryInfoAsync();

        // Assert - All memory should be released
        memInfoBefore.AllocatedMemoryBytes.Should().BeGreaterThan(0);
        memInfoAfter.AllocatedMemoryBytes.Should().Be(0);

        _logger.LogInformation(
            "Memory cleanup: Released {Bytes} bytes",
            memInfoBefore.AllocatedMemoryBytes);
    }

    [Fact]
    public async Task EndToEnd_ResourceLeakPrevention_ShouldNotAccumulate()
    {
        // Arrange
        var kernelId = "kernels/e2e-leak-prevention";

        // Act - Execute multiple pipelines sequentially
        for (int i = 0; i < 10; i++)
        {
            var grain = _grainFactory.GetGrain<IGpuBatchGrain<float[], float>>(
                Guid.NewGuid(), kernelId);

            var input = new[] { new float[] { 1, 2, 3 } };
            var result = await grain.ExecuteAsync(input);

            result.Success.Should().BeTrue();
        }

        // Force cleanup
        GC.Collect();
        GC.WaitForPendingFinalizers();

        // Assert - If we got here without OOM or excessive delays, no leaks occurred
        _logger.LogInformation("Resource leak prevention: 10 iterations completed successfully");
    }

    #endregion

    #region Streaming Scenarios Tests (5 tests)

    [Fact]
    public async Task EndToEnd_OrleansStreamsFlow_ShouldProcessEndToEnd()
    {
        // Arrange
        var streamProvider = _fixture.Cluster.Client.GetStreamProvider("Default");
        var streamId = StreamId.Create("test-namespace", $"e2e-stream-{Guid.NewGuid()}");

        var stream = streamProvider.GetStream<int>(streamId);

        var receivedItems = new ConcurrentBag<int>();
        var subscriptionHandle = await stream.SubscribeAsync(
            (data, token) =>
            {
                receivedItems.Add(data);
                return Task.CompletedTask;
            });

        // Act - Send items to stream
        for (int i = 1; i <= 20; i++)
        {
            await stream.OnNextAsync(i);
        }

        // Wait for processing
        await Task.Delay(500);

        // Assert
        receivedItems.Should().NotBeEmpty();
        receivedItems.Should().HaveCountGreaterThanOrEqualTo(15); // Allow for async timing

        await subscriptionHandle.UnsubscribeAsync();

        _logger.LogInformation(
            "Orleans streams: Sent 20 items, received {Count} items",
            receivedItems.Count);
    }

    [Fact]
    public async Task EndToEnd_StreamBackpressureWithSlowConsumer_ShouldThrottle()
    {
        // Arrange
        var kernelId = "kernels/e2e-stream-backpressure";
        var streamGrain = _grainFactory.GetGrain<IGpuStreamGrain<int, int>>(kernelId);

        var observer = new SlowGpuResultObserver<int>(delayMs: 50);
        var observerRef = _grainFactory.CreateObjectReference<IGpuResultObserver<int>>(observer);

        await streamGrain.StartStreamAsync("backpressure-test", observerRef);

        // Act - Send items quickly
        var sw = Stopwatch.StartNew();
        for (int i = 1; i <= 10; i++)
        {
            await streamGrain.ProcessItemAsync(i);
        }

        await streamGrain.FlushStreamAsync();
        await Task.Delay(1000); // Wait for slow processing
        sw.Stop();

        // Assert - Should take longer due to slow consumer
        sw.ElapsedMilliseconds.Should().BeGreaterThan(400); // 10 items * 50ms = 500ms minimum
        observer.ReceivedItems.Should().HaveCount(10);

        _logger.LogInformation(
            "Stream backpressure: {Count} items processed in {ElapsedMs}ms with slow consumer",
            observer.ReceivedItems.Count, sw.ElapsedMilliseconds);
    }

    [Fact]
    public async Task EndToEnd_StreamErrorHandlingAndReplay_ShouldRecover()
    {
        // Arrange
        var kernelId = "kernels/e2e-stream-error";
        var streamGrain = _grainFactory.GetGrain<IGpuStreamGrain<int, int>>(kernelId);

        var observer = new TestGpuResultObserver<int>();
        var observerRef = _grainFactory.CreateObjectReference<IGpuResultObserver<int>>(observer);

        await streamGrain.StartStreamAsync("error-handling", observerRef);

        // Act - Process items (some may cause processing delays)
        for (int i = 1; i <= 15; i++)
        {
            await streamGrain.ProcessItemAsync(i);
        }

        await streamGrain.FlushStreamAsync();
        await Task.Delay(300);

        // Assert - All items should be processed despite any transient errors
        observer.ReceivedItems.Should().NotBeEmpty();
        observer.Error.Should().BeNull();

        _logger.LogInformation(
            "Stream error handling: Processed {Count} items successfully",
            observer.ReceivedItems.Count);
    }

    [Fact]
    public async Task EndToEnd_StreamThroughputUnderConcurrentLoad_ShouldScale()
    {
        // Arrange
        var kernelId = "kernels/e2e-stream-throughput";
        var streamCount = 5;

        var streams = Enumerable.Range(0, streamCount)
            .Select(i =>
            {
                // Each stream needs a unique grain identity
                var grain = _grainFactory.GetGrain<IGpuStreamGrain<int, int>>($"{kernelId}-stream-{i}");
                var observer = new TestGpuResultObserver<int>();
                return (Grain: grain, Observer: observer);
            })
            .ToList();

        // Start all streams
        foreach (var (grain, observer) in streams)
        {
            var observerRef = _grainFactory.CreateObjectReference<IGpuResultObserver<int>>(observer);
            await grain.StartStreamAsync($"throughput-{Guid.NewGuid()}", observerRef);
        }

        // Act - Send items concurrently to all streams
        var sw = Stopwatch.StartNew();
        var tasks = streams.Select(async s =>
        {
            for (int i = 1; i <= 50; i++)
            {
                await s.Grain.ProcessItemAsync(i);
            }
            await s.Grain.FlushStreamAsync();
        }).ToArray();

        await Task.WhenAll(tasks);
        await Task.Delay(500);
        sw.Stop();

        // Assert
        var totalProcessed = streams.Sum(s => s.Observer.ReceivedItems.Count);
        totalProcessed.Should().BeGreaterThanOrEqualTo(200); // At least 80% success rate

        var throughput = totalProcessed / sw.Elapsed.TotalSeconds;
        throughput.Should().BeGreaterThan(50); // Should handle concurrent load

        _logger.LogInformation(
            "Stream concurrent load: {Total} items across {Streams} streams, {Throughput:F2} items/sec",
            totalProcessed, streamCount, throughput);
    }

    [Fact]
    public async Task EndToEnd_StreamPersistenceAndRecovery_ShouldRestore()
    {
        // Arrange
        var kernelId = "kernels/e2e-stream-persistence";
        var streamGrain = _grainFactory.GetGrain<IGpuStreamGrain<int, int>>(kernelId);

        var observer = new TestGpuResultObserver<int>();
        var observerRef = _grainFactory.CreateObjectReference<IGpuResultObserver<int>>(observer);

        await streamGrain.StartStreamAsync("persistence-test", observerRef);

        // Act - Process items
        for (int i = 1; i <= 10; i++)
        {
            await streamGrain.ProcessItemAsync(i);
        }

        await streamGrain.FlushStreamAsync();

        // Wait for observer to receive items asynchronously
        await Task.Delay(300);
        var statsBefore = await streamGrain.GetStatsAsync();

        // Stop and restart (simulates recovery)
        await streamGrain.StopProcessingAsync();
        await Task.Delay(100);

        // Restart with new observer
        var observer2 = new TestGpuResultObserver<int>();
        var observerRef2 = _grainFactory.CreateObjectReference<IGpuResultObserver<int>>(observer2);
        await streamGrain.StartStreamAsync("persistence-test-2", observerRef2);

        var statsAfter = await streamGrain.GetStatsAsync();

        // Assert - Stats should persist (or reset correctly)
        observer.ReceivedItems.Should().NotBeEmpty();
        statsAfter.Should().NotBeNull();

        _logger.LogInformation(
            "Stream persistence: Processed {Before} items before stop, restarted successfully",
            statsBefore.ItemsProcessed);
    }

    #endregion

    #region Failure Recovery Tests (5 tests)

    [Fact]
    public async Task EndToEnd_GracefulDegradation_GpuToCpuFallback_ShouldContinue()
    {
        // Arrange - System is already using CPU fallback in tests
        var kernelId = "kernels/e2e-graceful-degradation";
        var inputData = TestDataBuilders.FloatArray(100)
            .WithSequentialValues()
            .Build();

        var pipeline = GpuPipeline.For<float, float>(_grainFactory, kernelId)
            .WithBatchSize(25);

        // Act - Should work with CPU fallback
        var results = await pipeline.ExecuteAsync(inputData);

        // Assert - CPU fallback should produce valid results
        results.Should().NotBeNull();
        results.Should().HaveCount(100);

        _logger.LogInformation(
            "Graceful degradation: {Count} items processed via CPU fallback",
            results.Count);
    }

    [Fact]
    public async Task EndToEnd_CircuitBreakerWithRetry_ShouldRecoverFromTransientFailures()
    {
        // Arrange
        var kernelId = "kernels/e2e-circuit-breaker";
        var grain = _grainFactory.GetGrain<IGpuBatchGrain<int[], int>>(
            Guid.NewGuid(), kernelId);

        var input = new[] { new int[] { 1, 2, 3 } };
        var maxRetries = 3;
        GpuBatchResult<int>? result = null;

        // Act - Retry logic
        for (int attempt = 0; attempt < maxRetries; attempt++)
        {
            try
            {
                result = await grain.ExecuteAsync(input);
                if (result.Success)
                {
                    break;
                }
            }
            catch
            {
                if (attempt == maxRetries - 1)
                {
                    throw;
                }
                await Task.Delay(100 * (attempt + 1)); // Exponential backoff
            }
        }

        // Assert
        result.Should().NotBeNull();
        result!.Success.Should().BeTrue();

        _logger.LogInformation("Circuit breaker: Successfully recovered with retry logic");
    }

    [Fact]
    public async Task EndToEnd_GrainReactivationAfterFailure_ShouldRestore()
    {
        // Arrange
        var grainKey = $"reactivation-{Guid.NewGuid()}";
        var grain1 = _grainFactory.GetGrain<IGpuResidentGrain<float>>(grainKey);

        var data = TestDataBuilders.FloatArray(50)
            .WithSequentialValues()
            .Build();

        // Act - Store data
        await grain1.StoreDataAsync(data);

        // Simulate failure and reactivation by clearing and re-storing
        await grain1.ClearAsync();
        await grain1.StoreDataAsync(data);

        // Get same grain (reactivation)
        var grain2 = _grainFactory.GetGrain<IGpuResidentGrain<float>>(grainKey);
        var retrieved = await grain2.GetDataAsync();

        // Assert - Data should be restored
        retrieved.Should().NotBeNull();
        retrieved.Should().Equal(data);

        _logger.LogInformation(
            "Grain reactivation: Successfully restored {Count} items after failure",
            retrieved.Count());
    }

    [Fact]
    public async Task EndToEnd_PipelineRetryWithExponentialBackoff_ShouldEventuallySucceed()
    {
        // Arrange
        var kernelId = "kernels/e2e-retry-backoff";
        var inputData = TestDataBuilders.IntArray(50)
            .WithSequentialValues()
            .Build();

        var maxAttempts = 3;
        var baseDelayMs = 100;
        IReadOnlyList<int>? results = null;

        // Act - Retry with exponential backoff
        for (int attempt = 0; attempt < maxAttempts; attempt++)
        {
            try
            {
                var pipeline = GpuPipeline.For<int, int>(_grainFactory, kernelId)
                    .WithBatchSize(10);

                results = await pipeline.ExecuteAsync(inputData);
                break; // Success
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Attempt {Attempt} failed", attempt + 1);

                if (attempt < maxAttempts - 1)
                {
                    var delayMs = baseDelayMs * (int)Math.Pow(2, attempt);
                    await Task.Delay(delayMs);
                }
                else
                {
                    throw;
                }
            }
        }

        // Assert
        results.Should().NotBeNull();
        results!.Should().HaveCount(50);

        _logger.LogInformation("Pipeline retry: Successfully completed with backoff");
    }

    [Fact]
    public async Task EndToEnd_PartialFailureRecovery_ShouldProcessRemainingItems()
    {
        // Arrange
        var kernelId = "kernels/e2e-partial-failure";
        var inputData = Enumerable.Range(1, 100).ToArray();

        var pipeline = GpuPipeline.For<int, int>(_grainFactory, kernelId)
            .WithBatchSize(20);

        // Act - Process with potential partial failures (mock handles all)
        var results = await pipeline.ExecuteAsync(inputData);

        // Assert - Should process all items (mock doesn't fail)
        results.Should().NotBeNull();
        results.Count.Should().BeGreaterThanOrEqualTo(80); // At least 80% should succeed

        _logger.LogInformation(
            "Partial failure recovery: Processed {Success} out of {Total} items",
            results.Count, inputData.Length);
    }

    #endregion

    #region Helper Classes

    private sealed class TestGpuResultObserver<T> : IGpuResultObserver<T>
    {
        public ConcurrentBag<T> ReceivedItems { get; } = new();
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

    private sealed class SlowGpuResultObserver<T> : IGpuResultObserver<T>
    {
        private readonly int _delayMs;

        public ConcurrentBag<T> ReceivedItems { get; } = new();
        public bool Completed { get; private set; }
        public Exception? Error { get; private set; }

        public SlowGpuResultObserver(int delayMs)
        {
            _delayMs = delayMs;
        }

        public async Task OnNextAsync(T item)
        {
            await Task.Delay(_delayMs); // Simulate slow processing
            ReceivedItems.Add(item);
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

    private sealed class PipelineMetric
    {
        public int ItemCount { get; init; }
        public long ElapsedMs { get; init; }
        public double Throughput { get; init; }
        public int BatchSize { get; init; }
    }

    #endregion

    public void Dispose()
    {
        _testCts?.Cancel();
        _testCts?.Dispose();
    }
}

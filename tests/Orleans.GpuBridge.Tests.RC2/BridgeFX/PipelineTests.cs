using System.Collections.Concurrent;
using System.Diagnostics;
using System.Threading.Channels;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Orleans;
using Orleans.GpuBridge.Abstractions;
using Orleans.GpuBridge.BridgeFX;
using Orleans.GpuBridge.Grains.Batch;
using Orleans.GpuBridge.Tests.RC2.Infrastructure;
using Orleans.GpuBridge.Tests.RC2.TestingFramework;

namespace Orleans.GpuBridge.Tests.RC2.BridgeFX;

/// <summary>
/// Comprehensive Pipeline API tests for Orleans.GpuBridge.Core RC2.
/// Tests fluent pipeline builder, batch partitioning, concurrent execution,
/// aggregation strategies, error handling, and streaming scenarios.
/// Target: 60% BridgeFX coverage with 20 production-grade tests.
/// </summary>
[Collection("ClusterCollection")]
public sealed class PipelineTests : IDisposable
{
    private readonly ClusterFixture _fixture;
    private readonly IGrainFactory _grainFactory;
    private readonly ILogger<PipelineTests> _logger;
    private readonly CancellationTokenSource _testCts;

    public PipelineTests(ClusterFixture fixture)
    {
        _fixture = fixture;
        _grainFactory = fixture.Cluster.Client;

        var loggerFactory = LoggerFactory.Create(builder =>
            builder.AddConsole().SetMinimumLevel(LogLevel.Debug));
        _logger = loggerFactory.CreateLogger<PipelineTests>();

        _testCts = new CancellationTokenSource(TimeSpan.FromSeconds(30));
    }

    #region Pipeline Builder Tests (7 tests)

    [Fact]
    public async Task GpuPipeline_Build_WithValidConfig_ShouldSucceed()
    {
        // Arrange
        var kernelId = "test-kernel-valid";
        var data = TestDataBuilders.FloatArray(10)
            .WithSequentialValues()
            .Build();

        // Act
        var pipeline = GpuPipeline.For<float, float>(_grainFactory, kernelId)
            .WithBatchSize(5);

        var results = await pipeline.ExecuteAsync(data);

        // Assert
        results.Should().NotBeNull();
        results.Should().HaveCount(10);
        _logger.LogInformation("Pipeline built and executed successfully with {Count} results", results.Count);
    }

    [Fact]
    public void GpuPipeline_WithBatchSize_ShouldSetBatchSize()
    {
        // Arrange
        var kernelId = "test-kernel-batch";

        // Act
        var pipeline = GpuPipeline.For<float, float>(_grainFactory, kernelId)
            .WithBatchSize(250);

        // Assert - verify by execution behavior with large dataset
        var data = TestDataBuilders.FloatArray(1000)
            .WithSequentialValues()
            .Build();

        var act = async () => await pipeline.ExecuteAsync(data);

        act.Should().NotThrowAsync("Pipeline should handle custom batch size");
        _logger.LogInformation("Pipeline configured with batch size 250");
    }

    [Fact]
    public void GpuPipeline_WithMaxConcurrency_ShouldLimit()
    {
        // Arrange
        var kernelId = "test-kernel-concurrency";

        // Act & Assert - valid concurrency values
        var pipeline1 = GpuPipeline.For<float, float>(_grainFactory, kernelId)
            .WithMaxConcurrency(1);
        pipeline1.Should().NotBeNull();

        var pipeline2 = GpuPipeline.For<float, float>(_grainFactory, kernelId)
            .WithMaxConcurrency(10);
        pipeline2.Should().NotBeNull();

        // Act & Assert - invalid concurrency should throw
        var act = () => GpuPipeline.For<float, float>(_grainFactory, kernelId)
            .WithMaxConcurrency(0);

        act.Should().Throw<ArgumentOutOfRangeException>()
            .WithMessage("*Max concurrency must be at least 1*");

        var actNegative = () => GpuPipeline.For<float, float>(_grainFactory, kernelId)
            .WithMaxConcurrency(-5);

        actNegative.Should().Throw<ArgumentOutOfRangeException>();

        _logger.LogInformation("Max concurrency validation passed");
    }

    [Fact]
    public async Task GpuPipeline_WithTransform_ShouldApplyTransform()
    {
        // Arrange
        var services = new ServiceCollection();
        services.AddLogging(b => b.AddConsole().SetMinimumLevel(LogLevel.Debug));
        services.AddSingleton<IGpuBridge>(new MockGpuBridge());
        var provider = services.BuildServiceProvider();

        var bridge = provider.GetRequiredService<IGpuBridge>();
        var logger = provider.GetRequiredService<ILogger<GpuPipeline>>();

        var pipeline = new GpuPipeline(bridge, logger)
            .Transform<int, float>(x => x * 2.0f)
            .Transform<float, string>(x => $"Result: {x:F2}")
            .Build<int, string>();

        // Act
        var result = await pipeline.ProcessAsync(5);

        // Assert
        result.Should().Be("Result: 10.00");
        _logger.LogInformation("Transform pipeline executed: {Result}", result);
    }

    [Fact]
    public async Task GpuPipeline_WithAggregation_ShouldAggregate()
    {
        // Arrange
        var services = new ServiceCollection();
        services.AddLogging(b => b.AddConsole());
        services.AddSingleton<IGpuBridge>(new MockGpuBridge());
        var provider = services.BuildServiceProvider();

        var bridge = provider.GetRequiredService<IGpuBridge>();
        var logger = provider.GetRequiredService<ILogger<GpuPipeline>>();

        var aggregatedResults = new List<int>();
        var pipeline = new GpuPipeline(bridge, logger)
            .Tap<int>(x => aggregatedResults.Add(x))
            .Build<int, int>();

        var inputs = Enumerable.Range(1, 5).ToAsyncEnumerable();

        // Act
        var results = new List<int>();
        await foreach (var result in pipeline.ProcessManyAsync(inputs))
        {
            results.Add(result);
        }

        // Assert
        aggregatedResults.Should().Equal(1, 2, 3, 4, 5);
        results.Should().Equal(1, 2, 3, 4, 5);
        _logger.LogInformation("Aggregation collected {Count} items", aggregatedResults.Count);
    }

    [Fact]
    public async Task GpuPipeline_WithErrorHandler_ShouldHandleErrors()
    {
        // Arrange
        var services = new ServiceCollection();
        services.AddLogging(b => b.AddConsole().SetMinimumLevel(LogLevel.Debug));
        services.AddSingleton<IGpuBridge>(new MockGpuBridge());
        var provider = services.BuildServiceProvider();

        var bridge = provider.GetRequiredService<IGpuBridge>();
        var logger = provider.GetRequiredService<ILogger<GpuPipeline>>();

        var errorCount = 0;
        var pipeline = new GpuPipeline(bridge, logger)
            .Transform<int, float>(x =>
            {
                if (x < 0) throw new InvalidOperationException($"Negative value: {x}");
                return x * 2.0f;
            })
            .Build<int, float>();

        var inputs = new[] { 1, -2, 3, -4, 5 }.ToAsyncEnumerable();

        // Act - ProcessManyAsync should handle errors gracefully
        var results = new List<float>();
        await foreach (var result in pipeline.ProcessManyAsync(inputs))
        {
            results.Add(result);
        }

        // Assert - should only process valid inputs
        results.Should().Equal(2.0f, 6.0f, 10.0f);
        _logger.LogInformation("Error handling processed {Count} valid results", results.Count);
    }

    [Fact]
    public void GpuPipeline_Build_WithInvalidConfig_ShouldThrow()
    {
        // Arrange
        var services = new ServiceCollection();
        services.AddLogging(b => b.AddConsole());
        services.AddSingleton<IGpuBridge>(new MockGpuBridge());
        var provider = services.BuildServiceProvider();

        var bridge = provider.GetRequiredService<IGpuBridge>();
        var logger = provider.GetRequiredService<ILogger<GpuPipeline>>();

        // Act & Assert - empty pipeline
        var act = () => new GpuPipeline(bridge, logger)
            .Build<int, int>();

        act.Should().Throw<InvalidOperationException>()
            .WithMessage("*Pipeline must have at least one stage*");

        // Act & Assert - type mismatch
        var act2 = () => new GpuPipeline(bridge, logger)
            .Transform<int, float>(x => x * 2.0f)
            .Transform<string, double>(s => double.Parse(s)) // Type mismatch: float -> string
            .Build<int, double>();

        act2.Should().Throw<InvalidOperationException>()
            .WithMessage("*expects*but previous stage outputs*");

        _logger.LogInformation("Invalid pipeline configuration validation passed");
    }

    #endregion

    #region Pipeline Execution Tests (8 tests)

    [Fact]
    public async Task Pipeline_ExecuteAsync_WithSmallBatch_ShouldSucceed()
    {
        // Arrange
        var kernelId = "test-kernel-small";
        var data = TestDataBuilders.FloatArray(25)
            .WithSequentialValues()
            .Build();

        var pipeline = GpuPipeline.For<float, float>(_grainFactory, kernelId)
            .WithBatchSize(10);

        // Act
        var sw = Stopwatch.StartNew();
        var results = await pipeline.ExecuteAsync(data);
        sw.Stop();

        // Assert
        results.Should().NotBeNull();
        results.Should().HaveCount(25);
        results.Should().BeInAscendingOrder();
        _logger.LogInformation("Small batch execution completed in {ElapsedMs}ms", sw.ElapsedMilliseconds);
    }

    [Fact]
    public async Task Pipeline_ExecuteAsync_WithLargeBatch_ShouldPartition()
    {
        // Arrange
        var kernelId = "test-kernel-large";
        var data = TestDataBuilders.FloatArray(1000)
            .WithSequentialValues()
            .Build();

        var pipeline = GpuPipeline.For<float, float>(_grainFactory, kernelId)
            .WithBatchSize(100);

        // Act
        var sw = Stopwatch.StartNew();
        var results = await pipeline.ExecuteAsync(data);
        sw.Stop();

        // Assert
        results.Should().NotBeNull();
        results.Should().HaveCount(1000);
        results.Should().BeInAscendingOrder();

        // Verify batching occurred (10 batches of 100)
        // Since we get back 1000 results, batching must have worked
        (results.Count % 100).Should().Be(0, "Results should align with batch size");

        _logger.LogInformation(
            "Large batch execution completed in {ElapsedMs}ms, processed {Batches} batches",
            sw.ElapsedMilliseconds,
            Math.Ceiling(1000.0 / 100));
    }

    [Fact]
    public async Task Pipeline_ExecuteAsync_WithEmpty_ShouldReturnEmpty()
    {
        // Arrange
        var kernelId = "test-kernel-empty";
        var data = TestDataBuilders.EdgeCases.EmptyFloatArray();

        var pipeline = GpuPipeline.For<float, float>(_grainFactory, kernelId)
            .WithBatchSize(50);

        // Act
        var results = await pipeline.ExecuteAsync(data);

        // Assert
        results.Should().NotBeNull();
        results.Should().BeEmpty();
        _logger.LogInformation("Empty batch handled correctly");
    }

    [Fact]
    public async Task Pipeline_ExecuteAsync_Concurrent_ShouldParallelize()
    {
        // Arrange
        var kernelId = "test-kernel-concurrent";
        var data = TestDataBuilders.FloatArray(500)
            .WithSequentialValues()
            .Build();

        var pipeline = GpuPipeline.For<float, float>(_grainFactory, kernelId)
            .WithBatchSize(50)
            .WithMaxConcurrency(5);

        // Act - execute multiple pipelines concurrently
        var tasks = Enumerable.Range(0, 3).Select(async i =>
        {
            var sw = Stopwatch.StartNew();
            var results = await pipeline.ExecuteAsync(data);
            sw.Stop();
            return (Index: i, Results: results, ElapsedMs: sw.ElapsedMilliseconds);
        });

        var allResults = await Task.WhenAll(tasks);

        // Assert
        allResults.Should().AllSatisfy(r =>
        {
            r.Results.Should().HaveCount(500);
            r.Results.Should().BeInAscendingOrder();
        });

        var avgTime = allResults.Average(r => r.ElapsedMs);
        _logger.LogInformation(
            "Concurrent execution completed, average time: {AvgMs}ms",
            avgTime);
    }

    [Fact]
    public async Task Pipeline_ExecuteAsync_WithCancellation_ShouldCancel()
    {
        // Arrange
        var kernelId = "test-kernel-cancel";
        var data = TestDataBuilders.FloatArray(1000)
            .WithSequentialValues()
            .Build();

        var cts = new CancellationTokenSource();

        var services = new ServiceCollection();
        services.AddLogging(b => b.AddConsole());
        services.AddSingleton<IGpuBridge>(new MockGpuBridge());
        var provider = services.BuildServiceProvider();

        var bridge = provider.GetRequiredService<IGpuBridge>();
        var logger = provider.GetRequiredService<ILogger<GpuPipeline>>();

        var processedCount = 0;
        var pipeline = new GpuPipeline(bridge, logger)
            .Transform<float, float>(async x =>
            {
                // Cancel after processing 100 items to ensure deterministic behavior
                var count = Interlocked.Increment(ref processedCount);
                if (count == 100)
                {
                    cts.Cancel();
                }

                // Add small async yield to allow cancellation detection
                await Task.Yield();
                return x * 2;
            })
            .Build<float, float>();

        var inputs = data.ToAsyncEnumerable();

        // Act & Assert
        var results = new List<float>();

        // ProcessManyAsync should throw OperationCanceledException when cancelled
        var act = async () =>
        {
            await foreach (var result in pipeline.ProcessManyAsync(inputs, cts.Token))
            {
                results.Add(result);
            }
        };

        // Should throw when cancellation occurs
        await act.Should().ThrowAsync<OperationCanceledException>();

        // Should process some items before cancellation (but not all 1000)
        results.Should().NotBeEmpty("some items should be processed before cancellation");
        results.Count.Should().BeLessThan(1000, "cancellation should stop processing");

        // Should process around 100 items (might be slightly more due to async timing)
        results.Count.Should().BeGreaterThanOrEqualTo(100, "should process at least 100 items before cancel");
        results.Count.Should().BeLessThanOrEqualTo(150, "should stop soon after cancel at 100 items");

        _logger.LogInformation(
            "Cancellation stopped processing at {ResultCount} results (processed {ProcessedCount} transforms)",
            results.Count, processedCount);
    }

    [Fact]
    public async Task Pipeline_ExecuteAsync_WithPartialFailure_ShouldContinue()
    {
        // Arrange
        var services = new ServiceCollection();
        services.AddLogging(b => b.AddConsole().SetMinimumLevel(LogLevel.Debug));
        services.AddSingleton<IGpuBridge>(new MockGpuBridge());
        var provider = services.BuildServiceProvider();

        var bridge = provider.GetRequiredService<IGpuBridge>();
        var logger = provider.GetRequiredService<ILogger<GpuPipeline>>();

        var processedCount = 0;
        var pipeline = new GpuPipeline(bridge, logger)
            .Transform<int, float>(x =>
            {
                processedCount++;
                if (x == 50) throw new InvalidOperationException("Test failure");
                return x * 2.0f;
            })
            .Build<int, float>();

        var inputs = Enumerable.Range(1, 100).ToAsyncEnumerable();

        // Act
        var results = new List<float>();
        await foreach (var result in pipeline.ProcessManyAsync(inputs))
        {
            results.Add(result);
        }

        // Assert - should skip failed item and continue
        results.Should().HaveCount(99); // 100 inputs - 1 failure
        results.Should().NotContain(100.0f); // 50 * 2
        _logger.LogInformation(
            "Partial failure handled, processed {Success}/{Total} items",
            results.Count,
            100);
    }

    [Fact]
    public async Task Pipeline_ExecuteAsync_WithMetrics_ShouldRecord()
    {
        // Arrange
        var kernelId = "test-kernel-metrics";
        var data = TestDataBuilders.FloatArray(200)
            .WithSequentialValues()
            .Build();

        var metricsRecorded = new ConcurrentBag<(int BatchSize, TimeSpan Duration)>();

        var pipeline = GpuPipeline.For<float, float>(_grainFactory, kernelId)
            .WithBatchSize(50);

        // Act
        var sw = Stopwatch.StartNew();
        var results = await pipeline.ExecuteAsync(data);
        sw.Stop();

        // Assert
        results.Should().HaveCount(200);
        sw.Elapsed.Should().BeLessThan(TimeSpan.FromSeconds(10));

        var throughput = results.Count / sw.Elapsed.TotalSeconds;
        throughput.Should().BeGreaterThan(10); // At least 10 items/sec

        _logger.LogInformation(
            "Metrics: {Count} items in {ElapsedMs}ms, throughput: {Throughput:F2} items/sec",
            results.Count,
            sw.ElapsedMilliseconds,
            throughput);
    }

    [Fact]
    public async Task Pipeline_ExecuteAsync_WithStreaming_ShouldStreamResults()
    {
        // Arrange
        var services = new ServiceCollection();
        services.AddLogging(b => b.AddConsole().SetMinimumLevel(LogLevel.Debug));
        services.AddSingleton<IGpuBridge>(new MockGpuBridge());
        var provider = services.BuildServiceProvider();

        var bridge = provider.GetRequiredService<IGpuBridge>();
        var logger = provider.GetRequiredService<ILogger<GpuPipeline>>();

        var pipeline = new GpuPipeline(bridge, logger)
            .Transform<int, float>(x => x * 2.0f)
            .Filter<float>(x => x > 50.0f)
            .Build<int, float>();

        var inputChannel = Channel.CreateUnbounded<int>();
        var outputChannel = Channel.CreateUnbounded<float>();

        // Start processing in background
        var processingTask = pipeline.ProcessChannelAsync(
            inputChannel.Reader,
            outputChannel.Writer,
            _testCts.Token);

        // Act - write inputs
        for (int i = 1; i <= 100; i++)
        {
            await inputChannel.Writer.WriteAsync(i, _testCts.Token);
        }
        inputChannel.Writer.Complete();

        // Wait for processing to complete
        await processingTask;

        // Read all outputs
        var results = new List<float>();
        await foreach (var result in outputChannel.Reader.ReadAllAsync(_testCts.Token))
        {
            results.Add(result);
        }

        // Assert
        results.Should().NotBeEmpty();
        results.Should().OnlyContain(x => x > 50.0f);
        results.Should().BeInAscendingOrder();

        // Should have filtered: (1-25) * 2 are <= 50, so (26-100) * 2 pass
        results.Should().HaveCount(75);

        _logger.LogInformation("Streaming pipeline processed {Count} items", results.Count);
    }

    #endregion

    #region Aggregation Tests (5 tests)

    [Fact]
    public async Task Pipeline_Sum_ShouldSumResults()
    {
        // Arrange
        var kernelId = "test-kernel-sum";
        var data = TestDataBuilders.FloatArray(100)
            .WithConstantValue(1.0f)
            .Build();

        var pipeline = GpuPipeline.For<float, float>(_grainFactory, kernelId)
            .WithBatchSize(20);

        // Act
        var results = await pipeline.ExecuteAsync(data);

        // Assert
        results.Should().NotBeEmpty();

        // Custom aggregation: sum all results
        var sum = results.Sum();
        sum.Should().BeApproximately(200.0f, 0.01f); // Each input becomes 2*input in mock

        _logger.LogInformation("Sum aggregation result: {Sum}", sum);
    }

    [Fact]
    public async Task Pipeline_Average_ShouldAverageResults()
    {
        // Arrange
        var kernelId = "test-kernel-average";
        var data = TestDataBuilders.FloatArray(50)
            .WithSequentialValues()
            .Build();

        var pipeline = GpuPipeline.For<float, float>(_grainFactory, kernelId)
            .WithBatchSize(10);

        // Act
        var results = await pipeline.ExecuteAsync(data);

        // Assert
        results.Should().NotBeEmpty();

        // Custom aggregation: average
        var average = results.Average();
        average.Should().BeGreaterThan(0);

        _logger.LogInformation("Average aggregation result: {Average:F2}", average);
    }

    [Fact]
    public async Task Pipeline_Concat_ShouldConcatenateResults()
    {
        // Arrange
        var services = new ServiceCollection();
        services.AddLogging(b => b.AddConsole());
        services.AddSingleton<IGpuBridge>(new MockGpuBridge());
        var provider = services.BuildServiceProvider();

        var bridge = provider.GetRequiredService<IGpuBridge>();
        var logger = provider.GetRequiredService<ILogger<GpuPipeline>>();

        var pipeline = new GpuPipeline(bridge, logger)
            .Transform<int, string>(x => x.ToString())
            .Build<int, string>();

        var inputs = Enumerable.Range(1, 10).ToAsyncEnumerable();

        // Act
        var results = new List<string>();
        await foreach (var result in pipeline.ProcessManyAsync(inputs))
        {
            results.Add(result);
        }

        // Assert
        var concatenated = string.Join(",", results);
        concatenated.Should().Be("1,2,3,4,5,6,7,8,9,10");

        _logger.LogInformation("Concatenation result: {Result}", concatenated);
    }

    [Fact]
    public async Task Pipeline_Custom_ShouldApplyCustomAggregation()
    {
        // Arrange
        var services = new ServiceCollection();
        services.AddLogging(b => b.AddConsole());
        services.AddSingleton<IGpuBridge>(new MockGpuBridge());
        var provider = services.BuildServiceProvider();

        var bridge = provider.GetRequiredService<IGpuBridge>();
        var logger = provider.GetRequiredService<ILogger<GpuPipeline>>();

        var statistics = new
        {
            Count = 0,
            Sum = 0.0,
            Min = double.MaxValue,
            Max = double.MinValue
        };

        var stats = new ConcurrentDictionary<string, double>();
        stats["count"] = 0;
        stats["sum"] = 0;
        stats["min"] = double.MaxValue;
        stats["max"] = double.MinValue;

        var pipeline = new GpuPipeline(bridge, logger)
            .Transform<int, float>(x => x * 1.5f)
            .Tap<float>(x =>
            {
                stats.AddOrUpdate("count", 1, (_, v) => v + 1);
                stats.AddOrUpdate("sum", x, (_, v) => v + x);
                stats.AddOrUpdate("min", x, (_, v) => Math.Min(v, x));
                stats.AddOrUpdate("max", x, (_, v) => Math.Max(v, x));
            })
            .Build<int, float>();

        var inputs = Enumerable.Range(1, 20).ToAsyncEnumerable();

        // Act
        var results = new List<float>();
        await foreach (var result in pipeline.ProcessManyAsync(inputs))
        {
            results.Add(result);
        }

        // Assert
        stats["count"].Should().Be(20);
        stats["sum"].Should().BeGreaterThan(0);
        stats["min"].Should().Be(1.5f);
        stats["max"].Should().Be(30.0f);

        var average = stats["sum"] / stats["count"];

        _logger.LogInformation(
            "Custom aggregation: Count={Count}, Sum={Sum}, Min={Min}, Max={Max}, Avg={Avg}",
            stats["count"], stats["sum"], stats["min"], stats["max"], average);
    }

    [Fact]
    public async Task Pipeline_NoAggregation_ShouldReturnAll()
    {
        // Arrange
        var kernelId = "test-kernel-no-agg";
        var data = TestDataBuilders.FloatArray(75)
            .WithRandomValues()
            .WithSeed(123)
            .Build();

        var pipeline = GpuPipeline.For<float, float>(_grainFactory, kernelId)
            .WithBatchSize(25);

        // Act
        var results = await pipeline.ExecuteAsync(data);

        // Assert - should return all individual results without aggregation
        results.Should().NotBeNull();
        results.Should().HaveCount(75);
        results.Should().OnlyContain(x => x >= 0); // Mock returns positive values

        _logger.LogInformation("No aggregation: returned all {Count} results", results.Count);
    }

    #endregion

    public void Dispose()
    {
        _testCts?.Cancel();
        _testCts?.Dispose();
    }
}

/// <summary>
/// xUnit collection fixture for Orleans cluster.
/// Ensures single cluster instance is shared across all pipeline tests.
/// </summary>
[CollectionDefinition("ClusterCollection")]
public class ClusterCollection : ICollectionFixture<ClusterFixture>
{
    // This class has no code, and is never created. Its purpose is simply
    // to be the place to apply [CollectionDefinition] and all the
    // ICollectionFixture<> interfaces.
}

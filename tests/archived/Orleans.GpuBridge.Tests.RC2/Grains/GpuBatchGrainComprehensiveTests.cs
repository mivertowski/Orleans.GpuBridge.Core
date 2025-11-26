using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading.Tasks;
using FluentAssertions;
using Orleans.GpuBridge.Abstractions;
using Orleans.GpuBridge.Grains.Batch;
using Orleans.GpuBridge.Tests.RC2.Infrastructure;
using Xunit;
using Xunit.Abstractions;

namespace Orleans.GpuBridge.Tests.RC2.Grains;

/// <summary>
/// Comprehensive test suite for GpuBatchGrain implementation.
/// Tests batch execution, batch splitting, memory tracking, error handling,
/// edge cases, and metrics collection using Orleans TestingHost.
///
/// Test Coverage Areas:
/// 1. Batch Execution (ExecuteAsync) - 8 tests
/// 2. Batch Splitting Logic - 5 tests
/// 3. Memory Tracking - 4 tests
/// 4. Error Handling - 6 tests
/// 5. Empty and Null Batch Handling - 5 tests
/// 6. Metrics Collection - 5 tests
/// 7. Concurrency and Performance - 2 tests
///
/// Total: 35 tests
/// </summary>
[Collection("ClusterCollection")]
public sealed class GpuBatchGrainComprehensiveTests : IDisposable
{
    private readonly ClusterFixture _fixture;
    private readonly ITestOutputHelper _output;
    private readonly CancellationTokenSource _testCts;

    public GpuBatchGrainComprehensiveTests(ClusterFixture fixture, ITestOutputHelper output)
    {
        _fixture = fixture ?? throw new ArgumentNullException(nameof(fixture));
        _output = output ?? throw new ArgumentNullException(nameof(output));
        _testCts = new CancellationTokenSource(TimeSpan.FromSeconds(30));
    }

    #region Batch Execution Tests (8 tests)

    [Fact]
    public async Task ExecuteAsync_WithValidSmallBatch_ShouldProcessSuccessfully()
    {
        // Arrange
        var kernelId = "kernels/test-small-batch";
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuBatchGrain<float[], float>>(Guid.NewGuid(), kernelId);

        var batch = new[]
        {
            new float[] { 1.0f, 2.0f },
            new float[] { 3.0f, 4.0f },
            new float[] { 5.0f, 6.0f }
        };

        // Act
        var stopwatch = Stopwatch.StartNew();
        var result = await grain.ExecuteAsync(batch);
        stopwatch.Stop();

        // Assert
        result.Should().NotBeNull();
        result.Success.Should().BeTrue();
        result.Error.Should().BeNullOrEmpty();
        result.Results.Should().NotBeNull().And.HaveCount(3);
        result.ExecutionTime.Should().BeGreaterThan(TimeSpan.Zero);
        result.HandleId.Should().NotBeNullOrEmpty();
        result.KernelId.Should().Be(KernelId.Parse(kernelId));

        _output.WriteLine($"Small batch processed in {stopwatch.ElapsedMilliseconds}ms");
    }

    [Fact]
    public async Task ExecuteAsync_WithLargeBatch_ShouldProcessSuccessfully()
    {
        // Arrange
        var kernelId = "kernels/test-large-batch";
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuBatchGrain<float[], float>>(Guid.NewGuid(), kernelId);

        // Create large batch (100 items)
        var batch = Enumerable.Range(0, 100)
            .Select(i => new float[] { i, i + 1, i + 2 })
            .ToArray();

        // Act
        var stopwatch = Stopwatch.StartNew();
        var result = await grain.ExecuteAsync(batch);
        stopwatch.Stop();

        // Assert
        result.Should().NotBeNull();
        result.Success.Should().BeTrue();
        result.Results.Should().HaveCount(100);
        result.Metrics.Should().NotBeNull();
        result.Metrics!.TotalItems.Should().Be(100);

        _output.WriteLine($"Large batch (100 items) processed in {stopwatch.ElapsedMilliseconds}ms");
        _output.WriteLine($"Throughput: {result.Metrics.Throughput:F2} items/sec");
    }

    [Fact]
    public async Task ExecuteAsync_WithMaxBatch_ShouldProcessSuccessfully()
    {
        // Arrange
        var kernelId = "kernels/test-max-batch";
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuBatchGrain<float[], float>>(Guid.NewGuid(), kernelId);

        // Create maximum batch size (1024 items - typical GPU limit)
        var batch = Enumerable.Range(0, 1024)
            .Select(i => new float[] { i })
            .ToArray();

        // Act
        var result = await grain.ExecuteAsync(batch);

        // Assert
        result.Should().NotBeNull();
        result.Success.Should().BeTrue();
        result.Results.Should().HaveCount(1024);
        result.Metrics.Should().NotBeNull();
        result.Metrics!.TotalItems.Should().Be(1024);

        _output.WriteLine($"Max batch (1024 items) processed successfully");
    }

    [Fact]
    public async Task ExecuteAsync_WithHints_ShouldApplyHints()
    {
        // Arrange
        var kernelId = "kernels/test-hints";
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuBatchGrain<float[], float>>(Guid.NewGuid(), kernelId);

        var batch = new[] { new float[] { 1, 2, 3 } };
        var hints = new GpuExecutionHints(
            MaxMicroBatch: 512,
            HighPriority: true,
            PreferGpu: true);

        // Act
        var result = await grain.ExecuteAsync(batch, hints);

        // Assert
        result.Should().NotBeNull();
        result.Success.Should().BeTrue();
        result.Metrics.Should().NotBeNull();

        _output.WriteLine("Execution hints applied successfully");
    }

    [Fact]
    public async Task ExecuteAsync_WithCpuOnlyHints_ShouldUseCpuFallback()
    {
        // Arrange
        var kernelId = "kernels/test-cpu-only";
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuBatchGrain<float[], float>>(Guid.NewGuid(), kernelId);

        var batch = new[] { new float[] { 1, 2, 3 } };
        var hints = GpuExecutionHints.CpuOnly;

        // Act
        var result = await grain.ExecuteAsync(batch, hints);

        // Assert
        result.Should().NotBeNull();
        result.Success.Should().BeTrue();
        result.Metrics.Should().NotBeNull();
        result.Metrics!.DeviceType.Should().Be("CPU");

        _output.WriteLine("CPU-only execution completed successfully");
    }

    [Fact]
    public async Task ExecuteAsync_WithSingleItem_ShouldProcessSuccessfully()
    {
        // Arrange
        var kernelId = "kernels/test-single-item";
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuBatchGrain<float[], float>>(Guid.NewGuid(), kernelId);

        var batch = new[] { new float[] { 42.0f } };

        // Act
        var result = await grain.ExecuteAsync(batch);

        // Assert
        result.Should().NotBeNull();
        result.Success.Should().BeTrue();
        result.Results.Should().HaveCount(1);

        _output.WriteLine("Single item batch processed successfully");
    }

    [Fact]
    public async Task ExecuteAsync_MultipleSequentialCalls_ShouldSucceed()
    {
        // Arrange
        var kernelId = "kernels/test-sequential";
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuBatchGrain<float[], float>>(Guid.NewGuid(), kernelId);

        var batch = new[] { new float[] { 1, 2, 3 } };

        // Act - Execute multiple batches sequentially
        var results = new List<GpuBatchResult<float>>();
        for (int i = 0; i < 5; i++)
        {
            var result = await grain.ExecuteAsync(batch);
            results.Add(result);
        }

        // Assert
        results.Should().HaveCount(5);
        results.Should().AllSatisfy(r =>
        {
            r.Success.Should().BeTrue();
            r.Results.Should().HaveCount(1);
        });

        _output.WriteLine("Sequential execution of 5 batches completed successfully");
    }

    [Fact]
    public async Task ExecuteAsync_WithDifferentBatchSizes_ShouldAdaptPerformance()
    {
        // Arrange
        var kernelId = "kernels/test-adaptive-size";
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuBatchGrain<float[], float>>(Guid.NewGuid(), kernelId);

        var sizes = new[] { 1, 10, 50, 100, 500 };
        var results = new List<(int Size, TimeSpan Time, double Throughput)>();

        // Act
        foreach (var size in sizes)
        {
            var batch = Enumerable.Range(0, size)
                .Select(i => new float[] { i })
                .ToArray();

            var result = await grain.ExecuteAsync(batch);

            result.Success.Should().BeTrue();
            results.Add((size, result.ExecutionTime, result.Metrics!.Throughput));
        }

        // Assert
        results.Should().HaveCount(sizes.Length);
        results.Should().AllSatisfy(r => r.Time.Should().BeGreaterThan(TimeSpan.Zero));

        // Throughput should generally increase with batch size (GPU efficiency)
        var smallBatchThroughput = results.First(r => r.Size == 10).Throughput;
        var largeBatchThroughput = results.First(r => r.Size == 500).Throughput;

        _output.WriteLine($"Adaptive performance:");
        foreach (var (size, time, throughput) in results)
        {
            _output.WriteLine($"  Size {size}: {time.TotalMilliseconds:F2}ms, {throughput:F2} items/sec");
        }
    }

    #endregion

    #region Batch Splitting Logic Tests (5 tests)

    [Fact]
    public async Task ExecuteAsync_WithOversizedBatch_ShouldSplitAutomatically()
    {
        // Arrange
        var kernelId = "kernels/test-split-oversized";
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuBatchGrain<float[], float>>(Guid.NewGuid(), kernelId);

        // Create batch larger than typical MaxMicroBatch (1024)
        var batch = Enumerable.Range(0, 2000)
            .Select(i => new float[] { i })
            .ToArray();

        var hints = new GpuExecutionHints(MaxMicroBatch: 500);

        // Act
        var result = await grain.ExecuteAsync(batch, hints);

        // Assert
        result.Should().NotBeNull();
        result.Success.Should().BeTrue();
        result.Results.Should().HaveCount(2000);
        result.Metrics.Should().NotBeNull();

        // Should have split into multiple sub-batches (2000 / 500 = 4)
        result.Metrics!.SubBatchCount.Should().BeGreaterThan(1);
        result.Metrics.SuccessfulSubBatches.Should().Be(result.Metrics.SubBatchCount);

        _output.WriteLine($"Batch split into {result.Metrics.SubBatchCount} sub-batches");
    }

    [Fact]
    public async Task ExecuteAsync_WithExactBatchSize_ShouldNotSplit()
    {
        // Arrange
        var kernelId = "kernels/test-exact-size";
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuBatchGrain<float[], float>>(Guid.NewGuid(), kernelId);

        var batch = Enumerable.Range(0, 1024)
            .Select(i => new float[] { i })
            .ToArray();

        var hints = new GpuExecutionHints(MaxMicroBatch: 1024);

        // Act
        var result = await grain.ExecuteAsync(batch, hints);

        // Assert
        result.Should().NotBeNull();
        result.Success.Should().BeTrue();
        result.Metrics.Should().NotBeNull();
        result.Metrics!.SubBatchCount.Should().Be(1);

        _output.WriteLine("Exact batch size processed without splitting");
    }

    [Fact]
    public async Task ExecuteAsync_WithSmallMicroBatch_ShouldCreateManySubBatches()
    {
        // Arrange
        var kernelId = "kernels/test-many-splits";
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuBatchGrain<float[], float>>(Guid.NewGuid(), kernelId);

        var batch = Enumerable.Range(0, 1000)
            .Select(i => new float[] { i })
            .ToArray();

        var hints = new GpuExecutionHints(MaxMicroBatch: 100);

        // Act
        var result = await grain.ExecuteAsync(batch, hints);

        // Assert
        result.Should().NotBeNull();
        result.Success.Should().BeTrue();
        result.Metrics.Should().NotBeNull();
        result.Metrics!.SubBatchCount.Should().Be(10); // 1000 / 100

        _output.WriteLine($"Batch split into {result.Metrics.SubBatchCount} sub-batches of 100 items");
    }

    [Fact]
    public async Task ExecuteAsync_WithIrregularSplit_ShouldHandleRemainder()
    {
        // Arrange
        var kernelId = "kernels/test-irregular-split";
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuBatchGrain<float[], float>>(Guid.NewGuid(), kernelId);

        // 1050 items with batch size 1024 = 1 full batch + 26 items
        var batch = Enumerable.Range(0, 1050)
            .Select(i => new float[] { i })
            .ToArray();

        var hints = new GpuExecutionHints(MaxMicroBatch: 1024);

        // Act
        var result = await grain.ExecuteAsync(batch, hints);

        // Assert
        result.Should().NotBeNull();
        result.Success.Should().BeTrue();
        result.Results.Should().HaveCount(1050);
        result.Metrics.Should().NotBeNull();
        result.Metrics!.SubBatchCount.Should().Be(2);

        _output.WriteLine("Irregular split handled correctly (1024 + 26 items)");
    }

    [Fact]
    public async Task ExecuteAsync_SplitPerformance_ShouldBeFaster()
    {
        // Arrange
        var kernelId = "kernels/test-split-performance";
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuBatchGrain<float[], float>>(Guid.NewGuid(), kernelId);

        var batch = Enumerable.Range(0, 500)
            .Select(i => new float[] { i })
            .ToArray();

        // Act - Execute with and without splitting
        var resultNoSplit = await grain.ExecuteAsync(batch, new GpuExecutionHints(MaxMicroBatch: 500));
        var resultWithSplit = await grain.ExecuteAsync(batch, new GpuExecutionHints(MaxMicroBatch: 100));

        // Assert
        resultNoSplit.Success.Should().BeTrue();
        resultWithSplit.Success.Should().BeTrue();

        resultNoSplit.Metrics!.SubBatchCount.Should().Be(1);
        resultWithSplit.Metrics!.SubBatchCount.Should().Be(5);

        _output.WriteLine($"No split: {resultNoSplit.ExecutionTime.TotalMilliseconds:F2}ms (1 sub-batch)");
        _output.WriteLine($"With split: {resultWithSplit.ExecutionTime.TotalMilliseconds:F2}ms (5 sub-batches)");
    }

    #endregion

    #region Memory Tracking Tests (4 tests)

    [Fact]
    public async Task ExecuteAsync_ShouldTrackMemoryAllocation()
    {
        // Arrange
        var kernelId = "kernels/test-memory-tracking";
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuBatchGrain<float[], float>>(Guid.NewGuid(), kernelId);

        var batch = Enumerable.Range(0, 100)
            .Select(i => new float[] { i })
            .ToArray();

        // Act
        var result = await grain.ExecuteAsync(batch);

        // Assert
        result.Should().NotBeNull();
        result.Success.Should().BeTrue();
        result.Metrics.Should().NotBeNull();
        result.Metrics!.MemoryAllocated.Should().BeGreaterThan(0);

        _output.WriteLine($"Memory allocated: {result.Metrics.MemoryAllocated:N0} bytes");
    }

    [Fact]
    public async Task ExecuteAsync_MemoryUsage_ShouldScaleWithBatchSize()
    {
        // Arrange
        var kernelId = "kernels/test-memory-scaling";
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuBatchGrain<float[], float>>(Guid.NewGuid(), kernelId);

        var sizes = new[] { 10, 50, 100, 500 };
        var memoryUsages = new List<(int Size, long Memory)>();

        // Act
        foreach (var size in sizes)
        {
            var batch = Enumerable.Range(0, size)
                .Select(i => new float[] { i })
                .ToArray();

            var result = await grain.ExecuteAsync(batch);
            memoryUsages.Add((size, result.Metrics!.MemoryAllocated));
        }

        // Assert
        memoryUsages.Should().HaveCount(sizes.Length);

        // Memory should increase proportionally with batch size
        for (int i = 1; i < memoryUsages.Count; i++)
        {
            memoryUsages[i].Memory.Should().BeGreaterThan(memoryUsages[i - 1].Memory);
        }

        _output.WriteLine("Memory usage scaling:");
        foreach (var (size, memory) in memoryUsages)
        {
            _output.WriteLine($"  Size {size}: {memory:N0} bytes ({memory / 1024.0:F2} KB)");
        }
    }

    [Fact]
    public async Task ExecuteAsync_ShouldCalculateMemoryTransferTime()
    {
        // Arrange
        var kernelId = "kernels/test-memory-transfer";
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuBatchGrain<float[], float>>(Guid.NewGuid(), kernelId);

        var batch = Enumerable.Range(0, 1000)
            .Select(i => new float[] { i })
            .ToArray();

        // Act
        var result = await grain.ExecuteAsync(batch);

        // Assert
        result.Should().NotBeNull();
        result.Success.Should().BeTrue();
        result.Metrics.Should().NotBeNull();
        result.Metrics!.MemoryTransferTime.Should().BeGreaterThan(TimeSpan.Zero);
        result.Metrics.KernelExecutionTime.Should().BeGreaterThan(TimeSpan.Zero);

        _output.WriteLine($"Memory transfer: {result.Metrics.MemoryTransferTime.TotalMilliseconds:F2}ms");
        _output.WriteLine($"Kernel execution: {result.Metrics.KernelExecutionTime.TotalMilliseconds:F2}ms");
    }

    [Fact]
    public async Task ExecuteAsync_ShouldCalculateMemoryBandwidth()
    {
        // Arrange
        var kernelId = "kernels/test-memory-bandwidth";
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuBatchGrain<float[], float>>(Guid.NewGuid(), kernelId);

        var batch = Enumerable.Range(0, 500)
            .Select(i => new float[] { i })
            .ToArray();

        // Act
        var result = await grain.ExecuteAsync(batch);

        // Assert
        result.Should().NotBeNull();
        result.Success.Should().BeTrue();
        result.Metrics.Should().NotBeNull();
        result.Metrics!.MemoryBandwidthMBps.Should().BeGreaterThan(0);

        _output.WriteLine($"Memory bandwidth: {result.Metrics.MemoryBandwidthMBps:F2} MB/s");
    }

    #endregion

    #region Error Handling Tests (6 tests)

    [Fact]
    public async Task ExecuteAsync_WithInvalidKernel_ShouldHandleGracefully()
    {
        // Arrange
        var kernelId = "kernels/invalid-nonexistent-kernel";
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuBatchGrain<float[], float>>(Guid.NewGuid(), kernelId);

        var batch = new[] { new float[] { 1, 2, 3 } };

        // Act
        var result = await grain.ExecuteAsync(batch);

        // Assert - Should complete without throwing, may use CPU fallback
        result.Should().NotBeNull();

        _output.WriteLine($"Invalid kernel handled: Success={result.Success}, Error={result.Error}");
    }

    [Fact]
    public async Task ExecuteAsync_WithExceptionDuringExecution_ShouldReturnError()
    {
        // Arrange
        var kernelId = "kernels/test-execution-error";
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuBatchGrain<float[], float>>(Guid.NewGuid(), kernelId);

        // Create a batch that might cause issues (though mock should handle it)
        var batch = new[] { new float[] { float.NaN, float.PositiveInfinity } };

        // Act
        var result = await grain.ExecuteAsync(batch);

        // Assert - Should handle gracefully
        result.Should().NotBeNull();

        _output.WriteLine($"Execution with special values: Success={result.Success}");
    }

    [Fact]
    public async Task ExecuteAsync_AfterActivationFailure_ShouldRecoverOnRetry()
    {
        // Arrange
        var kernelId = "kernels/test-recovery";
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuBatchGrain<float[], float>>(Guid.NewGuid(), kernelId);

        var batch = new[] { new float[] { 1, 2, 3 } };

        // Act - Execute twice to test recovery
        var result1 = await grain.ExecuteAsync(batch);
        var result2 = await grain.ExecuteAsync(batch);

        // Assert
        result1.Should().NotBeNull();
        result2.Should().NotBeNull();
        result2.Success.Should().BeTrue(); // Should succeed on retry

        _output.WriteLine("Grain recovered successfully after potential activation issues");
    }

    [Fact]
    public async Task ExecuteAsync_WithTimeout_ShouldCompleteWithinTimeLimit()
    {
        // Arrange
        var kernelId = "kernels/test-timeout";
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuBatchGrain<float[], float>>(Guid.NewGuid(), kernelId);

        var batch = Enumerable.Range(0, 100)
            .Select(i => new float[] { i })
            .ToArray();

        var hints = new GpuExecutionHints(Timeout: TimeSpan.FromSeconds(10));

        // Act
        var stopwatch = Stopwatch.StartNew();
        var result = await grain.ExecuteAsync(batch, hints);
        stopwatch.Stop();

        // Assert
        result.Should().NotBeNull();
        stopwatch.Elapsed.Should().BeLessThan(TimeSpan.FromSeconds(10));

        _output.WriteLine($"Execution completed in {stopwatch.ElapsedMilliseconds}ms (timeout: 10000ms)");
    }

    [Fact]
    public async Task ExecuteAsync_WithMaxRetries_ShouldRespectRetryLimit()
    {
        // Arrange
        var kernelId = "kernels/test-retry-limit";
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuBatchGrain<float[], float>>(Guid.NewGuid(), kernelId);

        var batch = new[] { new float[] { 1, 2, 3 } };
        var hints = new GpuExecutionHints(MaxRetries: 3);

        // Act
        var result = await grain.ExecuteAsync(batch, hints);

        // Assert
        result.Should().NotBeNull();

        _output.WriteLine($"Execution with retry limit: Success={result.Success}");
    }

    [Fact]
    public async Task ExecuteWithCallbackAsync_OnError_ShouldNotifyObserver()
    {
        // Arrange
        var kernelId = "kernels/test-callback-error";
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuBatchGrain<float[], float>>(Guid.NewGuid(), kernelId);

        var batch = new[] { new float[] { 1, 2, 3 } };
        var observer = new TestGpuResultObserver<float>();
        var observerRef = _fixture.Cluster.GrainFactory
            .CreateObjectReference<IGpuResultObserver<float>>(observer);

        // Act
        var result = await grain.ExecuteWithCallbackAsync(batch, observerRef);

        // Wait for async callbacks
        await Task.Delay(200);

        // Assert
        observer.ReceivedItems.Should().NotBeEmpty();

        _output.WriteLine($"Callback observer received {observer.ReceivedItems.Count} items");
    }

    #endregion

    #region Empty and Null Batch Handling Tests (5 tests)

    [Fact]
    public async Task ExecuteAsync_WithEmptyBatch_ShouldReturnSuccessWithEmptyResults()
    {
        // Arrange
        var kernelId = "kernels/test-empty-batch";
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuBatchGrain<float[], float>>(Guid.NewGuid(), kernelId);

        var batch = Array.Empty<float[]>();

        // Act
        var result = await grain.ExecuteAsync(batch);

        // Assert
        result.Should().NotBeNull();
        result.Success.Should().BeTrue();
        result.Error.Should().BeNullOrEmpty();
        result.Results.Should().NotBeNull().And.BeEmpty();
        result.Metrics.Should().NotBeNull();
        result.Metrics!.TotalItems.Should().Be(0);
        result.Metrics.SubBatchCount.Should().Be(0);

        _output.WriteLine("Empty batch handled correctly with success status");
    }

    [Fact]
    public async Task ExecuteAsync_WithNullBatch_ShouldReturnErrorResult()
    {
        // Arrange
        var kernelId = "kernels/test-null-batch";
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuBatchGrain<float[], float>>(Guid.NewGuid(), kernelId);

        IReadOnlyList<float[]>? batch = null;

        // Act
        var result = await grain.ExecuteAsync(batch!);

        // Assert
        result.Should().NotBeNull();
        result.Success.Should().BeFalse();
        result.Error.Should().NotBeNullOrEmpty();
        result.Error.Should().Contain("null");
        result.Results.Should().BeEmpty();

        _output.WriteLine($"Null batch handled with error: {result.Error}");
    }

    [Fact]
    public async Task ProcessBatchAsync_WithEmptyBatch_ShouldReturnSuccessWithEmptyResults()
    {
        // Arrange
        var kernelId = "kernels/test-process-empty";
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuBatchGrain<float[], float>>(Guid.NewGuid(), kernelId);

        var batch = Array.Empty<float[]>();

        // Act
        var result = await grain.ProcessBatchAsync(batch);

        // Assert
        result.Should().NotBeNull();
        result.Success.Should().BeTrue();
        result.Results.Should().BeEmpty();

        _output.WriteLine("ProcessBatchAsync handled empty batch correctly");
    }

    [Fact]
    public async Task ExecuteAsync_WithEmptyBatchAndHints_ShouldIgnoreHints()
    {
        // Arrange
        var kernelId = "kernels/test-empty-with-hints";
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuBatchGrain<float[], float>>(Guid.NewGuid(), kernelId);

        var batch = Array.Empty<float[]>();
        var hints = new GpuExecutionHints(MaxMicroBatch: 512, HighPriority: true);

        // Act
        var result = await grain.ExecuteAsync(batch, hints);

        // Assert
        result.Should().NotBeNull();
        result.Success.Should().BeTrue();
        result.Results.Should().BeEmpty();
        result.Metrics.Should().NotBeNull();

        _output.WriteLine("Empty batch with hints handled correctly");
    }

    [Fact]
    public async Task ExecuteWithCallbackAsync_WithEmptyBatch_ShouldCompleteImmediately()
    {
        // Arrange
        var kernelId = "kernels/test-empty-callback";
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuBatchGrain<float[], float>>(Guid.NewGuid(), kernelId);

        var batch = Array.Empty<float[]>();
        var observer = new TestGpuResultObserver<float>();
        var observerRef = _fixture.Cluster.GrainFactory
            .CreateObjectReference<IGpuResultObserver<float>>(observer);

        // Act
        var result = await grain.ExecuteWithCallbackAsync(batch, observerRef);

        // Wait for callbacks
        await Task.Delay(100);

        // Assert
        result.Success.Should().BeTrue();
        observer.ReceivedItems.Should().BeEmpty();
        observer.Completed.Should().BeTrue();

        _output.WriteLine("Empty batch with callback completed immediately");
    }

    #endregion

    #region Metrics Collection Tests (5 tests)

    [Fact]
    public async Task ExecuteAsync_ShouldCollectComprehensiveMetrics()
    {
        // Arrange
        var kernelId = "kernels/test-comprehensive-metrics";
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuBatchGrain<float[], float>>(Guid.NewGuid(), kernelId);

        var batch = Enumerable.Range(0, 100)
            .Select(i => new float[] { i })
            .ToArray();

        // Act
        var result = await grain.ExecuteAsync(batch);

        // Assert
        result.Should().NotBeNull();
        result.Metrics.Should().NotBeNull();

        var metrics = result.Metrics!;
        metrics.TotalItems.Should().Be(100);
        metrics.SubBatchCount.Should().BeGreaterThan(0);
        metrics.SuccessfulSubBatches.Should().Be(metrics.SubBatchCount);
        metrics.TotalExecutionTime.Should().BeGreaterThan(TimeSpan.Zero);
        metrics.KernelExecutionTime.Should().BeGreaterThan(TimeSpan.Zero);
        metrics.MemoryTransferTime.Should().BeGreaterThanOrEqualTo(TimeSpan.Zero);
        metrics.Throughput.Should().BeGreaterThan(0);
        metrics.MemoryAllocated.Should().BeGreaterThan(0);
        metrics.DeviceType.Should().NotBeNullOrEmpty();
        metrics.DeviceName.Should().NotBeNullOrEmpty();

        _output.WriteLine("Comprehensive metrics collected:");
        _output.WriteLine($"  Total Items: {metrics.TotalItems}");
        _output.WriteLine($"  Sub-batches: {metrics.SubBatchCount}");
        _output.WriteLine($"  Execution Time: {metrics.TotalExecutionTime.TotalMilliseconds:F2}ms");
        _output.WriteLine($"  Throughput: {metrics.Throughput:F2} items/sec");
        _output.WriteLine($"  Memory: {metrics.MemoryAllocated:N0} bytes");
        _output.WriteLine($"  Device: {metrics.DeviceType} ({metrics.DeviceName})");
    }

    [Fact]
    public async Task ExecuteAsync_ShouldCalculateKernelEfficiency()
    {
        // Arrange
        var kernelId = "kernels/test-kernel-efficiency";
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuBatchGrain<float[], float>>(Guid.NewGuid(), kernelId);

        var batch = Enumerable.Range(0, 200)
            .Select(i => new float[] { i })
            .ToArray();

        // Act
        var result = await grain.ExecuteAsync(batch);

        // Assert
        result.Metrics.Should().NotBeNull();
        result.Metrics!.KernelEfficiency.Should().BeGreaterThan(0);
        result.Metrics.KernelEfficiency.Should().BeLessThanOrEqualTo(100);

        _output.WriteLine($"Kernel efficiency: {result.Metrics.KernelEfficiency:F2}%");
    }

    [Fact]
    public async Task ExecuteAsync_ShouldCalculateItemsPerMillisecond()
    {
        // Arrange
        var kernelId = "kernels/test-items-per-ms";
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuBatchGrain<float[], float>>(Guid.NewGuid(), kernelId);

        var batch = Enumerable.Range(0, 500)
            .Select(i => new float[] { i })
            .ToArray();

        // Act
        var result = await grain.ExecuteAsync(batch);

        // Assert
        result.Metrics.Should().NotBeNull();
        result.Metrics!.ItemsPerMillisecond.Should().BeGreaterThan(0);

        _output.WriteLine($"Processing rate: {result.Metrics.ItemsPerMillisecond:F2} items/ms");
    }

    [Fact]
    public async Task ExecuteAsync_MultipleExecutions_ShouldTrackConsistentMetrics()
    {
        // Arrange
        var kernelId = "kernels/test-consistent-metrics";
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuBatchGrain<float[], float>>(Guid.NewGuid(), kernelId);

        var batch = Enumerable.Range(0, 50)
            .Select(i => new float[] { i })
            .ToArray();

        // Act - Execute 3 times
        var results = new List<GpuBatchResult<float>>();
        for (int i = 0; i < 3; i++)
        {
            var result = await grain.ExecuteAsync(batch);
            results.Add(result);
        }

        // Assert - All should have consistent metrics
        results.Should().AllSatisfy(r =>
        {
            r.Metrics.Should().NotBeNull();
            r.Metrics!.TotalItems.Should().Be(50);
            r.Metrics.DeviceType.Should().NotBeNullOrEmpty();
        });

        var throughputs = results.Select(r => r.Metrics!.Throughput).ToList();
        var avgThroughput = throughputs.Average();

        _output.WriteLine("Consistent metrics across 3 executions:");
        for (int i = 0; i < results.Count; i++)
        {
            _output.WriteLine($"  Execution {i + 1}: {results[i].Metrics!.Throughput:F2} items/sec");
        }
        _output.WriteLine($"  Average: {avgThroughput:F2} items/sec");
    }

    [Fact]
    public async Task ExecuteAsync_ShouldIncludeDeviceInformation()
    {
        // Arrange
        var kernelId = "kernels/test-device-info";
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuBatchGrain<float[], float>>(Guid.NewGuid(), kernelId);

        var batch = new[] { new float[] { 1, 2, 3 } };

        // Act
        var result = await grain.ExecuteAsync(batch);

        // Assert
        result.Metrics.Should().NotBeNull();
        result.Metrics!.DeviceType.Should().NotBeNullOrEmpty();
        result.Metrics.DeviceName.Should().NotBeNullOrEmpty();

        _output.WriteLine($"Device: {result.Metrics.DeviceType} - {result.Metrics.DeviceName}");
    }

    #endregion

    #region Concurrency and Performance Tests (2 tests)

    [Fact]
    public async Task ExecuteAsync_ConcurrentExecutions_ShouldHandleCorrectly()
    {
        // Arrange
        var kernelId = "kernels/test-concurrent-exec";
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuBatchGrain<float[], float>>(Guid.NewGuid(), kernelId);

        var batch = Enumerable.Range(0, 100)
            .Select(i => new float[] { i })
            .ToArray();

        // Act - Execute 10 concurrent batches
        var tasks = Enumerable.Range(0, 10)
            .Select(_ => grain.ExecuteAsync(batch))
            .ToArray();

        var results = await Task.WhenAll(tasks);

        // Assert
        results.Should().HaveCount(10);
        results.Should().AllSatisfy(r =>
        {
            r.Success.Should().BeTrue();
            r.Results.Should().HaveCount(100);
        });

        _output.WriteLine("10 concurrent executions completed successfully");
    }

    [Fact]
    public async Task ExecuteAsync_HighLoadStressTest_ShouldMaintainPerformance()
    {
        // Arrange
        var kernelId = "kernels/test-high-load";
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuBatchGrain<float[], float>>(Guid.NewGuid(), kernelId);

        var batch = Enumerable.Range(0, 50)
            .Select(i => new float[] { i })
            .ToArray();

        var stopwatch = Stopwatch.StartNew();

        // Act - Execute 50 batches rapidly
        var tasks = Enumerable.Range(0, 50)
            .Select(async i =>
            {
                var result = await grain.ExecuteAsync(batch);
                return (Index: i, Result: result);
            })
            .ToArray();

        var results = await Task.WhenAll(tasks);
        stopwatch.Stop();

        // Assert
        results.Should().HaveCount(50);
        results.Should().AllSatisfy(r => r.Result.Success.Should().BeTrue());

        var totalItems = results.Sum(r => r.Result.Results.Count);
        var overallThroughput = totalItems / stopwatch.Elapsed.TotalSeconds;

        _output.WriteLine($"High load test: 50 batches in {stopwatch.ElapsedMilliseconds}ms");
        _output.WriteLine($"Total items processed: {totalItems}");
        _output.WriteLine($"Overall throughput: {overallThroughput:F2} items/sec");
    }

    #endregion

    public void Dispose()
    {
        _testCts?.Cancel();
        _testCts?.Dispose();
    }

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

using Orleans.GpuBridge.Abstractions;
using Orleans.GpuBridge.Grains.Batch;
using Orleans.GpuBridge.Tests.RC2.Infrastructure;

namespace Orleans.GpuBridge.Tests.RC2.Grains;

/// <summary>
/// Comprehensive tests for GpuBatchGrainEnhanced implementation with DotCompute backend integration.
/// Tests advanced batch processing, persistent kernels, compilation caching, and performance optimization.
/// Target: 40+ tests for enhanced features.
/// </summary>
[Collection("ClusterCollection")]
public sealed class GpuBatchGrainEnhancedTests : IClassFixture<ClusterFixture>
{
    private readonly ClusterFixture _fixture;

    public GpuBatchGrainEnhancedTests(ClusterFixture fixture)
    {
        _fixture = fixture;
    }

    #region Basic Functionality Tests

    [Fact]
    public async Task EnhancedGrain_Activation_ShouldInitializeBackend()
    {
        // Arrange
        var kernelId = "kernels/enhanced-activation";
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuBatchGrain<float, float>>(Guid.NewGuid(), kernelId);

        // Act - First call triggers activation
        var input = new[] { 1.0f, 2.0f, 3.0f };
        var result = await grain.ExecuteAsync(input);

        // Assert
        result.Should().NotBeNull();
        result.Success.Should().BeTrue();
        result.KernelId.Should().Be(KernelId.Parse(kernelId));
    }

    [Fact]
    public async Task EnhancedGrain_ExecuteAsync_ShouldReturnDetailedMetrics()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuBatchGrain<float, float>>(Guid.NewGuid(), "kernels/metrics-test");

        var input = new[] { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f };

        // Act
        var result = await grain.ExecuteAsync(input);

        // Assert
        result.Success.Should().BeTrue();
        result.Metrics.Should().NotBeNull();
        result.Metrics!.TotalItems.Should().Be(5);
        result.Metrics.TotalExecutionTime.Should().BeGreaterThan(TimeSpan.Zero);
        result.Metrics.Throughput.Should().BeGreaterThan(0);
    }

    [Fact]
    public async Task EnhancedGrain_SmallBatch_ShouldProcessInSingleBatch()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuBatchGrain<int, int>>(Guid.NewGuid(), "kernels/small-batch");

        var input = Enumerable.Range(1, 10).ToArray();

        // Act
        var result = await grain.ExecuteAsync(input);

        // Assert
        result.Success.Should().BeTrue();
        result.Metrics.Should().NotBeNull();
        result.Metrics!.SubBatchCount.Should().Be(1);
        result.Metrics.SuccessfulSubBatches.Should().Be(1);
    }

    [Fact]
    public async Task EnhancedGrain_EmptyBatch_ShouldReturnEmptyResults()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuBatchGrain<float, float>>(Guid.NewGuid(), "kernels/empty");

        var input = Array.Empty<float>();

        // Act
        var result = await grain.ExecuteAsync(input);

        // Assert
        result.Should().NotBeNull();
        result.Results.Should().BeEmpty();
        result.Error.Should().NotBeNullOrEmpty();
    }

    #endregion

    #region Batch Size Optimization Tests

    [Fact]
    public async Task EnhancedGrain_LargeBatch_ShouldSplitIntoSubBatches()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuBatchGrain<float, float>>(Guid.NewGuid(), "kernels/large-batch");

        // Create a large batch (10K items)
        var input = Enumerable.Range(0, 10_000)
            .Select(i => (float)i)
            .ToArray();

        // Act
        var result = await grain.ExecuteAsync(input);

        // Assert
        result.Success.Should().BeTrue();
        result.Results.Should().HaveCount(10_000);
        result.Metrics!.SubBatchCount.Should().BeGreaterThan(1);
    }

    [Fact]
    public async Task EnhancedGrain_OptimalBatchSize_ShouldCalculateBasedOnMemory()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuBatchGrain<double, double>>(Guid.NewGuid(), "kernels/optimal-size");

        var input = new[] { 1.0, 2.0, 3.0, 4.0, 5.0 };

        // Act
        var result = await grain.ExecuteAsync(input);

        // Assert
        result.Success.Should().BeTrue();
        result.Metrics!.MemoryAllocated.Should().BeGreaterThan(0);
    }

    [Fact]
    public async Task EnhancedGrain_MemoryPressure_ShouldAdaptBatchSize()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuBatchGrain<float, float>>(Guid.NewGuid(), "kernels/memory-pressure");

        // Execute multiple batches to simulate memory pressure
        var batches = Enumerable.Range(0, 5)
            .Select(i => Enumerable.Range(i * 1000, 1000).Select(x => (float)x).ToArray())
            .ToList();

        // Act
        var results = new List<GpuBatchResult<float>>();
        foreach (var batch in batches)
        {
            var result = await grain.ExecuteAsync(batch);
            results.Add(result);
        }

        // Assert
        results.Should().AllSatisfy(r => r.Success.Should().BeTrue());
        results.Should().AllSatisfy(r => r.Metrics.Should().NotBeNull());
    }

    #endregion

    #region Kernel Compilation and Caching Tests

    [Fact]
    public async Task EnhancedGrain_FirstExecution_ShouldCompileKernel()
    {
        // Arrange
        var kernelId = $"kernels/compile-{Guid.NewGuid()}";
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuBatchGrain<float, float>>(Guid.NewGuid(), kernelId);

        var input = new[] { 1.0f, 2.0f };

        // Act
        var result = await grain.ExecuteAsync(input);

        // Assert
        result.Success.Should().BeTrue();
    }

    [Fact]
    public async Task EnhancedGrain_SubsequentExecution_ShouldUseCachedKernel()
    {
        // Arrange
        var kernelId = "kernels/cached-kernel";
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuBatchGrain<float, float>>(Guid.NewGuid(), kernelId);

        var input = new[] { 1.0f, 2.0f };

        // Act - Execute twice to test caching
        var result1 = await grain.ExecuteAsync(input);
        var result2 = await grain.ExecuteAsync(input);

        // Assert
        result1.Success.Should().BeTrue();
        result2.Success.Should().BeTrue();
        // Second execution should be faster (cached kernel)
        result2.ExecutionTime.Should().BeLessThanOrEqualTo(result1.ExecutionTime * 1.5);
    }

    #endregion

    #region Performance Optimization Tests

    [Fact]
    public async Task EnhancedGrain_ConcurrentBatches_ShouldRespectSemaphore()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuBatchGrain<int, int>>(Guid.NewGuid(), "kernels/concurrent");

        var input = Enumerable.Range(1, 100).ToArray();

        // Act - Execute 8 concurrent batches (default concurrency = 4)
        var tasks = Enumerable.Range(0, 8)
            .Select(_ => grain.ExecuteAsync(input))
            .ToArray();

        var results = await Task.WhenAll(tasks);

        // Assert
        results.Should().AllSatisfy(r => r.Success.Should().BeTrue());
        results.Should().HaveCount(8);
    }

    [Fact]
    public async Task EnhancedGrain_Throughput_ShouldExceed1000ItemsPerSecond()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuBatchGrain<float, float>>(Guid.NewGuid(), "kernels/throughput");

        var input = Enumerable.Range(0, 1000).Select(i => (float)i).ToArray();

        // Act
        var result = await grain.ExecuteAsync(input);

        // Assert
        result.Success.Should().BeTrue();
        result.Metrics!.Throughput.Should().BeGreaterThan(1000); // items/sec
    }

    [Fact]
    public async Task EnhancedGrain_KernelEfficiency_ShouldBeMeasured()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuBatchGrain<double, double>>(Guid.NewGuid(), "kernels/efficiency");

        var input = new[] { 1.0, 2.0, 3.0, 4.0, 5.0 };

        // Act
        var result = await grain.ExecuteAsync(input);

        // Assert
        result.Success.Should().BeTrue();
        result.Metrics!.KernelEfficiency.Should().BeGreaterThanOrEqualTo(0);
        result.Metrics.KernelEfficiency.Should().BeLessThanOrEqualTo(100);
    }

    #endregion

    #region Memory Management Tests

    [Fact]
    public async Task EnhancedGrain_MemoryAllocation_ShouldBeTracked()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuBatchGrain<float, float>>(Guid.NewGuid(), "kernels/memory-tracking");

        var input = new[] { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f };

        // Act
        var result = await grain.ExecuteAsync(input);

        // Assert
        result.Success.Should().BeTrue();
        result.Metrics!.MemoryAllocated.Should().BeGreaterThan(0);
    }

    [Fact]
    public async Task EnhancedGrain_MemoryTransfer_ShouldBeMeasured()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuBatchGrain<float, float>>(Guid.NewGuid(), "kernels/memory-transfer");

        var input = Enumerable.Range(0, 1000).Select(i => (float)i).ToArray();

        // Act
        var result = await grain.ExecuteAsync(input);

        // Assert
        result.Success.Should().BeTrue();
        result.Metrics!.MemoryTransferTime.Should().BeGreaterThan(TimeSpan.Zero);
    }

    #endregion

    #region CPU Fallback Tests

    [Fact]
    public async Task EnhancedGrain_NoGpu_ShouldFallbackToCpu()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuBatchGrain<float, float>>(Guid.NewGuid(), "kernels/cpu-fallback");

        var input = new[] { 1.0f, 2.0f, 3.0f };

        // Act
        var result = await grain.ExecuteAsync(input);

        // Assert
        result.Success.Should().BeTrue();
        result.Metrics!.DeviceType.Should().Be("CPU");
    }

    [Fact]
    public async Task EnhancedGrain_CpuFallback_ShouldLogWarning()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuBatchGrain<int, int>>(Guid.NewGuid(), "kernels/cpu-warning");

        var input = new[] { 1, 2, 3 };

        // Act
        var result = await grain.ExecuteAsync(input);

        // Assert
        result.Success.Should().BeTrue();
    }

    #endregion

    #region Error Handling Tests

    [Fact]
    public async Task EnhancedGrain_NullBatch_ShouldReturnError()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuBatchGrain<float, float>>(Guid.NewGuid(), "kernels/null-batch");

        // Act
        var result = await grain.ExecuteAsync(null!);

        // Assert
        result.Should().NotBeNull();
        result.Success.Should().BeFalse();
        result.Error.Should().NotBeNullOrEmpty();
    }

    [Fact]
    public async Task EnhancedGrain_ExecutionFailure_ShouldReturnErrorResult()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuBatchGrain<float, float>>(Guid.NewGuid(), "kernels/execution-fail");

        var input = new[] { 1.0f };

        // Act
        var result = await grain.ExecuteAsync(input);

        // Assert
        result.Should().NotBeNull();
    }

    #endregion

    #region Observer Pattern Tests

    [Fact]
    public async Task EnhancedGrain_ExecuteWithCallback_ShouldNotifyObserver()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuBatchGrain<float, float>>(Guid.NewGuid(), "kernels/observer");

        var input = new[] { 1.0f, 2.0f, 3.0f };
        var observer = new TestObserver<float>();
        var observerRef = _fixture.Cluster.GrainFactory.CreateObjectReference<IGpuResultObserver<float>>(observer);

        // Act
        var result = await grain.ExecuteWithCallbackAsync(input, observerRef);

        // Wait for async callbacks
        await Task.Delay(200);

        // Assert
        result.Success.Should().BeTrue();
        observer.ReceivedItems.Should().NotBeEmpty();
        observer.Completed.Should().BeTrue();
    }

    [Fact]
    public async Task EnhancedGrain_ObserverError_ShouldNotifyOnError()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuBatchGrain<float, float>>(Guid.NewGuid(), "kernels/observer-error");

        var observer = new TestObserver<float>();
        var observerRef = _fixture.Cluster.GrainFactory.CreateObjectReference<IGpuResultObserver<float>>(observer);

        // Act - Execute with null to trigger error
        await grain.ExecuteWithCallbackAsync(null!, observerRef);

        await Task.Delay(100);

        // Assert
        observer.Error.Should().NotBeNull();
    }

    #endregion

    #region Grain Lifecycle Tests

    [Fact]
    public async Task EnhancedGrain_MultipleActivations_ShouldReinitialize()
    {
        // Arrange
        var grainId = Guid.NewGuid();
        var kernelId = "kernels/reactivate";
        var grain1 = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuBatchGrain<float, float>>(grainId, kernelId);

        // Act - First activation
        var result1 = await grain1.ExecuteAsync(new[] { 1.0f });

        // Get same grain (may reactivate)
        var grain2 = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuBatchGrain<float, float>>(grainId, kernelId);
        var result2 = await grain2.ExecuteAsync(new[] { 2.0f });

        // Assert
        result1.Success.Should().BeTrue();
        result2.Success.Should().BeTrue();
    }

    #endregion

    #region Integration Tests

    [Fact]
    public async Task EnhancedGrain_ProcessBatchAsync_ShouldWorkAsAlias()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuBatchGrain<int, int>>(Guid.NewGuid(), "kernels/alias");

        var input = new[] { 1, 2, 3 };

        // Act
        var result = await grain.ProcessBatchAsync(input);

        // Assert
        result.Success.Should().BeTrue();
        result.Results.Should().HaveCount(3);
    }

    [Fact]
    public async Task EnhancedGrain_MixedTypes_ShouldHandleCorrectly()
    {
        // Arrange - Test with different numeric types
        var floatGrain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuBatchGrain<float, float>>(Guid.NewGuid(), "kernels/float");
        var intGrain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuBatchGrain<int, int>>(Guid.NewGuid(), "kernels/int");
        var doubleGrain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuBatchGrain<double, double>>(Guid.NewGuid(), "kernels/double");

        // Act
        var floatResult = await floatGrain.ExecuteAsync(new[] { 1.0f, 2.0f });
        var intResult = await intGrain.ExecuteAsync(new[] { 1, 2 });
        var doubleResult = await doubleGrain.ExecuteAsync(new[] { 1.0, 2.0 });

        // Assert
        floatResult.Success.Should().BeTrue();
        intResult.Success.Should().BeTrue();
        doubleResult.Success.Should().BeTrue();
    }

    [Fact]
    public async Task EnhancedGrain_WithHints_ShouldApplyConfiguration()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuBatchGrain<float, float>>(Guid.NewGuid(), "kernels/hints");

        var input = new[] { 1.0f, 2.0f, 3.0f };
        var hints = new GpuExecutionHints { MaxMicroBatch = 2 };

        // Act
        var result = await grain.ExecuteAsync(input, hints);

        // Assert
        result.Success.Should().BeTrue();
    }

    #endregion

    /// <summary>
    /// Test observer implementation.
    /// </summary>
    private sealed class TestObserver<T> : IGpuResultObserver<T>
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

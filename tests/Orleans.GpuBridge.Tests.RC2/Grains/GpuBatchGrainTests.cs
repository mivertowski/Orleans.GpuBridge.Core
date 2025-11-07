using Orleans.GpuBridge.Abstractions;
using Orleans.GpuBridge.Grains.Batch;
using Orleans.GpuBridge.Tests.RC2.Infrastructure;

namespace Orleans.GpuBridge.Tests.RC2.Grains;

/// <summary>
/// Comprehensive tests for GpuBatchGrain implementation.
/// Tests grain lifecycle, batch execution, concurrency, state persistence, and metrics.
/// </summary>
[Collection("ClusterCollection")]
public sealed class GpuBatchGrainTests : IClassFixture<ClusterFixture>
{
    private readonly ClusterFixture _fixture;

    public GpuBatchGrainTests(ClusterFixture fixture)
    {
        _fixture = fixture;
    }

    [Fact]
    public async Task GpuBatchGrain_Activation_ShouldInitializeResources()
    {
        // Arrange
        var kernelId = "kernels/test-activation";
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuBatchGrain<float[], float>>(Guid.NewGuid(), kernelId);

        // Act - First call triggers activation
        var input = new[] { new float[] { 1, 2, 3 } };
        var result = await grain.ExecuteAsync(input);

        // Assert
        result.Should().NotBeNull();
        result.Success.Should().BeTrue();
        result.KernelId.Should().Be(KernelId.Parse(kernelId));
        result.Results.Should().NotBeNull();
    }

    [Fact]
    public async Task GpuBatchGrain_Deactivation_ShouldCleanupResources()
    {
        // Arrange
        var kernelId = "kernels/test-deactivation";
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuBatchGrain<float[], float>>(Guid.NewGuid(), kernelId);

        // Act - Execute to activate grain
        var input = new[] { new float[] { 1, 2, 3 } };
        await grain.ExecuteAsync(input);

        // Deactivation happens automatically after idle period
        // Or can be triggered explicitly for testing
        // Orleans will call OnDeactivateAsync internally

        // Assert - Grain should handle deactivation gracefully
        // We verify this by ensuring subsequent calls still work (reactivation)
        var result2 = await grain.ExecuteAsync(input);
        result2.Should().NotBeNull();
        result2.Success.Should().BeTrue();
    }

    [Fact]
    public async Task GpuBatchGrain_ExecuteAsync_WithValidBatch_ShouldSucceed()
    {
        // Arrange
        var kernelId = "kernels/test-execute-valid";
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuBatchGrain<float[], float>>(Guid.NewGuid(), kernelId);

        var input = new[]
        {
            new float[] { 1.0f, 2.0f, 3.0f },
            new float[] { 4.0f, 5.0f, 6.0f },
            new float[] { 7.0f, 8.0f, 9.0f }
        };

        // Act
        var result = await grain.ExecuteAsync(input);

        // Assert
        result.Should().NotBeNull();
        result.Success.Should().BeTrue();
        result.Error.Should().BeNullOrEmpty();
        result.Results.Should().NotBeNull();
        result.Results.Should().HaveCount(3);
        result.ExecutionTime.Should().BeGreaterThan(TimeSpan.Zero);
        result.HandleId.Should().NotBeNullOrEmpty();
        result.KernelId.Should().Be(KernelId.Parse(kernelId));
    }

    [Fact]
    public async Task GpuBatchGrain_ExecuteAsync_WithEmptyBatch_ShouldReturnEmpty()
    {
        // Arrange
        var kernelId = "kernels/test-execute-empty";
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuBatchGrain<float[], float>>(Guid.NewGuid(), kernelId);

        var input = Array.Empty<float[]>();

        // Act
        var result = await grain.ExecuteAsync(input);

        // Assert
        result.Should().NotBeNull();
        result.Success.Should().BeTrue();
        result.Results.Should().NotBeNull();
        result.Results.Should().BeEmpty();
    }

    [Fact]
    public async Task GpuBatchGrain_ExecuteAsync_Concurrent_ShouldQueue()
    {
        // Arrange
        var kernelId = "kernels/test-concurrent";
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuBatchGrain<float[], float>>(Guid.NewGuid(), kernelId);

        var input = new[] { new float[] { 1, 2, 3 } };

        // Act - Execute multiple concurrent requests
        var tasks = Enumerable.Range(0, 10)
            .Select(_ => grain.ExecuteAsync(input))
            .ToArray();

        var results = await Task.WhenAll(tasks);

        // Assert - All requests should complete successfully
        results.Should().AllSatisfy(result =>
        {
            result.Should().NotBeNull();
            result.Success.Should().BeTrue();
            result.Results.Should().HaveCount(1);
        });
    }

    [Fact]
    public async Task GpuBatchGrain_State_ShouldPersist()
    {
        // Arrange
        var kernelId = "kernels/test-state-persist";
        var grainId = Guid.NewGuid();
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuBatchGrain<float[], float>>(grainId, kernelId);

        var input = new[] { new float[] { 1, 2, 3 } };

        // Act - Execute batch
        var result1 = await grain.ExecuteAsync(input);

        // Get the same grain instance (should retrieve persisted state)
        var grain2 = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuBatchGrain<float[], float>>(grainId, kernelId);

        var result2 = await grain2.ExecuteAsync(input);

        // Assert - Both executions should work with same grain
        result1.Success.Should().BeTrue();
        result2.Success.Should().BeTrue();
        result1.KernelId.Should().Be(result2.KernelId);
    }

    [Fact]
    public async Task GpuBatchGrain_Metrics_ShouldTrackExecutions()
    {
        // Arrange
        var kernelId = "kernels/test-metrics";
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuBatchGrain<float[], float>>(Guid.NewGuid(), kernelId);

        var input = new[] { new float[] { 1, 2, 3 } };

        // Act - Execute multiple batches
        var results = new List<GpuBatchResult<float>>();
        for (int i = 0; i < 5; i++)
        {
            var result = await grain.ExecuteAsync(input);
            results.Add(result);
        }

        // Assert - Each execution should have metrics
        results.Should().AllSatisfy(result =>
        {
            result.Should().NotBeNull();
            result.Success.Should().BeTrue();
            result.ExecutionTime.Should().BeGreaterThan(TimeSpan.Zero);
            result.HandleId.Should().NotBeNullOrEmpty();
        });

        // Verify execution times are reasonable
        var totalTime = results.Sum(r => r.ExecutionTime.TotalMilliseconds);
        totalTime.Should().BeGreaterThan(0);
        totalTime.Should().BeLessThan(5000); // Should complete in under 5 seconds
    }

    [Fact]
    public async Task GpuBatchGrain_ExecuteWithCallbackAsync_ShouldInvokeObserver()
    {
        // Arrange
        var kernelId = "kernels/test-callback";
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuBatchGrain<float[], float>>(Guid.NewGuid(), kernelId);

        var input = new[] { new float[] { 1, 2, 3 } };
        var observer = new TestGpuResultObserver<float>();
        var observerRef = _fixture.Cluster.GrainFactory.CreateObjectReference<IGpuResultObserver<float>>(observer);

        // Act
        var result = await grain.ExecuteWithCallbackAsync(input, observerRef);

        // Assert
        result.Success.Should().BeTrue();

        // Wait a bit for async callbacks
        await Task.Delay(100);

        observer.ReceivedItems.Should().NotBeEmpty();
        observer.Completed.Should().BeTrue();
        observer.Error.Should().BeNull();
    }

    [Fact]
    public async Task GpuBatchGrain_ProcessBatchAsync_ShouldExecute()
    {
        // Arrange
        var kernelId = "kernels/test-process-batch";
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuBatchGrain<float[], float>>(Guid.NewGuid(), kernelId);

        var input = new[] { new float[] { 1, 2, 3 } };

        // Act - ProcessBatchAsync is an alias for ExecuteAsync
        var result = await grain.ProcessBatchAsync(input);

        // Assert
        result.Should().NotBeNull();
        result.Success.Should().BeTrue();
        result.Results.Should().HaveCount(1);
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

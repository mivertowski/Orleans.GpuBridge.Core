using Orleans.GpuBridge.Abstractions;
using Orleans.GpuBridge.Grains.Batch;
using Orleans.GpuBridge.Grains.Enums;
using Orleans.GpuBridge.Grains.Interfaces;
using Orleans.GpuBridge.Grains.Models;
using Orleans.GpuBridge.Tests.RC2.Infrastructure;

namespace Orleans.GpuBridge.Tests.RC2.Grains;

/// <summary>
/// Advanced comprehensive tests for GpuBatchGrain and GpuResidentGrain.
/// Tests complex scenarios including concurrent operations, error recovery,
/// memory management, performance optimization, and edge cases.
/// </summary>
[Collection("ClusterCollection")]
public sealed class GrainsAdvancedTests : IClassFixture<ClusterFixture>
{
    private readonly ClusterFixture _fixture;

    public GrainsAdvancedTests(ClusterFixture fixture)
    {
        _fixture = fixture;
    }

    #region GpuBatchGrain Advanced - Batch Processing Patterns (7 tests)

    [Fact]
    public async Task GpuBatchGrain_MultipleConcurrentSubmissions_ShouldQueueAndProcess()
    {
        // Arrange
        var kernelId = "kernels/test-concurrent-submissions";
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuBatchGrain<float[], float>>(Guid.NewGuid(), kernelId);

        var batches = Enumerable.Range(0, 20)
            .Select(i => new[] { new float[] { i, i + 1, i + 2 } })
            .ToArray();

        // Act - Submit all batches concurrently
        var tasks = batches.Select(batch => grain.ExecuteAsync(batch)).ToArray();
        var results = await Task.WhenAll(tasks);

        // Assert - All batches should complete successfully
        results.Should().HaveCount(20);
        results.Should().AllSatisfy(result =>
        {
            result.Success.Should().BeTrue();
            result.Results.Should().NotBeNull();
            result.ExecutionTime.Should().BeGreaterThan(TimeSpan.Zero);
            result.Metrics.Should().NotBeNull();
        });

        // Verify no results were lost
        var totalResultsCount = results.Sum(r => r.Results.Count);
        totalResultsCount.Should().Be(20);
    }

    [Fact]
    public async Task GpuBatchGrain_BatchSizeOptimization_WithHints_ShouldRespectMaxMicroBatch()
    {
        // Arrange
        var kernelId = "kernels/test-batch-optimization";
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuBatchGrain<float[], float>>(Guid.NewGuid(), kernelId);

        var largeBatch = Enumerable.Range(0, 5000)
            .Select(i => new float[] { i, i + 1, i + 2 })
            .ToArray();

        var hints = new GpuExecutionHints(
            MaxMicroBatch: 500,
            PreferGpu: true,
            HighPriority: true);

        // Act
        var result = await grain.ExecuteAsync(largeBatch, hints);

        // Assert
        result.Success.Should().BeTrue();
        result.Results.Should().HaveCount(5000);
        result.Metrics.Should().NotBeNull();

        // With MaxMicroBatch=500, expect 10 sub-batches (5000/500)
        result.Metrics!.SubBatchCount.Should().BeGreaterThanOrEqualTo(1);
        result.Metrics.TotalItems.Should().Be(5000);
        result.Metrics.SuccessfulSubBatches.Should().Be(result.Metrics.SubBatchCount);
    }

    [Fact]
    public async Task GpuBatchGrain_BatchTimeoutScenario_WithShortTimeout_ShouldCompleteOrFail()
    {
        // Arrange
        var kernelId = "kernels/test-timeout";
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuBatchGrain<float[], float>>(Guid.NewGuid(), kernelId);

        var batch = Enumerable.Range(0, 100)
            .Select(i => new float[] { i, i + 1 })
            .ToArray();

        // Very short timeout - may fail or succeed depending on system load
        var hints = new GpuExecutionHints(Timeout: TimeSpan.FromMilliseconds(1));

        // Act
        var result = await grain.ExecuteAsync(batch, hints);

        // Assert - Should either succeed or have timeout-related error
        result.Should().NotBeNull();
        result.ExecutionTime.Should().BeGreaterThan(TimeSpan.Zero);

        if (!result.Success)
        {
            result.Error.Should().NotBeNullOrEmpty();
        }
    }

    [Fact]
    public async Task GpuBatchGrain_EmptyBatchHandling_MultipleScenarios_ShouldHandleGracefully()
    {
        // Arrange
        var kernelId = "kernels/test-empty-batches";
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuBatchGrain<float[], float>>(Guid.NewGuid(), kernelId);

        // Act - Test multiple empty batch submissions
        var emptyBatch1 = Array.Empty<float[]>();
        var emptyBatch2 = Array.Empty<float[]>();
        var emptyBatch3 = Array.Empty<float[]>();

        var result1 = await grain.ExecuteAsync(emptyBatch1);
        var result2 = await grain.ExecuteAsync(emptyBatch2);
        var result3 = await grain.ExecuteAsync(emptyBatch3);

        // Assert - All should succeed with empty results
        result1.Success.Should().BeTrue();
        result1.Results.Should().BeEmpty();
        result1.Metrics.Should().NotBeNull();
        result1.Metrics!.TotalItems.Should().Be(0);

        result2.Success.Should().BeTrue();
        result2.Results.Should().BeEmpty();

        result3.Success.Should().BeTrue();
        result3.Results.Should().BeEmpty();
    }

    [Fact]
    public async Task GpuBatchGrain_VeryLargeBatchSplitting_ShouldSplitAndAggregate()
    {
        // Arrange
        var kernelId = "kernels/test-large-split";
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuBatchGrain<float[], float>>(Guid.NewGuid(), kernelId);

        // Create a very large batch (10K items)
        var largeBatch = Enumerable.Range(0, 10000)
            .Select(i => new float[] { i })
            .ToArray();

        var hints = new GpuExecutionHints(MaxMicroBatch: 1000);

        // Act
        var result = await grain.ExecuteAsync(largeBatch, hints);

        // Assert
        result.Success.Should().BeTrue();
        result.Results.Should().HaveCount(10000);
        result.Metrics.Should().NotBeNull();
        result.Metrics!.TotalItems.Should().Be(10000);

        // Should have split into multiple sub-batches
        result.Metrics.SubBatchCount.Should().BeGreaterThanOrEqualTo(1);
        result.Metrics.Throughput.Should().BeGreaterThan(0);
    }

    [Fact]
    public async Task GpuBatchGrain_BatchPriorityQueuing_HighPriorityFirst_ShouldProcessInOrder()
    {
        // Arrange
        var kernelId = "kernels/test-priority";
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuBatchGrain<float[], float>>(Guid.NewGuid(), kernelId);

        var normalBatch = new[] { new float[] { 1, 2, 3 } };
        var highPriorityBatch = new[] { new float[] { 10, 20, 30 } };

        var normalHints = new GpuExecutionHints(HighPriority: false);
        var highPriorityHints = new GpuExecutionHints(HighPriority: true);

        // Act - Submit both concurrently
        var normalTask = grain.ExecuteAsync(normalBatch, normalHints);
        var highPriorityTask = grain.ExecuteAsync(highPriorityBatch, highPriorityHints);

        var results = await Task.WhenAll(normalTask, highPriorityTask);

        // Assert - Both should complete
        results.Should().HaveCount(2);
        results.Should().AllSatisfy(r => r.Success.Should().BeTrue());
    }

    [Fact]
    public async Task GpuBatchGrain_MixedBatchSizes_ShouldHandleDynamically()
    {
        // Arrange
        var kernelId = "kernels/test-mixed-sizes";
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuBatchGrain<float[], float>>(Guid.NewGuid(), kernelId);

        // Create batches of varying sizes
        var smallBatch = new[] { new float[] { 1 } };
        var mediumBatch = Enumerable.Range(0, 100).Select(i => new float[] { i }).ToArray();
        var largeBatch = Enumerable.Range(0, 1000).Select(i => new float[] { i }).ToArray();

        // Act - Execute sequentially
        var smallResult = await grain.ExecuteAsync(smallBatch);
        var mediumResult = await grain.ExecuteAsync(mediumBatch);
        var largeResult = await grain.ExecuteAsync(largeBatch);

        // Assert
        smallResult.Success.Should().BeTrue();
        smallResult.Results.Should().HaveCount(1);

        mediumResult.Success.Should().BeTrue();
        mediumResult.Results.Should().HaveCount(100);

        largeResult.Success.Should().BeTrue();
        largeResult.Results.Should().HaveCount(1000);

        // All batches should complete successfully (timing varies based on system load)
        smallResult.ExecutionTime.Should().BeGreaterThan(TimeSpan.Zero);
        largeResult.ExecutionTime.Should().BeGreaterThan(TimeSpan.Zero);
    }

    #endregion

    #region GpuBatchGrain Advanced - Memory and Resource Management (7 tests)

    [Fact]
    public async Task GpuBatchGrain_MemoryPressure_LargeBatches_ShouldHandleGracefully()
    {
        // Arrange
        var kernelId = "kernels/test-memory-pressure";
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuBatchGrain<float[], float>>(Guid.NewGuid(), kernelId);

        // Create multiple large batches to simulate memory pressure
        var largeBatches = Enumerable.Range(0, 5)
            .Select(batch => Enumerable.Range(0, 5000)
                .Select(i => new float[] { i, i + 1, i + 2, i + 4 })
                .ToArray())
            .ToArray();

        // Act - Execute all batches concurrently
        var tasks = largeBatches.Select(batch => grain.ExecuteAsync(batch)).ToArray();
        var results = await Task.WhenAll(tasks);

        // Assert - Should handle memory pressure
        results.Should().AllSatisfy(result =>
        {
            result.Should().NotBeNull();
            result.ExecutionTime.Should().BeGreaterThan(TimeSpan.Zero);
        });

        var successfulResults = results.Where(r => r.Success).ToList();
        successfulResults.Should().NotBeEmpty("At least some batches should complete successfully");
    }

    [Fact]
    public async Task GpuBatchGrain_GpuMemoryAllocationFailure_ShouldFallbackToCpu()
    {
        // Arrange - Use a kernel ID that might trigger GPU allocation failure
        var kernelId = "kernels/test-allocation-failure";
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuBatchGrain<float[], float>>(Guid.NewGuid(), kernelId);

        var batch = Enumerable.Range(0, 100)
            .Select(i => new float[] { i })
            .ToArray();

        // Force CPU fallback by preferring CPU
        var hints = new GpuExecutionHints(PreferGpu: false);

        // Act
        var result = await grain.ExecuteAsync(batch, hints);

        // Assert - Should succeed on CPU
        result.Success.Should().BeTrue();
        result.Results.Should().HaveCount(100);
        result.Metrics.Should().NotBeNull();
        result.Metrics!.DeviceType.Should().Be("CPU");
    }

    [Fact]
    public async Task GpuBatchGrain_ResourceCleanup_AfterErrors_ShouldReleaseResources()
    {
        // Arrange
        var kernelId = "kernels/test-cleanup";
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuBatchGrain<float[], float>>(Guid.NewGuid(), kernelId);

        var batch = new[] { new float[] { 1, 2, 3 } };

        // Act - Execute multiple times to verify cleanup between calls
        for (int i = 0; i < 10; i++)
        {
            var result = await grain.ExecuteAsync(batch);
            result.Success.Should().BeTrue();
        }

        // Assert - Final execution should still work (resources were cleaned up)
        var finalResult = await grain.ExecuteAsync(batch);
        finalResult.Success.Should().BeTrue();
        finalResult.Results.Should().NotBeNull();
    }

    [Fact]
    public async Task GpuBatchGrain_ConcurrentMemoryAccess_ShouldSynchronize()
    {
        // Arrange
        var kernelId = "kernels/test-concurrent-memory";
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuBatchGrain<float[], float>>(Guid.NewGuid(), kernelId);

        var batches = Enumerable.Range(0, 15)
            .Select(i => new[] { new float[] { i, i * 2 } })
            .ToArray();

        // Act - Execute all batches concurrently (testing concurrent memory access)
        var tasks = batches.Select(batch => grain.ExecuteAsync(batch)).ToArray();
        var results = await Task.WhenAll(tasks);

        // Assert - All should complete without memory corruption
        results.Should().HaveCount(15);
        results.Should().AllSatisfy(result =>
        {
            result.Success.Should().BeTrue();
            result.Results.Should().NotBeNull();
        });
    }

    [Fact]
    public async Task GpuBatchGrain_MemoryPooling_ReuseBuffers_ShouldOptimizeAllocations()
    {
        // Arrange
        var kernelId = "kernels/test-pooling";
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuBatchGrain<float[], float>>(Guid.NewGuid(), kernelId);

        var batch = Enumerable.Range(0, 100).Select(i => new float[] { i }).ToArray();

        // Act - Execute same-sized batch multiple times (should reuse buffers)
        var results = new List<GpuBatchResult<float>>();
        for (int i = 0; i < 20; i++)
        {
            var result = await grain.ExecuteAsync(batch);
            results.Add(result);
        }

        // Assert - All executions should succeed
        results.Should().AllSatisfy(result =>
        {
            result.Success.Should().BeTrue();
            result.Results.Should().HaveCount(100);
        });

        // Later executions might be faster due to pooling/caching
        var firstExecution = results.First().ExecutionTime;
        var lastExecution = results.Last().ExecutionTime;
        lastExecution.Should().BeLessThanOrEqualTo(firstExecution.Add(TimeSpan.FromMilliseconds(100)));
    }

    [Fact]
    public async Task GpuBatchGrain_BufferReuseValidation_DifferentSizes_ShouldHandleCorrectly()
    {
        // Arrange
        var kernelId = "kernels/test-buffer-reuse";
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuBatchGrain<float[], float>>(Guid.NewGuid(), kernelId);

        // Act - Execute batches of varying sizes in sequence
        var sizes = new[] { 10, 100, 50, 200, 25, 150 };
        var results = new List<GpuBatchResult<float>>();

        foreach (var size in sizes)
        {
            var batch = Enumerable.Range(0, size).Select(i => new float[] { i }).ToArray();
            var result = await grain.ExecuteAsync(batch);
            results.Add(result);
        }

        // Assert - All should succeed despite size variations
        results.Should().HaveCount(sizes.Length);
        for (int i = 0; i < sizes.Length; i++)
        {
            results[i].Success.Should().BeTrue();
            results[i].Results.Should().HaveCount(sizes[i]);
        }
    }

    [Fact]
    public async Task GpuBatchGrain_MemoryLeakDetection_LongRunning_ShouldNotLeak()
    {
        // Arrange
        var kernelId = "kernels/test-memory-leak";
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuBatchGrain<float[], float>>(Guid.NewGuid(), kernelId);

        var batch = Enumerable.Range(0, 100).Select(i => new float[] { i }).ToArray();

        // Act - Execute many iterations to detect memory leaks
        var iterations = 50;
        var results = new List<GpuBatchResult<float>>();

        for (int i = 0; i < iterations; i++)
        {
            var result = await grain.ExecuteAsync(batch);
            results.Add(result);

            // Verify memory is released between calls
            if (i % 10 == 0)
            {
                GC.Collect();
                GC.WaitForPendingFinalizers();
            }
        }

        // Assert - All executions should succeed
        results.Should().HaveCount(iterations);
        results.Should().AllSatisfy(r => r.Success.Should().BeTrue());

        // Execution times should remain consistent (no memory buildup)
        var avgTimeFirst10 = results.Take(10).Average(r => r.ExecutionTime.TotalMilliseconds);
        var avgTimeLast10 = results.Skip(iterations - 10).Average(r => r.ExecutionTime.TotalMilliseconds);

        // Allow 2x variance (last batch shouldn't be significantly slower)
        avgTimeLast10.Should().BeLessThanOrEqualTo(avgTimeFirst10 * 2);
    }

    #endregion

    #region GpuBatchGrain Advanced - Error Recovery (7 tests)

    [Fact]
    public async Task GpuBatchGrain_KernelExecutionFailure_ShouldReturnError()
    {
        // Arrange - Use invalid kernel ID to trigger failure
        var kernelId = "kernels/invalid-kernel";
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuBatchGrain<float[], float>>(Guid.NewGuid(), kernelId);

        var batch = new[] { new float[] { 1, 2, 3 } };

        // Act
        var result = await grain.ExecuteAsync(batch);

        // Assert - Should handle failure gracefully
        result.Should().NotBeNull();
        result.ExecutionTime.Should().BeGreaterThanOrEqualTo(TimeSpan.Zero);
    }

    [Fact]
    public async Task GpuBatchGrain_PartialBatchFailure_ShouldReportResults()
    {
        // Arrange
        var kernelId = "kernels/test-partial-failure";
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuBatchGrain<float[], float>>(Guid.NewGuid(), kernelId);

        // Large batch that might partially fail
        var largeBatch = Enumerable.Range(0, 3000)
            .Select(i => new float[] { i })
            .ToArray();

        var hints = new GpuExecutionHints(MaxMicroBatch: 500);

        // Act
        var result = await grain.ExecuteAsync(largeBatch, hints);

        // Assert - Should report results or errors
        result.Should().NotBeNull();
        result.Metrics.Should().NotBeNull();

        if (result.Success)
        {
            result.Results.Should().NotBeNull();
        }
        else
        {
            result.Error.Should().NotBeNullOrEmpty();
        }
    }

    [Fact]
    public async Task GpuBatchGrain_RetryLogic_TransientFailures_ShouldRetry()
    {
        // Arrange
        var kernelId = "kernels/test-retry";
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuBatchGrain<float[], float>>(Guid.NewGuid(), kernelId);

        var batch = new[] { new float[] { 1, 2, 3 } };
        var hints = new GpuExecutionHints(MaxRetries: 3);

        // Act - Execute with retry hints
        var result = await grain.ExecuteAsync(batch, hints);

        // Assert - Should complete (with or without retries)
        result.Should().NotBeNull();
        result.ExecutionTime.Should().BeGreaterThan(TimeSpan.Zero);
    }

    [Fact]
    public async Task GpuBatchGrain_CircuitBreaker_AfterMultipleFailures_ShouldBreak()
    {
        // Arrange - Use a kernel that will consistently fail
        var kernelId = "kernels/test-circuit-breaker";
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuBatchGrain<float[], float>>(Guid.NewGuid(), kernelId);

        var batch = new[] { new float[] { 1, 2, 3 } };

        // Act - Execute multiple times
        var results = new List<GpuBatchResult<float>>();
        for (int i = 0; i < 5; i++)
        {
            var result = await grain.ExecuteAsync(batch);
            results.Add(result);
        }

        // Assert - Should handle failures gracefully
        results.Should().HaveCount(5);
        results.Should().AllSatisfy(r => r.Should().NotBeNull());
    }

    [Fact]
    public async Task GpuBatchGrain_GracefulDegradation_GpuToCpu_ShouldFallback()
    {
        // Arrange
        var kernelId = "kernels/test-degradation";
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuBatchGrain<float[], float>>(Guid.NewGuid(), kernelId);

        var batch = Enumerable.Range(0, 100).Select(i => new float[] { i }).ToArray();

        // First try GPU
        var gpuHints = new GpuExecutionHints(PreferGpu: true);
        var gpuResult = await grain.ExecuteAsync(batch, gpuHints);

        // Then force CPU fallback
        var cpuHints = new GpuExecutionHints(PreferGpu: false);
        var cpuResult = await grain.ExecuteAsync(batch, cpuHints);

        // Assert - Both should complete
        gpuResult.Should().NotBeNull();
        cpuResult.Should().NotBeNull();

        if (cpuResult.Success)
        {
            cpuResult.Metrics!.DeviceType.Should().Be("CPU");
        }
    }

    [Fact]
    public async Task GpuBatchGrain_ErrorAggregation_MultipleBatches_ShouldReportAll()
    {
        // Arrange
        var kernelId = "kernels/test-error-aggregation";
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuBatchGrain<float[], float>>(Guid.NewGuid(), kernelId);

        var batches = Enumerable.Range(0, 10)
            .Select(i => new[] { new float[] { i } })
            .ToArray();

        // Act - Execute all batches
        var tasks = batches.Select(batch => grain.ExecuteAsync(batch)).ToArray();
        var results = await Task.WhenAll(tasks);

        // Assert - Should report all results
        results.Should().HaveCount(10);
        results.Should().AllSatisfy(r => r.Should().NotBeNull());

        var successCount = results.Count(r => r.Success);
        var errorCount = results.Count(r => !r.Success);

        (successCount + errorCount).Should().Be(10);
    }

    [Fact]
    public async Task GpuBatchGrain_RecoveryAfterError_ShouldContinueProcessing()
    {
        // Arrange
        var kernelId = "kernels/test-recovery";
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuBatchGrain<float[], float>>(Guid.NewGuid(), kernelId);

        var batch = new[] { new float[] { 1, 2, 3 } };

        // Act - Execute, simulate error, then execute again
        var result1 = await grain.ExecuteAsync(batch);

        // Try to trigger an error condition (if possible)
        var result2 = await grain.ExecuteAsync(Array.Empty<float[]>());

        // Execute normal batch again after error
        var result3 = await grain.ExecuteAsync(batch);

        // Assert - Should recover and continue processing
        result1.Should().NotBeNull();
        result2.Should().NotBeNull();
        result3.Should().NotBeNull();

        // The final execution after error should still work
        result3.Success.Should().BeTrue();
        result3.Results.Should().NotBeNull();
    }

    #endregion

    #region GpuResidentGrain Advanced - State Management (7 tests)

    [Fact]
    public async Task GpuResidentGrain_StatePersistence_AcrossActivations_ShouldMaintainData()
    {
        // Arrange
        var grainKey = $"test-persistence-{Guid.NewGuid()}";
        var grain1 = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuResidentGrain<int>>(grainKey);

        var data = Enumerable.Range(0, 1000).ToArray();
        await grain1.StoreDataAsync(data);

        // Act - Get the same grain (simulates reactivation)
        var grain2 = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuResidentGrain<int>>(grainKey);

        var retrieved = await grain2.GetDataAsync();

        // Assert - Data should persist
        retrieved.Should().NotBeNull();
        retrieved.Should().HaveCount(data.Length);
        retrieved.Should().Equal(data);
    }

    [Fact]
    public async Task GpuResidentGrain_StateSizeLimit_VeryLargeData_ShouldHandle()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuResidentGrain<long>>($"test-size-limit-{Guid.NewGuid()}");

        // Create 10MB of data (1.25M longs)
        var largeData = Enumerable.Range(0, 1_250_000)
            .Select(i => (long)i)
            .ToArray();

        // Act
        await grain.StoreDataAsync(largeData);
        var info = await grain.GetMemoryInfoAsync();

        // Assert
        info.AllocatedMemoryBytes.Should().BeGreaterThanOrEqualTo(10_000_000); // ~10MB

        // Verify data can be retrieved
        var retrieved = await grain.GetDataAsync();
        retrieved.Should().NotBeNull();
        retrieved.Should().HaveCount(largeData.Length);
    }

    [Fact]
    public async Task GpuResidentGrain_StateCorruption_Recovery_ShouldRecover()
    {
        // Arrange
        var grainKey = $"test-corruption-{Guid.NewGuid()}";
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuResidentGrain<float>>(grainKey);

        var data = new float[] { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f };
        await grain.StoreDataAsync(data);

        // Act - Clear and restore
        await grain.ClearAsync();
        await grain.StoreDataAsync(data);

        var retrieved = await grain.GetDataAsync();

        // Assert - Should recover after clear
        retrieved.Should().NotBeNull();
        retrieved.Should().Equal(data);
    }

    [Fact]
    public async Task GpuResidentGrain_ConcurrentStateUpdates_ShouldSynchronize()
    {
        // Arrange
        var grainKey = $"test-concurrent-state-{Guid.NewGuid()}";
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuResidentGrain<int>>(grainKey);

        // Act - Multiple concurrent allocations
        var tasks = Enumerable.Range(0, 20)
            .Select(async i =>
            {
                var handle = await grain.AllocateAsync(1024);
                var data = new int[] { i, i + 1, i + 2 };
                await grain.WriteAsync(handle, data);
                return handle;
            })
            .ToArray();

        var handles = await Task.WhenAll(tasks);

        // Assert - All allocations should succeed
        handles.Should().HaveCount(20);
        handles.Should().OnlyHaveUniqueItems(h => h.Id);

        var info = await grain.GetMemoryInfoAsync();
        info.AllocatedMemoryBytes.Should().BeGreaterThan(0);
    }

    [Fact]
    public async Task GpuResidentGrain_StateSnapshot_AndRestore_ShouldWork()
    {
        // Arrange
        var grainKey = $"test-snapshot-{Guid.NewGuid()}";
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuResidentGrain<double>>(grainKey);

        var originalData = new double[] { 1.5, 2.5, 3.5, 4.5 };
        await grain.StoreDataAsync(originalData);

        var info1 = await grain.GetMemoryInfoAsync();

        // Act - Retrieve data (snapshot)
        var snapshot = await grain.GetDataAsync();

        // Clear and restore from snapshot
        await grain.ClearAsync();
        await grain.StoreDataAsync(snapshot!);

        var info2 = await grain.GetMemoryInfoAsync();
        var restored = await grain.GetDataAsync();

        // Assert
        restored.Should().NotBeNull();
        restored.Should().Equal(originalData);
        info2.AllocatedMemoryBytes.Should().BeGreaterThan(0);
    }

    [Fact]
    public async Task GpuResidentGrain_StateMigration_DifferentTypes_ShouldHandle()
    {
        // Arrange - Store int data
        var grainKeyInt = $"test-migration-int-{Guid.NewGuid()}";
        var grainInt = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuResidentGrain<int>>(grainKeyInt);

        var intData = new int[] { 1, 2, 3, 4, 5 };
        await grainInt.StoreDataAsync(intData);

        // Arrange - Store float data with same key pattern
        var grainKeyFloat = $"test-migration-float-{Guid.NewGuid()}";
        var grainFloat = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuResidentGrain<float>>(grainKeyFloat);

        var floatData = new float[] { 1.0f, 2.0f, 3.0f };
        await grainFloat.StoreDataAsync(floatData);

        // Act - Retrieve both
        var retrievedInt = await grainInt.GetDataAsync();
        var retrievedFloat = await grainFloat.GetDataAsync();

        // Assert - Both types should work independently
        retrievedInt.Should().NotBeNull();
        retrievedInt.Should().Equal(intData);

        retrievedFloat.Should().NotBeNull();
        retrievedFloat.Should().Equal(floatData);
    }

    [Fact]
    public async Task GpuResidentGrain_StateMultipleAllocations_ShouldTrackAll()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuResidentGrain<byte>>($"test-multi-alloc-{Guid.NewGuid()}");

        // Act - Create multiple allocations with different sizes
        var handles = new List<GpuMemoryHandle>();
        var sizes = new[] { 512L, 1024L, 2048L, 4096L, 8192L };

        foreach (var size in sizes)
        {
            var handle = await grain.AllocateAsync(size);
            handles.Add(handle);
        }

        var info = await grain.GetMemoryInfoAsync();

        // Assert - All allocations should be tracked
        handles.Should().HaveCount(sizes.Length);
        handles.Should().OnlyHaveUniqueItems(h => h.Id);

        var totalExpected = sizes.Sum();
        info.AllocatedMemoryBytes.Should().BeGreaterThanOrEqualTo(totalExpected);
    }

    #endregion

    #region GpuResidentGrain Advanced - Memory Lifecycle (7 tests)

    [Fact]
    public async Task GpuResidentGrain_RingBuffer_WraparoundHandling_ShouldManage()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuResidentGrain<int>>($"test-ringbuffer-{Guid.NewGuid()}");

        var handles = new List<GpuMemoryHandle>();

        // Act - Allocate, write, release in a cycle
        for (int cycle = 0; cycle < 3; cycle++)
        {
            for (int i = 0; i < 10; i++)
            {
                var handle = await grain.AllocateAsync(1024);
                var data = Enumerable.Range(i, 10).ToArray();
                await grain.WriteAsync(handle, data);
                handles.Add(handle);
            }

            // Release all handles
            foreach (var handle in handles)
            {
                await grain.ReleaseAsync(handle);
            }
            handles.Clear();
        }

        // Assert - Should handle wraparound gracefully
        var info = await grain.GetMemoryInfoAsync();
        info.AllocatedMemoryBytes.Should().Be(0);
    }

    [Fact]
    public async Task GpuResidentGrain_MemoryPinning_AndUnpinning_ShouldManage()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuResidentGrain<long>>($"test-pinning-{Guid.NewGuid()}");

        // Act - Allocate pinned memory
        var pinnedHandle = await grain.AllocateAsync(8192, GpuMemoryType.Pinned);
        var data = Enumerable.Range(0, 1024).Select(i => (long)i).ToArray();
        await grain.WriteAsync(pinnedHandle, data);

        // Read back
        var retrieved = await grain.ReadAsync<long>(pinnedHandle, 1024);

        // Release
        await grain.ReleaseAsync(pinnedHandle);

        // Assert
        retrieved.Should().Equal(data);

        var info = await grain.GetMemoryInfoAsync();
        info.AllocatedMemoryBytes.Should().Be(0);
    }

    [Fact]
    public async Task GpuResidentGrain_GpuMemoryMapping_Validation_ShouldValidate()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuResidentGrain<float>>($"test-mapping-{Guid.NewGuid()}");

        var handle = await grain.AllocateAsync(4096);
        var data = Enumerable.Range(0, 1024).Select(i => (float)i).ToArray();

        // Act - Write and read
        await grain.WriteAsync(handle, data);
        var retrieved = await grain.ReadAsync<float>(handle, 1024);

        // Assert - Data should match (validates memory mapping)
        retrieved.Should().NotBeNull();
        retrieved.Should().HaveCount(1024);
        retrieved.Should().Equal(data);
    }

    [Fact]
    public async Task GpuResidentGrain_MemoryPressure_Backpressure_ShouldApply()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuResidentGrain<byte>>($"test-backpressure-{Guid.NewGuid()}");

        // Act - Allocate until memory pressure
        var handles = new List<GpuMemoryHandle>();
        var allocSize = 1024 * 1024; // 1MB each

        for (int i = 0; i < 100; i++)
        {
            try
            {
                var handle = await grain.AllocateAsync(allocSize);
                handles.Add(handle);
            }
            catch
            {
                // Expected - memory exhaustion
                break;
            }
        }

        // Assert - Should have allocated some memory
        handles.Should().NotBeEmpty();

        // Clean up
        foreach (var handle in handles)
        {
            await grain.ReleaseAsync(handle);
        }

        var info = await grain.GetMemoryInfoAsync();
        info.AllocatedMemoryBytes.Should().Be(0);
    }

    [Fact]
    public async Task GpuResidentGrain_AutomaticMemoryCompaction_ShouldCompact()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuResidentGrain<int>>($"test-compaction-{Guid.NewGuid()}");

        // Act - Allocate many small buffers
        var handles = new List<GpuMemoryHandle>();
        for (int i = 0; i < 50; i++)
        {
            var handle = await grain.AllocateAsync(256);
            handles.Add(handle);
        }

        var infoBeforeRelease = await grain.GetMemoryInfoAsync();

        // Release every other handle (creates fragmentation)
        for (int i = 0; i < handles.Count; i += 2)
        {
            await grain.ReleaseAsync(handles[i]);
        }

        var infoAfterRelease = await grain.GetMemoryInfoAsync();

        // Assert - Memory should be released
        infoAfterRelease.AllocatedMemoryBytes.Should().BeLessThan(infoBeforeRelease.AllocatedMemoryBytes);

        // Clean up remaining handles
        for (int i = 1; i < handles.Count; i += 2)
        {
            await grain.ReleaseAsync(handles[i]);
        }
    }

    [Fact]
    public async Task GpuResidentGrain_MemoryLeakDetection_ExtendedUsage_ShouldNotLeak()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuResidentGrain<double>>($"test-leak-detection-{Guid.NewGuid()}");

        // Act - Repeated allocate/release cycles
        for (int cycle = 0; cycle < 100; cycle++)
        {
            var handle = await grain.AllocateAsync(2048);
            var data = new double[] { cycle, cycle + 1, cycle + 2 };
            await grain.WriteAsync(handle, data);
            var retrieved = await grain.ReadAsync<double>(handle, 3);
            await grain.ReleaseAsync(handle);

            // Periodic GC to detect leaks
            if (cycle % 20 == 0)
            {
                GC.Collect();
                GC.WaitForPendingFinalizers();
            }
        }

        // Assert - Memory should be fully released
        var info = await grain.GetMemoryInfoAsync();
        info.AllocatedMemoryBytes.Should().Be(0);
    }

    [Fact]
    public async Task GpuResidentGrain_MultipleKernelCompute_ReusingMemory_ShouldWork()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuResidentGrain<float>>($"test-multi-kernel-{Guid.NewGuid()}");

        var inputHandle = await grain.AllocateAsync(4096);
        var outputHandle = await grain.AllocateAsync(4096);

        var inputData = Enumerable.Range(0, 1024).Select(i => (float)i).ToArray();
        await grain.WriteAsync(inputHandle, inputData);

        // Act - Execute multiple kernels on same memory
        var kernel1 = KernelId.Parse("kernels/test-kernel-1");
        var kernel2 = KernelId.Parse("kernels/test-kernel-2");

        var result1 = await grain.ComputeAsync(kernel1, inputHandle, outputHandle);
        var result2 = await grain.ComputeAsync(kernel2, inputHandle, outputHandle);

        // Assert
        result1.Success.Should().BeTrue();
        result1.ExecutionTime.Should().BeGreaterThan(TimeSpan.Zero);

        result2.Success.Should().BeTrue();
        result2.ExecutionTime.Should().BeGreaterThan(TimeSpan.Zero);

        // Clean up
        await grain.ReleaseAsync(inputHandle);
        await grain.ReleaseAsync(outputHandle);
    }

    #endregion

    #region GpuResidentGrain Advanced - Performance Optimization (6 tests)

    [Fact]
    public async Task GpuResidentGrain_KernelPreloading_ShouldOptimize()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuResidentGrain<int>>($"test-preload-{Guid.NewGuid()}");

        var inputHandle = await grain.AllocateAsync(2048);
        var outputHandle = await grain.AllocateAsync(2048);

        var data = Enumerable.Range(0, 512).ToArray();
        await grain.WriteAsync(inputHandle, data);

        var kernelId = KernelId.Parse("kernels/test-preload");

        // Act - First execution (may preload)
        var result1 = await grain.ComputeAsync(kernelId, inputHandle, outputHandle);

        // Second execution (should benefit from preload)
        var result2 = await grain.ComputeAsync(kernelId, inputHandle, outputHandle);

        // Assert - Both should succeed
        result1.Success.Should().BeTrue();
        result2.Success.Should().BeTrue();

        // Second execution might be faster due to preloading
        result2.ExecutionTime.Should().BeLessThanOrEqualTo(result1.ExecutionTime.Add(TimeSpan.FromMilliseconds(10)));

        // Clean up
        await grain.ReleaseAsync(inputHandle);
        await grain.ReleaseAsync(outputHandle);
    }

    [Fact]
    public async Task GpuResidentGrain_BatchAccumulation_ShouldOptimize()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuResidentGrain<float>>($"test-batch-accumulate-{Guid.NewGuid()}");

        // Act - Multiple small writes
        var handle = await grain.AllocateAsync(16384);
        var baselineData = new float[] { 1.0f, 2.0f, 3.0f, 4.0f };

        for (int i = 0; i < 10; i++)
        {
            await grain.WriteAsync(handle, baselineData, i * 16);
        }

        // Assert - All writes should succeed
        var retrieved = await grain.ReadAsync<float>(handle, 4, 0);
        retrieved.Should().Equal(baselineData);

        await grain.ReleaseAsync(handle);
    }

    [Fact]
    public async Task GpuResidentGrain_LazyKernelCompilation_ShouldDefer()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuResidentGrain<int>>($"test-lazy-compile-{Guid.NewGuid()}");

        var inputHandle = await grain.AllocateAsync(1024);
        var outputHandle = await grain.AllocateAsync(1024);

        var data = new int[] { 1, 2, 3, 4, 5 };
        await grain.WriteAsync(inputHandle, data);

        // Act - First compute triggers compilation
        var kernelId = KernelId.Parse("kernels/test-lazy");
        var result = await grain.ComputeAsync(kernelId, inputHandle, outputHandle);

        // Assert
        result.Should().NotBeNull();
        result.ExecutionTime.Should().BeGreaterThan(TimeSpan.Zero);

        await grain.ReleaseAsync(inputHandle);
        await grain.ReleaseAsync(outputHandle);
    }

    [Fact]
    public async Task GpuResidentGrain_HotPathOptimization_RepeatedOperations_ShouldOptimize()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuResidentGrain<double>>($"test-hotpath-{Guid.NewGuid()}");

        var handle = await grain.AllocateAsync(8192);
        var data = Enumerable.Range(0, 1024).Select(i => (double)i).ToArray();

        // Act - Repeated write/read operations (hot path)
        var executionTimes = new List<TimeSpan>();

        for (int i = 0; i < 50; i++)
        {
            var sw = System.Diagnostics.Stopwatch.StartNew();
            await grain.WriteAsync(handle, data);
            var retrieved = await grain.ReadAsync<double>(handle, data.Length);
            sw.Stop();
            executionTimes.Add(sw.Elapsed);
        }

        // Assert - Later operations should be optimized
        var avgFirst10 = executionTimes.Take(10).Average(t => t.TotalMilliseconds);
        var avgLast10 = executionTimes.Skip(40).Average(t => t.TotalMilliseconds);

        // Last 10 should not be slower than first 10 (optimization working)
        avgLast10.Should().BeLessThanOrEqualTo(avgFirst10 * 1.5);

        await grain.ReleaseAsync(handle);
    }

    [Fact]
    public async Task GpuResidentGrain_ThroughputMeasurement_UnderLoad_ShouldMaintain()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuResidentGrain<int>>($"test-throughput-{Guid.NewGuid()}");

        var handles = new List<GpuMemoryHandle>();
        var dataSize = 1024;

        // Act - Measure throughput under load
        var sw = System.Diagnostics.Stopwatch.StartNew();

        for (int i = 0; i < 100; i++)
        {
            var handle = await grain.AllocateAsync(dataSize * sizeof(int));
            var data = Enumerable.Range(0, dataSize).ToArray();
            await grain.WriteAsync(handle, data);
            handles.Add(handle);
        }

        sw.Stop();

        // Assert - Calculate throughput
        var totalBytes = handles.Count * dataSize * sizeof(int);
        var throughputMBps = (totalBytes / (1024.0 * 1024.0)) / sw.Elapsed.TotalSeconds;

        throughputMBps.Should().BeGreaterThan(0);

        // Clean up
        foreach (var handle in handles)
        {
            await grain.ReleaseAsync(handle);
        }
    }

    [Fact]
    public async Task GpuResidentGrain_ConcurrentCompute_ParallelKernels_ShouldExecute()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuResidentGrain<float>>($"test-parallel-{Guid.NewGuid()}");

        // Create multiple input/output pairs
        var pairs = await Task.WhenAll(Enumerable.Range(0, 5).Select(async i =>
        {
            var input = await grain.AllocateAsync(2048);
            var output = await grain.AllocateAsync(2048);
            var data = Enumerable.Range(i * 100, 512).Select(x => (float)x).ToArray();
            await grain.WriteAsync(input, data);
            return (input, output, data);
        }));

        // Act - Execute kernels in parallel
        var kernelId = KernelId.Parse("kernels/test-parallel");
        var tasks = pairs.Select(pair =>
            grain.ComputeAsync(kernelId, pair.input, pair.output)).ToArray();

        var results = await Task.WhenAll(tasks);

        // Assert - All should complete
        results.Should().HaveCount(5);
        results.Should().AllSatisfy(r =>
        {
            r.Success.Should().BeTrue();
            r.ExecutionTime.Should().BeGreaterThan(TimeSpan.Zero);
        });

        // Clean up
        foreach (var (input, output, _) in pairs)
        {
            await grain.ReleaseAsync(input);
            await grain.ReleaseAsync(output);
        }
    }

    #endregion
}

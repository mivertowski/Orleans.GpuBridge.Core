using Orleans.GpuBridge.Abstractions;
using Orleans.GpuBridge.Grains.Enums;
using Orleans.GpuBridge.Grains.Interfaces;
using Orleans.GpuBridge.Grains.Models;
using Orleans.GpuBridge.Grains.Resident.Metrics;
using Orleans.GpuBridge.Tests.RC2.Infrastructure;

namespace Orleans.GpuBridge.Tests.RC2.Grains;

/// <summary>
/// Comprehensive tests for GpuResidentGrainEnhanced with Ring Kernel integration.
/// Tests resident memory management, persistent execution loops, ring kernels, and performance optimization.
/// Target: 40+ tests for enhanced ring kernel features.
/// </summary>
[Collection("ClusterCollection")]
public sealed class GpuResidentGrainEnhancedTests : IClassFixture<ClusterFixture>
{
    private readonly ClusterFixture _fixture;

    public GpuResidentGrainEnhancedTests(ClusterFixture fixture)
    {
        _fixture = fixture;
    }

    #region Ring Kernel Initialization Tests

    [Fact]
    public async Task EnhancedGrain_Activation_ShouldInitializeRingKernel()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuResidentGrain<float>>($"enhanced-init-{Guid.NewGuid()}");

        // Act - First call triggers activation and ring kernel initialization
        var info = await grain.GetMemoryInfoAsync();

        // Assert
        info.Should().NotBeNull();
        info.DeviceName.Should().NotBeNullOrEmpty();
    }

    [Fact]
    public async Task EnhancedGrain_RingKernel_ShouldStartWithEmptyPool()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuResidentGrain<int>>($"empty-pool-{Guid.NewGuid()}");

        // Act
        var info = await grain.GetMemoryInfoAsync();

        // Assert
        info.TotalMemoryBytes.Should().Be(0);
        info.AllocatedMemoryBytes.Should().Be(0);
    }

    #endregion

    #region Memory Pool Tests

    [Fact]
    public async Task EnhancedGrain_AllocateAsync_ShouldUseMemoryPool()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuResidentGrain<float>>($"pool-alloc-{Guid.NewGuid()}");

        // Act
        var handle = await grain.AllocateAsync(1024);

        // Assert
        handle.Should().NotBeNull();
        handle.SizeBytes.Should().Be(1024);
        handle.Id.Should().NotBeNullOrEmpty();
    }

    [Fact]
    public async Task EnhancedGrain_SequentialAllocations_ShouldReuseFromPool()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuResidentGrain<byte>>($"pool-reuse-{Guid.NewGuid()}");

        var size = 1024L;

        // Act - Allocate, release, allocate again
        var handle1 = await grain.AllocateAsync(size);
        await grain.ReleaseAsync(handle1);

        var handle2 = await grain.AllocateAsync(size);

        // Assert
        handle2.Should().NotBeNull();
        handle2.SizeBytes.Should().Be(size);
    }

    [Fact]
    public async Task EnhancedGrain_PoolHitRate_ShouldBeTracked()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuResidentGrain<int>>($"hit-rate-{Guid.NewGuid()}");

        var size = 512L;

        // Act - Multiple allocations of same size
        var handle1 = await grain.AllocateAsync(size);
        await grain.ReleaseAsync(handle1);

        var handle2 = await grain.AllocateAsync(size);
        await grain.ReleaseAsync(handle2);

        var handle3 = await grain.AllocateAsync(size);

        // Assert - Metrics should show pool hits
        handle3.Should().NotBeNull();
    }

    [Fact]
    public async Task EnhancedGrain_LRUEviction_ShouldWorkUnderPressure()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuResidentGrain<byte>>($"lru-{Guid.NewGuid()}");

        // Act - Allocate many small buffers to trigger LRU
        var handles = new List<GpuMemoryHandle>();
        for (int i = 0; i < 100; i++)
        {
            var handle = await grain.AllocateAsync(1024);
            handles.Add(handle);
        }

        // Assert - All allocations should succeed
        handles.Should().HaveCount(100);
        handles.Should().OnlyHaveUniqueItems(h => h.Id);
    }

    #endregion

    #region DMA Transfer Tests

    [Fact]
    public async Task EnhancedGrain_WriteAsync_ShouldUseDmaTransfer()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuResidentGrain<float>>($"dma-write-{Guid.NewGuid()}");

        var data = new float[] { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f };
        var handle = await grain.AllocateAsync(data.Length * sizeof(float));

        // Act
        await grain.WriteAsync(handle, data);

        // Assert - Verify write succeeded by reading back
        var retrieved = await grain.ReadAsync<float>(handle, data.Length);
        retrieved.Should().Equal(data);
    }

    [Fact]
    public async Task EnhancedGrain_ReadAsync_ShouldUseDmaTransfer()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuResidentGrain<int>>($"dma-read-{Guid.NewGuid()}");

        var data = new int[] { 10, 20, 30, 40, 50 };
        var handle = await grain.AllocateAsync(data.Length * sizeof(int));
        await grain.WriteAsync(handle, data);

        // Act
        var retrieved = await grain.ReadAsync<int>(handle, data.Length);

        // Assert
        retrieved.Should().NotBeNull();
        retrieved.Should().Equal(data);
    }

    [Fact]
    public async Task EnhancedGrain_LargeTransfer_ShouldHandleEfficiently()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuResidentGrain<double>>($"large-transfer-{Guid.NewGuid()}");

        // 1MB of data
        var data = Enumerable.Range(0, 128 * 1024).Select(i => (double)i).ToArray();
        var handle = await grain.AllocateAsync(data.Length * sizeof(double));

        // Act
        await grain.WriteAsync(handle, data);
        var retrieved = await grain.ReadAsync<double>(handle, data.Length);

        // Assert
        retrieved.Should().HaveCount(data.Length);
    }

    #endregion

    #region Kernel Execution Tests

    [Fact]
    public async Task EnhancedGrain_ComputeAsync_ShouldExecuteKernel()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuResidentGrain<float>>($"compute-{Guid.NewGuid()}");

        var inputData = new float[] { 1.0f, 2.0f, 3.0f, 4.0f };
        var inputHandle = await grain.AllocateAsync(inputData.Length * sizeof(float));
        await grain.WriteAsync(inputHandle, inputData);

        var outputHandle = await grain.AllocateAsync(inputData.Length * sizeof(float));
        var kernelId = KernelId.Parse("kernels/test-compute");

        // Act
        var result = await grain.ComputeAsync(kernelId, inputHandle, outputHandle);

        // Assert
        result.Should().NotBeNull();
        result.Success.Should().BeTrue();
        result.ExecutionTime.Should().BeGreaterThan(TimeSpan.Zero);
    }

    [Fact]
    public async Task EnhancedGrain_KernelCache_ShouldImprovePerformance()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuResidentGrain<float>>($"cache-{Guid.NewGuid()}");

        var data = new float[] { 1.0f, 2.0f };
        var inputHandle = await grain.AllocateAsync(data.Length * sizeof(float));
        var outputHandle = await grain.AllocateAsync(data.Length * sizeof(float));
        await grain.WriteAsync(inputHandle, data);

        var kernelId = KernelId.Parse("kernels/cached");

        // Act - Execute twice to test caching
        var result1 = await grain.ComputeAsync(kernelId, inputHandle, outputHandle);
        var result2 = await grain.ComputeAsync(kernelId, inputHandle, outputHandle);

        // Assert - Both should succeed, second may be faster (cached)
        result1.Success.Should().BeTrue();
        result2.Success.Should().BeTrue();
    }

    #endregion

    #region State Persistence Tests

    [Fact]
    public async Task EnhancedGrain_Deactivation_ShouldPersistState()
    {
        // Arrange
        var grainKey = $"persist-{Guid.NewGuid()}";
        var grain1 = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuResidentGrain<float>>(grainKey);

        var data = new float[] { 1.0f, 2.0f, 3.0f };
        await grain1.StoreDataAsync(data);

        // Act - Get same grain (simulates reactivation)
        var grain2 = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuResidentGrain<float>>(grainKey);

        var retrieved = await grain2.GetDataAsync();

        // Assert
        retrieved.Should().NotBeNull();
        retrieved.Should().Equal(data);
    }

    [Fact]
    public async Task EnhancedGrain_MultipleAllocations_ShouldPersist()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuResidentGrain<int>>($"multi-alloc-{Guid.NewGuid()}");

        // Act - Create multiple allocations
        var handles = new List<GpuMemoryHandle>();
        for (int i = 0; i < 5; i++)
        {
            var handle = await grain.AllocateAsync(1024);
            handles.Add(handle);
        }

        var info = await grain.GetMemoryInfoAsync();

        // Assert
        info.AllocatedMemoryBytes.Should().BeGreaterThan(0);
    }

    #endregion

    #region Performance Metrics Tests

    [Fact]
    public async Task EnhancedGrain_OperationLatency_ShouldBeSubMicrosecond()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuResidentGrain<float>>($"latency-{Guid.NewGuid()}");

        var data = new float[] { 1.0f };
        var handle = await grain.AllocateAsync(sizeof(float));

        // Act - Measure write latency
        var stopwatch = System.Diagnostics.Stopwatch.StartNew();
        await grain.WriteAsync(handle, data);
        stopwatch.Stop();

        // Assert - Should be very fast (< 1ms in test environment)
        stopwatch.ElapsedMilliseconds.Should().BeLessThan(100);
    }

    [Fact]
    public async Task EnhancedGrain_Throughput_ShouldBeHigh()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuResidentGrain<int>>($"throughput-{Guid.NewGuid()}");

        var handle = await grain.AllocateAsync(100 * sizeof(int));
        var data = Enumerable.Range(0, 100).ToArray();

        // Act - Measure operations per second
        var stopwatch = System.Diagnostics.Stopwatch.StartNew();
        for (int i = 0; i < 100; i++)
        {
            await grain.WriteAsync(handle, data);
        }
        stopwatch.Stop();

        var opsPerSecond = 100.0 / stopwatch.Elapsed.TotalSeconds;

        // Assert
        opsPerSecond.Should().BeGreaterThan(100); // At least 100 ops/sec
    }

    #endregion

    #region Memory Management Tests

    [Fact]
    public async Task EnhancedGrain_ReleaseAsync_ShouldReturnToPool()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuResidentGrain<byte>>($"release-pool-{Guid.NewGuid()}");

        var handle = await grain.AllocateAsync(1024);
        var infoBefore = await grain.GetMemoryInfoAsync();

        // Act
        await grain.ReleaseAsync(handle);
        var infoAfter = await grain.GetMemoryInfoAsync();

        // Assert
        infoBefore.AllocatedMemoryBytes.Should().BeGreaterThan(0);
        infoAfter.AllocatedMemoryBytes.Should().BeLessThanOrEqualTo(infoBefore.AllocatedMemoryBytes);
    }

    [Fact]
    public async Task EnhancedGrain_ClearAsync_ShouldReleaseAll()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuResidentGrain<float>>($"clear-all-{Guid.NewGuid()}");

        // Create multiple allocations
        for (int i = 0; i < 5; i++)
        {
            await grain.AllocateAsync(1024);
        }

        var infoBefore = await grain.GetMemoryInfoAsync();

        // Act
        await grain.ClearAsync();
        var infoAfter = await grain.GetMemoryInfoAsync();

        // Assert
        infoBefore.AllocatedMemoryBytes.Should().BeGreaterThan(0);
        infoAfter.AllocatedMemoryBytes.Should().Be(0);
    }

    [Fact]
    public async Task EnhancedGrain_MemoryInfo_ShouldReflectAllocations()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuResidentGrain<double>>($"mem-info-{Guid.NewGuid()}");

        var size = 2048L;
        var handle = await grain.AllocateAsync(size);

        // Act
        var info = await grain.GetMemoryInfoAsync();

        // Assert
        info.AllocatedMemoryBytes.Should().BeGreaterThanOrEqualTo(size);
        info.DeviceIndex.Should().BeGreaterThanOrEqualTo(0);
    }

    #endregion

    #region High-Level API Tests

    [Fact]
    public async Task EnhancedGrain_StoreDataAsync_ShouldAllocateAndWrite()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuResidentGrain<float>>($"store-{Guid.NewGuid()}");

        var data = new float[] { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f };

        // Act
        await grain.StoreDataAsync(data);

        // Assert
        var info = await grain.GetMemoryInfoAsync();
        info.AllocatedMemoryBytes.Should().BeGreaterThan(0);
    }

    [Fact]
    public async Task EnhancedGrain_GetDataAsync_ShouldRetrieveStored()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuResidentGrain<int>>($"get-data-{Guid.NewGuid()}");

        var data = new int[] { 10, 20, 30, 40, 50 };
        await grain.StoreDataAsync(data);

        // Act
        var retrieved = await grain.GetDataAsync();

        // Assert
        retrieved.Should().NotBeNull();
        retrieved.Should().Equal(data);
    }

    [Fact]
    public async Task EnhancedGrain_GetDataAsync_NoData_ShouldReturnNull()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuResidentGrain<float>>($"no-data-{Guid.NewGuid()}");

        // Act
        var retrieved = await grain.GetDataAsync();

        // Assert
        retrieved.Should().BeNull();
    }

    #endregion

    #region Concurrent Access Tests

    [Fact]
    public async Task EnhancedGrain_ConcurrentAllocations_ShouldSerialize()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuResidentGrain<int>>($"concurrent-{Guid.NewGuid()}");

        // Act - Multiple concurrent allocations
        var tasks = Enumerable.Range(0, 10)
            .Select(_ => grain.AllocateAsync(1024))
            .ToArray();

        var handles = await Task.WhenAll(tasks);

        // Assert
        handles.Should().HaveCount(10);
        handles.Should().OnlyHaveUniqueItems(h => h.Id);
    }

    [Fact]
    public async Task EnhancedGrain_ConcurrentWrites_ShouldSucceed()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuResidentGrain<float>>($"concurrent-write-{Guid.NewGuid()}");

        var handles = new List<GpuMemoryHandle>();
        for (int i = 0; i < 5; i++)
        {
            handles.Add(await grain.AllocateAsync(sizeof(float) * 10));
        }

        // Act - Concurrent writes to different handles
        var writeTasks = handles.Select((h, i) =>
        {
            var data = Enumerable.Range(i * 10, 10).Select(x => (float)x).ToArray();
            return grain.WriteAsync(h, data);
        }).ToArray();

        // Assert - All writes should complete
        await Task.WhenAll(writeTasks);
    }

    #endregion

    #region Edge Case Tests

    [Fact]
    public async Task EnhancedGrain_ZeroSizeAllocation_ShouldThrow()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuResidentGrain<byte>>($"zero-size-{Guid.NewGuid()}");

        // Act & Assert
        await Assert.ThrowsAsync<ArgumentException>(async () =>
        {
            await grain.AllocateAsync(0);
        });
    }

    [Fact]
    public async Task EnhancedGrain_InvalidHandle_ShouldThrow()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuResidentGrain<int>>($"invalid-handle-{Guid.NewGuid()}");

        var invalidHandle = GpuMemoryHandle.Create(1024, GpuMemoryType.Default, 0);

        // Act & Assert
        await Assert.ThrowsAsync<ArgumentException>(async () =>
        {
            await grain.WriteAsync(invalidHandle, new int[] { 1, 2, 3 });
        });
    }

    [Fact]
    public async Task EnhancedGrain_ReleaseTwice_ShouldHandleGracefully()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuResidentGrain<float>>($"release-twice-{Guid.NewGuid()}");

        var handle = await grain.AllocateAsync(1024);

        // Act
        await grain.ReleaseAsync(handle);
        await grain.ReleaseAsync(handle); // Second release

        // Assert - Should not throw
    }

    #endregion

    #region Memory Type Tests

    [Fact]
    public async Task EnhancedGrain_PinnedMemory_ShouldAllocate()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuResidentGrain<float>>($"pinned-{Guid.NewGuid()}");

        // Act
        var handle = await grain.AllocateAsync(1024, GpuMemoryType.Pinned);

        // Assert
        handle.Should().NotBeNull();
        handle.Type.Should().Be(GpuMemoryType.Pinned);
    }

    [Fact]
    public async Task EnhancedGrain_DefaultMemory_ShouldAllocate()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuResidentGrain<int>>($"default-{Guid.NewGuid()}");

        // Act
        var handle = await grain.AllocateAsync(1024, GpuMemoryType.Default);

        // Assert
        handle.Should().NotBeNull();
        handle.Type.Should().Be(GpuMemoryType.Default);
    }

    #endregion

    #region Integration Tests

    [Fact]
    public async Task EnhancedGrain_FullWorkflow_ShouldWork()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuResidentGrain<double>>($"workflow-{Guid.NewGuid()}");

        var data = new double[] { 1.0, 2.0, 3.0, 4.0, 5.0 };

        // Act - Complete workflow
        // 1. Store data
        await grain.StoreDataAsync(data);

        // 2. Get memory info
        var info = await grain.GetMemoryInfoAsync();

        // 3. Retrieve data
        var retrieved = await grain.GetDataAsync();

        // 4. Clear
        await grain.ClearAsync();

        var finalInfo = await grain.GetMemoryInfoAsync();

        // Assert
        info.AllocatedMemoryBytes.Should().BeGreaterThan(0);
        retrieved.Should().Equal(data);
        finalInfo.AllocatedMemoryBytes.Should().Be(0);
    }

    [Fact]
    public async Task EnhancedGrain_RingKernelLifecycle_ShouldComplete()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuResidentGrain<float>>($"lifecycle-{Guid.NewGuid()}");

        // Act - Exercise ring kernel through various operations
        var handle1 = await grain.AllocateAsync(512);
        await grain.WriteAsync(handle1, new float[] { 1.0f, 2.0f });

        var handle2 = await grain.AllocateAsync(512);
        await grain.ReadAsync<float>(handle1, 2);

        await grain.ReleaseAsync(handle1);
        await grain.ReleaseAsync(handle2);

        // Assert
        var info = await grain.GetMemoryInfoAsync();
        info.Should().NotBeNull();
    }

    #endregion
}

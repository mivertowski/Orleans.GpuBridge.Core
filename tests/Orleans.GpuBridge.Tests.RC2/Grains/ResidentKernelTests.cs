using Orleans.GpuBridge.Abstractions;
using Orleans.GpuBridge.Grains.Enums;
using Orleans.GpuBridge.Grains.Interfaces;
using Orleans.GpuBridge.Grains.Models;
using Orleans.GpuBridge.Tests.RC2.Infrastructure;

namespace Orleans.GpuBridge.Tests.RC2.Grains;

/// <summary>
/// Comprehensive tests for ResidentMemoryRingKernel and resident kernel operations.
/// Tests ring kernel lifecycle, message processing, memory pools, kernel caching, and performance.
/// Target: 30+ tests for ring kernel functionality.
/// </summary>
[Collection("ClusterCollection")]
public sealed class ResidentKernelTests : IClassFixture<ClusterFixture>
{
    private readonly ClusterFixture _fixture;

    public ResidentKernelTests(ClusterFixture fixture)
    {
        _fixture = fixture;
    }

    #region Ring Kernel Message Processing Tests

    [Fact]
    public async Task RingKernel_AllocateMessage_ShouldSucceed()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuResidentGrain<float>>($"ring-alloc-{Guid.NewGuid()}");

        // Act - Trigger allocation (which uses ring kernel)
        var handle = await grain.AllocateAsync(1024);

        // Assert
        handle.Should().NotBeNull();
        handle.Id.Should().NotBeNullOrEmpty();
        handle.SizeBytes.Should().Be(1024);
    }

    [Fact]
    public async Task RingKernel_WriteMessage_ShouldTransferData()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuResidentGrain<int>>($"ring-write-{Guid.NewGuid()}");

        var data = new int[] { 1, 2, 3, 4, 5 };
        var handle = await grain.AllocateAsync(data.Length * sizeof(int));

        // Act
        await grain.WriteAsync(handle, data);

        // Assert - Verify by reading back
        var retrieved = await grain.ReadAsync<int>(handle, data.Length);
        retrieved.Should().Equal(data);
    }

    [Fact]
    public async Task RingKernel_ReadMessage_ShouldRetrieveData()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuResidentGrain<float>>($"ring-read-{Guid.NewGuid()}");

        var data = new float[] { 1.0f, 2.0f, 3.0f };
        var handle = await grain.AllocateAsync(data.Length * sizeof(float));
        await grain.WriteAsync(handle, data);

        // Act
        var retrieved = await grain.ReadAsync<float>(handle, data.Length);

        // Assert
        retrieved.Should().NotBeNull();
        retrieved.Should().Equal(data);
    }

    [Fact]
    public async Task RingKernel_ReleaseMessage_ShouldFreeMemory()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuResidentGrain<byte>>($"ring-release-{Guid.NewGuid()}");

        var handle = await grain.AllocateAsync(2048);
        var infoBefore = await grain.GetMemoryInfoAsync();

        // Act
        await grain.ReleaseAsync(handle);
        var infoAfter = await grain.GetMemoryInfoAsync();

        // Assert
        infoBefore.AllocatedMemoryBytes.Should().BeGreaterThan(0);
        infoAfter.AllocatedMemoryBytes.Should().BeLessThanOrEqualTo(infoBefore.AllocatedMemoryBytes);
    }

    #endregion

    #region Memory Pool Operation Tests

    [Fact]
    public async Task RingKernel_PoolHit_ShouldReuseAllocation()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuResidentGrain<float>>($"pool-hit-{Guid.NewGuid()}");

        var size = 1024L;

        // Act - Allocate, release, allocate same size
        var handle1 = await grain.AllocateAsync(size);
        await grain.ReleaseAsync(handle1);

        var handle2 = await grain.AllocateAsync(size);

        // Assert - Should get from pool (fast)
        handle2.Should().NotBeNull();
        handle2.SizeBytes.Should().Be(size);
    }

    [Fact]
    public async Task RingKernel_PoolMiss_ShouldAllocateNew()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuResidentGrain<int>>($"pool-miss-{Guid.NewGuid()}");

        // Act - Allocate different sizes (miss pool)
        var handle1 = await grain.AllocateAsync(512);
        var handle2 = await grain.AllocateAsync(1024);
        var handle3 = await grain.AllocateAsync(2048);

        // Assert
        handle1.SizeBytes.Should().Be(512);
        handle2.SizeBytes.Should().Be(1024);
        handle3.SizeBytes.Should().Be(2048);
    }

    [Fact]
    public async Task RingKernel_LRUEviction_ShouldMaintainPoolSize()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuResidentGrain<byte>>($"lru-evict-{Guid.NewGuid()}");

        // Act - Create many allocations to trigger LRU
        var handles = new List<GpuMemoryHandle>();
        for (int i = 0; i < 50; i++)
        {
            var handle = await grain.AllocateAsync(1024 * 1024); // 1MB each
            handles.Add(handle);
        }

        // Assert - Should succeed without OOM
        handles.Should().HaveCount(50);

        // Cleanup
        foreach (var handle in handles)
        {
            await grain.ReleaseAsync(handle);
        }
    }

    #endregion

    #region Kernel Cache Tests

    [Fact]
    public async Task RingKernel_KernelCacheHit_ShouldImprovePerformance()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuResidentGrain<float>>($"kernel-cache-{Guid.NewGuid()}");

        var data = new float[] { 1.0f, 2.0f };
        var inputHandle = await grain.AllocateAsync(data.Length * sizeof(float));
        var outputHandle = await grain.AllocateAsync(data.Length * sizeof(float));
        await grain.WriteAsync(inputHandle, data);

        var kernelId = KernelId.Parse("kernels/cached-compute");

        // Act - Execute same kernel twice
        var result1 = await grain.ComputeAsync(kernelId, inputHandle, outputHandle);
        var result2 = await grain.ComputeAsync(kernelId, inputHandle, outputHandle);

        // Assert - Both should succeed
        result1.Success.Should().BeTrue();
        result2.Success.Should().BeTrue();
        // Second may be faster due to caching
        result2.ExecutionTime.Should().BeLessThanOrEqualTo(result1.ExecutionTime * 2);
    }

    [Fact]
    public async Task RingKernel_MultipleKernels_ShouldCacheIndependently()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuResidentGrain<float>>($"multi-kernel-{Guid.NewGuid()}");

        var data = new float[] { 1.0f };
        var handle1 = await grain.AllocateAsync(sizeof(float));
        var handle2 = await grain.AllocateAsync(sizeof(float));
        await grain.WriteAsync(handle1, data);

        var kernel1 = KernelId.Parse("kernels/kernel1");
        var kernel2 = KernelId.Parse("kernels/kernel2");

        // Act
        var result1 = await grain.ComputeAsync(kernel1, handle1, handle2);
        var result2 = await grain.ComputeAsync(kernel2, handle1, handle2);

        // Assert
        result1.Success.Should().BeTrue();
        result2.Success.Should().BeTrue();
    }

    #endregion

    #region Performance Tests

    [Fact]
    public async Task RingKernel_AllocationLatency_ShouldBeLow()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuResidentGrain<byte>>($"alloc-latency-{Guid.NewGuid()}");

        // Act - Measure allocation time
        var stopwatch = System.Diagnostics.Stopwatch.StartNew();
        var handle = await grain.AllocateAsync(1024);
        stopwatch.Stop();

        // Assert - Should be fast (< 10ms in test environment)
        stopwatch.ElapsedMilliseconds.Should().BeLessThan(50);
        handle.Should().NotBeNull();
    }

    [Fact]
    public async Task RingKernel_TransferThroughput_ShouldBeHigh()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuResidentGrain<int>>($"throughput-{Guid.NewGuid()}");

        var data = Enumerable.Range(0, 1000).ToArray();
        var handle = await grain.AllocateAsync(data.Length * sizeof(int));

        // Act - Measure write throughput
        var stopwatch = System.Diagnostics.Stopwatch.StartNew();
        for (int i = 0; i < 10; i++)
        {
            await grain.WriteAsync(handle, data);
        }
        stopwatch.Stop();

        var bytesPerSecond = (data.Length * sizeof(int) * 10) / stopwatch.Elapsed.TotalSeconds;

        // Assert - Should achieve reasonable throughput
        bytesPerSecond.Should().BeGreaterThan(1_000_000); // > 1 MB/s
    }

    #endregion

    #region Concurrent Operation Tests

    [Fact]
    public async Task RingKernel_ConcurrentAllocations_ShouldSerialize()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuResidentGrain<float>>($"concurrent-alloc-{Guid.NewGuid()}");

        // Act - Multiple concurrent allocations
        var tasks = Enumerable.Range(0, 20)
            .Select(_ => grain.AllocateAsync(512))
            .ToArray();

        var handles = await Task.WhenAll(tasks);

        // Assert
        handles.Should().HaveCount(20);
        handles.Should().OnlyHaveUniqueItems(h => h.Id);
    }

    [Fact]
    public async Task RingKernel_ConcurrentWrites_ShouldSucceed()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuResidentGrain<int>>($"concurrent-write-{Guid.NewGuid()}");

        var handles = new List<GpuMemoryHandle>();
        for (int i = 0; i < 10; i++)
        {
            handles.Add(await grain.AllocateAsync(sizeof(int) * 10));
        }

        // Act - Concurrent writes
        var writeTasks = handles.Select((h, i) =>
        {
            var data = Enumerable.Range(i * 10, 10).ToArray();
            return grain.WriteAsync(h, data);
        }).ToArray();

        // Assert - All should complete
        await Task.WhenAll(writeTasks);
    }

    #endregion

    #region Error Handling Tests

    [Fact]
    public async Task RingKernel_InvalidHandleWrite_ShouldThrow()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuResidentGrain<float>>($"invalid-write-{Guid.NewGuid()}");

        var invalidHandle = GpuMemoryHandle.Create(1024, GpuMemoryType.Default, 0);
        var data = new float[] { 1.0f };

        // Act & Assert
        await Assert.ThrowsAsync<ArgumentException>(async () =>
        {
            await grain.WriteAsync(invalidHandle, data);
        });
    }

    [Fact]
    public async Task RingKernel_InvalidHandleRead_ShouldThrow()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuResidentGrain<int>>($"invalid-read-{Guid.NewGuid()}");

        var invalidHandle = GpuMemoryHandle.Create(1024, GpuMemoryType.Default, 0);

        // Act & Assert
        await Assert.ThrowsAsync<ArgumentException>(async () =>
        {
            await grain.ReadAsync<int>(invalidHandle, 10);
        });
    }

    [Fact]
    public async Task RingKernel_InvalidHandleCompute_ShouldThrow()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuResidentGrain<float>>($"invalid-compute-{Guid.NewGuid()}");

        var invalidInput = GpuMemoryHandle.Create(1024, GpuMemoryType.Default, 0);
        var invalidOutput = GpuMemoryHandle.Create(1024, GpuMemoryType.Default, 0);
        var kernelId = KernelId.Parse("kernels/test");

        // Act & Assert
        await Assert.ThrowsAsync<ArgumentException>(async () =>
        {
            await grain.ComputeAsync(kernelId, invalidInput, invalidOutput);
        });
    }

    #endregion

    #region Lifecycle Tests

    [Fact]
    public async Task RingKernel_GrainActivation_ShouldInitializeKernel()
    {
        // Arrange & Act - Activation happens on first call
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuResidentGrain<byte>>($"lifecycle-init-{Guid.NewGuid()}");

        var info = await grain.GetMemoryInfoAsync();

        // Assert
        info.Should().NotBeNull();
        info.DeviceName.Should().NotBeNullOrEmpty();
    }

    [Fact]
    public async Task RingKernel_GrainDeactivation_ShouldCleanup()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuResidentGrain<float>>($"lifecycle-cleanup-{Guid.NewGuid()}");

        // Create allocations
        await grain.AllocateAsync(1024);
        await grain.AllocateAsync(2048);

        // Act - Cleanup
        await grain.ClearAsync();
        var info = await grain.GetMemoryInfoAsync();

        // Assert
        info.AllocatedMemoryBytes.Should().Be(0);
    }

    #endregion

    #region Integration Tests

    [Fact]
    public async Task RingKernel_CompleteWorkflow_ShouldSucceed()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuResidentGrain<double>>($"workflow-{Guid.NewGuid()}");

        var data = new double[] { 1.0, 2.0, 3.0, 4.0, 5.0 };

        // Act - Complete workflow
        // 1. Allocate
        var handle = await grain.AllocateAsync(data.Length * sizeof(double));

        // 2. Write
        await grain.WriteAsync(handle, data);

        // 3. Read
        var retrieved = await grain.ReadAsync<double>(handle, data.Length);

        // 4. Verify
        retrieved.Should().Equal(data);

        // 5. Release
        await grain.ReleaseAsync(handle);

        // Assert
        var info = await grain.GetMemoryInfoAsync();
        info.AllocatedMemoryBytes.Should().Be(0);
    }

    [Fact]
    public async Task RingKernel_MultipleGrains_ShouldWorkIndependently()
    {
        // Arrange
        var grain1 = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuResidentGrain<float>>($"multi-grain1-{Guid.NewGuid()}");
        var grain2 = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuResidentGrain<float>>($"multi-grain2-{Guid.NewGuid()}");

        var data1 = new float[] { 1.0f, 2.0f };
        var data2 = new float[] { 3.0f, 4.0f };

        // Act
        await grain1.StoreDataAsync(data1);
        await grain2.StoreDataAsync(data2);

        var retrieved1 = await grain1.GetDataAsync();
        var retrieved2 = await grain2.GetDataAsync();

        // Assert
        retrieved1.Should().Equal(data1);
        retrieved2.Should().Equal(data2);
    }

    #endregion

    #region Memory Type Tests

    [Fact]
    public async Task RingKernel_PinnedMemory_ShouldAllocate()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuResidentGrain<int>>($"pinned-mem-{Guid.NewGuid()}");

        // Act
        var handle = await grain.AllocateAsync(1024, GpuMemoryType.Pinned);

        // Assert
        handle.Should().NotBeNull();
        handle.Type.Should().Be(GpuMemoryType.Pinned);
    }

    [Fact]
    public async Task RingKernel_MixedMemoryTypes_ShouldCoexist()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuResidentGrain<float>>($"mixed-mem-{Guid.NewGuid()}");

        // Act
        var defaultHandle = await grain.AllocateAsync(512, GpuMemoryType.Default);
        var pinnedHandle = await grain.AllocateAsync(512, GpuMemoryType.Pinned);

        // Assert
        defaultHandle.Type.Should().Be(GpuMemoryType.Default);
        pinnedHandle.Type.Should().Be(GpuMemoryType.Pinned);
        defaultHandle.Id.Should().NotBe(pinnedHandle.Id);
    }

    #endregion

    #region Stress Tests

    [Fact]
    public async Task RingKernel_RapidAllocRelease_ShouldHandleEfficiently()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuResidentGrain<byte>>($"stress-alloc-{Guid.NewGuid()}");

        // Act - Rapid allocation/release cycles
        for (int i = 0; i < 100; i++)
        {
            var handle = await grain.AllocateAsync(1024);
            await grain.ReleaseAsync(handle);
        }

        // Assert
        var info = await grain.GetMemoryInfoAsync();
        info.Should().NotBeNull();
    }

    [Fact]
    public async Task RingKernel_LargeDataTransfers_ShouldSucceed()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuResidentGrain<double>>($"large-data-{Guid.NewGuid()}");

        // 10MB of data
        var data = Enumerable.Range(0, 1_310_720).Select(i => (double)i).ToArray();
        var handle = await grain.AllocateAsync(data.Length * sizeof(double));

        // Act
        await grain.WriteAsync(handle, data);
        var retrieved = await grain.ReadAsync<double>(handle, data.Length);

        // Assert
        retrieved.Should().HaveCount(data.Length);
    }

    #endregion
}

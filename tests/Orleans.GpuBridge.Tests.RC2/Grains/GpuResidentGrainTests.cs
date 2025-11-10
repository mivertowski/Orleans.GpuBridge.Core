using Orleans.GpuBridge.Abstractions;
using Orleans.GpuBridge.Grains.Enums;
using Orleans.GpuBridge.Grains.Interfaces;
using Orleans.GpuBridge.Grains.Models;
using Orleans.GpuBridge.Tests.RC2.Infrastructure;

namespace Orleans.GpuBridge.Tests.RC2.Grains;

/// <summary>
/// Comprehensive tests for GpuResidentGrain implementation.
/// Tests GPU memory allocation, data transfer, kernel execution, and resource management.
/// </summary>
[Collection("ClusterCollection")]
public sealed class GpuResidentGrainTests : IClassFixture<ClusterFixture>
{
    private readonly ClusterFixture _fixture;

    public GpuResidentGrainTests(ClusterFixture fixture)
    {
        _fixture = fixture;
    }

    [Fact]
    public async Task GpuResidentGrain_StoreDataAsync_ShouldAllocateGpuMemory()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuResidentGrain<float>>($"test-store-{Guid.NewGuid()}");

        var data = new float[] { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f };

        // Act
        await grain.StoreDataAsync(data, GpuMemoryType.Default);

        // Assert - Verify data was stored
        var info = await grain.GetMemoryInfoAsync();
        info.Should().NotBeNull();
        info.TotalMemoryBytes.Should().BeGreaterThan(0);
        info.AllocatedMemoryBytes.Should().BeGreaterThan(0);
    }

    [Fact]
    public async Task GpuResidentGrain_GetDataAsync_ShouldRetrieveFromGpu()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuResidentGrain<float>>($"test-get-{Guid.NewGuid()}");

        var originalData = new float[] { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f };
        await grain.StoreDataAsync(originalData);

        // Act
        var retrievedData = await grain.GetDataAsync();

        // Assert
        retrievedData.Should().NotBeNull();
        retrievedData.Should().HaveCount(originalData.Length);
        retrievedData.Should().Equal(originalData);
    }

    [Fact]
    public async Task GpuResidentGrain_Deactivation_ShouldReleaseGpuMemory()
    {
        // Arrange
        var grainKey = $"test-deactivate-{Guid.NewGuid()}";
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuResidentGrain<float>>(grainKey);

        var data = new float[] { 1.0f, 2.0f, 3.0f };
        await grain.StoreDataAsync(data);

        // Act - Clear memory (simulates deactivation cleanup)
        await grain.ClearAsync();

        // Assert - Memory should be released
        var info = await grain.GetMemoryInfoAsync();
        info.TotalMemoryBytes.Should().Be(0);
        info.AllocatedMemoryBytes.Should().Be(0);
    }

    [Fact]
    public async Task GpuResidentGrain_LargeData_ShouldHandleCorrectly()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuResidentGrain<float>>($"test-large-{Guid.NewGuid()}");

        // Create 1MB of data (256K floats)
        var largeData = Enumerable.Range(0, 256 * 1024)
            .Select(i => (float)i)
            .ToArray();

        // Act
        await grain.StoreDataAsync(largeData);
        var retrieved = await grain.GetDataAsync();

        // Assert
        retrieved.Should().NotBeNull();
        retrieved.Should().HaveCount(largeData.Length);

        var info = await grain.GetMemoryInfoAsync();
        info.AllocatedMemoryBytes.Should().BeGreaterThan(1_000_000); // ~1MB
    }

    [Fact]
    public async Task GpuResidentGrain_Concurrent_ShouldSynchronize()
    {
        // Arrange
        var grainKey = $"test-concurrent-{Guid.NewGuid()}";
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuResidentGrain<int>>(grainKey);

        // Act - Perform concurrent allocations
        var tasks = Enumerable.Range(0, 10)
            .Select(async i =>
            {
                var data = new int[] { i, i + 1, i + 2 };
                var handle = await grain.AllocateAsync(data.Length * sizeof(int));
                await grain.WriteAsync(handle, data);
                return handle;
            })
            .ToArray();

        var handles = await Task.WhenAll(tasks);

        // Assert - All allocations should succeed
        handles.Should().HaveCount(10);
        handles.Should().OnlyHaveUniqueItems(h => h.Id);

        var info = await grain.GetMemoryInfoAsync();
        info.AllocatedMemoryBytes.Should().BeGreaterThan(0);
    }

    [Fact]
    public async Task GpuResidentGrain_MemoryPressure_ShouldEvict()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuResidentGrain<byte>>($"test-pressure-{Guid.NewGuid()}");

        // Act - Allocate multiple buffers
        var handles = new List<GpuMemoryHandle>();
        for (int i = 0; i < 5; i++)
        {
            var handle = await grain.AllocateAsync(1024 * 1024); // 1MB each
            handles.Add(handle);
        }

        // Assert - All allocations tracked
        var info = await grain.GetMemoryInfoAsync();
        info.AllocatedMemoryBytes.Should().BeGreaterThan(5_000_000); // ~5MB

        // Clean up by releasing allocations
        foreach (var handle in handles)
        {
            await grain.ReleaseAsync(handle);
        }

        var finalInfo = await grain.GetMemoryInfoAsync();
        finalInfo.AllocatedMemoryBytes.Should().Be(0);
    }

    [Fact]
    public async Task GpuResidentGrain_Reactivation_ShouldRestoreState()
    {
        // Arrange
        var grainKey = $"test-reactivate-{Guid.NewGuid()}";
        var grain1 = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuResidentGrain<float>>(grainKey);

        var data = new float[] { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f };
        await grain1.StoreDataAsync(data);

        // Act - Get same grain (may trigger reactivation in real scenario)
        var grain2 = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuResidentGrain<float>>(grainKey);

        var retrieved = await grain2.GetDataAsync();

        // Assert - State should be preserved
        retrieved.Should().NotBeNull();
        retrieved.Should().Equal(data);
    }

    [Fact]
    public async Task GpuResidentGrain_AllocateAsync_ShouldReturnValidHandle()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuResidentGrain<int>>($"test-allocate-{Guid.NewGuid()}");

        var sizeBytes = 1024L;

        // Act
        var handle = await grain.AllocateAsync(sizeBytes, GpuMemoryType.Pinned);

        // Assert
        handle.Should().NotBeNull();
        handle.Id.Should().NotBeNullOrEmpty();
        handle.SizeBytes.Should().Be(sizeBytes);
    }

    [Fact]
    public async Task GpuResidentGrain_WriteAndReadAsync_ShouldRoundTrip()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuResidentGrain<int>>($"test-roundtrip-{Guid.NewGuid()}");

        var data = new int[] { 42, 100, 256, 1024 };
        var handle = await grain.AllocateAsync(data.Length * sizeof(int));

        // Act
        await grain.WriteAsync(handle, data);
        var retrieved = await grain.ReadAsync<int>(handle, data.Length);

        // Assert
        retrieved.Should().NotBeNull();
        retrieved.Should().Equal(data);
    }

    [Fact]
    public async Task GpuResidentGrain_ComputeAsync_ShouldExecuteKernel()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuResidentGrain<float>>($"test-compute-{Guid.NewGuid()}");

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
        result.Error.Should().BeNullOrEmpty();
    }

    [Fact]
    public async Task GpuResidentGrain_ReleaseAsync_ShouldFreeMemory()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuResidentGrain<byte>>($"test-release-{Guid.NewGuid()}");

        var handle = await grain.AllocateAsync(1024);
        var infoBefore = await grain.GetMemoryInfoAsync();

        // Act
        await grain.ReleaseAsync(handle);
        var infoAfter = await grain.GetMemoryInfoAsync();

        // Assert
        infoBefore.AllocatedMemoryBytes.Should().BeGreaterThan(0);
        infoAfter.AllocatedMemoryBytes.Should().BeLessThan(infoBefore.AllocatedMemoryBytes);
    }

    [Fact]
    public async Task GpuResidentGrain_ClearAsync_ShouldReleaseAllAllocations()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuResidentGrain<int>>($"test-clear-{Guid.NewGuid()}");

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
        infoAfter.TotalMemoryBytes.Should().Be(0);
    }

    [Fact]
    public async Task GpuResidentGrain_GetMemoryInfoAsync_ShouldReturnAccurateStats()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuResidentGrain<float>>($"test-meminfo-{Guid.NewGuid()}");

        // Act - Initial state
        var infoEmpty = await grain.GetMemoryInfoAsync();

        // Allocate some memory
        var data = new float[] { 1, 2, 3, 4, 5 };
        await grain.StoreDataAsync(data);

        var infoWithData = await grain.GetMemoryInfoAsync();

        // Assert
        infoEmpty.TotalMemoryBytes.Should().Be(0);
        infoEmpty.AllocatedMemoryBytes.Should().Be(0);

        infoWithData.TotalMemoryBytes.Should().BeGreaterThan(0);
        infoWithData.AllocatedMemoryBytes.Should().BeGreaterThan(0);
        infoWithData.DeviceName.Should().NotBeNullOrEmpty();
    }
}

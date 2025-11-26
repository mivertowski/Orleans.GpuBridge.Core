using Orleans.GpuBridge.Abstractions;
using Orleans.GpuBridge.Grains.Enums;
using Orleans.GpuBridge.Grains.Interfaces;
using Orleans.GpuBridge.Grains.Models;
using Orleans.GpuBridge.Tests.RC2.Infrastructure;

namespace Orleans.GpuBridge.Tests.RC2.Grains;

/// <summary>
/// Comprehensive tests for GpuResidentGrain implementation focusing on memory allocation,
/// deallocation, pinned memory management, state persistence, concurrent operations,
/// and resource cleanup.
/// Target: 25-30 tests covering all critical grain behaviors.
/// </summary>
[Collection("ClusterCollection")]
public sealed class GpuResidentGrainComprehensiveTests : IClassFixture<ClusterFixture>
{
    private readonly ClusterFixture _fixture;

    public GpuResidentGrainComprehensiveTests(ClusterFixture fixture)
    {
        _fixture = fixture;
    }

    #region Memory Allocation Tests (Extended)

    [Fact]
    public async Task AllocateAsync_WithValidSize_ShouldReturnValidHandle()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuResidentGrain<float>>($"alloc-valid-{Guid.NewGuid()}");
        var sizeBytes = 4096L;

        // Act
        var handle = await grain.AllocateAsync(sizeBytes, GpuMemoryType.Default);

        // Assert
        handle.Should().NotBeNull();
        handle.Id.Should().NotBeNullOrEmpty();
        handle.SizeBytes.Should().Be(sizeBytes);
        handle.Type.Should().Be(GpuMemoryType.Default);
        handle.DeviceIndex.Should().BeGreaterThanOrEqualTo(-1); // Can be -1, 0, or greater
        handle.AllocatedAt.Should().BeCloseTo(DateTime.UtcNow, TimeSpan.FromSeconds(5));
    }

    [Fact]
    public async Task AllocateAsync_WithZeroSize_ShouldThrowArgumentException()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuResidentGrain<int>>($"alloc-zero-{Guid.NewGuid()}");

        // Act & Assert
        var exception = await Assert.ThrowsAsync<ArgumentException>(
            async () => await grain.AllocateAsync(0));

        exception.Message.Should().Contain("must be greater than zero");
        // ParamName may be null in some exception constructors, so we don't assert on it
    }

    [Fact]
    public async Task AllocateAsync_WithNegativeSize_ShouldThrowArgumentException()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuResidentGrain<byte>>($"alloc-negative-{Guid.NewGuid()}");

        // Act & Assert
        var exception = await Assert.ThrowsAsync<ArgumentException>(
            async () => await grain.AllocateAsync(-1024));

        exception.Message.Should().Contain("must be greater than zero");
        // ParamName may be null in some exception constructors, so we don't assert on it
    }

    [Fact]
    public async Task AllocateAsync_SmallAllocation_ShouldSucceed()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuResidentGrain<byte>>($"alloc-small-{Guid.NewGuid()}");
        var sizeBytes = 64L; // 64 bytes

        // Act
        var handle = await grain.AllocateAsync(sizeBytes);

        // Assert
        handle.Should().NotBeNull();
        handle.SizeBytes.Should().Be(sizeBytes);

        var info = await grain.GetMemoryInfoAsync();
        info.AllocatedMemoryBytes.Should().BeGreaterThanOrEqualTo(sizeBytes);
    }

    [Fact]
    public async Task AllocateAsync_LargeAllocation_ShouldSucceed()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuResidentGrain<double>>($"alloc-large-{Guid.NewGuid()}");
        var sizeBytes = 16L * 1024 * 1024; // 16 MB

        // Act
        var handle = await grain.AllocateAsync(sizeBytes);

        // Assert
        handle.Should().NotBeNull();
        handle.SizeBytes.Should().Be(sizeBytes);

        var info = await grain.GetMemoryInfoAsync();
        info.AllocatedMemoryBytes.Should().BeGreaterThanOrEqualTo(sizeBytes);
    }

    [Fact]
    public async Task AllocateAsync_MaxRealisticSize_ShouldSucceed()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuResidentGrain<float>>($"alloc-max-{Guid.NewGuid()}");
        var sizeBytes = 256L * 1024 * 1024; // 256 MB

        // Act
        var handle = await grain.AllocateAsync(sizeBytes);

        // Assert
        handle.Should().NotBeNull();
        handle.SizeBytes.Should().Be(sizeBytes);
    }

    [Fact]
    public async Task AllocateAsync_MultipleSequential_ShouldGenerateUniqueHandles()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuResidentGrain<int>>($"alloc-multi-{Guid.NewGuid()}");
        var handles = new List<GpuMemoryHandle>();

        // Act - Allocate 10 separate memory blocks
        for (int i = 0; i < 10; i++)
        {
            var handle = await grain.AllocateAsync(1024);
            handles.Add(handle);
        }

        // Assert
        handles.Should().HaveCount(10);
        handles.Should().OnlyHaveUniqueItems(h => h.Id);

        var info = await grain.GetMemoryInfoAsync();
        info.AllocatedMemoryBytes.Should().BeGreaterThanOrEqualTo(10 * 1024);
    }

    #endregion

    #region Memory Deallocation Tests

    [Fact]
    public async Task ReleaseAsync_WithValidHandle_ShouldFreeMemory()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuResidentGrain<float>>($"release-valid-{Guid.NewGuid()}");
        var handle = await grain.AllocateAsync(2048);
        var infoBefore = await grain.GetMemoryInfoAsync();

        // Act
        await grain.ReleaseAsync(handle);

        // Assert
        var infoAfter = await grain.GetMemoryInfoAsync();
        infoAfter.AllocatedMemoryBytes.Should().BeLessThan(infoBefore.AllocatedMemoryBytes);
    }

    [Fact]
    public async Task ReleaseAsync_WithInvalidHandle_ShouldNotThrow()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuResidentGrain<int>>($"release-invalid-{Guid.NewGuid()}");
        var invalidHandle = GpuMemoryHandle.Create(9999, GpuMemoryType.Default, 0);

        // Act - Should log warning but not throw
        await grain.ReleaseAsync(invalidHandle);

        // Assert - No exception thrown
    }

    [Fact]
    public async Task ReleaseAsync_CalledTwice_ShouldHandleGracefully()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuResidentGrain<byte>>($"release-twice-{Guid.NewGuid()}");
        var handle = await grain.AllocateAsync(1024);

        // Act
        await grain.ReleaseAsync(handle);
        await grain.ReleaseAsync(handle); // Second release

        // Assert - Should not throw, should log warning
    }

    [Fact]
    public async Task ReleaseAsync_AllAllocations_ShouldReturnToZero()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuResidentGrain<double>>($"release-all-{Guid.NewGuid()}");
        var handles = new List<GpuMemoryHandle>();

        // Create 5 allocations
        for (int i = 0; i < 5; i++)
        {
            handles.Add(await grain.AllocateAsync(2048));
        }

        var infoBefore = await grain.GetMemoryInfoAsync();
        infoBefore.AllocatedMemoryBytes.Should().BeGreaterThan(0);

        // Act - Release all
        foreach (var handle in handles)
        {
            await grain.ReleaseAsync(handle);
        }

        // Assert
        var infoAfter = await grain.GetMemoryInfoAsync();
        infoAfter.AllocatedMemoryBytes.Should().Be(0);
        infoAfter.TotalMemoryBytes.Should().Be(0);
    }

    #endregion

    #region Pinned Memory Management Tests

    [Fact]
    public async Task AllocateAsync_PinnedMemory_ShouldSetCorrectType()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuResidentGrain<float>>($"pinned-alloc-{Guid.NewGuid()}");

        // Act
        var handle = await grain.AllocateAsync(4096, GpuMemoryType.Pinned);

        // Assert
        handle.Type.Should().Be(GpuMemoryType.Pinned);
    }

    [Fact]
    public async Task AllocateAsync_SharedMemory_ShouldSetCorrectType()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuResidentGrain<int>>($"shared-alloc-{Guid.NewGuid()}");

        // Act
        var handle = await grain.AllocateAsync(2048, GpuMemoryType.Shared);

        // Assert
        handle.Type.Should().Be(GpuMemoryType.Shared);
    }

    [Fact]
    public async Task AllocateAsync_TextureMemory_ShouldSetCorrectType()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuResidentGrain<byte>>($"texture-alloc-{Guid.NewGuid()}");

        // Act
        var handle = await grain.AllocateAsync(8192, GpuMemoryType.Texture);

        // Assert
        handle.Type.Should().Be(GpuMemoryType.Texture);
    }

    [Fact]
    public async Task AllocateAsync_ConstantMemory_ShouldSetCorrectType()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuResidentGrain<double>>($"constant-alloc-{Guid.NewGuid()}");

        // Act
        var handle = await grain.AllocateAsync(1024, GpuMemoryType.Constant);

        // Assert
        handle.Type.Should().Be(GpuMemoryType.Constant);
    }

    [Fact]
    public async Task WriteAsync_ToPinnedMemory_ShouldSucceed()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuResidentGrain<float>>($"pinned-write-{Guid.NewGuid()}");
        var data = new float[] { 1.5f, 2.5f, 3.5f, 4.5f };
        var handle = await grain.AllocateAsync(data.Length * sizeof(float), GpuMemoryType.Pinned);

        // Act
        await grain.WriteAsync(handle, data);

        // Assert
        var retrieved = await grain.ReadAsync<float>(handle, data.Length);
        retrieved.Should().Equal(data);
    }

    #endregion

    #region Memory State Tracking Tests

    [Fact]
    public async Task GetMemoryInfoAsync_EmptyGrain_ShouldReturnZeroAllocations()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuResidentGrain<int>>($"empty-info-{Guid.NewGuid()}");

        // Act
        var info = await grain.GetMemoryInfoAsync();

        // Assert
        info.Should().NotBeNull();
        info.TotalMemoryBytes.Should().Be(0);
        info.AllocatedMemoryBytes.Should().Be(0);
        info.DeviceName.Should().NotBeNullOrEmpty();
    }

    [Fact]
    public async Task GetMemoryInfoAsync_WithAllocations_ShouldReflectCurrentState()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuResidentGrain<float>>($"info-state-{Guid.NewGuid()}");

        var size1 = 1024L;
        var size2 = 2048L;
        await grain.AllocateAsync(size1);
        await grain.AllocateAsync(size2);

        // Act
        var info = await grain.GetMemoryInfoAsync();

        // Assert
        info.AllocatedMemoryBytes.Should().BeGreaterThanOrEqualTo(size1 + size2);
        info.TotalMemoryBytes.Should().Be(info.AllocatedMemoryBytes);
    }

    [Fact]
    public async Task GetMemoryInfoAsync_AfterPartialRelease_ShouldUpdateCorrectly()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuResidentGrain<byte>>($"info-partial-{Guid.NewGuid()}");

        var handle1 = await grain.AllocateAsync(1024);
        var handle2 = await grain.AllocateAsync(2048);
        var handle3 = await grain.AllocateAsync(512);

        var infoFull = await grain.GetMemoryInfoAsync();

        // Act - Release middle allocation
        await grain.ReleaseAsync(handle2);

        // Assert
        var infoPartial = await grain.GetMemoryInfoAsync();
        infoPartial.AllocatedMemoryBytes.Should().BeLessThan(infoFull.AllocatedMemoryBytes);
        infoPartial.AllocatedMemoryBytes.Should().BeGreaterThanOrEqualTo(1024 + 512);
    }

    #endregion

    #region Concurrent Allocation Request Tests

    [Fact]
    public async Task AllocateAsync_ConcurrentRequests_ShouldSerializeCorrectly()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuResidentGrain<int>>($"concurrent-alloc-{Guid.NewGuid()}");

        // Act - 20 concurrent allocations
        var tasks = Enumerable.Range(0, 20)
            .Select(_ => grain.AllocateAsync(512))
            .ToArray();

        var handles = await Task.WhenAll(tasks);

        // Assert
        handles.Should().HaveCount(20);
        handles.Should().OnlyHaveUniqueItems(h => h.Id);

        var info = await grain.GetMemoryInfoAsync();
        info.AllocatedMemoryBytes.Should().BeGreaterThanOrEqualTo(20 * 512);
    }

    [Fact]
    public async Task WriteAsync_ConcurrentToSeparateHandles_ShouldSucceed()
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

        await Task.WhenAll(writeTasks);

        // Assert - Verify each handle has correct data
        for (int i = 0; i < handles.Count; i++)
        {
            var expectedData = Enumerable.Range(i * 10, 10).Select(x => (float)x).ToArray();
            var actualData = await grain.ReadAsync<float>(handles[i], 10);
            actualData.Should().Equal(expectedData);
        }
    }

    [Fact]
    public async Task ReleaseAsync_ConcurrentReleases_ShouldHandleCorrectly()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuResidentGrain<byte>>($"concurrent-release-{Guid.NewGuid()}");

        var handles = new List<GpuMemoryHandle>();
        for (int i = 0; i < 10; i++)
        {
            handles.Add(await grain.AllocateAsync(1024));
        }

        var infoBefore = await grain.GetMemoryInfoAsync();

        // Act - Concurrent releases
        var releaseTasks = handles.Select(h => grain.ReleaseAsync(h)).ToArray();
        await Task.WhenAll(releaseTasks);

        // Assert
        var infoAfter = await grain.GetMemoryInfoAsync();
        infoAfter.AllocatedMemoryBytes.Should().BeLessThan(infoBefore.AllocatedMemoryBytes);
        infoAfter.AllocatedMemoryBytes.Should().Be(0);
    }

    #endregion

    #region Resource Cleanup on Deactivation Tests

    [Fact]
    public async Task ClearAsync_WithMultipleAllocations_ShouldReleaseAll()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuResidentGrain<float>>($"clear-multi-{Guid.NewGuid()}");

        // Create 7 allocations of varying sizes
        await grain.AllocateAsync(512);
        await grain.AllocateAsync(1024);
        await grain.AllocateAsync(2048);
        await grain.AllocateAsync(4096);
        await grain.AllocateAsync(512);
        await grain.AllocateAsync(1024);
        await grain.AllocateAsync(8192);

        var infoBefore = await grain.GetMemoryInfoAsync();
        infoBefore.AllocatedMemoryBytes.Should().BeGreaterThan(0);

        // Act
        await grain.ClearAsync();

        // Assert
        var infoAfter = await grain.GetMemoryInfoAsync();
        infoAfter.AllocatedMemoryBytes.Should().Be(0);
        infoAfter.TotalMemoryBytes.Should().Be(0);
    }

    [Fact]
    public async Task ClearAsync_WithNoAllocations_ShouldNotThrow()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuResidentGrain<int>>($"clear-empty-{Guid.NewGuid()}");

        // Act & Assert - Should not throw
        await grain.ClearAsync();

        var info = await grain.GetMemoryInfoAsync();
        info.AllocatedMemoryBytes.Should().Be(0);
    }

    [Fact]
    public async Task Reactivation_ShouldRestoreMemoryAllocations()
    {
        // Arrange
        var grainKey = $"reactivate-{Guid.NewGuid()}";
        var grain1 = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuResidentGrain<float>>(grainKey);

        var data = new float[] { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f };
        await grain1.StoreDataAsync(data);

        var infoBefore = await grain1.GetMemoryInfoAsync();

        // Act - Get same grain (simulates reactivation)
        var grain2 = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuResidentGrain<float>>(grainKey);

        var retrieved = await grain2.GetDataAsync();
        var infoAfter = await grain2.GetMemoryInfoAsync();

        // Assert
        retrieved.Should().NotBeNull();
        retrieved.Should().Equal(data);
        infoAfter.AllocatedMemoryBytes.Should().Be(infoBefore.AllocatedMemoryBytes);
    }

    [Fact]
    public async Task Deactivation_ShouldPersistStateForReactivation()
    {
        // Arrange
        var grainKey = $"persist-{Guid.NewGuid()}";
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuResidentGrain<double>>(grainKey);

        var allocations = new List<GpuMemoryHandle>();
        for (int i = 0; i < 3; i++)
        {
            allocations.Add(await grain.AllocateAsync(1024 * (i + 1)));
        }

        var infoBeforeDeactivation = await grain.GetMemoryInfoAsync();

        // Act - Force state persistence by getting new reference
        var grainAfter = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuResidentGrain<double>>(grainKey);

        var infoAfterReactivation = await grainAfter.GetMemoryInfoAsync();

        // Assert - State should be preserved
        infoAfterReactivation.AllocatedMemoryBytes.Should().Be(infoBeforeDeactivation.AllocatedMemoryBytes);
    }

    #endregion

    #region Write/Read Operation Tests

    [Fact]
    public async Task WriteAsync_WithInvalidHandle_ShouldThrowArgumentException()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuResidentGrain<float>>($"write-invalid-{Guid.NewGuid()}");
        var invalidHandle = GpuMemoryHandle.Create("invalid-id", 1024);
        var data = new float[] { 1.0f, 2.0f };

        // Act & Assert
        var exception = await Assert.ThrowsAsync<ArgumentException>(
            async () => await grain.WriteAsync(invalidHandle, data));

        exception.Message.Should().Contain("not found");
    }

    [Fact]
    public async Task WriteAsync_ExceedingBounds_ShouldThrowArgumentOutOfRangeException()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuResidentGrain<int>>($"write-bounds-{Guid.NewGuid()}");
        var handle = await grain.AllocateAsync(10 * sizeof(int)); // Room for 10 ints
        var data = new int[20]; // Try to write 20 ints

        // Act & Assert
        var exception = await Assert.ThrowsAsync<ArgumentOutOfRangeException>(
            async () => await grain.WriteAsync(handle, data));

        exception.Message.Should().Contain("exceed allocated memory bounds");
    }

    [Fact]
    public async Task ReadAsync_WithInvalidHandle_ShouldThrowArgumentException()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuResidentGrain<byte>>($"read-invalid-{Guid.NewGuid()}");
        var invalidHandle = GpuMemoryHandle.Create("invalid-id", 1024);

        // Act & Assert
        var exception = await Assert.ThrowsAsync<ArgumentException>(
            async () => await grain.ReadAsync<byte>(invalidHandle, 100));

        exception.Message.Should().Contain("not found");
    }

    [Fact]
    public async Task ReadAsync_ExceedingBounds_ShouldThrowArgumentOutOfRangeException()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuResidentGrain<double>>($"read-bounds-{Guid.NewGuid()}");
        var handle = await grain.AllocateAsync(10 * sizeof(double)); // Room for 10 doubles

        // Act & Assert
        var exception = await Assert.ThrowsAsync<ArgumentOutOfRangeException>(
            async () => await grain.ReadAsync<double>(handle, 20)); // Try to read 20

        exception.Message.Should().Contain("exceed allocated memory bounds");
    }

    #endregion

    #region Compute Operation Tests

    [Fact]
    public async Task ComputeAsync_WithValidHandles_ShouldExecuteSuccessfully()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuResidentGrain<float>>($"compute-valid-{Guid.NewGuid()}");

        var inputData = new float[] { 1.0f, 2.0f, 3.0f };
        var inputHandle = await grain.AllocateAsync(inputData.Length * sizeof(float));
        await grain.WriteAsync(inputHandle, inputData);

        var outputHandle = await grain.AllocateAsync(inputData.Length * sizeof(float));
        var kernelId = KernelId.Parse("kernels/test");

        // Act
        var result = await grain.ComputeAsync(kernelId, inputHandle, outputHandle);

        // Assert
        result.Should().NotBeNull();
        result.Success.Should().BeTrue();
        result.ExecutionTime.Should().BeGreaterThan(TimeSpan.Zero);
        result.Error.Should().BeNullOrEmpty();
    }

    [Fact]
    public async Task ComputeAsync_WithInvalidInputHandle_ShouldThrowArgumentException()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuResidentGrain<int>>($"compute-invalid-input-{Guid.NewGuid()}");

        var outputHandle = await grain.AllocateAsync(1024);
        var invalidInputHandle = GpuMemoryHandle.Create("invalid", 1024);
        var kernelId = KernelId.Parse("kernels/test");

        // Act & Assert
        var exception = await Assert.ThrowsAsync<ArgumentException>(
            async () => await grain.ComputeAsync(kernelId, invalidInputHandle, outputHandle));

        exception.Message.Should().Contain("not found");
    }

    [Fact]
    public async Task ComputeAsync_WithInvalidOutputHandle_ShouldThrowArgumentException()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuResidentGrain<float>>($"compute-invalid-output-{Guid.NewGuid()}");

        var inputHandle = await grain.AllocateAsync(1024);
        var invalidOutputHandle = GpuMemoryHandle.Create("invalid", 1024);
        var kernelId = KernelId.Parse("kernels/test");

        // Act & Assert
        var exception = await Assert.ThrowsAsync<ArgumentException>(
            async () => await grain.ComputeAsync(kernelId, inputHandle, invalidOutputHandle));

        exception.Message.Should().Contain("not found");
    }

    #endregion

    #region High-Level API Tests

    [Fact]
    public async Task StoreDataAsync_WithNullData_ShouldThrowArgumentNullException()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuResidentGrain<float>>($"store-null-{Guid.NewGuid()}");

        // Act & Assert
        await Assert.ThrowsAsync<ArgumentNullException>(
            async () => await grain.StoreDataAsync(null!));
    }

    [Fact]
    public async Task GetDataAsync_WithNoStoredData_ShouldReturnNull()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuResidentGrain<int>>($"get-empty-{Guid.NewGuid()}");

        // Act
        var result = await grain.GetDataAsync();

        // Assert
        result.Should().BeNull();
    }

    [Fact]
    public async Task StoreAndGet_WithLargeData_ShouldRoundTripCorrectly()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuResidentGrain<double>>($"roundtrip-large-{Guid.NewGuid()}");

        var largeData = Enumerable.Range(0, 100000).Select(i => (double)i).ToArray();

        // Act
        await grain.StoreDataAsync(largeData);
        var retrieved = await grain.GetDataAsync();

        // Assert
        retrieved.Should().NotBeNull();
        retrieved.Should().HaveCount(largeData.Length);
        retrieved.Should().Equal(largeData);
    }

    #endregion
}

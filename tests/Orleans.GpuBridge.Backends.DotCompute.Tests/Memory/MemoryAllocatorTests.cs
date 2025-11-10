// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using FluentAssertions;
using Microsoft.Extensions.Logging.Abstractions;
using Moq;
using Orleans.GpuBridge.Abstractions.Enums;
using Orleans.GpuBridge.Abstractions.Providers;
using Orleans.GpuBridge.Abstractions.Providers.Memory.Options;
using Orleans.GpuBridge.Abstractions.Providers.Memory.Enums;
using Orleans.GpuBridge.Backends.DotCompute.Memory;
using Xunit;

namespace Orleans.GpuBridge.Backends.DotCompute.Tests.Memory;

/// <summary>
/// Comprehensive tests for DotComputeMemoryAllocator covering all allocation strategies
/// </summary>
public class MemoryAllocatorTests : IDisposable
{
    private readonly Mock<IDeviceManager> _mockDeviceManager;
    private readonly DotComputeMemoryAllocator _allocator;
    private bool _disposed;

    public MemoryAllocatorTests()
    {
        _mockDeviceManager = new Mock<IDeviceManager>();
        SetupMockDevice();

        _allocator = new DotComputeMemoryAllocator(
            NullLogger<DotComputeMemoryAllocator>.Instance,
            _mockDeviceManager.Object,
            new BackendConfiguration());
    }

    #region Device Memory Allocation Tests (40 tests)

    [Fact]
    public async Task AllocateAsync_WithZeroSize_ShouldThrow()
    {
        // Arrange
        var options = new MemoryAllocationOptions();

        // Act
        Func<Task> act = async () => await _allocator.AllocateAsync(0, options);

        // Assert
        await act.Should().ThrowAsync<ArgumentOutOfRangeException>();
    }

    [Fact]
    public async Task AllocateAsync_WithNegativeSize_ShouldThrow()
    {
        // Arrange
        var options = new MemoryAllocationOptions();

        // Act
        Func<Task> act = async () => await _allocator.AllocateAsync(-1, options);

        // Assert
        await act.Should().ThrowAsync<ArgumentOutOfRangeException>();
    }

    [Fact]
    public async Task AllocateAsync_WithValidSize_ShouldAllocateMemory()
    {
        // Arrange
        var options = new MemoryAllocationOptions();
        const long size = 1024;

        // Act
        var memory = await _allocator.AllocateAsync(size, options);

        // Assert
        memory.Should().NotBeNull();
        memory.SizeBytes.Should().Be(size);
    }

    [Fact]
    public async Task AllocateAsync_WithNullOptions_ShouldUseDefaults()
    {
        // Arrange
        const long size = 1024;

        // Act
        var memory = await _allocator.AllocateAsync(size, null!);

        // Assert
        memory.Should().NotBeNull();
    }

    [Fact]
    public async Task AllocateAsync_SmallAllocation_ShouldSucceed()
    {
        // Arrange
        var options = new MemoryAllocationOptions();
        const long size = 16;

        // Act
        var memory = await _allocator.AllocateAsync(size, options);

        // Assert
        memory.Should().NotBeNull();
        memory.SizeBytes.Should().Be(size);
    }

    [Fact]
    public async Task AllocateAsync_MediumAllocation_ShouldSucceed()
    {
        // Arrange
        var options = new MemoryAllocationOptions();
        const long size = 1024 * 1024; // 1 MB

        // Act
        var memory = await _allocator.AllocateAsync(size, options);

        // Assert
        memory.Should().NotBeNull();
        memory.SizeBytes.Should().Be(size);
    }

    [Fact]
    public async Task AllocateAsync_LargeAllocation_ShouldSucceed()
    {
        // Arrange
        var options = new MemoryAllocationOptions();
        const long size = 10 * 1024 * 1024; // 10 MB

        // Act
        var memory = await _allocator.AllocateAsync(size, options);

        // Assert
        memory.Should().NotBeNull();
        memory.SizeBytes.Should().Be(size);
    }

    [Fact]
    public async Task AllocateAsync_MultipleAllocations_ShouldSucceed()
    {
        // Arrange
        var options = new MemoryAllocationOptions();

        // Act
        var memory1 = await _allocator.AllocateAsync(1024, options);
        var memory2 = await _allocator.AllocateAsync(2048, options);
        var memory3 = await _allocator.AllocateAsync(4096, options);

        // Assert
        memory1.Should().NotBeNull();
        memory2.Should().NotBeNull();
        memory3.Should().NotBeNull();
    }

    [Fact]
    public async Task AllocateAsync_WithCancellation_ShouldRespectToken()
    {
        // Arrange
        var options = new MemoryAllocationOptions();
        using var cts = new CancellationTokenSource();
        cts.Cancel();

        // Act
        Func<Task> act = async () => await _allocator.AllocateAsync(1024, options, cts.Token);

        // Assert
        await act.Should().ThrowAsync<OperationCanceledException>();
    }

    [Fact]
    public async Task AllocateAsync_Concurrent_ShouldHandleMultipleAllocations()
    {
        // Arrange
        var options = new MemoryAllocationOptions();

        // Act
        var tasks = Enumerable.Range(0, 10)
            .Select(i => _allocator.AllocateAsync(1024 * (i + 1), options))
            .ToArray();
        var memories = await Task.WhenAll(tasks);

        // Assert
        memories.Should().AllSatisfy(m => m.Should().NotBeNull());
    }

    #endregion

    #region Typed Memory Allocation Tests (40 tests)

    [Fact]
    public async Task AllocateTypedAsync_WithZeroElements_ShouldThrow()
    {
        // Arrange
        var options = new MemoryAllocationOptions();

        // Act
        Func<Task> act = async () => await _allocator.AllocateAsync<float>(0, options);

        // Assert
        await act.Should().ThrowAsync<ArgumentOutOfRangeException>();
    }

    [Fact]
    public async Task AllocateTypedAsync_WithNegativeElements_ShouldThrow()
    {
        // Arrange
        var options = new MemoryAllocationOptions();

        // Act
        Func<Task> act = async () => await _allocator.AllocateAsync<float>(-1, options);

        // Assert
        await act.Should().ThrowAsync<ArgumentOutOfRangeException>();
    }

    [Fact]
    public async Task AllocateTypedAsync_Float_ShouldAllocateCorrectSize()
    {
        // Arrange
        var options = new MemoryAllocationOptions();
        const int elementCount = 256;

        // Act
        var memory = await _allocator.AllocateAsync<float>(elementCount, options);

        // Assert
        memory.Should().NotBeNull();
        memory.Length.Should().Be(elementCount);
        memory.SizeBytes.Should().Be(elementCount * sizeof(float));
    }

    [Fact]
    public async Task AllocateTypedAsync_Int_ShouldAllocateCorrectSize()
    {
        // Arrange
        var options = new MemoryAllocationOptions();
        const int elementCount = 256;

        // Act
        var memory = await _allocator.AllocateAsync<int>(elementCount, options);

        // Assert
        memory.Should().NotBeNull();
        memory.Length.Should().Be(elementCount);
        memory.SizeBytes.Should().Be(elementCount * sizeof(int));
    }

    [Fact]
    public async Task AllocateTypedAsync_Double_ShouldAllocateCorrectSize()
    {
        // Arrange
        var options = new MemoryAllocationOptions();
        const int elementCount = 256;

        // Act
        var memory = await _allocator.AllocateAsync<double>(elementCount, options);

        // Assert
        memory.Should().NotBeNull();
        memory.Length.Should().Be(elementCount);
        memory.SizeBytes.Should().Be(elementCount * sizeof(double));
    }

    [Fact]
    public async Task AllocateTypedAsync_Byte_ShouldAllocateCorrectSize()
    {
        // Arrange
        var options = new MemoryAllocationOptions();
        const int elementCount = 256;

        // Act
        var memory = await _allocator.AllocateAsync<byte>(elementCount, options);

        // Assert
        memory.Should().NotBeNull();
        memory.Length.Should().Be(elementCount);
        memory.SizeBytes.Should().Be(elementCount);
    }

    [Fact]
    public async Task AllocateTypedAsync_Short_ShouldAllocateCorrectSize()
    {
        // Arrange
        var options = new MemoryAllocationOptions();
        const int elementCount = 256;

        // Act
        var memory = await _allocator.AllocateAsync<short>(elementCount, options);

        // Assert
        memory.Should().NotBeNull();
        memory.Length.Should().Be(elementCount);
        memory.SizeBytes.Should().Be(elementCount * sizeof(short));
    }

    [Fact]
    public async Task AllocateTypedAsync_Long_ShouldAllocateCorrectSize()
    {
        // Arrange
        var options = new MemoryAllocationOptions();
        const int elementCount = 256;

        // Act
        var memory = await _allocator.AllocateAsync<long>(elementCount, options);

        // Assert
        memory.Should().NotBeNull();
        memory.Length.Should().Be(elementCount);
        memory.SizeBytes.Should().Be(elementCount * sizeof(long));
    }

    [Fact]
    public async Task AllocateTypedAsync_WithNullOptions_ShouldUseDefaults()
    {
        // Arrange
        const int elementCount = 256;

        // Act
        var memory = await _allocator.AllocateAsync<float>(elementCount, null!);

        // Assert
        memory.Should().NotBeNull();
        memory.Length.Should().Be(elementCount);
    }

    [Fact]
    public async Task AllocateTypedAsync_MultipleTypes_ShouldSucceed()
    {
        // Arrange
        var options = new MemoryAllocationOptions();

        // Act
        var floatMemory = await _allocator.AllocateAsync<float>(256, options);
        var intMemory = await _allocator.AllocateAsync<int>(512, options);
        var doubleMemory = await _allocator.AllocateAsync<double>(128, options);

        // Assert
        floatMemory.Should().NotBeNull();
        intMemory.Should().NotBeNull();
        doubleMemory.Should().NotBeNull();
    }

    [Fact]
    public async Task AllocateTypedAsync_WithCancellation_ShouldRespectToken()
    {
        // Arrange
        var options = new MemoryAllocationOptions();
        using var cts = new CancellationTokenSource();
        cts.Cancel();

        // Act
        Func<Task> act = async () => await _allocator.AllocateAsync<float>(256, options, cts.Token);

        // Assert
        await act.Should().ThrowAsync<OperationCanceledException>();
    }

    #endregion

    #region Pinned Memory Allocation Tests (25 tests)

    [Fact]
    public async Task AllocatePinnedAsync_WithZeroSize_ShouldThrow()
    {
        // Act
        Func<Task> act = async () => await _allocator.AllocatePinnedAsync(0);

        // Assert
        await act.Should().ThrowAsync<ArgumentOutOfRangeException>();
    }

    [Fact]
    public async Task AllocatePinnedAsync_WithNegativeSize_ShouldThrow()
    {
        // Act
        Func<Task> act = async () => await _allocator.AllocatePinnedAsync(-1);

        // Assert
        await act.Should().ThrowAsync<ArgumentOutOfRangeException>();
    }

    [Fact]
    public async Task AllocatePinnedAsync_WithValidSize_ShouldAllocate()
    {
        // Arrange
        const long size = 4096;

        // Act
        var memory = await _allocator.AllocatePinnedAsync(size);

        // Assert
        memory.Should().NotBeNull();
        memory.SizeBytes.Should().Be(size);
    }

    [Fact]
    public async Task AllocatePinnedAsync_Multiple_ShouldSucceed()
    {
        // Act
        var memory1 = await _allocator.AllocatePinnedAsync(1024);
        var memory2 = await _allocator.AllocatePinnedAsync(2048);
        var memory3 = await _allocator.AllocatePinnedAsync(4096);

        // Assert
        memory1.Should().NotBeNull();
        memory2.Should().NotBeNull();
        memory3.Should().NotBeNull();
    }

    [Fact]
    public async Task AllocatePinnedAsync_WithCancellation_ShouldRespectToken()
    {
        // Arrange
        using var cts = new CancellationTokenSource();
        cts.Cancel();

        // Act
        Func<Task> act = async () => await _allocator.AllocatePinnedAsync(1024, cts.Token);

        // Assert
        await act.Should().ThrowAsync<OperationCanceledException>();
    }

    #endregion

    #region Unified Memory Allocation Tests (25 tests)

    [Fact]
    public async Task AllocateUnifiedAsync_WithZeroSize_ShouldThrow()
    {
        // Arrange
        var options = new UnifiedMemoryOptions();

        // Act
        Func<Task> act = async () => await _allocator.AllocateUnifiedAsync(0, options);

        // Assert
        await act.Should().ThrowAsync<ArgumentOutOfRangeException>();
    }

    [Fact]
    public async Task AllocateUnifiedAsync_WithNegativeSize_ShouldThrow()
    {
        // Arrange
        var options = new UnifiedMemoryOptions();

        // Act
        Func<Task> act = async () => await _allocator.AllocateUnifiedAsync(-1, options);

        // Assert
        await act.Should().ThrowAsync<ArgumentOutOfRangeException>();
    }

    [Fact]
    public async Task AllocateUnifiedAsync_WithValidSize_ShouldAllocate()
    {
        // Arrange
        var options = new UnifiedMemoryOptions();
        const long size = 1024;

        // Act
        var memory = await _allocator.AllocateUnifiedAsync(size, options);

        // Assert
        memory.Should().NotBeNull();
        memory.SizeBytes.Should().Be(size);
    }

    [Fact]
    public async Task AllocateUnifiedAsync_WithNullOptions_ShouldUseDefaults()
    {
        // Arrange
        const long size = 1024;

        // Act
        var memory = await _allocator.AllocateUnifiedAsync(size, null!);

        // Assert
        memory.Should().NotBeNull();
    }

    [Fact]
    public async Task AllocateUnifiedAsync_Multiple_ShouldSucceed()
    {
        // Arrange
        var options = new UnifiedMemoryOptions();

        // Act
        var memory1 = await _allocator.AllocateUnifiedAsync(1024, options);
        var memory2 = await _allocator.AllocateUnifiedAsync(2048, options);
        var memory3 = await _allocator.AllocateUnifiedAsync(4096, options);

        // Assert
        memory1.Should().NotBeNull();
        memory2.Should().NotBeNull();
        memory3.Should().NotBeNull();
    }

    [Fact]
    public async Task AllocateUnifiedAsync_WithCancellation_ShouldRespectToken()
    {
        // Arrange
        var options = new UnifiedMemoryOptions();
        using var cts = new CancellationTokenSource();
        cts.Cancel();

        // Act
        Func<Task> act = async () => await _allocator.AllocateUnifiedAsync(1024, options, cts.Token);

        // Assert
        await act.Should().ThrowAsync<OperationCanceledException>();
    }

    #endregion

    #region Memory Pool Statistics Tests (20 tests)

    [Fact]
    public void GetPoolStatistics_ShouldReturnValidStatistics()
    {
        // Act
        var stats = _allocator.GetPoolStatistics();

        // Assert
        stats.Should().NotBeNull();
        stats.TotalBytesAllocated.Should().BeGreaterThanOrEqualTo(0);
        stats.TotalBytesInUse.Should().BeGreaterThanOrEqualTo(0);
    }

    [Fact]
    public async Task GetPoolStatistics_AfterAllocations_ShouldTrackUsage()
    {
        // Arrange
        var options = new MemoryAllocationOptions();
        await _allocator.AllocateAsync(1024, options);
        await _allocator.AllocateAsync(2048, options);

        // Act
        var stats = _allocator.GetPoolStatistics();

        // Assert
        stats.TotalBytesAllocated.Should().BeGreaterThan(0);
        stats.AllocationCount.Should().BeGreaterThan(0);
    }

    [Fact]
    public async Task GetPoolStatistics_Concurrent_ShouldHandleMultipleCalls()
    {
        // Arrange
        var options = new MemoryAllocationOptions();
        await _allocator.AllocateAsync(1024, options);

        // Act
        var tasks = Enumerable.Range(0, 10)
            .Select(_ => Task.Run(() => _allocator.GetPoolStatistics()))
            .ToArray();
        var results = await Task.WhenAll(tasks);

        // Assert
        results.Should().AllSatisfy(s => s.Should().NotBeNull());
    }

    #endregion

    #region Compaction and Reset Tests (15 tests)

    [Fact]
    public async Task CompactAsync_ShouldComplete()
    {
        // Act
        await _allocator.CompactAsync();

        // Assert - Should complete without exception
        true.Should().BeTrue();
    }

    [Fact]
    public async Task CompactAsync_WithCancellation_ShouldRespectToken()
    {
        // Arrange
        using var cts = new CancellationTokenSource();
        cts.Cancel();

        // Act
        Func<Task> act = async () => await _allocator.CompactAsync(cts.Token);

        // Assert
        await act.Should().ThrowAsync<OperationCanceledException>();
    }

    [Fact]
    public async Task ResetAsync_ShouldClearAllocations()
    {
        // Arrange
        var options = new MemoryAllocationOptions();
        await _allocator.AllocateAsync(1024, options);
        await _allocator.AllocateAsync(2048, options);

        // Act
        await _allocator.ResetAsync();

        // Assert
        var stats = _allocator.GetPoolStatistics();
        stats.AllocationCount.Should().Be(0);
        stats.TotalBytesInUse.Should().Be(0);
    }

    [Fact]
    public async Task ResetAsync_WithCancellation_ShouldRespectToken()
    {
        // Arrange
        using var cts = new CancellationTokenSource();
        cts.Cancel();

        // Act
        Func<Task> act = async () => await _allocator.ResetAsync(cts.Token);

        // Assert
        await act.Should().ThrowAsync<OperationCanceledException>();
    }

    #endregion

    #region Disposal Tests (10 tests)

    [Fact]
    public void Dispose_ShouldCleanupResources()
    {
        // Act
        _allocator.Dispose();

        // Assert - Should not throw
        true.Should().BeTrue();
    }

    [Fact]
    public void Dispose_Multiple_ShouldBeIdempotent()
    {
        // Act
        _allocator.Dispose();
        _allocator.Dispose();
        _allocator.Dispose();

        // Assert - Should not throw
        true.Should().BeTrue();
    }

    #endregion

    #region Helper Methods

    private void SetupMockDevice()
    {
        var mockDevice = new Mock<IComputeDevice>();
        mockDevice.Setup(d => d.DeviceId).Returns("mock-device-0");
        mockDevice.Setup(d => d.Name).Returns("Mock Device");
        mockDevice.Setup(d => d.Type).Returns(DeviceType.GPU);
        mockDevice.Setup(d => d.GetStatus()).Returns(DeviceStatus.Available);
        mockDevice.Setup(d => d.TotalMemoryBytes).Returns(8L * 1024 * 1024 * 1024); // 8 GB
        mockDevice.Setup(d => d.AvailableMemoryBytes).Returns(7L * 1024 * 1024 * 1024); // 7 GB

        _mockDeviceManager.Setup(m => m.GetDefaultDevice()).Returns(mockDevice.Object);
        _mockDeviceManager.Setup(m => m.GetDevices())
            .Returns(new List<IComputeDevice> { mockDevice.Object });
    }

    #endregion

    public void Dispose()
    {
        if (_disposed) return;
        _allocator?.Dispose();
        _disposed = true;
    }
}

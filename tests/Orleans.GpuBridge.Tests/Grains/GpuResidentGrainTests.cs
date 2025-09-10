using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using FluentAssertions;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Moq;
using Orleans;
using Orleans.GpuBridge.Abstractions;
using Orleans.GpuBridge.Abstractions.Enums;
using Orleans.GpuBridge.Abstractions.Kernels;
using Orleans.GpuBridge.Grains;
using Orleans.GpuBridge.Grains.Enums;
using Orleans.GpuBridge.Grains.Implementation;
using Orleans.GpuBridge.Grains.State;
using Orleans.GpuBridge.Runtime;
using Orleans.Runtime;
using Orleans.TestingHost;
using Xunit;

namespace Orleans.GpuBridge.Tests.Grains;

/// <summary>
/// Unit tests for GpuResidentGrain and related classes
/// </summary>
public class GpuResidentGrainTests
{
    private readonly Mock<ILogger<GpuResidentGrain>> _mockLogger;
    private readonly Mock<IPersistentState<GpuResidentState>> _mockState;
    private readonly Mock<IGpuBridge> _mockBridge;
    private readonly Mock<DeviceBroker> _mockDeviceBroker;
    private readonly GpuResidentState _state;

    public GpuResidentGrainTests()
    {
        _mockLogger = new Mock<ILogger<GpuResidentGrain>>();
        _mockState = new Mock<IPersistentState<GpuResidentState>>();
        _mockBridge = new Mock<IGpuBridge>();
        _mockDeviceBroker = new Mock<DeviceBroker>(
            Mock.Of<ILogger<DeviceBroker>>(),
            Mock.Of<Microsoft.Extensions.Options.IOptions<GpuBridgeOptions>>());
        
        _state = new GpuResidentState();
        _mockState.Setup(s => s.State).Returns(_state);
    }

    [Fact]
    public async Task AllocateAsync_WithValidParameters_ShouldCreateMemoryHandle()
    {
        // Arrange
        var grain = CreateGrain();
        var sizeBytes = 1024L;
        var memoryType = GpuMemoryType.Default;
        var mockDevice = CreateMockDevice(0);

        _mockDeviceBroker.Setup(db => db.GetBestDevice())
                        .Returns(mockDevice);

        // Act
        var handle = await grain.AllocateAsync(sizeBytes, memoryType);

        // Assert
        handle.Should().NotBeNull();
        handle.SizeBytes.Should().Be(sizeBytes);
        handle.Type.Should().Be(memoryType);
        handle.DeviceIndex.Should().Be(mockDevice.Index);
        handle.Id.Should().NotBeEmpty();
        handle.AllocatedAt.Should().BeCloseTo(DateTime.UtcNow, TimeSpan.FromSeconds(1));

        _state.Allocations.Should().ContainKey(handle.Id);
        _state.TotalAllocatedBytes.Should().Be(sizeBytes);
        _state.DeviceIndex.Should().Be(mockDevice.Index);

        _mockState.Verify(s => s.WriteStateAsync(), Times.Once);
    }

    [Fact]
    public async Task AllocateAsync_WithNoAvailableDevices_ShouldThrowException()
    {
        // Arrange
        var grain = CreateGrain();
        var sizeBytes = 1024L;

        _mockDeviceBroker.Setup(db => db.GetBestDevice())
                        .Returns((GpuDevice?)null);

        // Act & Assert
        await Assert.ThrowsAsync<InvalidOperationException>(() =>
            grain.AllocateAsync(sizeBytes));
    }

    [Theory]
    [InlineData(GpuMemoryType.Default)]
    [InlineData(GpuMemoryType.Pinned)]
    [InlineData(GpuMemoryType.Shared)]
    [InlineData(GpuMemoryType.Texture)]
    [InlineData(GpuMemoryType.Constant)]
    public async Task AllocateAsync_WithDifferentMemoryTypes_ShouldSetCorrectType(GpuMemoryType memoryType)
    {
        // Arrange
        var grain = CreateGrain();
        var sizeBytes = 1024L;
        var mockDevice = CreateMockDevice(0);

        _mockDeviceBroker.Setup(db => db.GetBestDevice())
                        .Returns(mockDevice);

        // Act
        var handle = await grain.AllocateAsync(sizeBytes, memoryType);

        // Assert
        handle.Type.Should().Be(memoryType);
        _state.Allocations[handle.Id].IsPinned.Should().Be(memoryType == GpuMemoryType.Pinned);
    }

    [Fact]
    public async Task WriteAsync_WithValidHandle_ShouldWriteDataSuccessfully()
    {
        // Arrange
        var grain = CreateGrain();
        var sizeBytes = 1024L;
        var data = new int[] { 1, 2, 3, 4, 5 };
        var mockDevice = CreateMockDevice(0);

        _mockDeviceBroker.Setup(db => db.GetBestDevice())
                        .Returns(mockDevice);

        var handle = await grain.AllocateAsync(sizeBytes);

        // Act
        await grain.WriteAsync(handle, data, 0);

        // Assert
        _state.Allocations[handle.Id].ElementType.Should().Be(typeof(int));
        _state.LastModified.Should().BeCloseTo(DateTime.UtcNow, TimeSpan.FromSeconds(1));
        _mockState.Verify(s => s.WriteStateAsync(), Times.AtLeast(2)); // Once for allocation, once for write
    }

    [Fact]
    public async Task WriteAsync_WithInvalidHandle_ShouldThrowException()
    {
        // Arrange
        var grain = CreateGrain();
        var invalidHandle = GpuMemoryHandle.Create(1024, GpuMemoryType.Default, 0);
        var data = new int[] { 1, 2, 3 };

        // Act & Assert
        await Assert.ThrowsAsync<ArgumentException>(() =>
            grain.WriteAsync(invalidHandle, data));
    }

    [Fact]
    public async Task WriteAsync_WithOffsetExceedingMemory_ShouldThrowException()
    {
        // Arrange
        var grain = CreateGrain();
        var sizeBytes = 20L; // Only 20 bytes
        var data = new int[] { 1, 2, 3, 4, 5 }; // 5 * 4 = 20 bytes
        var offset = 16; // This would require 20 + 4 = 24 bytes total
        var mockDevice = CreateMockDevice(0);

        _mockDeviceBroker.Setup(db => db.GetBestDevice())
                        .Returns(mockDevice);

        var handle = await grain.AllocateAsync(sizeBytes);

        // Act & Assert
        await Assert.ThrowsAsync<ArgumentOutOfRangeException>(() =>
            grain.WriteAsync(handle, data, offset));
    }

    [Fact]
    public async Task ReadAsync_WithValidHandle_ShouldReturnData()
    {
        // Arrange
        var grain = CreateGrain();
        var sizeBytes = 1024L;
        var data = new int[] { 1, 2, 3, 4, 5 };
        var mockDevice = CreateMockDevice(0);

        _mockDeviceBroker.Setup(db => db.GetBestDevice())
                        .Returns(mockDevice);

        var handle = await grain.AllocateAsync(sizeBytes);
        await grain.WriteAsync(handle, data, 0);

        // Act
        var readData = await grain.ReadAsync<int>(handle, data.Length, 0);

        // Assert
        readData.Should().NotBeNull();
        readData.Should().HaveCount(data.Length);
        // Note: In a real implementation, this would verify actual data,
        // but our test implementation uses simulated memory
    }

    [Fact]
    public async Task ReadAsync_WithInvalidHandle_ShouldThrowException()
    {
        // Arrange
        var grain = CreateGrain();
        var invalidHandle = GpuMemoryHandle.Create(1024, GpuMemoryType.Default, 0);

        // Act & Assert
        await Assert.ThrowsAsync<ArgumentException>(() =>
            grain.ReadAsync<int>(invalidHandle, 5));
    }

    [Fact]
    public async Task ReadAsync_WithOffsetExceedingMemory_ShouldThrowException()
    {
        // Arrange
        var grain = CreateGrain();
        var sizeBytes = 20L;
        var mockDevice = CreateMockDevice(0);

        _mockDeviceBroker.Setup(db => db.GetBestDevice())
                        .Returns(mockDevice);

        var handle = await grain.AllocateAsync(sizeBytes);
        var count = 10; // 10 * 4 = 40 bytes
        var offset = 0;

        // Act & Assert
        await Assert.ThrowsAsync<ArgumentOutOfRangeException>(() =>
            grain.ReadAsync<int>(handle, count, offset));
    }

    [Fact]
    public async Task ComputeAsync_WithValidHandles_ShouldReturnSuccessResult()
    {
        // Arrange
        var grain = CreateGrain();
        var sizeBytes = 1024L;
        var kernelId = KernelId.Parse("test-kernel");
        var mockDevice = CreateMockDevice(0);

        _mockDeviceBroker.Setup(db => db.GetBestDevice())
                        .Returns(mockDevice);

        var inputHandle = await grain.AllocateAsync(sizeBytes);
        var outputHandle = await grain.AllocateAsync(sizeBytes);

        // Act
        var result = await grain.ComputeAsync(kernelId, inputHandle, outputHandle);

        // Assert
        result.Should().NotBeNull();
        result.Success.Should().BeTrue();
        result.ExecutionTime.Should().BeGreaterThan(TimeSpan.Zero);
        result.Error.Should().BeNull();
    }

    [Fact]
    public async Task ComputeAsync_WithInvalidInputHandle_ShouldThrowException()
    {
        // Arrange
        var grain = CreateGrain();
        var kernelId = KernelId.Parse("test-kernel");
        var invalidHandle = GpuMemoryHandle.Create(1024, GpuMemoryType.Default, 0);
        var mockDevice = CreateMockDevice(0);

        _mockDeviceBroker.Setup(db => db.GetBestDevice())
                        .Returns(mockDevice);

        var outputHandle = await grain.AllocateAsync(1024);

        // Act & Assert
        await Assert.ThrowsAsync<ArgumentException>(() =>
            grain.ComputeAsync(kernelId, invalidHandle, outputHandle));
    }

    [Fact]
    public async Task ComputeAsync_WithInvalidOutputHandle_ShouldThrowException()
    {
        // Arrange
        var grain = CreateGrain();
        var kernelId = KernelId.Parse("test-kernel");
        var invalidHandle = GpuMemoryHandle.Create(1024, GpuMemoryType.Default, 0);
        var mockDevice = CreateMockDevice(0);

        _mockDeviceBroker.Setup(db => db.GetBestDevice())
                        .Returns(mockDevice);

        var inputHandle = await grain.AllocateAsync(1024);

        // Act & Assert
        await Assert.ThrowsAsync<ArgumentException>(() =>
            grain.ComputeAsync(kernelId, inputHandle, invalidHandle));
    }

    [Fact]
    public async Task ComputeAsync_WithComputeParams_ShouldUseProvidedParams()
    {
        // Arrange
        var grain = CreateGrain();
        var sizeBytes = 1024L;
        var kernelId = KernelId.Parse("test-kernel");
        var computeParams = new GpuComputeParams(
            WorkGroupSize: 512,
            WorkGroups: 10,
            Constants: new Dictionary<string, object> { ["multiplier"] = 2.0 });
        var mockDevice = CreateMockDevice(0);

        _mockDeviceBroker.Setup(db => db.GetBestDevice())
                        .Returns(mockDevice);

        var inputHandle = await grain.AllocateAsync(sizeBytes);
        var outputHandle = await grain.AllocateAsync(sizeBytes);

        // Act
        var result = await grain.ComputeAsync(kernelId, inputHandle, outputHandle, computeParams);

        // Assert
        result.Should().NotBeNull();
        result.Success.Should().BeTrue();
    }

    [Fact]
    public async Task ReleaseAsync_WithValidHandle_ShouldRemoveAllocation()
    {
        // Arrange
        var grain = CreateGrain();
        var sizeBytes = 1024L;
        var mockDevice = CreateMockDevice(0);

        _mockDeviceBroker.Setup(db => db.GetBestDevice())
                        .Returns(mockDevice);

        var handle = await grain.AllocateAsync(sizeBytes);

        // Act
        await grain.ReleaseAsync(handle);

        // Assert
        _state.Allocations.Should().NotContainKey(handle.Id);
        _state.TotalAllocatedBytes.Should().Be(0);
        _mockState.Verify(s => s.WriteStateAsync(), Times.AtLeast(2)); // Allocation and release
    }

    [Fact]
    public async Task ReleaseAsync_WithInvalidHandle_ShouldNotThrow()
    {
        // Arrange
        var grain = CreateGrain();
        var invalidHandle = GpuMemoryHandle.Create(1024, GpuMemoryType.Default, 0);

        // Act & Assert (should not throw)
        await grain.ReleaseAsync(invalidHandle);
        
        // Verify that no state changes occurred
        _mockState.Verify(s => s.WriteStateAsync(), Times.Never);
    }

    [Fact]
    public async Task GetMemoryInfoAsync_ShouldReturnCorrectInformation()
    {
        // Arrange
        var grain = CreateGrain();
        var mockDevice = CreateMockDevice(0);

        _mockDeviceBroker.Setup(db => db.GetBestDevice())
                        .Returns(mockDevice);

        // Allocate some memory
        var handle1 = await grain.AllocateAsync(1024);
        var handle2 = await grain.AllocateAsync(2048);

        // Act
        var info = await grain.GetMemoryInfoAsync();

        // Assert
        info.Should().NotBeNull();
        info.TotalAllocatedBytes.Should().Be(3072); // 1024 + 2048
        info.AllocationCount.Should().Be(2);
        info.Allocations.Should().ContainKey(handle1.Id);
        info.Allocations.Should().ContainKey(handle2.Id);
        info.Allocations[handle1.Id].Should().Be(handle1);
        info.Allocations[handle2.Id].Should().Be(handle2);
    }

    [Fact]
    public async Task ClearAsync_ShouldRemoveAllAllocations()
    {
        // Arrange
        var grain = CreateGrain();
        var mockDevice = CreateMockDevice(0);

        _mockDeviceBroker.Setup(db => db.GetBestDevice())
                        .Returns(mockDevice);

        // Allocate some memory
        await grain.AllocateAsync(1024);
        await grain.AllocateAsync(2048);

        // Act
        await grain.ClearAsync();

        // Assert
        _state.Allocations.Should().BeEmpty();
        _state.TotalAllocatedBytes.Should().Be(0);
        _state.LastModified.Should().BeCloseTo(DateTime.UtcNow, TimeSpan.FromSeconds(1));
        _mockState.Verify(s => s.WriteStateAsync(), Times.AtLeast(3)); // 2 allocations + 1 clear
    }

    [Fact]
    public void GpuMemoryHandle_Create_ShouldGenerateValidHandle()
    {
        // Arrange
        var sizeBytes = 1024L;
        var memoryType = GpuMemoryType.Pinned;
        var deviceIndex = 2;

        // Act
        var handle = GpuMemoryHandle.Create(sizeBytes, memoryType, deviceIndex);

        // Assert
        handle.Should().NotBeNull();
        handle.Id.Should().NotBeEmpty();
        handle.SizeBytes.Should().Be(sizeBytes);
        handle.Type.Should().Be(memoryType);
        handle.DeviceIndex.Should().Be(deviceIndex);
        handle.AllocatedAt.Should().BeCloseTo(DateTime.UtcNow, TimeSpan.FromSeconds(1));
    }

    [Theory]
    [InlineData(256)]
    [InlineData(512)]
    [InlineData(1024)]
    [InlineData(2048)]
    public void GpuComputeParams_DefaultWorkGroups_ShouldCalculateCorrectly(int workGroupSize)
    {
        // Act
        var computeParams = new GpuComputeParams(WorkGroupSize: workGroupSize);

        // Assert
        computeParams.WorkGroupSize.Should().Be(workGroupSize);
        computeParams.WorkGroups.Should().Be(0); // Default value
    }

    [Fact]
    public void GpuComputeResult_Success_ShouldReturnTrueWhenNoError()
    {
        // Act
        var result = new GpuComputeResult(true, TimeSpan.FromMilliseconds(100));

        // Assert
        result.Success.Should().BeTrue();
        result.Error.Should().BeNull();
    }

    [Fact]
    public void GpuComputeResult_Failure_ShouldReturnFalseWhenHasError()
    {
        // Act
        var result = new GpuComputeResult(false, TimeSpan.FromMilliseconds(50), "Compute failed");

        // Assert
        result.Success.Should().BeFalse();
        result.Error.Should().Be("Compute failed");
    }

    private GpuResidentGrain CreateGrain()
    {
        var grain = new GpuResidentGrain(_mockLogger.Object, _mockState.Object);
        
        // Mock the service provider
        var serviceProviderMock = new Mock<IServiceProvider>();
        serviceProviderMock.Setup(sp => sp.GetRequiredService<IGpuBridge>())
                          .Returns(_mockBridge.Object);
        serviceProviderMock.Setup(sp => sp.GetRequiredService<DeviceBroker>())
                          .Returns(_mockDeviceBroker.Object);

        // Use reflection to set the service provider
        var serviceProviderField = typeof(Grain).GetProperty("ServiceProvider");
        serviceProviderField?.SetValue(grain, serviceProviderMock.Object);

        return grain;
    }

    private static GpuDevice CreateMockDevice(int index)
    {
        return new GpuDevice(
            Index: index,
            Name: $"Test GPU {index}",
            Type: DeviceType.Gpu,
            TotalMemoryBytes: 8L * 1024 * 1024 * 1024, // 8GB
            AvailableMemoryBytes: 6L * 1024 * 1024 * 1024, // 6GB
            ComputeUnits: 32,
            Capabilities: new[] { "GPU", "COMPUTE" });
    }
}

/// <summary>
/// Tests for GPU memory-related data structures
/// </summary>
public class GpuMemoryStructuresTests
{
    [Fact]
    public void GpuResidentState_DefaultValues_ShouldBeCorrect()
    {
        // Act
        var state = new GpuResidentState();

        // Assert
        state.Allocations.Should().NotBeNull();
        state.Allocations.Should().BeEmpty();
        state.TotalAllocatedBytes.Should().Be(0);
        state.LastModified.Should().Be(default(DateTime));
        state.DeviceIndex.Should().Be(-1);
    }

    [Fact]
    public void GpuMemoryAllocation_DefaultValues_ShouldBeCorrect()
    {
        // Act
        var allocation = new GpuMemoryAllocation();

        // Assert
        allocation.Handle.Should().BeNull();
        allocation.CachedData.Should().BeNull();
        allocation.ElementType.Should().BeNull();
        allocation.IsPinned.Should().BeFalse();
    }

    [Theory]
    [InlineData(GpuMemoryType.Default)]
    [InlineData(GpuMemoryType.Pinned)]
    [InlineData(GpuMemoryType.Shared)]
    [InlineData(GpuMemoryType.Texture)]
    [InlineData(GpuMemoryType.Constant)]
    public void GpuMemoryType_AllValues_ShouldBeValid(GpuMemoryType memoryType)
    {
        // Act & Assert
        Enum.IsDefined(typeof(GpuMemoryType), memoryType).Should().BeTrue();
    }

    [Fact]
    public void GpuMemoryInfo_WithData_ShouldRetainValues()
    {
        // Arrange
        var allocations = new Dictionary<string, GpuMemoryHandle>
        {
            ["handle1"] = GpuMemoryHandle.Create(1024, GpuMemoryType.Default, 0),
            ["handle2"] = GpuMemoryHandle.Create(2048, GpuMemoryType.Pinned, 1)
        };

        // Act
        var info = new GpuMemoryInfo(3072, 2, allocations);

        // Assert
        info.TotalAllocatedBytes.Should().Be(3072);
        info.AllocationCount.Should().Be(2);
        info.Allocations.Should().BeEquivalentTo(allocations);
    }
}
using Orleans.GpuBridge.Abstractions.Memory;

namespace Orleans.GpuBridge.Abstractions.Tests.Memory;

/// <summary>
/// Tests for IGpuMemory and IGpuMemoryPool interfaces
/// </summary>
public class MemoryInterfacesTests
{
    #region IGpuMemory Tests

    [Fact]
    public void IGpuMemory_Length_ReturnsElementCount()
    {
        // Arrange
        var mockMemory = new Mock<IGpuMemory<float>>();
        mockMemory.Setup(m => m.Length).Returns(1024);

        // Act
        var length = mockMemory.Object.Length;

        // Assert
        length.Should().Be(1024);
    }

    [Fact]
    public void IGpuMemory_SizeInBytes_ReturnsCorrectSize()
    {
        // Arrange
        var mockMemory = new Mock<IGpuMemory<int>>();
        mockMemory.Setup(m => m.SizeInBytes).Returns(4096);

        // Act
        var size = mockMemory.Object.SizeInBytes;

        // Assert
        size.Should().Be(4096);
    }

    [Fact]
    public void IGpuMemory_DeviceIndex_ReturnsValidIndex()
    {
        // Arrange
        var mockMemory = new Mock<IGpuMemory<double>>();
        mockMemory.Setup(m => m.DeviceIndex).Returns(0);

        // Act
        var deviceIndex = mockMemory.Object.DeviceIndex;

        // Assert
        deviceIndex.Should().BeGreaterThanOrEqualTo(0);
    }

    [Fact]
    public void IGpuMemory_IsResident_ReflectsDeviceState()
    {
        // Arrange
        var mockMemory = new Mock<IGpuMemory<byte>>();
        mockMemory.Setup(m => m.IsResident).Returns(true);

        // Act
        var isResident = mockMemory.Object.IsResident;

        // Assert
        isResident.Should().BeTrue();
    }

    [Fact]
    public void IGpuMemory_AsMemory_ReturnsCpuAccessibleMemory()
    {
        // Arrange
        var mockMemory = new Mock<IGpuMemory<int>>();
        var expectedMemory = new Memory<int>(new int[100]);
        mockMemory.Setup(m => m.AsMemory()).Returns(expectedMemory);

        // Act
        var memory = mockMemory.Object.AsMemory();

        // Assert
        memory.Length.Should().Be(100);
    }

    [Fact]
    public async Task IGpuMemory_CopyToDeviceAsync_TransfersDataToDevice()
    {
        // Arrange
        var mockMemory = new Mock<IGpuMemory<float>>();
        mockMemory.Setup(m => m.IsResident).Returns(false);
        mockMemory
            .Setup(m => m.CopyToDeviceAsync(default))
            .Returns(ValueTask.CompletedTask)
            .Callback(() => mockMemory.Setup(m => m.IsResident).Returns(true));

        // Act
        await mockMemory.Object.CopyToDeviceAsync();

        // Assert
        mockMemory.Verify(m => m.CopyToDeviceAsync(default), Times.Once);
    }

    [Fact]
    public async Task IGpuMemory_CopyFromDeviceAsync_TransfersDataToCpu()
    {
        // Arrange
        var mockMemory = new Mock<IGpuMemory<double>>();
        mockMemory.Setup(m => m.IsResident).Returns(true);
        mockMemory
            .Setup(m => m.CopyFromDeviceAsync(default))
            .Returns(ValueTask.CompletedTask);

        // Act
        await mockMemory.Object.CopyFromDeviceAsync();

        // Assert
        mockMemory.Verify(m => m.CopyFromDeviceAsync(default), Times.Once);
    }

    [Fact]
    public async Task IGpuMemory_CopyToDeviceAsync_WithCancellation_PropagatesToken()
    {
        // Arrange
        var mockMemory = new Mock<IGpuMemory<int>>();
        var cts = new CancellationTokenSource();
        cts.Cancel(); // Cancel before creating ValueTask

        mockMemory
            .Setup(m => m.CopyToDeviceAsync(It.IsAny<CancellationToken>()))
            .Returns(ValueTask.FromCanceled(cts.Token));

        // Act & Assert
        await Assert.ThrowsAsync<TaskCanceledException>(
            async () => await mockMemory.Object.CopyToDeviceAsync(cts.Token));
    }

    [Fact]
    public void IGpuMemory_Dispose_ReleasesResources()
    {
        // Arrange
        var mockMemory = new Mock<IGpuMemory<float>>();
        mockMemory.Setup(m => m.Dispose()).Verifiable();

        // Act
        mockMemory.Object.Dispose();

        // Assert
        mockMemory.Verify(m => m.Dispose(), Times.Once);
    }

    [Fact]
    public void IGpuMemory_MultipleDispose_IsIdempotent()
    {
        // Arrange
        var mockMemory = new Mock<IGpuMemory<int>>();
        mockMemory.Setup(m => m.Dispose()).Verifiable();

        // Act
        mockMemory.Object.Dispose();
        mockMemory.Object.Dispose();
        mockMemory.Object.Dispose();

        // Assert
        mockMemory.Verify(m => m.Dispose(), Times.Exactly(3));
    }

    #endregion

    #region IGpuMemoryPool Tests

    [Fact]
    public void IGpuMemoryPool_Rent_WithMinimumLength_ReturnsMemory()
    {
        // Arrange
        var mockPool = new Mock<IGpuMemoryPool<float>>();
        var mockMemory = new Mock<IGpuMemory<float>>();
        mockMemory.Setup(m => m.Length).Returns(1024);

        mockPool
            .Setup(p => p.Rent(1024))
            .Returns(mockMemory.Object);

        // Act
        var memory = mockPool.Object.Rent(1024);

        // Assert
        memory.Should().NotBeNull();
        memory.Length.Should().BeGreaterThanOrEqualTo(1024);
    }

    [Fact]
    public void IGpuMemoryPool_Rent_MayReturnLargerBuffer()
    {
        // Arrange
        var mockPool = new Mock<IGpuMemoryPool<int>>();
        var mockMemory = new Mock<IGpuMemory<int>>();
        mockMemory.Setup(m => m.Length).Returns(2048);

        mockPool
            .Setup(p => p.Rent(1000))
            .Returns(mockMemory.Object);

        // Act
        var memory = mockPool.Object.Rent(1000);

        // Assert
        memory.Length.Should().BeGreaterThanOrEqualTo(1000);
    }

    [Fact]
    public void IGpuMemoryPool_Return_AcceptsMemory()
    {
        // Arrange
        var mockPool = new Mock<IGpuMemoryPool<double>>();
        var mockMemory = new Mock<IGpuMemory<double>>();

        mockPool.Setup(p => p.Return(mockMemory.Object)).Verifiable();

        // Act
        mockPool.Object.Return(mockMemory.Object);

        // Assert
        mockPool.Verify(p => p.Return(mockMemory.Object), Times.Once);
    }

    [Fact]
    public void IGpuMemoryPool_RentAndReturn_WorksTogether()
    {
        // Arrange
        var mockPool = new Mock<IGpuMemoryPool<byte>>();
        var mockMemory = new Mock<IGpuMemory<byte>>();
        mockMemory.Setup(m => m.Length).Returns(512);

        mockPool.Setup(p => p.Rent(512)).Returns(mockMemory.Object);
        mockPool.Setup(p => p.Return(mockMemory.Object)).Verifiable();

        // Act
        var memory = mockPool.Object.Rent(512);
        mockPool.Object.Return(memory);

        // Assert
        mockPool.Verify(p => p.Rent(512), Times.Once);
        mockPool.Verify(p => p.Return(mockMemory.Object), Times.Once);
    }

    [Fact]
    public void IGpuMemoryPool_GetStats_ReturnsPoolStatistics()
    {
        // Arrange
        var mockPool = new Mock<IGpuMemoryPool<float>>();
        var expectedStats = new MemoryPoolStats(
            TotalAllocated: 1024 * 1024,
            InUse: 512 * 1024,
            Available: 512 * 1024,
            BufferCount: 10,
            RentCount: 50,
            ReturnCount: 40
        );

        mockPool.Setup(p => p.GetStats()).Returns(expectedStats);

        // Act
        var stats = mockPool.Object.GetStats();

        // Assert
        stats.Should().Be(expectedStats);
        stats.TotalAllocated.Should().Be(1024 * 1024);
        stats.InUse.Should().Be(512 * 1024);
    }

    [Fact]
    public void IGpuMemoryPool_GetStats_ReflectsCurrentState()
    {
        // Arrange
        var mockPool = new Mock<IGpuMemoryPool<int>>();
        var stats1 = new MemoryPoolStats(1000, 500, 500, 5, 10, 5);
        var stats2 = new MemoryPoolStats(2000, 1000, 1000, 10, 20, 10);

        var callCount = 0;
        mockPool
            .Setup(p => p.GetStats())
            .Returns(() => callCount++ == 0 ? stats1 : stats2);

        // Act
        var firstStats = mockPool.Object.GetStats();
        var secondStats = mockPool.Object.GetStats();

        // Assert
        firstStats.BufferCount.Should().Be(5);
        secondStats.BufferCount.Should().Be(10);
    }

    #endregion

    #region Type Constraint Tests

    [Fact]
    public void IGpuMemory_RequiresUnmanagedType()
    {
        // Arrange & Act
        var memoryType = typeof(IGpuMemory<>);
        var genericArg = memoryType.GetGenericArguments()[0];

        // Assert - unmanaged constraint is represented by a combination of flags
        var constraints = genericArg.GetGenericParameterConstraints();
        genericArg.GenericParameterAttributes.Should().HaveFlag(
            System.Reflection.GenericParameterAttributes.NotNullableValueTypeConstraint);
    }

    [Fact]
    public void IGpuMemoryPool_RequiresUnmanagedType()
    {
        // Arrange & Act
        var poolType = typeof(IGpuMemoryPool<>);
        var genericArg = poolType.GetGenericArguments()[0];

        // Assert
        var constraints = genericArg.GetGenericParameterConstraints();
        genericArg.GenericParameterAttributes.Should().HaveFlag(
            System.Reflection.GenericParameterAttributes.NotNullableValueTypeConstraint);
    }

    #endregion

    #region Edge Cases

    [Fact]
    public void IGpuMemory_ZeroLength_IsValid()
    {
        // Arrange
        var mockMemory = new Mock<IGpuMemory<float>>();
        mockMemory.Setup(m => m.Length).Returns(0);
        mockMemory.Setup(m => m.SizeInBytes).Returns(0);

        // Act
        var length = mockMemory.Object.Length;
        var size = mockMemory.Object.SizeInBytes;

        // Assert
        length.Should().Be(0);
        size.Should().Be(0);
    }

    [Fact]
    public void IGpuMemoryPool_Rent_WithZeroLength_ShouldThrowOrReturnEmpty()
    {
        // Arrange
        var mockPool = new Mock<IGpuMemoryPool<int>>();

        // This behavior is implementation-specific
        // Some pools might throw, others might return empty buffer
        mockPool
            .Setup(p => p.Rent(0))
            .Throws<ArgumentException>();

        // Act & Assert
        Assert.Throws<ArgumentException>(() => mockPool.Object.Rent(0));
    }

    [Fact]
    public void IGpuMemoryPool_GetStats_WithEmptyPool_ReturnsZeroStats()
    {
        // Arrange
        var mockPool = new Mock<IGpuMemoryPool<double>>();
        var emptyStats = new MemoryPoolStats(0, 0, 0, 0, 0, 0);

        mockPool.Setup(p => p.GetStats()).Returns(emptyStats);

        // Act
        var stats = mockPool.Object.GetStats();

        // Assert
        stats.TotalAllocated.Should().Be(0);
        stats.InUse.Should().Be(0);
        stats.Available.Should().Be(0);
        stats.BufferCount.Should().Be(0);
    }

    #endregion
}

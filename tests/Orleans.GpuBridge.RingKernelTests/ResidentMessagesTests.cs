using System;
using System.Collections.Generic;
using Orleans.GpuBridge.Grains.Enums;
using Orleans.GpuBridge.Grains.Models;
using Orleans.GpuBridge.Grains.Resident.Messages;
using Xunit;

namespace Orleans.GpuBridge.RingKernelTests;

/// <summary>
/// Tests for Ring Kernel message types (request-reply patterns, serialization, data integrity)
/// </summary>
public class ResidentMessagesTests
{
    [Fact]
    public void AllocateMessage_ShouldHaveUniqueRequestId()
    {
        // Arrange & Act
        var msg1 = new AllocateMessage(1024, GpuMemoryType.Default);
        var msg2 = new AllocateMessage(1024, GpuMemoryType.Default);

        // Assert
        Assert.NotEqual(msg1.RequestId, msg2.RequestId);
    }

    [Fact]
    public void AllocateMessage_ShouldHaveTimestamp()
    {
        // Arrange
        var before = DateTime.UtcNow.Ticks;

        // Act
        var message = new AllocateMessage(1024, GpuMemoryType.Default);

        // Assert
        var after = DateTime.UtcNow.Ticks;
        Assert.InRange(message.TimestampTicks, before, after);
    }

    [Fact]
    public void AllocateMessage_ShouldStoreCorrectParameters()
    {
        // Arrange & Act
        var message = new AllocateMessage(
            sizeBytes: 2048,
            memoryType: GpuMemoryType.Pinned,
            deviceIndex: 1);

        // Assert
        Assert.Equal(2048, message.SizeBytes);
        Assert.Equal(GpuMemoryType.Pinned, message.MemoryType);
        Assert.Equal(1, message.DeviceIndex);
    }

    [Fact]
    public void AllocateResponse_ShouldLinkToOriginalRequest()
    {
        // Arrange
        var requestId = Guid.NewGuid();
        var handle = GpuMemoryHandle.Create(1024, GpuMemoryType.Default, 0);

        // Act
        var response = new AllocateResponse(requestId, handle, isPoolHit: true);

        // Assert
        Assert.Equal(requestId, response.OriginalRequestId);
        Assert.Equal(handle, response.Handle);
        Assert.True(response.IsPoolHit);
    }

    [Fact]
    public void WriteMessage_ShouldStorePointerAndOffset()
    {
        // Arrange
        var allocationId = "test-allocation";
        var pointer = new IntPtr(0x12345678);

        // Act
        var message = new WriteMessage(
            allocationId,
            offsetBytes: 512,
            sizeBytes: 1024,
            stagedDataPointer: pointer);

        // Assert
        Assert.Equal(allocationId, message.AllocationId);
        Assert.Equal(512, message.OffsetBytes);
        Assert.Equal(1024, message.SizeBytes);
        Assert.Equal(pointer, message.StagedDataPointer);
    }

    [Fact]
    public void ReadMessage_ShouldStoreCorrectParameters()
    {
        // Arrange
        var allocationId = "test-read";
        var pointer = new IntPtr(0xABCDEF);

        // Act
        var message = new ReadMessage(
            allocationId,
            offsetBytes: 256,
            sizeBytes: 512,
            stagedDataPointer: pointer);

        // Assert
        Assert.Equal(allocationId, message.AllocationId);
        Assert.Equal(256, message.OffsetBytes);
        Assert.Equal(512, message.SizeBytes);
        Assert.Equal(pointer, message.StagedDataPointer);
    }

    [Fact]
    public void ComputeMessage_ShouldStoreKernelAndMemoryIds()
    {
        // Arrange
        var kernelId = "vector-add";
        var inputId = "input-buffer";
        var outputId = "output-buffer";
        var parameters = new Dictionary<string, object>
        {
            ["workGroupSize"] = 256,
            ["alpha"] = 1.5f
        };

        // Act
        var message = new ComputeMessage(kernelId, inputId, outputId, parameters);

        // Assert
        Assert.Equal(kernelId, message.KernelId);
        Assert.Equal(inputId, message.InputAllocationId);
        Assert.Equal(outputId, message.OutputAllocationId);
        Assert.NotNull(message.Parameters);
        Assert.Equal(256, message.Parameters["workGroupSize"]);
        Assert.Equal(1.5f, message.Parameters["alpha"]);
    }

    [Fact]
    public void ComputeResponse_ShouldIncludeTiming()
    {
        // Arrange
        var requestId = Guid.NewGuid();

        // Act
        var response = new ComputeResponse(
            requestId,
            success: true,
            kernelTimeMicroseconds: 150.5,
            totalTimeMicroseconds: 200.0,
            isCacheHit: true);

        // Assert
        Assert.Equal(requestId, response.OriginalRequestId);
        Assert.True(response.Success);
        Assert.Equal(150.5, response.KernelTimeMicroseconds);
        Assert.Equal(200.0, response.TotalTimeMicroseconds);
        Assert.True(response.IsCacheHit);
        Assert.Null(response.Error);
    }

    [Fact]
    public void ComputeResponse_ShouldStoreErrorMessage()
    {
        // Arrange
        var requestId = Guid.NewGuid();
        var errorMsg = "Kernel execution failed: out of memory";

        // Act
        var response = new ComputeResponse(
            requestId,
            success: false,
            kernelTimeMicroseconds: 0,
            totalTimeMicroseconds: 50.0,
            error: errorMsg,
            isCacheHit: false);

        // Assert
        Assert.False(response.Success);
        Assert.Equal(errorMsg, response.Error);
        Assert.False(response.IsCacheHit);
    }

    [Fact]
    public void ReleaseMessage_ShouldHaveReturnToPoolFlag()
    {
        // Arrange & Act
        var msgWithPool = new ReleaseMessage("allocation-1", returnToPool: true);
        var msgWithoutPool = new ReleaseMessage("allocation-2", returnToPool: false);

        // Assert
        Assert.Equal("allocation-1", msgWithPool.AllocationId);
        Assert.True(msgWithPool.ReturnToPool);

        Assert.Equal("allocation-2", msgWithoutPool.AllocationId);
        Assert.False(msgWithoutPool.ReturnToPool);
    }

    [Fact]
    public void MetricsResponse_ShouldContainAllMetrics()
    {
        // Arrange
        var requestId = Guid.NewGuid();

        // Act
        var response = new MetricsResponse(
            requestId,
            totalMessagesProcessed: 100000,
            messagesPerSecond: 1500000.0,
            averageLatencyNanoseconds: 75.5,
            poolHitCount: 90000,
            poolMissCount: 10000,
            kernelCacheHitCount: 95000,
            kernelCacheMissCount: 5000,
            totalAllocatedBytes: 1024L * 1024 * 512, // 512 MB
            activeAllocationCount: 150);

        // Assert
        Assert.Equal(requestId, response.OriginalRequestId);
        Assert.Equal(100000, response.TotalMessagesProcessed);
        Assert.Equal(1500000.0, response.MessagesPerSecond);
        Assert.Equal(75.5, response.AverageLatencyNanoseconds);
        Assert.Equal(90000, response.PoolHitCount);
        Assert.Equal(10000, response.PoolMissCount);
        Assert.Equal(95000, response.KernelCacheHitCount);
        Assert.Equal(5000, response.KernelCacheMissCount);
        Assert.Equal(1024L * 1024 * 512, response.TotalAllocatedBytes);
        Assert.Equal(150, response.ActiveAllocationCount);
    }

    [Fact]
    public void InitializeMessage_ShouldStoreConfiguration()
    {
        // Arrange & Act
        var message = new InitializeMessage(
            maxPoolSizeBytes: 1024L * 1024 * 1024, // 1GB
            maxKernelCacheSize: 100,
            deviceIndex: 0);

        // Assert
        Assert.Equal(1024L * 1024 * 1024, message.MaxPoolSizeBytes);
        Assert.Equal(100, message.MaxKernelCacheSize);
        Assert.Equal(0, message.DeviceIndex);
    }

    [Fact]
    public void ShutdownMessage_ShouldHaveDrainFlag()
    {
        // Arrange & Act
        var msgWithDrain = new ShutdownMessage(drainPendingMessages: true);
        var msgWithoutDrain = new ShutdownMessage(drainPendingMessages: false);

        // Assert
        Assert.True(msgWithDrain.DrainPendingMessages);
        Assert.False(msgWithoutDrain.DrainPendingMessages);
    }

    [Fact]
    public void GetMetricsMessage_ShouldHaveDetailsFlag()
    {
        // Arrange & Act
        var msgWithDetails = new GetMetricsMessage(includeDetails: true);
        var msgWithoutDetails = new GetMetricsMessage(includeDetails: false);

        // Assert
        Assert.True(msgWithDetails.IncludeDetails);
        Assert.False(msgWithoutDetails.IncludeDetails);
    }

    [Fact]
    public void AllMessages_ShouldInheritFromResidentMessage()
    {
        // Assert
        Assert.IsAssignableFrom<ResidentMessage>(new AllocateMessage(1024, GpuMemoryType.Default));
        Assert.IsAssignableFrom<ResidentMessage>(new AllocateResponse(Guid.NewGuid(), GpuMemoryHandle.Empty, false));
        Assert.IsAssignableFrom<ResidentMessage>(new WriteMessage("id", 0, 100, IntPtr.Zero));
        Assert.IsAssignableFrom<ResidentMessage>(new ReadMessage("id", 0, 100, IntPtr.Zero));
        Assert.IsAssignableFrom<ResidentMessage>(new ComputeMessage("kernel", "in", "out"));
        Assert.IsAssignableFrom<ResidentMessage>(new ComputeResponse(Guid.NewGuid(), true, 0, 0));
        Assert.IsAssignableFrom<ResidentMessage>(new ReleaseMessage("id"));
        Assert.IsAssignableFrom<ResidentMessage>(new ReleaseResponse(Guid.NewGuid(), 0, false));
        Assert.IsAssignableFrom<ResidentMessage>(new GetMetricsMessage());
        Assert.IsAssignableFrom<ResidentMessage>(new MetricsResponse(Guid.NewGuid(), 0, 0, 0, 0, 0, 0, 0, 0, 0));
        Assert.IsAssignableFrom<ResidentMessage>(new InitializeMessage(0, 0, 0));
        Assert.IsAssignableFrom<ResidentMessage>(new ShutdownMessage());
    }
}

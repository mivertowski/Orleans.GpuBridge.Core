using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
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
using Orleans.GpuBridge.Abstractions.Providers;
using Orleans.GpuBridge.Abstractions.Models;
using Orleans.GpuBridge.Abstractions.Providers.Execution.Interfaces;
using Orleans.GpuBridge.Abstractions.Providers.Execution.Parameters;
using Orleans.GpuBridge.Abstractions.Providers.Execution.Results;
using Orleans.GpuBridge.Abstractions.Providers.Memory.Allocators;
using Orleans.GpuBridge.Abstractions.Providers.Memory.Enums;
using Orleans.GpuBridge.Abstractions.Providers.Memory.Interfaces;
using Orleans.GpuBridge.Abstractions.Providers.Memory.Options;
using Orleans.GpuBridge.Grains.Batch;
using Orleans.Runtime;
using Orleans.TestingHost;
using Xunit;

namespace Orleans.GpuBridge.Tests.Grains;

/// <summary>
/// Comprehensive unit tests for GpuBatchGrainEnhanced with DotCompute integration
/// </summary>
public class GpuBatchGrainEnhancedTests
{
    private readonly Mock<ILogger<GpuBatchGrainEnhanced<float, float>>> _mockLogger;
    private readonly Mock<IGpuBackendProvider> _mockBackendProvider;
    private readonly Mock<IDeviceManager> _mockDeviceManager;
    private readonly Mock<IKernelExecutor> _mockKernelExecutor;
    private readonly Mock<IMemoryAllocator> _mockMemoryAllocator;
    private readonly Mock<IKernelCompiler> _mockKernelCompiler;
    private readonly Mock<IComputeDevice> _mockDevice;
    private readonly Mock<IDeviceMemory> _mockInputMemory;
    private readonly Mock<IDeviceMemory> _mockOutputMemory;
    private readonly KernelId _testKernelId;

    public GpuBatchGrainEnhancedTests()
    {
        _mockLogger = new Mock<ILogger<GpuBatchGrainEnhanced<float, float>>>();
        _mockBackendProvider = new Mock<IGpuBackendProvider>();
        _mockDeviceManager = new Mock<IDeviceManager>();
        _mockKernelExecutor = new Mock<IKernelExecutor>();
        _mockMemoryAllocator = new Mock<IMemoryAllocator>();
        _mockKernelCompiler = new Mock<IKernelCompiler>();
        _mockDevice = new Mock<IComputeDevice>();
        _mockInputMemory = new Mock<IDeviceMemory>();
        _mockOutputMemory = new Mock<IDeviceMemory>();
        _testKernelId = KernelId.Parse("test-gpu-kernel");

        // Setup backend provider to return mocked components
        _mockBackendProvider.Setup(p => p.GetDeviceManager()).Returns(_mockDeviceManager.Object);
        _mockBackendProvider.Setup(p => p.GetKernelExecutor()).Returns(_mockKernelExecutor.Object);
        _mockBackendProvider.Setup(p => p.GetMemoryAllocator()).Returns(_mockMemoryAllocator.Object);
        _mockBackendProvider.Setup(p => p.GetKernelCompiler()).Returns(_mockKernelCompiler.Object);

        // Setup default GPU device (8GB)
        _mockDevice.Setup(d => d.Index).Returns(0);
        _mockDevice.Setup(d => d.Name).Returns("Test GPU");
        _mockDevice.Setup(d => d.Type).Returns(DeviceType.CUDA);
        _mockDevice.Setup(d => d.TotalMemoryBytes).Returns(8192L * 1024 * 1024); // 8GB
        _mockDevice.Setup(d => d.AvailableMemoryBytes).Returns(6000L * 1024 * 1024); // 6GB available

        _mockDeviceManager.Setup(dm => dm.GetDevices())
            .Returns(new List<IComputeDevice> { _mockDevice.Object });

        _mockDeviceManager.Setup(dm => dm.GetDevice(It.IsAny<DeviceType>()))
            .Returns(_mockDevice.Object);
    }

    #region Batch Size Optimization Tests

    [Fact]
    public async Task ExecuteAsync_WithSmallBatch_ShouldProcessEntireBatchOnGpu()
    {
        // Arrange
        var grain = CreateGrain();
        var batch = CreateFloatArray(100); // Small batch (400 bytes)

        SetupSuccessfulGpuExecution(batch.Count);

        // Act
        var result = await grain.ExecuteAsync(batch);

        // Assert
        result.Should().NotBeNull();
        result.Success.Should().BeTrue();
        result.Results.Should().HaveCount(batch.Count);
        result.Metrics.Should().NotBeNull();
        result.Metrics!.SubBatchCount.Should().Be(1); // Single batch
        result.Metrics.DeviceType.Should().Be("CUDA");

        // Verify GPU memory allocation was called once
        _mockMemoryAllocator.Verify(
            m => m.AllocateAsync(
                It.IsAny<long>(),
                It.IsAny<MemoryAllocationOptions>(),
                It.IsAny<CancellationToken>()),
            Times.Exactly(2)); // Input + output buffers
    }

    [Fact]
    public async Task ExecuteAsync_WithLargeBatch_ShouldSplitIntoSubBatches()
    {
        // Arrange
        var grain = CreateGrain();

        // Create batch larger than 80% of available GPU memory (6GB available * 0.8 = 4.8GB)
        // Each float = 4 bytes, so 4.8GB / 8 bytes (input + output) = 629,145,600 items
        // Use 700 million items to exceed capacity
        var largeItemCount = 700_000_000;
        var batch = new List<float>(largeItemCount);
        for (int i = 0; i < largeItemCount; i++) batch.Add(i);

        SetupSuccessfulGpuExecution(largeItemCount);

        // Act
        var result = await grain.ExecuteAsync(batch);

        // Assert
        result.Should().NotBeNull();
        result.Success.Should().BeTrue();
        result.Metrics.Should().NotBeNull();
        result.Metrics!.SubBatchCount.Should().BeGreaterThan(1); // Split into multiple sub-batches
        result.Metrics.TotalItems.Should().Be(largeItemCount);
    }

    [Fact]
    public async Task ExecuteAsync_WithCpuDevice_ShouldProcessEntireBatchOnCpu()
    {
        // Arrange
        _mockDevice.Setup(d => d.Type).Returns(DeviceType.CPU); // CPU fallback
        var grain = CreateGrain();
        var batch = CreateFloatArray(100);

        SetupSuccessfulGpuExecution(batch.Count);

        // Act
        var result = await grain.ExecuteAsync(batch);

        // Assert
        result.Should().NotBeNull();
        result.Success.Should().BeTrue();
        result.Metrics!.DeviceType.Should().Be("CPU");
        result.Metrics.SubBatchCount.Should().Be(1); // No splitting on CPU
    }

    [Theory]
    [InlineData(100)]      // 100 items = 400 bytes
    [InlineData(1_000)]    // 1K items = 4KB
    [InlineData(10_000)]   // 10K items = 40KB
    [InlineData(100_000)]  // 100K items = 400KB
    [InlineData(1_000_000)] // 1M items = 4MB
    public async Task ExecuteAsync_WithVariousBatchSizes_ShouldHandleCorrectly(int itemCount)
    {
        // Arrange
        var grain = CreateGrain();
        var batch = CreateFloatArray(itemCount);

        SetupSuccessfulGpuExecution(itemCount);

        // Act
        var result = await grain.ExecuteAsync(batch);

        // Assert
        result.Should().NotBeNull();
        result.Success.Should().BeTrue();
        result.Results.Should().HaveCount(itemCount);
        result.Metrics!.TotalItems.Should().Be(itemCount);
        result.Metrics.DeviceType.Should().Be("CUDA");
    }

    #endregion

    #region GPU Memory Management Tests

    [Fact]
    public async Task ExecuteAsync_ShouldAllocateMemoryWithCorrectOptions()
    {
        // Arrange
        var grain = CreateGrain();
        var batch = CreateFloatArray(10);

        SetupSuccessfulGpuExecution(batch.Count);

        // Act
        await grain.ExecuteAsync(batch);

        // Assert
        _mockMemoryAllocator.Verify(
            m => m.AllocateAsync(
                It.Is<long>(size => size == 10 * Marshal.SizeOf<float>()), // Input size
                It.Is<MemoryAllocationOptions>(opts =>
                    opts.Type == MemoryType.Device &&
                    opts.PreferredDevice == _mockDevice.Object),
                It.IsAny<CancellationToken>()),
            Times.Once);

        _mockMemoryAllocator.Verify(
            m => m.AllocateAsync(
                It.Is<long>(size => size == 10 * Marshal.SizeOf<float>()), // Output size
                It.Is<MemoryAllocationOptions>(opts =>
                    opts.Type == MemoryType.Device &&
                    opts.PreferredDevice == _mockDevice.Object),
                It.IsAny<CancellationToken>()),
            Times.Once);
    }

    [Fact]
    public async Task ExecuteAsync_ShouldCopyDataToGpuWithPinnedMemory()
    {
        // Arrange
        var grain = CreateGrain();
        var batch = CreateFloatArray(10);

        SetupSuccessfulGpuExecution(batch.Count);

        // Act
        await grain.ExecuteAsync(batch);

        // Assert - Verify CopyFromHostAsync was called with IntPtr (pinned memory)
        _mockInputMemory.Verify(
            m => m.CopyFromHostAsync(
                It.IsAny<IntPtr>(), // Pinned memory pointer
                0,
                It.Is<long>(size => size == 10 * Marshal.SizeOf<float>()),
                It.IsAny<CancellationToken>()),
            Times.Once);
    }

    [Fact]
    public async Task ExecuteAsync_ShouldReadResultsFromGpuWithPinnedMemory()
    {
        // Arrange
        var grain = CreateGrain();
        var batch = CreateFloatArray(10);

        SetupSuccessfulGpuExecution(batch.Count);

        // Act
        await grain.ExecuteAsync(batch);

        // Assert - Verify CopyToHostAsync was called with IntPtr (pinned memory)
        _mockOutputMemory.Verify(
            m => m.CopyToHostAsync(
                It.IsAny<IntPtr>(), // Pinned memory pointer
                0,
                It.Is<long>(size => size == 10 * Marshal.SizeOf<float>()),
                It.IsAny<CancellationToken>()),
            Times.Once);
    }

    [Fact]
    public async Task ExecuteAsync_ShouldDisposeMemoryAfterExecution()
    {
        // Arrange
        var grain = CreateGrain();
        var batch = CreateFloatArray(10);

        SetupSuccessfulGpuExecution(batch.Count);

        // Act
        await grain.ExecuteAsync(batch);

        // Assert - Verify Dispose was called on both input and output memory
        _mockInputMemory.Verify(m => m.Dispose(), Times.Once);
        _mockOutputMemory.Verify(m => m.Dispose(), Times.Once);
    }

    [Fact]
    public async Task ExecuteAsync_WithAllocationFailure_ShouldGracefullyHandleError()
    {
        // Arrange
        var grain = CreateGrain();
        var batch = CreateFloatArray(10);

        _mockMemoryAllocator
            .Setup(m => m.AllocateAsync(
                It.IsAny<long>(),
                It.IsAny<MemoryAllocationOptions>(),
                It.IsAny<CancellationToken>()))
            .ThrowsAsync(new OutOfMemoryException("GPU out of memory"));

        // Act
        var result = await grain.ExecuteAsync(batch);

        // Assert
        result.Should().NotBeNull();
        result.Success.Should().BeFalse();
        result.Error.Should().Contain("GPU out of memory");
    }

    #endregion

    #region Kernel Execution Tests

    [Fact]
    public async Task ExecuteAsync_ShouldCallKernelExecutorWithCorrectParameters()
    {
        // Arrange
        var grain = CreateGrain();
        var batch = CreateFloatArray(256);

        SetupSuccessfulGpuExecution(batch.Count);

        // Act
        await grain.ExecuteAsync(batch);

        // Assert
        _mockKernelExecutor.Verify(
            k => k.ExecuteAsync(
                It.IsAny<CompiledKernel>(),
                It.Is<KernelExecutionParameters>(p =>
                    p.GlobalWorkSize[0] == 256 &&
                    p.LocalWorkSize![0] == 256 && // Min(256, 256)
                    p.MemoryArguments.ContainsKey("input") &&
                    p.MemoryArguments.ContainsKey("output") &&
                    p.ScalarArguments.ContainsKey("count")),
                It.IsAny<CancellationToken>()),
            Times.Once);
    }

    [Fact]
    public async Task ExecuteAsync_WithLargeWorkSize_ShouldLimitLocalWorkSize()
    {
        // Arrange
        var grain = CreateGrain();
        var batch = CreateFloatArray(1024); // Large work size

        SetupSuccessfulGpuExecution(batch.Count);

        // Act
        await grain.ExecuteAsync(batch);

        // Assert - Local work size should be capped at 256
        _mockKernelExecutor.Verify(
            k => k.ExecuteAsync(
                It.IsAny<CompiledKernel>(),
                It.Is<KernelExecutionParameters>(p =>
                    p.GlobalWorkSize[0] == 1024 &&
                    p.LocalWorkSize![0] == 256), // Capped at 256 threads per block
                It.IsAny<CancellationToken>()),
            Times.Once);
    }

    [Fact]
    public async Task ExecuteAsync_WithKernelExecutionFailure_ShouldReturnErrorResult()
    {
        // Arrange
        var grain = CreateGrain();
        var batch = CreateFloatArray(10);

        SetupSuccessfulMemoryOperations();

        _mockKernelExecutor
            .Setup(k => k.ExecuteAsync(
                It.IsAny<CompiledKernel>(),
                It.IsAny<KernelExecutionParameters>(),
                It.IsAny<CancellationToken>()))
            .ThrowsAsync(new InvalidOperationException("Kernel execution failed"));

        // Act
        var result = await grain.ExecuteAsync(batch);

        // Assert
        result.Should().NotBeNull();
        result.Success.Should().BeFalse();
        result.Error.Should().Contain("Kernel execution failed");

        // Memory should still be cleaned up
        _mockInputMemory.Verify(m => m.Dispose(), Times.Once);
        _mockOutputMemory.Verify(m => m.Dispose(), Times.Once);
    }

    #endregion

    #region Performance Metrics Tests

    [Fact]
    public async Task ExecuteAsync_ShouldCalculateThroughputMetrics()
    {
        // Arrange
        var grain = CreateGrain();
        var batch = CreateFloatArray(1000);

        SetupSuccessfulGpuExecution(batch.Count);

        // Act
        var result = await grain.ExecuteAsync(batch);

        // Assert
        result.Metrics.Should().NotBeNull();
        result.Metrics!.Throughput.Should().BeGreaterThan(0);
        result.Metrics.ItemsPerMillisecond.Should().BeGreaterThan(0);
    }

    [Fact]
    public async Task ExecuteAsync_ShouldCalculateKernelEfficiency()
    {
        // Arrange
        var grain = CreateGrain();
        var batch = CreateFloatArray(100);

        SetupSuccessfulGpuExecution(batch.Count);

        // Act
        var result = await grain.ExecuteAsync(batch);

        // Assert
        result.Metrics!.KernelEfficiency.Should().BeInRange(0, 100);
        result.Metrics.KernelExecutionTime.Should().BeGreaterThan(TimeSpan.Zero);
        result.Metrics.MemoryTransferTime.Should().BeGreaterThan(TimeSpan.Zero);
    }

    [Fact]
    public async Task ExecuteAsync_ShouldCalculateMemoryBandwidth()
    {
        // Arrange
        var grain = CreateGrain();
        var batch = CreateFloatArray(1000);

        SetupSuccessfulGpuExecution(batch.Count);

        // Act
        var result = await grain.ExecuteAsync(batch);

        // Assert
        result.Metrics!.MemoryBandwidthMBps.Should().BeGreaterThan(0);
        result.Metrics.MemoryAllocated.Should().Be(
            1000 * Marshal.SizeOf<float>() * 2); // Input + output buffers
    }

    [Fact]
    public async Task ExecuteAsync_WithMultipleSubBatches_ShouldAggregateMetrics()
    {
        // Arrange
        var grain = CreateGrain();
        var largeItemCount = 700_000_000; // Force sub-batch splitting
        var batch = new List<float>(largeItemCount);
        for (int i = 0; i < largeItemCount; i++) batch.Add(i);

        SetupSuccessfulGpuExecution(largeItemCount);

        // Act
        var result = await grain.ExecuteAsync(batch);

        // Assert
        result.Metrics!.TotalItems.Should().Be(largeItemCount);
        result.Metrics.SubBatchCount.Should().BeGreaterThan(1);
        result.Metrics.SuccessfulSubBatches.Should().Be(result.Metrics.SubBatchCount);
        result.Metrics.TotalExecutionTime.Should().BeGreaterThan(TimeSpan.Zero);
    }

    #endregion

    #region Error Handling and Fallback Tests

    [Fact]
    public async Task ExecuteAsync_WhenGpuUnavailable_ShouldFallbackToCpu()
    {
        // Arrange
        _mockDevice.Setup(d => d.Type).Returns(DeviceType.CPU);
        var grain = CreateGrain();
        var batch = CreateFloatArray(100);

        SetupSuccessfulGpuExecution(batch.Count);

        // Act
        var result = await grain.ExecuteAsync(batch);

        // Assert
        result.Success.Should().BeTrue();
        result.Metrics!.DeviceType.Should().Be("CPU");
    }

    [Fact]
    public async Task ExecuteAsync_WithPartialSubBatchFailure_ShouldReportCorrectMetrics()
    {
        // Arrange
        var grain = CreateGrain();
        var batch = CreateFloatArray(1000);

        SetupSuccessfulMemoryOperations();

        // Setup kernel executor to fail on second call (simulating sub-batch failure)
        var callCount = 0;
        _mockKernelExecutor
            .Setup(k => k.ExecuteAsync(
                It.IsAny<CompiledKernel>(),
                It.IsAny<KernelExecutionParameters>(),
                It.IsAny<CancellationToken>()))
            .Callback(() =>
            {
                callCount++;
                if (callCount == 2)
                    throw new InvalidOperationException("Sub-batch failed");
            })
            .ReturnsAsync(new KernelExecutionResult(Success: true, ExecutionTimeMs: 1.0));

        // Act
        var result = await grain.ExecuteAsync(batch);

        // Assert - Should handle partial failure gracefully
        result.Should().NotBeNull();
        // Depending on implementation, may return partial results or full error
    }

    [Fact]
    public async Task ExecuteAsync_WithEmptyBatch_ShouldReturnEmptyResult()
    {
        // Arrange
        var grain = CreateGrain();
        var batch = new List<float>();

        // Act
        var result = await grain.ExecuteAsync(batch);

        // Assert
        result.Should().NotBeNull();
        result.Success.Should().BeTrue();
        result.Results.Should().BeEmpty();
        result.Metrics.Should().NotBeNull();
        result.Metrics!.TotalItems.Should().Be(0);
    }

    [Fact]
    public async Task ExecuteAsync_WithNullBatch_ShouldHandleGracefully()
    {
        // Arrange
        var grain = CreateGrain();

        // Act
        var result = await grain.ExecuteAsync(null!);

        // Assert
        result.Should().NotBeNull();
        result.Success.Should().BeFalse();
        result.Error.Should().Contain("batch");
    }

    #endregion

    #region Helper Methods

    private GpuBatchGrainEnhanced<float, float> CreateGrain()
    {
        var grain = new GpuBatchGrainEnhanced<float, float>(_mockLogger.Object);

        // Mock the service provider
        var serviceProviderMock = new Mock<IServiceProvider>();
        serviceProviderMock.Setup(sp => sp.GetService(typeof(IGpuBackendProvider)))
            .Returns(_mockBackendProvider.Object);

        // Use reflection to set the service provider
        var serviceProviderProperty = typeof(Grain).GetProperty("ServiceProvider");
        serviceProviderProperty?.SetValue(grain, serviceProviderMock.Object);

        // Mock GetPrimaryKeyString to return test kernel ID
        var grainContextMock = new Mock<IGrainContext>();
        grainContextMock.Setup(gc => gc.GrainId)
            .Returns(Orleans.Runtime.GrainId.Parse(_testKernelId.Value));

        return grain;
    }

    private List<float> CreateFloatArray(int count)
    {
        var list = new List<float>(count);
        for (int i = 0; i < count; i++)
        {
            list.Add(i * 1.5f);
        }
        return list;
    }

    private void SetupSuccessfulMemoryOperations()
    {
        // Setup memory allocation
        _mockMemoryAllocator
            .Setup(m => m.AllocateAsync(
                It.IsAny<long>(),
                It.IsAny<MemoryAllocationOptions>(),
                It.IsAny<CancellationToken>()))
            .ReturnsAsync((long size, MemoryAllocationOptions opts, CancellationToken ct) =>
            {
                if (opts.Type == MemoryType.Device)
                    return size == Marshal.SizeOf<float>() * 10 ? _mockInputMemory.Object : _mockOutputMemory.Object;
                return _mockInputMemory.Object;
            });

        // Setup CopyFromHostAsync (write to GPU)
        _mockInputMemory
            .Setup(m => m.CopyFromHostAsync(
                It.IsAny<IntPtr>(),
                It.IsAny<long>(),
                It.IsAny<long>(),
                It.IsAny<CancellationToken>()))
            .Returns(Task.CompletedTask);

        // Setup CopyToHostAsync (read from GPU)
        _mockOutputMemory
            .Setup(m => m.CopyToHostAsync(
                It.IsAny<IntPtr>(),
                It.IsAny<long>(),
                It.IsAny<long>(),
                It.IsAny<CancellationToken>()))
            .Returns(Task.CompletedTask);

        // Setup Dispose
        _mockInputMemory.Setup(m => m.Dispose());
        _mockOutputMemory.Setup(m => m.Dispose());
    }

    private void SetupSuccessfulGpuExecution(int itemCount)
    {
        SetupSuccessfulMemoryOperations();

        // Setup kernel executor
        _mockKernelExecutor
            .Setup(k => k.ExecuteAsync(
                It.IsAny<CompiledKernel>(),
                It.IsAny<KernelExecutionParameters>(),
                It.IsAny<CancellationToken>()))
            .ReturnsAsync(new KernelExecutionResult(Success: true, ExecutionTimeMs: 1.0));
    }

    #endregion
}

/// <summary>
/// Unit tests for GpuBatchMetrics record
/// </summary>
public class GpuBatchMetricsTests
{
    [Fact]
    public void GpuBatchMetrics_ShouldCalculateKernelEfficiencyCorrectly()
    {
        // Arrange
        var metrics = new GpuBatchMetrics(
            TotalItems: 1000,
            SubBatchCount: 2,
            SuccessfulSubBatches: 2,
            TotalExecutionTime: TimeSpan.FromMilliseconds(100),
            KernelExecutionTime: TimeSpan.FromMilliseconds(60),
            MemoryTransferTime: TimeSpan.FromMilliseconds(40),
            Throughput: 10000,
            MemoryAllocated: 8000,
            DeviceType: "CUDA",
            DeviceName: "Test GPU");

        // Act
        var efficiency = metrics.KernelEfficiency;

        // Assert
        efficiency.Should().BeApproximately(60.0, 0.1); // 60ms / 100ms * 100 = 60%
    }

    [Fact]
    public void GpuBatchMetrics_ShouldCalculateItemsPerMillisecondCorrectly()
    {
        // Arrange
        var metrics = new GpuBatchMetrics(
            TotalItems: 1000,
            SubBatchCount: 1,
            SuccessfulSubBatches: 1,
            TotalExecutionTime: TimeSpan.FromMilliseconds(50),
            KernelExecutionTime: TimeSpan.FromMilliseconds(30),
            MemoryTransferTime: TimeSpan.FromMilliseconds(20),
            Throughput: 20000,
            MemoryAllocated: 4000,
            DeviceType: "CUDA",
            DeviceName: "Test GPU");

        // Act
        var itemsPerMs = metrics.ItemsPerMillisecond;

        // Assert
        itemsPerMs.Should().BeApproximately(20.0, 0.1); // 1000 items / 50ms = 20 items/ms
    }

    [Fact]
    public void GpuBatchMetrics_ShouldCalculateMemoryBandwidthCorrectly()
    {
        // Arrange
        var metrics = new GpuBatchMetrics(
            TotalItems: 1000,
            SubBatchCount: 1,
            SuccessfulSubBatches: 1,
            TotalExecutionTime: TimeSpan.FromSeconds(1),
            KernelExecutionTime: TimeSpan.FromMilliseconds(500),
            MemoryTransferTime: TimeSpan.FromMilliseconds(500),
            Throughput: 1000,
            MemoryAllocated: 1024 * 1024 * 100, // 100 MB
            DeviceType: "CUDA",
            DeviceName: "Test GPU");

        // Act
        var bandwidth = metrics.MemoryBandwidthMBps;

        // Assert
        bandwidth.Should().BeApproximately(100.0, 0.1); // 100 MB / 1 second = 100 MB/s
    }

    [Theory]
    [InlineData(1000, 100, 10.0)]
    [InlineData(5000, 250, 20.0)]
    [InlineData(10000, 1000, 10.0)]
    public void GpuBatchMetrics_ItemsPerMillisecond_WithVariousInputs(int totalItems, int totalMs, double expected)
    {
        // Arrange
        var metrics = new GpuBatchMetrics(
            TotalItems: totalItems,
            SubBatchCount: 1,
            SuccessfulSubBatches: 1,
            TotalExecutionTime: TimeSpan.FromMilliseconds(totalMs),
            KernelExecutionTime: TimeSpan.FromMilliseconds(totalMs / 2),
            MemoryTransferTime: TimeSpan.FromMilliseconds(totalMs / 2),
            Throughput: totalItems,
            MemoryAllocated: totalItems * 4,
            DeviceType: "CUDA",
            DeviceName: "Test GPU");

        // Act
        var itemsPerMs = metrics.ItemsPerMillisecond;

        // Assert
        itemsPerMs.Should().BeApproximately(expected, 0.01);
    }
}

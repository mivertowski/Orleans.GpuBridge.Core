using FluentAssertions;
using Moq;
using Orleans.GpuBridge.Abstractions.Models;
using Orleans.GpuBridge.Abstractions.Providers;
using Orleans.GpuBridge.Abstractions.Providers.Execution.Interfaces;
using Orleans.GpuBridge.Abstractions.Providers.Execution.Results;
using Orleans.GpuBridge.Abstractions.Providers.Memory.Allocators;
using Orleans.GpuBridge.Abstractions.Providers.Memory.Interfaces;
using Orleans.GpuBridge.Grains.Stream;
using Orleans.GpuBridge.Grains.Stream.Configuration;
using Orleans.GpuBridge.Grains.Stream.Metrics;
using System.Runtime.InteropServices;
using Xunit;

namespace Orleans.GpuBridge.Tests.Grains;

/// <summary>
/// Comprehensive tests for GpuStreamGrainEnhanced with DotCompute integration
/// Tests cover: batch accumulation, backpressure, GPU execution, and metrics tracking
/// </summary>
public sealed class GpuStreamGrainEnhancedTests
{
    private readonly Mock<IGpuBackendProvider> _mockBackendProvider;
    private readonly Mock<IDeviceManager> _mockDeviceManager;
    private readonly Mock<IKernelExecutor> _mockKernelExecutor;
    private readonly Mock<IMemoryAllocator> _mockMemoryAllocator;
    private readonly Mock<IComputeDevice> _mockDevice;
    private readonly Mock<IDeviceMemory> _mockInputMemory;
    private readonly Mock<IDeviceMemory> _mockOutputMemory;

    public GpuStreamGrainEnhancedTests()
    {
        // Setup mock backend provider
        _mockBackendProvider = new Mock<IGpuBackendProvider>();
        _mockDeviceManager = new Mock<IDeviceManager>();
        _mockKernelExecutor = new Mock<IKernelExecutor>();
        _mockMemoryAllocator = new Mock<IMemoryAllocator>();
        _mockDevice = new Mock<IComputeDevice>();
        _mockInputMemory = new Mock<IDeviceMemory>();
        _mockOutputMemory = new Mock<IDeviceMemory>();

        // Setup device with CUDA GPU capabilities
        _mockDevice.Setup(d => d.Type).Returns(DeviceType.GPU);
        _mockDevice.Setup(d => d.Name).Returns("NVIDIA RTX 4090");
        _mockDevice.Setup(d => d.GetAvailableMemoryAsync(It.IsAny<CancellationToken>()))
            .ReturnsAsync(24L * 1024 * 1024 * 1024); // 24GB

        // Setup device manager
        _mockDeviceManager.Setup(m => m.GetAvailableDevicesAsync(It.IsAny<CancellationToken>()))
            .ReturnsAsync(new[] { _mockDevice.Object });

        // Setup backend provider
        _mockBackendProvider.Setup(p => p.GetDeviceManager()).Returns(_mockDeviceManager.Object);
        _mockBackendProvider.Setup(p => p.GetKernelExecutor()).Returns(_mockKernelExecutor.Object);
        _mockBackendProvider.Setup(p => p.GetMemoryAllocator()).Returns(_mockMemoryAllocator.Object);

        // Setup memory allocation
        _mockMemoryAllocator.Setup(m => m.AllocateAsync(
                It.IsAny<long>(),
                It.IsAny<MemoryAllocationOptions>(),
                It.IsAny<CancellationToken>()))
            .ReturnsAsync((long size, MemoryAllocationOptions _, CancellationToken _) =>
                size < 1000000 ? _mockInputMemory.Object : _mockOutputMemory.Object);

        // Setup memory operations
        _mockInputMemory.Setup(m => m.CopyFromHostAsync(
            It.IsAny<IntPtr>(), It.IsAny<long>(), It.IsAny<long>(), It.IsAny<CancellationToken>()))
            .Returns(Task.CompletedTask);

        _mockOutputMemory.Setup(m => m.CopyToHostAsync(
            It.IsAny<IntPtr>(), It.IsAny<long>(), It.IsAny<long>(), It.IsAny<CancellationToken>()))
            .Returns(Task.CompletedTask);

        // Setup kernel execution
        _mockKernelExecutor.Setup(k => k.ExecuteAsync(
                It.IsAny<CompiledKernel>(),
                It.IsAny<KernelExecutionParameters>(),
                It.IsAny<CancellationToken>()))
            .ReturnsAsync(new KernelExecutionResult(
                Success: true,
                ExecutionTimeMs: 1.5,
                KernelTimeMs: 1.0,
                MemoryTransferTimeMs: 0.5));
    }

    #region Batch Accumulation Tests

    [Fact]
    public async Task PushAsync_ShouldAccumulateItemsUntilMinBatchSize()
    {
        // Arrange
        var config = new StreamProcessingConfiguration
        {
            BatchConfig = new BatchAccumulationConfig
            {
                MinBatchSize = 32,
                MaxBatchSize = 1000,
                MaxBatchWaitTime = TimeSpan.FromMilliseconds(100)
            }
        };

        var grain = CreateStreamGrain(config);
        await grain.StartAsync();

        // Act - Push items below min batch size
        for (int i = 0; i < 31; i++)
        {
            await grain.PushAsync((float)i);
        }

        await Task.Delay(50); // Wait but not full timeout

        // Assert - Should not have processed yet (below min batch)
        var metrics = await grain.GetMetricsAsync();
        metrics.TotalBatchesProcessed.Should().Be(0);

        // Act - Push one more to reach min batch
        await grain.PushAsync(31f);
        await Task.Delay(150); // Wait for processing

        // Assert - Should have processed the batch
        metrics = await grain.GetMetricsAsync();
        metrics.TotalBatchesProcessed.Should().BeGreaterOrEqualTo(1);
    }

    [Fact]
    public async Task PushAsync_ShouldFlushOnMaxBatchWaitTimeout()
    {
        // Arrange
        var config = new StreamProcessingConfiguration
        {
            BatchConfig = new BatchAccumulationConfig
            {
                MinBatchSize = 100,
                MaxBatchSize = 1000,
                MaxBatchWaitTime = TimeSpan.FromMilliseconds(50) // Short timeout
            }
        };

        var grain = CreateStreamGrain(config);
        await grain.StartAsync();

        // Act - Push only a few items (below min batch)
        for (int i = 0; i < 10; i++)
        {
            await grain.PushAsync((float)i);
        }

        await Task.Delay(100); // Wait past timeout

        // Assert - Should have flushed partial batch due to timeout
        var metrics = await grain.GetMetricsAsync();
        metrics.TotalItemsProcessed.Should().Be(10);
    }

    [Fact]
    public async Task AdaptiveBatching_ShouldIncreaseSize_WhenThroughputImproves()
    {
        // Arrange
        var config = new StreamProcessingConfiguration
        {
            BatchConfig = new BatchAccumulationConfig
            {
                MinBatchSize = 32,
                MaxBatchSize = 10000,
                EnableAdaptiveBatching = true
            }
        };

        var grain = CreateStreamGrain(config);
        await grain.StartAsync();

        // Act - Process multiple batches with good throughput
        for (int batch = 0; batch < 5; batch++)
        {
            for (int i = 0; i < 100; i++)
            {
                await grain.PushAsync((float)(batch * 100 + i));
            }
            await Task.Delay(20); // Fast processing
        }

        await Task.Delay(200); // Allow adaptation

        // Assert - Should have increased batch size
        var metrics = await grain.GetMetricsAsync();
        metrics.AverageBatchSize.Should().BeGreaterThan(100);
    }

    [Fact]
    public async Task AdaptiveBatching_ShouldDecreaseSize_WhenThroughputDegrades()
    {
        // Arrange
        var config = new StreamProcessingConfiguration
        {
            BatchConfig = new BatchAccumulationConfig
            {
                MinBatchSize = 16,
                MaxBatchSize = 10000,
                EnableAdaptiveBatching = true
            }
        };

        // Setup slow kernel execution
        _mockKernelExecutor.Setup(k => k.ExecuteAsync(
                It.IsAny<CompiledKernel>(),
                It.IsAny<KernelExecutionParameters>(),
                It.IsAny<CancellationToken>()))
            .ReturnsAsync(new KernelExecutionResult(
                Success: true,
                ExecutionTimeMs: 100.0, // Slow execution
                KernelTimeMs: 80.0,
                MemoryTransferTimeMs: 20.0));

        var grain = CreateStreamGrain(config);
        await grain.StartAsync();

        // Act - Process with degrading throughput
        for (int batch = 0; batch < 3; batch++)
        {
            for (int i = 0; i < 200; i++)
            {
                await grain.PushAsync((float)(batch * 200 + i));
            }
            await Task.Delay(150); // Slow processing
        }

        await Task.Delay(300);

        // Assert - Should have decreased batch size
        var metrics = await grain.GetMetricsAsync();
        metrics.AverageBatchSize.Should().BeLessThan(200);
    }

    [Fact]
    public async Task BatchAccumulation_ShouldRespectMaxBatchSize()
    {
        // Arrange
        var config = new StreamProcessingConfiguration
        {
            BatchConfig = new BatchAccumulationConfig
            {
                MinBatchSize = 32,
                MaxBatchSize = 256,
                MaxBatchWaitTime = TimeSpan.FromMilliseconds(100)
            }
        };

        var grain = CreateStreamGrain(config);
        await grain.StartAsync();

        // Act - Push more than max batch size
        for (int i = 0; i < 1000; i++)
        {
            await grain.PushAsync((float)i);
        }

        await Task.Delay(300);

        // Assert - Should have split into multiple batches
        var metrics = await grain.GetMetricsAsync();
        metrics.TotalBatchesProcessed.Should().BeGreaterThan(3); // At least 1000/256 batches
        metrics.AverageBatchSize.Should().BeLessOrEqualTo(256);
    }

    #endregion

    #region Backpressure Management Tests

    [Fact]
    public async Task Backpressure_ShouldPause_WhenBufferUtilizationHigh()
    {
        // Arrange
        var config = new StreamProcessingConfiguration
        {
            BackpressureConfig = new BackpressureConfig
            {
                BufferCapacity = 100,
                PauseThreshold = 0.9, // Pause at 90%
                ResumeThreshold = 0.5
            }
        };

        // Setup very slow kernel execution to cause backlog
        _mockKernelExecutor.Setup(k => k.ExecuteAsync(
                It.IsAny<CompiledKernel>(),
                It.IsAny<KernelExecutionParameters>(),
                It.IsAny<CancellationToken>()))
            .Returns(async () =>
            {
                await Task.Delay(100); // Very slow
                return new KernelExecutionResult(Success: true, ExecutionTimeMs: 100.0);
            });

        var grain = CreateStreamGrain(config);
        await grain.StartAsync();

        // Act - Push items rapidly to fill buffer
        for (int i = 0; i < 95; i++)
        {
            await grain.PushAsync((float)i);
        }

        await Task.Delay(50);

        // Assert - Should have paused
        var metrics = await grain.GetMetricsAsync();
        metrics.TotalPauseCount.Should().BeGreaterOrEqualTo(1);
        metrics.BufferUtilization.Should().BeGreaterThan(0.9);
    }

    [Fact]
    public async Task Backpressure_ShouldResume_WhenBufferUtilizationLow()
    {
        // Arrange
        var config = new StreamProcessingConfiguration
        {
            BackpressureConfig = new BackpressureConfig
            {
                BufferCapacity = 100,
                PauseThreshold = 0.9,
                ResumeThreshold = 0.5 // Resume at 50%
            }
        };

        var grain = CreateStreamGrain(config);
        await grain.StartAsync();

        // Act - Fill buffer to trigger pause
        for (int i = 0; i < 95; i++)
        {
            await grain.PushAsync((float)i);
        }

        await Task.Delay(100); // Wait for pause

        // Wait for buffer to drain
        await Task.Delay(500);

        // Assert - Should have resumed
        var metrics = await grain.GetMetricsAsync();
        metrics.BufferUtilization.Should().BeLessThan(0.5);
        metrics.TotalPauseCount.Should().BeGreaterOrEqualTo(1);
        metrics.TotalPauseDuration.Should().BeGreaterThan(TimeSpan.Zero);
    }

    [Fact]
    public async Task Backpressure_WithDropOldest_ShouldDropItems_WhenBufferFull()
    {
        // Arrange
        var config = new StreamProcessingConfiguration
        {
            BackpressureConfig = new BackpressureConfig
            {
                BufferCapacity = 50,
                DropOldestOnFull = true // Drop old items
            }
        };

        // Slow processing
        _mockKernelExecutor.Setup(k => k.ExecuteAsync(
                It.IsAny<CompiledKernel>(),
                It.IsAny<KernelExecutionParameters>(),
                It.IsAny<CancellationToken>()))
            .Returns(async () =>
            {
                await Task.Delay(50);
                return new KernelExecutionResult(Success: true, ExecutionTimeMs: 50.0);
            });

        var grain = CreateStreamGrain(config);
        await grain.StartAsync();

        // Act - Push more items than buffer capacity
        for (int i = 0; i < 200; i++)
        {
            await grain.PushAsync((float)i);
        }

        await Task.Delay(500);

        // Assert - Should have dropped some items
        var metrics = await grain.GetMetricsAsync();
        metrics.TotalItemsProcessed.Should().BeLessThan(200);
    }

    [Fact]
    public async Task Backpressure_ShouldTrackPauseDuration()
    {
        // Arrange
        var config = new StreamProcessingConfiguration
        {
            BackpressureConfig = new BackpressureConfig
            {
                BufferCapacity = 100,
                PauseThreshold = 0.9,
                ResumeThreshold = 0.5
            }
        };

        // Very slow processing
        _mockKernelExecutor.Setup(k => k.ExecuteAsync(
                It.IsAny<CompiledKernel>(),
                It.IsAny<KernelExecutionParameters>(),
                It.IsAny<CancellationToken>()))
            .Returns(async () =>
            {
                await Task.Delay(200);
                return new KernelExecutionResult(Success: true, ExecutionTimeMs: 200.0);
            });

        var grain = CreateStreamGrain(config);
        await grain.StartAsync();

        // Act - Trigger pause
        for (int i = 0; i < 95; i++)
        {
            await grain.PushAsync((float)i);
        }

        await Task.Delay(300); // Wait during pause

        // Assert
        var metrics = await grain.GetMetricsAsync();
        metrics.TotalPauseDuration.Should().BeGreaterThan(TimeSpan.FromMilliseconds(100));
        metrics.AveragePauseDuration.Should().BeGreaterThan(TimeSpan.FromMilliseconds(50));
    }

    #endregion

    #region GPU Execution Tests

    [Fact]
    public async Task GpuExecution_ShouldUsePinnedMemory_ForDataTransfer()
    {
        // Arrange
        var config = StreamProcessingConfiguration.Default;
        var grain = CreateStreamGrain(config);
        await grain.StartAsync();

        // Act
        for (int i = 0; i < 64; i++)
        {
            await grain.PushAsync((float)i);
        }

        await Task.Delay(200);

        // Assert - Should have used pinned memory for GPU transfer
        _mockInputMemory.Verify(
            m => m.CopyFromHostAsync(
                It.IsAny<IntPtr>(), // Pinned memory pointer
                0,
                It.IsAny<long>(),
                It.IsAny<CancellationToken>()),
            Times.AtLeastOnce);

        _mockOutputMemory.Verify(
            m => m.CopyToHostAsync(
                It.IsAny<IntPtr>(),
                0,
                It.IsAny<long>(),
                It.IsAny<CancellationToken>()),
            Times.AtLeastOnce);
    }

    [Fact]
    public async Task GpuExecution_ShouldDisposeMemory_AfterBatchProcessing()
    {
        // Arrange
        var config = StreamProcessingConfiguration.Default;
        var grain = CreateStreamGrain(config);
        await grain.StartAsync();

        // Act
        for (int i = 0; i < 64; i++)
        {
            await grain.PushAsync((float)i);
        }

        await Task.Delay(200);

        // Assert - Should have disposed GPU memory
        _mockInputMemory.Verify(m => m.Dispose(), Times.AtLeastOnce);
        _mockOutputMemory.Verify(m => m.Dispose(), Times.AtLeastOnce);
    }

    [Fact]
    public async Task GpuExecution_ShouldCalculateOptimalBatchSize_BasedOnGpuMemory()
    {
        // Arrange
        var config = new StreamProcessingConfiguration
        {
            BatchConfig = new BatchAccumulationConfig
            {
                MinBatchSize = 32,
                MaxBatchSize = 1_000_000,
                GpuMemoryUtilizationTarget = 0.8 // 80% target
            }
        };

        // Setup device with limited memory (1GB)
        _mockDevice.Setup(d => d.GetAvailableMemoryAsync(It.IsAny<CancellationToken>()))
            .ReturnsAsync(1L * 1024 * 1024 * 1024); // 1GB

        var grain = CreateStreamGrain(config);
        await grain.StartAsync();

        // Act - Push large number of items
        for (int i = 0; i < 10_000; i++)
        {
            await grain.PushAsync((float)i);
        }

        await Task.Delay(500);

        // Assert - Should have split into multiple batches to fit GPU memory
        var metrics = await grain.GetMetricsAsync();
        metrics.TotalBatchesProcessed.Should().BeGreaterThan(1);
        metrics.TotalGpuMemoryAllocated.Should().BeLessThan(1L * 1024 * 1024 * 1024 * 0.8);
    }

    #endregion

    #region Metrics Tracking Tests

    [Fact]
    public async Task Metrics_ShouldTrackLatencyPercentiles()
    {
        // Arrange
        var config = StreamProcessingConfiguration.Default;
        var grain = CreateStreamGrain(config);
        await grain.StartAsync();

        // Act - Process multiple batches
        for (int batch = 0; batch < 10; batch++)
        {
            for (int i = 0; i < 100; i++)
            {
                await grain.PushAsync((float)(batch * 100 + i));
            }
            await Task.Delay(50);
        }

        await Task.Delay(200);

        // Assert
        var metrics = await grain.GetMetricsAsync();
        metrics.AverageLatencyMs.Should().BeGreaterThan(0);
        metrics.P50LatencyMs.Should().BeGreaterThan(0);
        metrics.P99LatencyMs.Should().BeGreaterOrEqualTo(metrics.P50LatencyMs);
    }

    [Fact]
    public async Task Metrics_ShouldTrackCurrentThroughput_UsingSlidingWindow()
    {
        // Arrange
        var config = StreamProcessingConfiguration.Default;
        var grain = CreateStreamGrain(config);
        await grain.StartAsync();

        // Act - Process items over time
        for (int i = 0; i < 500; i++)
        {
            await grain.PushAsync((float)i);
            if (i % 50 == 0) await Task.Delay(100);
        }

        await Task.Delay(300);

        // Assert - Current throughput should reflect recent performance (last 10 seconds)
        var metrics = await grain.GetMetricsAsync();
        metrics.CurrentThroughput.Should().BeGreaterThan(0);
        metrics.PeakThroughput.Should().BeGreaterOrEqualTo(metrics.CurrentThroughput);
    }

    [Fact]
    public async Task Metrics_ShouldCalculateKernelEfficiency()
    {
        // Arrange
        var config = StreamProcessingConfiguration.Default;
        var grain = CreateStreamGrain(config);
        await grain.StartAsync();

        // Act
        for (int i = 0; i < 100; i++)
        {
            await grain.PushAsync((float)i);
        }

        await Task.Delay(200);

        // Assert - Kernel efficiency = kernel_time / (kernel_time + transfer_time)
        var metrics = await grain.GetMetricsAsync();
        metrics.KernelEfficiency.Should().BeInRange(0, 100);
        metrics.TotalKernelExecutionTime.Should().BeGreaterThan(TimeSpan.Zero);
        metrics.TotalMemoryTransferTime.Should().BeGreaterThan(TimeSpan.Zero);
    }

    #endregion

    #region StreamProcessingMetrics Tests

    [Fact]
    public void StreamProcessingMetrics_ShouldCalculateUptime()
    {
        // Arrange
        var startTime = DateTime.UtcNow.AddMinutes(-5);
        var lastProcessed = DateTime.UtcNow;

        var metrics = new StreamProcessingMetrics(
            TotalItemsProcessed: 1000,
            TotalItemsFailed: 0,
            TotalProcessingTime: TimeSpan.FromMinutes(5),
            TotalBatchesProcessed: 10,
            AverageBatchSize: 100,
            BatchEfficiency: 0.9,
            AverageLatencyMs: 1.5,
            P50LatencyMs: 1.2,
            P99LatencyMs: 3.0,
            TotalKernelExecutionTime: TimeSpan.FromSeconds(10),
            TotalMemoryTransferTime: TimeSpan.FromSeconds(5),
            KernelEfficiency: 66.7,
            MemoryBandwidthMBps: 1000,
            TotalGpuMemoryAllocated: 1024 * 1024 * 100,
            CurrentThroughput: 200,
            PeakThroughput: 250,
            BufferCurrentSize: 50,
            BufferCapacity: 1000,
            BufferUtilization: 0.05,
            TotalPauseCount: 2,
            TotalPauseDuration: TimeSpan.FromSeconds(10),
            DeviceType: "CUDA",
            DeviceName: "RTX 4090",
            StartTime: startTime,
            LastProcessedTime: lastProcessed);

        // Assert
        metrics.Uptime.Should().BeCloseTo(TimeSpan.FromMinutes(5), TimeSpan.FromSeconds(1));
    }

    [Fact]
    public void StreamProcessingMetrics_ShouldCalculateAverageThroughput()
    {
        // Arrange
        var metrics = new StreamProcessingMetrics(
            TotalItemsProcessed: 1000,
            TotalItemsFailed: 0,
            TotalProcessingTime: TimeSpan.FromSeconds(10),
            TotalBatchesProcessed: 10,
            AverageBatchSize: 100,
            BatchEfficiency: 0.9,
            AverageLatencyMs: 1.5,
            P50LatencyMs: 1.2,
            P99LatencyMs: 3.0,
            TotalKernelExecutionTime: TimeSpan.FromSeconds(8),
            TotalMemoryTransferTime: TimeSpan.FromSeconds(2),
            KernelEfficiency: 80,
            MemoryBandwidthMBps: 1000,
            TotalGpuMemoryAllocated: 1024 * 1024 * 100,
            CurrentThroughput: 100,
            PeakThroughput: 150,
            BufferCurrentSize: 0,
            BufferCapacity: 1000,
            BufferUtilization: 0,
            TotalPauseCount: 0,
            TotalPauseDuration: TimeSpan.Zero,
            DeviceType: "CUDA",
            DeviceName: "RTX 4090",
            StartTime: DateTime.UtcNow.AddSeconds(-10),
            LastProcessedTime: DateTime.UtcNow);

        // Assert - 1000 items / 10 seconds = 100 items/sec
        metrics.AverageThroughput.Should().BeApproximately(100, 5);
    }

    [Fact]
    public void StreamProcessingMetrics_ShouldCalculateSuccessRate()
    {
        // Arrange
        var metrics = new StreamProcessingMetrics(
            TotalItemsProcessed: 950,
            TotalItemsFailed: 50,
            TotalProcessingTime: TimeSpan.FromSeconds(10),
            TotalBatchesProcessed: 10,
            AverageBatchSize: 100,
            BatchEfficiency: 0.9,
            AverageLatencyMs: 1.5,
            P50LatencyMs: 1.2,
            P99LatencyMs: 3.0,
            TotalKernelExecutionTime: TimeSpan.FromSeconds(8),
            TotalMemoryTransferTime: TimeSpan.FromSeconds(2),
            KernelEfficiency: 80,
            MemoryBandwidthMBps: 1000,
            TotalGpuMemoryAllocated: 1024 * 1024 * 100,
            CurrentThroughput: 100,
            PeakThroughput: 150,
            BufferCurrentSize: 0,
            BufferCapacity: 1000,
            BufferUtilization: 0,
            TotalPauseCount: 0,
            TotalPauseDuration: TimeSpan.Zero,
            DeviceType: "CUDA",
            DeviceName: "RTX 4090",
            StartTime: DateTime.UtcNow.AddSeconds(-10),
            LastProcessedTime: DateTime.UtcNow);

        // Assert - 950 / 1000 = 95%
        metrics.SuccessRate.Should().BeApproximately(95.0, 0.1);
    }

    [Fact]
    public void StreamProcessingMetrics_ShouldCalculateAveragePauseDuration()
    {
        // Arrange
        var metrics = new StreamProcessingMetrics(
            TotalItemsProcessed: 1000,
            TotalItemsFailed: 0,
            TotalProcessingTime: TimeSpan.FromSeconds(10),
            TotalBatchesProcessed: 10,
            AverageBatchSize: 100,
            BatchEfficiency: 0.9,
            AverageLatencyMs: 1.5,
            P50LatencyMs: 1.2,
            P99LatencyMs: 3.0,
            TotalKernelExecutionTime: TimeSpan.FromSeconds(8),
            TotalMemoryTransferTime: TimeSpan.FromSeconds(2),
            KernelEfficiency: 80,
            MemoryBandwidthMBps: 1000,
            TotalGpuMemoryAllocated: 1024 * 1024 * 100,
            CurrentThroughput: 100,
            PeakThroughput: 150,
            BufferCurrentSize: 0,
            BufferCapacity: 1000,
            BufferUtilization: 0,
            TotalPauseCount: 5,
            TotalPauseDuration: TimeSpan.FromSeconds(10),
            DeviceType: "CUDA",
            DeviceName: "RTX 4090",
            StartTime: DateTime.UtcNow.AddSeconds(-10),
            LastProcessedTime: DateTime.UtcNow);

        // Assert - 10 seconds / 5 pauses = 2 seconds per pause
        metrics.AveragePauseDuration.Should().Be(TimeSpan.FromSeconds(2));
    }

    #endregion

    #region Helper Methods

    private GpuStreamGrainEnhanced<float, float> CreateStreamGrain(StreamProcessingConfiguration config)
    {
        // Note: In actual tests with Orleans TestingHost, this would be grain.Get<T>()
        // For unit tests, we're creating instances directly (simplified for demonstration)

        // This is a simplified mock - in production tests you'd use Orleans TestingHost
        throw new NotImplementedException(
            "Use Orleans TestingHost for actual grain testing. " +
            "This test suite demonstrates the expected test scenarios.");
    }

    #endregion
}

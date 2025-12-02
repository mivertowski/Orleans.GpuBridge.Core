// Orleans.GpuBridge - GPU-native distributed computing for Microsoft Orleans
// Copyright (c) 2025 Michael Ivertowski

using DotCompute.Abstractions.RingKernels;
using Microsoft.Extensions.Logging;
using Orleans.GpuBridge.Abstractions.Models;
using Orleans.GpuBridge.Abstractions.Placement;
using Orleans.GpuBridge.Abstractions.Providers;
using Orleans.GpuBridge.Runtime.Placement;

namespace Orleans.GpuBridge.Runtime.Tests.Placement;

/// <summary>
/// Unit tests for QueueDepthMonitor queue depth monitoring functionality.
/// </summary>
public sealed class QueueDepthMonitorTests : IDisposable
{
    private readonly Mock<IRingKernelRuntime> _mockRuntime;
    private readonly Mock<ILogger<QueueDepthMonitor>> _mockLogger;
    private readonly QueueDepthMonitor _monitor;

    public QueueDepthMonitorTests()
    {
        _mockRuntime = new Mock<IRingKernelRuntime>();
        _mockLogger = new Mock<ILogger<QueueDepthMonitor>>();
        _monitor = new QueueDepthMonitor(_mockRuntime.Object, _mockLogger.Object, "test-silo");
    }

    public void Dispose()
    {
        _monitor.Dispose();
    }

    #region GetQueueDepthAsync Tests

    [Fact]
    public async Task GetQueueDepthAsync_WithNoKernels_ReturnsEmptySnapshot()
    {
        // Arrange
        _mockRuntime.Setup(r => r.ListKernelsAsync())
            .ReturnsAsync(new List<string>());

        // Act
        var snapshot = await _monitor.GetQueueDepthAsync();

        // Assert
        snapshot.ActiveKernelCount.Should().Be(0);
        snapshot.TotalInputQueueDepth.Should().Be(0);
        snapshot.TotalOutputQueueDepth.Should().Be(0);
        snapshot.SiloId.Should().Be("test-silo");
    }

    [Fact]
    public async Task GetQueueDepthAsync_WithActiveKernels_ReturnsAggregatedMetrics()
    {
        // Arrange
        var kernelIds = new List<string> { "kernel-1", "kernel-2" };
        _mockRuntime.Setup(r => r.ListKernelsAsync())
            .ReturnsAsync(kernelIds);

        var metrics1 = CreateMockMetrics(inputUtil: 0.3, outputUtil: 0.2, throughput: 1000, gpuUtil: 40);
        var metrics2 = CreateMockMetrics(inputUtil: 0.5, outputUtil: 0.4, throughput: 2000, gpuUtil: 60);

        _mockRuntime.Setup(r => r.GetMetricsAsync("kernel-1", It.IsAny<CancellationToken>()))
            .ReturnsAsync(metrics1);
        _mockRuntime.Setup(r => r.GetMetricsAsync("kernel-2", It.IsAny<CancellationToken>()))
            .ReturnsAsync(metrics2);

        var status = CreateMockStatus(isLaunched: true);
        _mockRuntime.Setup(r => r.GetStatusAsync(It.IsAny<string>(), It.IsAny<CancellationToken>()))
            .ReturnsAsync(status);

        // Act
        var snapshot = await _monitor.GetQueueDepthAsync();

        // Assert
        snapshot.ActiveKernelCount.Should().Be(2);
        snapshot.ThroughputMsgsPerSec.Should().Be(3000); // 1000 + 2000
        snapshot.GpuUtilization.Should().BeApproximately(0.5, 0.01); // (40 + 60) / 2 / 100
    }

    [Fact]
    public async Task GetQueueDepthAsync_WithSpecificSilo_UsesSiloId()
    {
        // Arrange
        _mockRuntime.Setup(r => r.ListKernelsAsync())
            .ReturnsAsync(new List<string>());

        // Act
        var snapshot = await _monitor.GetQueueDepthAsync(siloId: "custom-silo");

        // Assert
        snapshot.SiloId.Should().Be("custom-silo");
    }

    [Fact]
    public async Task GetQueueDepthAsync_WithSpecificDevice_UsesDeviceIndex()
    {
        // Arrange
        _mockRuntime.Setup(r => r.ListKernelsAsync())
            .ReturnsAsync(new List<string>());

        // Act
        var snapshot = await _monitor.GetQueueDepthAsync(deviceIndex: 2);

        // Assert
        snapshot.DeviceIndex.Should().Be(2);
    }

    [Fact]
    public async Task GetQueueDepthAsync_WithUnlaunchedKernel_SkipsKernel()
    {
        // Arrange
        var kernelIds = new List<string> { "kernel-1", "kernel-2" };
        _mockRuntime.Setup(r => r.ListKernelsAsync())
            .ReturnsAsync(kernelIds);

        var metrics = CreateMockMetrics(inputUtil: 0.3, outputUtil: 0.2, throughput: 1000, gpuUtil: 40);
        _mockRuntime.Setup(r => r.GetMetricsAsync(It.IsAny<string>(), It.IsAny<CancellationToken>()))
            .ReturnsAsync(metrics);

        var launchedStatus = CreateMockStatus(isLaunched: true);
        var notLaunchedStatus = CreateMockStatus(isLaunched: false);

        _mockRuntime.Setup(r => r.GetStatusAsync("kernel-1", It.IsAny<CancellationToken>()))
            .ReturnsAsync(launchedStatus);
        _mockRuntime.Setup(r => r.GetStatusAsync("kernel-2", It.IsAny<CancellationToken>()))
            .ReturnsAsync(notLaunchedStatus);

        // Act
        var snapshot = await _monitor.GetQueueDepthAsync();

        // Assert
        snapshot.ActiveKernelCount.Should().Be(1);
    }

    [Fact]
    public async Task GetQueueDepthAsync_WithMetricsException_ReturnsEmptySnapshot()
    {
        // Arrange
        var kernelIds = new List<string> { "kernel-1" };
        _mockRuntime.Setup(r => r.ListKernelsAsync())
            .ReturnsAsync(kernelIds);

        _mockRuntime.Setup(r => r.GetMetricsAsync(It.IsAny<string>(), It.IsAny<CancellationToken>()))
            .ThrowsAsync(new InvalidOperationException("Metrics unavailable"));

        // Act
        var snapshot = await _monitor.GetQueueDepthAsync();

        // Assert - should return empty due to all kernels failing
        snapshot.ActiveKernelCount.Should().Be(0);
    }

    [Fact]
    public async Task GetQueueDepthAsync_CalculatesUtilizationCorrectly()
    {
        // Arrange
        _mockRuntime.Setup(r => r.ListKernelsAsync())
            .ReturnsAsync(new List<string> { "kernel-1" });

        var metrics = CreateMockMetrics(inputUtil: 0.6, outputUtil: 0.4, throughput: 1000, gpuUtil: 50);
        _mockRuntime.Setup(r => r.GetMetricsAsync(It.IsAny<string>(), It.IsAny<CancellationToken>()))
            .ReturnsAsync(metrics);

        var status = CreateMockStatus(isLaunched: true);
        _mockRuntime.Setup(r => r.GetStatusAsync(It.IsAny<string>(), It.IsAny<CancellationToken>()))
            .ReturnsAsync(status);

        // Act
        var snapshot = await _monitor.GetQueueDepthAsync();

        // Assert
        // Input utilization = 0.6, Output utilization = 0.4
        // Expected input depth = 0.6 * 4096 = 2458, output depth = 0.4 * 4096 = 1638
        snapshot.TotalInputQueueDepth.Should().Be((int)(0.6 * 4096));
        snapshot.TotalOutputQueueDepth.Should().Be((int)(0.4 * 4096));
        snapshot.AverageQueueUtilization.Should().BeApproximately(0.5, 0.01);
    }

    #endregion

    #region GetAggregatedMetricsAsync Tests

    [Fact]
    public async Task GetAggregatedMetricsAsync_WithNoKernels_ReturnsZeroMetrics()
    {
        // Arrange
        _mockRuntime.Setup(r => r.ListKernelsAsync())
            .ReturnsAsync(new List<string>());

        // Act
        var metrics = await _monitor.GetAggregatedMetricsAsync();

        // Assert
        metrics.KernelCount.Should().Be(0);
        metrics.MinQueueUtilization.Should().Be(0);
        metrics.MaxQueueUtilization.Should().Be(0);
        metrics.AvgQueueUtilization.Should().Be(0);
        metrics.IsLoadBalanced.Should().BeTrue();
    }

    [Fact]
    public async Task GetAggregatedMetricsAsync_WithMultipleKernels_CalculatesStatistics()
    {
        // Arrange
        var kernelIds = new List<string> { "kernel-1", "kernel-2", "kernel-3" };
        _mockRuntime.Setup(r => r.ListKernelsAsync())
            .ReturnsAsync(kernelIds);

        _mockRuntime.Setup(r => r.GetMetricsAsync("kernel-1", It.IsAny<CancellationToken>()))
            .ReturnsAsync(CreateMockMetrics(0.2, 0.2, 1000, 40));
        _mockRuntime.Setup(r => r.GetMetricsAsync("kernel-2", It.IsAny<CancellationToken>()))
            .ReturnsAsync(CreateMockMetrics(0.5, 0.5, 2000, 60));
        _mockRuntime.Setup(r => r.GetMetricsAsync("kernel-3", It.IsAny<CancellationToken>()))
            .ReturnsAsync(CreateMockMetrics(0.8, 0.8, 3000, 80));

        // Act
        var metrics = await _monitor.GetAggregatedMetricsAsync();

        // Assert
        metrics.KernelCount.Should().Be(3);
        metrics.MinQueueUtilization.Should().BeApproximately(0.2, 0.01);
        metrics.MaxQueueUtilization.Should().BeApproximately(0.8, 0.01);
        metrics.AvgQueueUtilization.Should().BeApproximately(0.5, 0.01);
        metrics.TotalThroughput.Should().Be(6000);
    }

    [Fact]
    public async Task GetAggregatedMetricsAsync_CalculatesP99LatencyCorrectly()
    {
        // Arrange - Use 10 kernels with increasing latencies to verify P99 calculation
        // P99 of 10 items = index 9 (last item = highest latency)
        var kernelIds = new List<string>
        {
            "kernel-0", "kernel-1", "kernel-2", "kernel-3", "kernel-4",
            "kernel-5", "kernel-6", "kernel-7", "kernel-8", "kernel-9"
        };
        _mockRuntime.Setup(r => r.ListKernelsAsync())
            .ReturnsAsync(kernelIds);

        // Latencies: 1ms, 2ms, 3ms, 4ms, 5ms, 6ms, 7ms, 8ms, 9ms, 10ms
        // Avg = 5.5ms, P99 = 10ms (last element)
        _mockRuntime.Setup(r => r.GetMetricsAsync("kernel-0", It.IsAny<CancellationToken>()))
            .ReturnsAsync(CreateMockMetrics(0.5, 0.5, 100, 50, 1.0));
        _mockRuntime.Setup(r => r.GetMetricsAsync("kernel-1", It.IsAny<CancellationToken>()))
            .ReturnsAsync(CreateMockMetrics(0.5, 0.5, 100, 50, 2.0));
        _mockRuntime.Setup(r => r.GetMetricsAsync("kernel-2", It.IsAny<CancellationToken>()))
            .ReturnsAsync(CreateMockMetrics(0.5, 0.5, 100, 50, 3.0));
        _mockRuntime.Setup(r => r.GetMetricsAsync("kernel-3", It.IsAny<CancellationToken>()))
            .ReturnsAsync(CreateMockMetrics(0.5, 0.5, 100, 50, 4.0));
        _mockRuntime.Setup(r => r.GetMetricsAsync("kernel-4", It.IsAny<CancellationToken>()))
            .ReturnsAsync(CreateMockMetrics(0.5, 0.5, 100, 50, 5.0));
        _mockRuntime.Setup(r => r.GetMetricsAsync("kernel-5", It.IsAny<CancellationToken>()))
            .ReturnsAsync(CreateMockMetrics(0.5, 0.5, 100, 50, 6.0));
        _mockRuntime.Setup(r => r.GetMetricsAsync("kernel-6", It.IsAny<CancellationToken>()))
            .ReturnsAsync(CreateMockMetrics(0.5, 0.5, 100, 50, 7.0));
        _mockRuntime.Setup(r => r.GetMetricsAsync("kernel-7", It.IsAny<CancellationToken>()))
            .ReturnsAsync(CreateMockMetrics(0.5, 0.5, 100, 50, 8.0));
        _mockRuntime.Setup(r => r.GetMetricsAsync("kernel-8", It.IsAny<CancellationToken>()))
            .ReturnsAsync(CreateMockMetrics(0.5, 0.5, 100, 50, 9.0));
        _mockRuntime.Setup(r => r.GetMetricsAsync("kernel-9", It.IsAny<CancellationToken>()))
            .ReturnsAsync(CreateMockMetrics(0.5, 0.5, 100, 50, 10.0));

        // Act
        var metrics = await _monitor.GetAggregatedMetricsAsync();

        // Assert
        // P99 should be the highest latency (10ms = 10_000_000 ns)
        // Avg should be 5.5ms = 5_500_000 ns
        metrics.P99ProcessingLatencyNanos.Should().BeGreaterThan(metrics.AvgProcessingLatencyNanos);
        metrics.P99ProcessingLatencyNanos.Should().Be(10_000_000); // 10ms in nanoseconds
        metrics.AvgProcessingLatencyNanos.Should().BeApproximately(5_500_000, 100); // 5.5ms in nanoseconds
    }

    [Fact]
    public async Task GetAggregatedMetricsAsync_DetectsLoadImbalance()
    {
        // Arrange
        var kernelIds = new List<string> { "kernel-1", "kernel-2" };
        _mockRuntime.Setup(r => r.ListKernelsAsync())
            .ReturnsAsync(kernelIds);

        // Very unbalanced load - kernel-2 at 0.95 to trigger HasOverloadedKernel (> 0.9)
        _mockRuntime.Setup(r => r.GetMetricsAsync("kernel-1", It.IsAny<CancellationToken>()))
            .ReturnsAsync(CreateMockMetrics(0.1, 0.1, 1000, 20));
        _mockRuntime.Setup(r => r.GetMetricsAsync("kernel-2", It.IsAny<CancellationToken>()))
            .ReturnsAsync(CreateMockMetrics(0.95, 0.95, 9000, 90));

        // Act
        var metrics = await _monitor.GetAggregatedMetricsAsync();

        // Assert
        metrics.StdDevQueueUtilization.Should().BeGreaterThan(0.1);
        metrics.IsLoadBalanced.Should().BeFalse();
        metrics.HasOverloadedKernel.Should().BeTrue(); // MaxQueueUtilization (0.95) > 0.9
    }

    #endregion

    #region HasCapacityAsync Tests

    [Fact]
    public async Task HasCapacityAsync_WhenBelowThreshold_ReturnsTrue()
    {
        // Arrange
        _mockRuntime.Setup(r => r.ListKernelsAsync())
            .ReturnsAsync(new List<string> { "kernel-1" });

        _mockRuntime.Setup(r => r.GetMetricsAsync(It.IsAny<string>(), It.IsAny<CancellationToken>()))
            .ReturnsAsync(CreateMockMetrics(0.3, 0.3, 1000, 30));

        var status = CreateMockStatus(isLaunched: true);
        _mockRuntime.Setup(r => r.GetStatusAsync(It.IsAny<string>(), It.IsAny<CancellationToken>()))
            .ReturnsAsync(status);

        // Act
        var hasCapacity = await _monitor.HasCapacityAsync(maxQueueUtilization: 0.8);

        // Assert
        hasCapacity.Should().BeTrue();
    }

    [Fact]
    public async Task HasCapacityAsync_WhenAboveThreshold_ReturnsFalse()
    {
        // Arrange
        _mockRuntime.Setup(r => r.ListKernelsAsync())
            .ReturnsAsync(new List<string> { "kernel-1" });

        _mockRuntime.Setup(r => r.GetMetricsAsync(It.IsAny<string>(), It.IsAny<CancellationToken>()))
            .ReturnsAsync(CreateMockMetrics(0.85, 0.85, 1000, 80));

        var status = CreateMockStatus(isLaunched: true);
        _mockRuntime.Setup(r => r.GetStatusAsync(It.IsAny<string>(), It.IsAny<CancellationToken>()))
            .ReturnsAsync(status);

        // Act
        var hasCapacity = await _monitor.HasCapacityAsync(maxQueueUtilization: 0.8);

        // Assert
        hasCapacity.Should().BeFalse();
    }

    [Fact]
    public async Task HasCapacityAsync_WithCustomThreshold_UsesThreshold()
    {
        // Arrange
        _mockRuntime.Setup(r => r.ListKernelsAsync())
            .ReturnsAsync(new List<string> { "kernel-1" });

        _mockRuntime.Setup(r => r.GetMetricsAsync(It.IsAny<string>(), It.IsAny<CancellationToken>()))
            .ReturnsAsync(CreateMockMetrics(0.55, 0.55, 1000, 50));

        var status = CreateMockStatus(isLaunched: true);
        _mockRuntime.Setup(r => r.GetStatusAsync(It.IsAny<string>(), It.IsAny<CancellationToken>()))
            .ReturnsAsync(status);

        // Act
        var hasCapacityAt60 = await _monitor.HasCapacityAsync(maxQueueUtilization: 0.6);
        var hasCapacityAt50 = await _monitor.HasCapacityAsync(maxQueueUtilization: 0.5);

        // Assert
        hasCapacityAt60.Should().BeTrue();
        hasCapacityAt50.Should().BeFalse();
    }

    #endregion

    #region GetHistoryAsync Tests

    [Fact]
    public async Task GetHistoryAsync_WithNoSamples_ReturnsEmptyHistory()
    {
        // Act
        var history = await _monitor.GetHistoryAsync();

        // Assert
        history.Samples.Should().BeEmpty();
        history.TrendDirection.Should().Be(0);
        history.SiloId.Should().Be("test-silo");
    }

    [Fact]
    public async Task GetHistoryAsync_WithCustomDuration_UsesDuration()
    {
        // Act
        var history = await _monitor.GetHistoryAsync(duration: TimeSpan.FromMinutes(10));

        // Assert - verify the duration property is calculated correctly
        history.Duration.Should().BeLessThanOrEqualTo(TimeSpan.FromMinutes(10));
    }

    #endregion

    #region SubscribeToAlerts Tests

    [Fact]
    public void SubscribeToAlerts_WithValidThreshold_ReturnsDisposable()
    {
        // Arrange
        Action<QueueDepthAlert> callback = _ => { };

        // Act
        var subscription = _monitor.SubscribeToAlerts(0.8, callback);

        // Assert
        subscription.Should().NotBeNull();
        subscription.Should().BeAssignableTo<IDisposable>();
    }

    [Fact]
    public void SubscribeToAlerts_WhenDisposed_ThrowsObjectDisposedException()
    {
        // Arrange
        _monitor.Dispose();
        Action<QueueDepthAlert> callback = _ => { };

        // Act
        var act = () => _monitor.SubscribeToAlerts(0.8, callback);

        // Assert
        act.Should().Throw<ObjectDisposedException>();
    }

    [Fact]
    public void SubscribeToAlerts_SubscriptionDispose_UnsubscribesCallback()
    {
        // Arrange
        var callCount = 0;
        Action<QueueDepthAlert> callback = _ => callCount++;

        // Act
        var subscription = _monitor.SubscribeToAlerts(0.8, callback);
        subscription.Dispose();

        // Assert - subscription should be disposed without error
        subscription.Should().NotBeNull();
    }

    #endregion

    #region Snapshot Calculation Tests

    [Fact]
    public void QueueDepthSnapshot_CalculatesInputUtilizationCorrectly()
    {
        // Arrange
        var snapshot = new QueueDepthSnapshot
        {
            TimestampNanos = 0,
            SiloId = "test",
            DeviceIndex = 0,
            ActiveKernelCount = 1,
            TotalInputQueueDepth = 1000,
            TotalOutputQueueDepth = 500,
            TotalInputQueueCapacity = 4096,
            TotalOutputQueueCapacity = 4096,
            ThroughputMsgsPerSec = 1000,
            GpuUtilization = 0.5,
            AvailableMemoryBytes = 4_000_000_000,
            TotalMemoryBytes = 8_000_000_000
        };

        // Act & Assert
        snapshot.InputQueueUtilization.Should().BeApproximately(1000.0 / 4096, 0.001);
        snapshot.OutputQueueUtilization.Should().BeApproximately(500.0 / 4096, 0.001);
        snapshot.AverageQueueUtilization.Should().BeApproximately((1000.0 / 4096 + 500.0 / 4096) / 2, 0.001);
    }

    [Fact]
    public void QueueDepthSnapshot_CalculatesMemoryUtilizationCorrectly()
    {
        // Arrange
        var snapshot = new QueueDepthSnapshot
        {
            TimestampNanos = 0,
            SiloId = "test",
            DeviceIndex = 0,
            ActiveKernelCount = 1,
            TotalInputQueueDepth = 0,
            TotalOutputQueueDepth = 0,
            TotalInputQueueCapacity = 1,
            TotalOutputQueueCapacity = 1,
            ThroughputMsgsPerSec = 0,
            GpuUtilization = 0,
            AvailableMemoryBytes = 2_000_000_000,
            TotalMemoryBytes = 8_000_000_000
        };

        // Act & Assert
        snapshot.MemoryUtilization.Should().BeApproximately(0.75, 0.001);
        snapshot.AvailableMemoryRatio.Should().BeApproximately(0.25, 0.001);
    }

    [Fact]
    public void QueueDepthSnapshot_WithZeroCapacity_ReturnsZeroUtilization()
    {
        // Arrange
        var snapshot = new QueueDepthSnapshot
        {
            TimestampNanos = 0,
            SiloId = "test",
            DeviceIndex = 0,
            ActiveKernelCount = 0,
            TotalInputQueueDepth = 0,
            TotalOutputQueueDepth = 0,
            TotalInputQueueCapacity = 0,
            TotalOutputQueueCapacity = 0,
            ThroughputMsgsPerSec = 0,
            GpuUtilization = 0,
            AvailableMemoryBytes = 0,
            TotalMemoryBytes = 0
        };

        // Act & Assert
        snapshot.InputQueueUtilization.Should().Be(0);
        snapshot.OutputQueueUtilization.Should().Be(0);
        snapshot.MemoryUtilization.Should().Be(0);
    }

    #endregion

    #region AggregatedQueueMetrics Tests

    [Fact]
    public void AggregatedQueueMetrics_IsLoadBalanced_WhenStdDevLow()
    {
        // Arrange
        var metrics = new AggregatedQueueMetrics
        {
            TimestampNanos = 0,
            SiloId = "test",
            DeviceIndex = 0,
            KernelCount = 3,
            MinQueueUtilization = 0.48,
            MaxQueueUtilization = 0.52,
            AvgQueueUtilization = 0.5,
            StdDevQueueUtilization = 0.02, // Low stddev
            TotalThroughput = 3000,
            AvgProcessingLatencyNanos = 1000,
            P99ProcessingLatencyNanos = 5000
        };

        // Act & Assert
        metrics.IsLoadBalanced.Should().BeTrue();
        metrics.HasOverloadedKernel.Should().BeFalse();
    }

    [Fact]
    public void AggregatedQueueMetrics_HasOverloadedKernel_WhenMaxHigh()
    {
        // Arrange
        var metrics = new AggregatedQueueMetrics
        {
            TimestampNanos = 0,
            SiloId = "test",
            DeviceIndex = 0,
            KernelCount = 3,
            MinQueueUtilization = 0.1,
            MaxQueueUtilization = 0.95, // Very high
            AvgQueueUtilization = 0.5,
            StdDevQueueUtilization = 0.35,
            TotalThroughput = 3000,
            AvgProcessingLatencyNanos = 1000,
            P99ProcessingLatencyNanos = 5000
        };

        // Act & Assert
        metrics.HasOverloadedKernel.Should().BeTrue();
        metrics.IsLoadBalanced.Should().BeFalse();
    }

    #endregion

    #region QueueDepthHistory Tests

    [Fact]
    public void QueueDepthHistory_CalculatesDurationCorrectly()
    {
        // Arrange
        var startNanos = 1_000_000_000L; // 1 second (1 billion nanoseconds)
        var endNanos = 6_000_000_000L;   // 6 seconds (6 billion nanoseconds)

        var history = new QueueDepthHistory
        {
            SiloId = "test",
            DeviceIndex = 0,
            StartTimestampNanos = startNanos,
            EndTimestampNanos = endNanos,
            Samples = Array.Empty<QueueDepthSample>(),
            TrendDirection = 0,
            PredictedUtilization1Min = 0
        };

        // Act & Assert
        history.Duration.TotalSeconds.Should().BeApproximately(5, 0.1);
    }

    #endregion

    #region Alert Severity Tests

    [Theory]
    [InlineData(QueueAlertSeverity.Info, 0)]
    [InlineData(QueueAlertSeverity.Warning, 1)]
    [InlineData(QueueAlertSeverity.Critical, 2)]
    [InlineData(QueueAlertSeverity.Emergency, 3)]
    public void QueueAlertSeverity_HasCorrectOrdinalValues(QueueAlertSeverity severity, int expectedValue)
    {
        // Assert
        ((int)severity).Should().Be(expectedValue);
    }

    #endregion

    #region Dispose Tests

    [Fact]
    public async Task GetQueueDepthAsync_WhenDisposed_ThrowsObjectDisposedException()
    {
        // Arrange
        _monitor.Dispose();

        // Act
        var act = async () => await _monitor.GetQueueDepthAsync();

        // Assert
        await act.Should().ThrowAsync<ObjectDisposedException>();
    }

    [Fact]
    public async Task GetAggregatedMetricsAsync_WhenDisposed_ThrowsObjectDisposedException()
    {
        // Arrange
        _monitor.Dispose();

        // Act
        var act = async () => await _monitor.GetAggregatedMetricsAsync();

        // Assert
        await act.Should().ThrowAsync<ObjectDisposedException>();
    }

    [Fact]
    public async Task HasCapacityAsync_WhenDisposed_ThrowsObjectDisposedException()
    {
        // Arrange
        _monitor.Dispose();

        // Act
        var act = async () => await _monitor.HasCapacityAsync();

        // Assert
        await act.Should().ThrowAsync<ObjectDisposedException>();
    }

    [Fact]
    public async Task GetHistoryAsync_WhenDisposed_ThrowsObjectDisposedException()
    {
        // Arrange
        _monitor.Dispose();

        // Act
        var act = async () => await _monitor.GetHistoryAsync();

        // Assert
        await act.Should().ThrowAsync<ObjectDisposedException>();
    }

    #endregion

    #region Device Manager Integration Tests

    [Fact]
    public async Task GetQueueDepthAsync_WithDeviceManager_UsesRealMemoryMetrics()
    {
        // Arrange
        var mockDeviceManager = new Mock<IDeviceManager>();
        var mockDevice = new Mock<IComputeDevice>();
        mockDevice.Setup(d => d.TotalMemoryBytes).Returns(16L * 1024 * 1024 * 1024); // 16GB

        var deviceMetrics = new DeviceMetrics
        {
            GpuUtilizationPercent = 50,
            MemoryUtilizationPercent = 40,
            UsedMemoryBytes = 6L * 1024 * 1024 * 1024, // 6GB used
            TemperatureCelsius = 65,
            PowerWatts = 150,
            FanSpeedPercent = 40,
            KernelsExecuted = 1000,
            BytesTransferred = 1_000_000_000,
            Uptime = TimeSpan.FromHours(2)
        };

        mockDeviceManager.Setup(dm => dm.GetDevice(It.IsAny<int>()))
            .Returns(mockDevice.Object);
        mockDeviceManager.Setup(dm => dm.GetDeviceMetricsAsync(It.IsAny<IComputeDevice>(), It.IsAny<CancellationToken>()))
            .ReturnsAsync(deviceMetrics);

        var monitorWithDevice = new QueueDepthMonitor(
            _mockRuntime.Object,
            _mockLogger.Object,
            "test-silo",
            mockDeviceManager.Object);

        _mockRuntime.Setup(r => r.ListKernelsAsync())
            .ReturnsAsync(new List<string>());

        try
        {
            // Act
            var snapshot = await monitorWithDevice.GetQueueDepthAsync();

            // Assert - should use real memory values
            snapshot.TotalMemoryBytes.Should().Be(16L * 1024 * 1024 * 1024);
            snapshot.AvailableMemoryBytes.Should().Be(10L * 1024 * 1024 * 1024); // 16GB - 6GB
        }
        finally
        {
            monitorWithDevice.Dispose();
        }
    }

    [Fact]
    public async Task GetQueueDepthAsync_WithoutDeviceManager_UsesFallbackMetrics()
    {
        // Arrange
        _mockRuntime.Setup(r => r.ListKernelsAsync())
            .ReturnsAsync(new List<string>());

        // Act - _monitor doesn't have device manager
        var snapshot = await _monitor.GetQueueDepthAsync();

        // Assert - should use 8GB fallback
        snapshot.TotalMemoryBytes.Should().Be(8L * 1024 * 1024 * 1024);
    }

    [Fact]
    public async Task GetQueueDepthAsync_WithDeviceManagerError_FallsBackToDefaults()
    {
        // Arrange
        var mockDeviceManager = new Mock<IDeviceManager>();
        mockDeviceManager.Setup(dm => dm.GetDevice(It.IsAny<int>()))
            .Throws(new InvalidOperationException("Device not available"));

        var monitorWithDevice = new QueueDepthMonitor(
            _mockRuntime.Object,
            _mockLogger.Object,
            "test-silo",
            mockDeviceManager.Object);

        _mockRuntime.Setup(r => r.ListKernelsAsync())
            .ReturnsAsync(new List<string>());

        try
        {
            // Act
            var snapshot = await monitorWithDevice.GetQueueDepthAsync();

            // Assert - should fall back to 8GB
            snapshot.TotalMemoryBytes.Should().Be(8L * 1024 * 1024 * 1024);
        }
        finally
        {
            monitorWithDevice.Dispose();
        }
    }

    [Fact]
    public async Task GetQueueDepthAsync_CachesDeviceMetrics()
    {
        // Arrange
        var mockDeviceManager = new Mock<IDeviceManager>();
        var mockDevice = new Mock<IComputeDevice>();
        mockDevice.Setup(d => d.TotalMemoryBytes).Returns(16L * 1024 * 1024 * 1024);

        var callCount = 0;
        mockDeviceManager.Setup(dm => dm.GetDevice(It.IsAny<int>()))
            .Returns(mockDevice.Object);
        mockDeviceManager.Setup(dm => dm.GetDeviceMetricsAsync(It.IsAny<IComputeDevice>(), It.IsAny<CancellationToken>()))
            .ReturnsAsync(() =>
            {
                callCount++;
                return new DeviceMetrics
                {
                    GpuUtilizationPercent = 50,
                    MemoryUtilizationPercent = 40,
                    UsedMemoryBytes = 6L * 1024 * 1024 * 1024,
                    TemperatureCelsius = 65,
                    PowerWatts = 150,
                    FanSpeedPercent = 40,
                    KernelsExecuted = 1000,
                    BytesTransferred = 1_000_000_000,
                    Uptime = TimeSpan.FromHours(2)
                };
            });

        var monitorWithDevice = new QueueDepthMonitor(
            _mockRuntime.Object,
            _mockLogger.Object,
            "test-silo",
            mockDeviceManager.Object);

        _mockRuntime.Setup(r => r.ListKernelsAsync())
            .ReturnsAsync(new List<string>());

        try
        {
            // Act - call twice in quick succession
            await monitorWithDevice.GetQueueDepthAsync();
            await monitorWithDevice.GetQueueDepthAsync();

            // Assert - should only call device manager once due to caching
            callCount.Should().Be(1);
        }
        finally
        {
            monitorWithDevice.Dispose();
        }
    }

    #endregion

    #region Helper Methods

    private static RingKernelMetrics CreateMockMetrics(
        double inputUtil,
        double outputUtil,
        double throughput,
        double gpuUtil,
        double avgProcessingTimeMs = 1.0)
    {
        return new RingKernelMetrics
        {
            InputQueueUtilization = inputUtil,
            OutputQueueUtilization = outputUtil,
            ThroughputMsgsPerSec = throughput,
            GpuUtilizationPercent = gpuUtil,
            AvgProcessingTimeMs = avgProcessingTimeMs,
            MessagesReceived = 1000,
            MessagesSent = 1000
        };
    }

    private static RingKernelStatus CreateMockStatus(bool isLaunched)
    {
        return new RingKernelStatus
        {
            IsLaunched = isLaunched,
            IsActive = isLaunched,
            IsTerminating = !isLaunched,
            MessagesPending = 0
        };
    }

    #endregion
}

// Orleans.GpuBridge - GPU-native distributed computing for Microsoft Orleans
// Copyright (c) 2025 Michael Ivertowski

using Orleans.GpuBridge.Diagnostics.Models;

namespace Orleans.GpuBridge.Diagnostics.Tests;

/// <summary>
/// Tests for <see cref="GpuDeviceMetrics"/> model class.
/// </summary>
public class GpuDeviceMetricsTests
{
    [Fact]
    public void MemoryAvailableMB_ShouldComputeCorrectly()
    {
        // Arrange
        var metrics = new GpuDeviceMetrics
        {
            MemoryTotalMB = 8192,
            MemoryUsedMB = 2048
        };

        // Act
        var available = metrics.MemoryAvailableMB;

        // Assert
        available.Should().Be(6144);
    }

    [Fact]
    public void MemoryUsagePercent_ShouldComputeCorrectly()
    {
        // Arrange
        var metrics = new GpuDeviceMetrics
        {
            MemoryTotalMB = 8000,
            MemoryUsedMB = 4000
        };

        // Act
        var usagePercent = metrics.MemoryUsagePercent;

        // Assert
        usagePercent.Should().BeApproximately(50.0, 0.01);
    }

    [Fact]
    public void MemoryUsagePercent_ShouldReturnZero_WhenTotalMemoryIsZero()
    {
        // Arrange
        var metrics = new GpuDeviceMetrics
        {
            MemoryTotalMB = 0,
            MemoryUsedMB = 0
        };

        // Act
        var usagePercent = metrics.MemoryUsagePercent;

        // Assert
        usagePercent.Should().Be(0);
    }

    [Fact]
    public void DefaultValues_ShouldBeSet()
    {
        // Arrange & Act
        var metrics = new GpuDeviceMetrics();

        // Assert
        metrics.DeviceIndex.Should().Be(0);
        metrics.DeviceName.Should().BeEmpty();
        metrics.GpuUtilization.Should().Be(0);
        metrics.MemoryUtilization.Should().Be(0);
        metrics.TemperatureCelsius.Should().Be(0);
        metrics.PowerUsageWatts.Should().Be(0);
    }

    [Fact]
    public void AllProperties_ShouldBeSettable()
    {
        // Arrange
        var timestamp = DateTimeOffset.UtcNow;
        var metrics = new GpuDeviceMetrics
        {
            DeviceIndex = 1,
            DeviceName = "RTX 4090",
            DeviceType = "NVIDIA",
            GpuUtilization = 85.5,
            MemoryUtilization = 70.0,
            MemoryUsedMB = 18432,
            MemoryTotalMB = 24576,
            TemperatureCelsius = 72.0,
            PowerUsageWatts = 350.0,
            Timestamp = timestamp
        };

        // Assert
        metrics.DeviceIndex.Should().Be(1);
        metrics.DeviceName.Should().Be("RTX 4090");
        metrics.DeviceType.Should().Be("NVIDIA");
        metrics.GpuUtilization.Should().Be(85.5);
        metrics.MemoryUtilization.Should().Be(70.0);
        metrics.MemoryUsedMB.Should().Be(18432);
        metrics.MemoryTotalMB.Should().Be(24576);
        metrics.TemperatureCelsius.Should().Be(72.0);
        metrics.PowerUsageWatts.Should().Be(350.0);
        metrics.Timestamp.Should().Be(timestamp);
    }

    [Theory]
    [InlineData(1000, 100, 10.0)]
    [InlineData(8192, 8192, 100.0)]
    [InlineData(16384, 0, 0.0)]
    [InlineData(24576, 6144, 25.0)]
    public void MemoryUsagePercent_ShouldHandleVariousValues(long totalMB, long usedMB, double expectedPercent)
    {
        // Arrange
        var metrics = new GpuDeviceMetrics
        {
            MemoryTotalMB = totalMB,
            MemoryUsedMB = usedMB
        };

        // Act
        var usagePercent = metrics.MemoryUsagePercent;

        // Assert
        usagePercent.Should().BeApproximately(expectedPercent, 0.01);
    }
}

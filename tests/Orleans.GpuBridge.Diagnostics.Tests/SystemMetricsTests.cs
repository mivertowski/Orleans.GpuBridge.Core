// Orleans.GpuBridge - GPU-native distributed computing for Microsoft Orleans
// Copyright (c) 2025 Michael Ivertowski

using Orleans.GpuBridge.Diagnostics.Models;

namespace Orleans.GpuBridge.Diagnostics.Tests;

/// <summary>
/// Tests for <see cref="SystemMetrics"/> model class.
/// </summary>
public class SystemMetricsTests
{
    [Fact]
    public void DefaultValues_ShouldBeSet()
    {
        // Arrange & Act
        var metrics = new SystemMetrics();

        // Assert
        metrics.ProcessCpuUsage.Should().Be(0);
        metrics.ProcessMemoryMB.Should().Be(0);
        metrics.ThreadCount.Should().Be(0);
        metrics.HandleCount.Should().Be(0);
    }

    [Fact]
    public void AllProperties_ShouldBeSettable()
    {
        // Arrange
        var timestamp = DateTimeOffset.UtcNow;
        var metrics = new SystemMetrics
        {
            ProcessCpuUsage = 45.5,
            ProcessMemoryMB = 2048,
            ThreadCount = 32,
            HandleCount = 512,
            Timestamp = timestamp
        };

        // Assert
        metrics.ProcessCpuUsage.Should().Be(45.5);
        metrics.ProcessMemoryMB.Should().Be(2048);
        metrics.ThreadCount.Should().Be(32);
        metrics.HandleCount.Should().Be(512);
        metrics.Timestamp.Should().Be(timestamp);
    }

    [Fact]
    public void Properties_ShouldAcceptEdgeCaseValues()
    {
        // Arrange & Act
        var metrics = new SystemMetrics
        {
            ProcessCpuUsage = 100.0,
            ProcessMemoryMB = long.MaxValue,
            ThreadCount = int.MaxValue,
            HandleCount = int.MaxValue
        };

        // Assert
        metrics.ProcessCpuUsage.Should().Be(100.0);
        metrics.ProcessMemoryMB.Should().Be(long.MaxValue);
        metrics.ThreadCount.Should().Be(int.MaxValue);
        metrics.HandleCount.Should().Be(int.MaxValue);
    }
}

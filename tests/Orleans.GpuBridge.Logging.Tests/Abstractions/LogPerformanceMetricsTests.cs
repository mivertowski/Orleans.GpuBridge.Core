// Orleans.GpuBridge - GPU-native distributed computing for Microsoft Orleans
// Copyright (c) 2025 Michael Ivertowski

using Orleans.GpuBridge.Logging.Abstractions;

namespace Orleans.GpuBridge.Logging.Tests.Abstractions;

/// <summary>
/// Tests for <see cref="LogPerformanceMetrics"/> record class.
/// </summary>
public class LogPerformanceMetricsTests
{
    [Fact]
    public void WithDuration_ShouldCreateMetricsWithDuration()
    {
        // Arrange
        var duration = TimeSpan.FromMilliseconds(500);

        // Act
        var metrics = LogPerformanceMetrics.WithDuration(duration);

        // Assert
        metrics.Duration.Should().Be(duration);
        metrics.MemoryUsage.Should().BeNull();
        metrics.CpuUsage.Should().BeNull();
    }

    [Fact]
    public void WithMemory_ShouldCreateMetricsWithMemory()
    {
        // Arrange
        var bytes = 1024 * 1024L; // 1 MB

        // Act
        var metrics = LogPerformanceMetrics.WithMemory(bytes);

        // Assert
        metrics.MemoryUsage.Should().Be(bytes);
        metrics.Duration.Should().BeNull();
        metrics.CpuUsage.Should().BeNull();
    }

    [Fact]
    public void WithCounters_ShouldCreateMetricsWithCounters()
    {
        // Arrange
        var counters = new Dictionary<string, double>
        {
            ["Requests"] = 100,
            ["Errors"] = 5
        };

        // Act
        var metrics = LogPerformanceMetrics.WithCounters(counters);

        // Assert
        metrics.Counters.Should().ContainKey("Requests").WhoseValue.Should().Be(100);
        metrics.Counters.Should().ContainKey("Errors").WhoseValue.Should().Be(5);
    }

    [Fact]
    public void DefaultValues_ShouldBeNull()
    {
        // Act
        var metrics = new LogPerformanceMetrics();

        // Assert
        metrics.Duration.Should().BeNull();
        metrics.MemoryUsage.Should().BeNull();
        metrics.CpuUsage.Should().BeNull();
        metrics.Counters.Should().BeEmpty();
    }

    [Fact]
    public void AllProperties_ShouldBeSettable()
    {
        // Arrange
        var metrics = new LogPerformanceMetrics
        {
            Duration = TimeSpan.FromSeconds(1),
            MemoryUsage = 1024,
            CpuUsage = 50.5,
            Counters = new Dictionary<string, double> { ["Test"] = 1 }
        };

        // Assert
        metrics.Duration.Should().Be(TimeSpan.FromSeconds(1));
        metrics.MemoryUsage.Should().Be(1024);
        metrics.CpuUsage.Should().Be(50.5);
        metrics.Counters.Should().ContainKey("Test");
    }

    [Fact]
    public void LogPerformanceMetrics_ShouldSupportEquality()
    {
        // Arrange
        var metrics1 = LogPerformanceMetrics.WithDuration(TimeSpan.FromSeconds(1));
        var metrics2 = LogPerformanceMetrics.WithDuration(TimeSpan.FromSeconds(1));

        // Assert - use BeEquivalentTo for record comparison
        metrics1.Should().BeEquivalentTo(metrics2);
    }

    [Fact]
    public void LogPerformanceMetrics_ShouldSupportWith()
    {
        // Arrange
        var original = LogPerformanceMetrics.WithDuration(TimeSpan.FromSeconds(1));

        // Act
        var modified = original with { MemoryUsage = 2048 };

        // Assert
        original.MemoryUsage.Should().BeNull();
        modified.MemoryUsage.Should().Be(2048);
        modified.Duration.Should().Be(TimeSpan.FromSeconds(1));
    }
}

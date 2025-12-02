// Orleans.GpuBridge - GPU-native distributed computing for Microsoft Orleans
// Copyright (c) 2025 Michael Ivertowski

using Microsoft.Extensions.Logging;
using Orleans.GpuBridge.Logging.Configuration;

namespace Orleans.GpuBridge.Logging.Tests.Configuration;

/// <summary>
/// Tests for <see cref="GpuBridgeLoggingOptions"/> configuration class.
/// </summary>
public class GpuBridgeLoggingOptionsTests
{
    [Fact]
    public void SectionName_ShouldBeCorrect()
    {
        // Assert
        GpuBridgeLoggingOptions.SectionName.Should().Be("GpuBridgeLogging");
    }

    [Fact]
    public void DefaultValues_ShouldBeReasonable()
    {
        // Act
        var options = new GpuBridgeLoggingOptions();

        // Assert
        options.MinimumLevel.Should().Be(LogLevel.Information);
        options.EnableStructuredLogging.Should().BeTrue();
        options.EnablePerformanceMetrics.Should().BeTrue();
        options.EnableCorrelationTracking.Should().BeTrue();
        options.Buffer.Should().NotBeNull();
        options.DelegateManager.Should().NotBeNull();
        options.Console.Should().NotBeNull();
        options.File.Should().NotBeNull();
        options.Telemetry.Should().NotBeNull();
        options.CategoryLevels.Should().BeEmpty();
    }

    [Fact]
    public void AllProperties_ShouldBeSettable()
    {
        // Arrange
        var options = new GpuBridgeLoggingOptions
        {
            MinimumLevel = LogLevel.Debug,
            EnableStructuredLogging = false,
            EnablePerformanceMetrics = false,
            EnableCorrelationTracking = false
        };

        // Assert
        options.MinimumLevel.Should().Be(LogLevel.Debug);
        options.EnableStructuredLogging.Should().BeFalse();
        options.EnablePerformanceMetrics.Should().BeFalse();
        options.EnableCorrelationTracking.Should().BeFalse();
    }

    [Theory]
    [InlineData(LogLevel.Trace)]
    [InlineData(LogLevel.Debug)]
    [InlineData(LogLevel.Information)]
    [InlineData(LogLevel.Warning)]
    [InlineData(LogLevel.Error)]
    [InlineData(LogLevel.Critical)]
    public void MinimumLevel_ShouldAcceptAllLogLevels(LogLevel level)
    {
        // Arrange
        var options = new GpuBridgeLoggingOptions { MinimumLevel = level };

        // Assert
        options.MinimumLevel.Should().Be(level);
    }

    [Fact]
    public void CategoryLevels_ShouldBeConfigurable()
    {
        // Arrange
        var options = new GpuBridgeLoggingOptions
        {
            CategoryLevels = new Dictionary<string, LogLevel>
            {
                ["Microsoft"] = LogLevel.Warning,
                ["Orleans"] = LogLevel.Debug,
                ["MyApp"] = LogLevel.Trace
            }
        };

        // Assert
        options.CategoryLevels.Should().HaveCount(3);
        options.CategoryLevels["Microsoft"].Should().Be(LogLevel.Warning);
        options.CategoryLevels["Orleans"].Should().Be(LogLevel.Debug);
        options.CategoryLevels["MyApp"].Should().Be(LogLevel.Trace);
    }

    [Fact]
    public void NestedConfigurations_ShouldBeInitialized()
    {
        // Act
        var options = new GpuBridgeLoggingOptions();

        // Assert - all nested configs should have their own defaults
        options.Buffer.Should().NotBeNull();
        options.DelegateManager.Should().NotBeNull();
        options.Console.Should().NotBeNull();
        options.File.Should().NotBeNull();
        options.Telemetry.Should().NotBeNull();
    }

    [Fact]
    public void EnableFlags_ShouldToggle()
    {
        // Arrange
        var options = new GpuBridgeLoggingOptions();

        // Act & Assert - defaults are true
        options.EnableStructuredLogging.Should().BeTrue();
        options.EnablePerformanceMetrics.Should().BeTrue();
        options.EnableCorrelationTracking.Should().BeTrue();

        // Toggle off
        options.EnableStructuredLogging = false;
        options.EnablePerformanceMetrics = false;
        options.EnableCorrelationTracking = false;

        options.EnableStructuredLogging.Should().BeFalse();
        options.EnablePerformanceMetrics.Should().BeFalse();
        options.EnableCorrelationTracking.Should().BeFalse();
    }
}

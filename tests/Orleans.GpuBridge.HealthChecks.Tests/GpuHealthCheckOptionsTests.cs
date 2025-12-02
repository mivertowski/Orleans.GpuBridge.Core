// Orleans.GpuBridge - GPU-native distributed computing for Microsoft Orleans
// Copyright (c) 2025 Michael Ivertowski

using Orleans.GpuBridge.HealthChecks.Configuration;

namespace Orleans.GpuBridge.HealthChecks.Tests;

/// <summary>
/// Tests for <see cref="GpuHealthCheckOptions"/> configuration class.
/// </summary>
public class GpuHealthCheckOptionsTests
{
    [Fact]
    public void DefaultValues_ShouldBeReasonable()
    {
        // Arrange & Act
        var options = new GpuHealthCheckOptions();

        // Assert
        options.RequireGpu.Should().BeFalse();
        options.TestKernelExecution.Should().BeTrue();
        options.MaxTemperatureCelsius.Should().Be(85.0);
        options.WarnTemperatureCelsius.Should().Be(75.0);
        options.MaxMemoryUsagePercent.Should().Be(95.0);
        options.WarnMemoryUsagePercent.Should().Be(80.0);
        options.MinUtilizationPercent.Should().Be(5.0);
    }

    [Fact]
    public void AllProperties_ShouldBeSettable()
    {
        // Arrange
        var options = new GpuHealthCheckOptions
        {
            RequireGpu = true,
            TestKernelExecution = false,
            MaxTemperatureCelsius = 90.0,
            WarnTemperatureCelsius = 80.0,
            MaxMemoryUsagePercent = 98.0,
            WarnMemoryUsagePercent = 85.0,
            MinUtilizationPercent = 10.0
        };

        // Assert
        options.RequireGpu.Should().BeTrue();
        options.TestKernelExecution.Should().BeFalse();
        options.MaxTemperatureCelsius.Should().Be(90.0);
        options.WarnTemperatureCelsius.Should().Be(80.0);
        options.MaxMemoryUsagePercent.Should().Be(98.0);
        options.WarnMemoryUsagePercent.Should().Be(85.0);
        options.MinUtilizationPercent.Should().Be(10.0);
    }

    [Theory]
    [InlineData(70.0, 60.0)]
    [InlineData(85.0, 75.0)]
    [InlineData(95.0, 85.0)]
    public void TemperatureThresholds_ShouldAcceptVariousValues(double maxTemp, double warnTemp)
    {
        // Arrange
        var options = new GpuHealthCheckOptions
        {
            MaxTemperatureCelsius = maxTemp,
            WarnTemperatureCelsius = warnTemp
        };

        // Assert
        options.MaxTemperatureCelsius.Should().Be(maxTemp);
        options.WarnTemperatureCelsius.Should().Be(warnTemp);
    }

    [Theory]
    [InlineData(90.0, 80.0)]
    [InlineData(95.0, 85.0)]
    [InlineData(99.0, 90.0)]
    public void MemoryThresholds_ShouldAcceptVariousValues(double maxMem, double warnMem)
    {
        // Arrange
        var options = new GpuHealthCheckOptions
        {
            MaxMemoryUsagePercent = maxMem,
            WarnMemoryUsagePercent = warnMem
        };

        // Assert
        options.MaxMemoryUsagePercent.Should().Be(maxMem);
        options.WarnMemoryUsagePercent.Should().Be(warnMem);
    }

    [Theory]
    [InlineData(1.0)]
    [InlineData(5.0)]
    [InlineData(20.0)]
    public void MinUtilization_ShouldAcceptVariousValues(double minUtil)
    {
        // Arrange
        var options = new GpuHealthCheckOptions { MinUtilizationPercent = minUtil };

        // Assert
        options.MinUtilizationPercent.Should().Be(minUtil);
    }

    [Fact]
    public void RequireGpu_ShouldToggle()
    {
        // Arrange
        var options = new GpuHealthCheckOptions();

        // Act & Assert - default is false
        options.RequireGpu.Should().BeFalse();

        // Act & Assert - set to true
        options.RequireGpu = true;
        options.RequireGpu.Should().BeTrue();

        // Act & Assert - set back to false
        options.RequireGpu = false;
        options.RequireGpu.Should().BeFalse();
    }

    [Fact]
    public void TestKernelExecution_ShouldToggle()
    {
        // Arrange
        var options = new GpuHealthCheckOptions();

        // Act & Assert - default is true
        options.TestKernelExecution.Should().BeTrue();

        // Act & Assert - set to false
        options.TestKernelExecution = false;
        options.TestKernelExecution.Should().BeFalse();

        // Act & Assert - set back to true
        options.TestKernelExecution = true;
        options.TestKernelExecution.Should().BeTrue();
    }
}

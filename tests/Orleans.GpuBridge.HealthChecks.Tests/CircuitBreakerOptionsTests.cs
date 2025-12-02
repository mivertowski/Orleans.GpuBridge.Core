// Orleans.GpuBridge - GPU-native distributed computing for Microsoft Orleans
// Copyright (c) 2025 Michael Ivertowski

using Orleans.GpuBridge.HealthChecks.CircuitBreaker;

namespace Orleans.GpuBridge.HealthChecks.Tests;

/// <summary>
/// Tests for <see cref="CircuitBreakerOptions"/> configuration class.
/// </summary>
public class CircuitBreakerOptionsTests
{
    [Fact]
    public void DefaultValues_ShouldBeReasonable()
    {
        // Arrange & Act
        var options = new CircuitBreakerOptions();

        // Assert
        options.FailureThreshold.Should().Be(3);
        options.BreakDuration.Should().Be(TimeSpan.FromSeconds(30));
        options.RetryCount.Should().Be(3);
        options.RetryDelayMs.Should().Be(100);
        options.OperationTimeout.Should().Be(TimeSpan.FromSeconds(30));
    }

    [Fact]
    public void AllProperties_ShouldBeSettable()
    {
        // Arrange
        var options = new CircuitBreakerOptions
        {
            FailureThreshold = 5,
            BreakDuration = TimeSpan.FromMinutes(1),
            RetryCount = 10,
            RetryDelayMs = 500,
            OperationTimeout = TimeSpan.FromMinutes(5)
        };

        // Assert
        options.FailureThreshold.Should().Be(5);
        options.BreakDuration.Should().Be(TimeSpan.FromMinutes(1));
        options.RetryCount.Should().Be(10);
        options.RetryDelayMs.Should().Be(500);
        options.OperationTimeout.Should().Be(TimeSpan.FromMinutes(5));
    }

    [Theory]
    [InlineData(1)]
    [InlineData(5)]
    [InlineData(10)]
    public void FailureThreshold_ShouldAcceptVariousValues(int threshold)
    {
        // Arrange
        var options = new CircuitBreakerOptions { FailureThreshold = threshold };

        // Assert
        options.FailureThreshold.Should().Be(threshold);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(50)]
    [InlineData(1000)]
    public void RetryDelayMs_ShouldAcceptVariousValues(int delayMs)
    {
        // Arrange
        var options = new CircuitBreakerOptions { RetryDelayMs = delayMs };

        // Assert
        options.RetryDelayMs.Should().Be(delayMs);
    }
}

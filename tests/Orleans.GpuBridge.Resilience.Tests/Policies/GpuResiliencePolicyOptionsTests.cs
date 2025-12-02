// Orleans.GpuBridge - GPU-native distributed computing for Microsoft Orleans
// Copyright (c) 2025 Michael Ivertowski

using Orleans.GpuBridge.Resilience.Policies;

namespace Orleans.GpuBridge.Resilience.Tests.Policies;

/// <summary>
/// Tests for <see cref="GpuResiliencePolicyOptions"/> configuration class.
/// </summary>
public class GpuResiliencePolicyOptionsTests
{
    [Fact]
    public void SectionName_ShouldBeCorrect()
    {
        // Assert
        GpuResiliencePolicyOptions.SectionName.Should().Be("GpuResilience");
    }

    [Fact]
    public void DefaultValues_ShouldBeInitialized()
    {
        // Act
        var options = new GpuResiliencePolicyOptions();

        // Assert
        options.RetryOptions.Should().NotBeNull();
        options.CircuitBreakerOptions.Should().NotBeNull();
        options.TimeoutOptions.Should().NotBeNull();
        options.BulkheadOptions.Should().NotBeNull();
        options.RateLimitingOptions.Should().NotBeNull();
        options.ChaosOptions.Should().NotBeNull();
    }

    [Fact]
    public void RetryOptions_ShouldHaveReasonableDefaults()
    {
        // Act
        var options = new GpuResiliencePolicyOptions();

        // Assert
        options.RetryOptions.MaxAttempts.Should().BeGreaterThan(0);
        options.RetryOptions.BaseDelay.Should().BeGreaterThan(TimeSpan.Zero);
        options.RetryOptions.MaxDelay.Should().BeGreaterThan(options.RetryOptions.BaseDelay);
    }

    [Fact]
    public void CircuitBreakerOptions_ShouldHaveReasonableDefaults()
    {
        // Act
        var options = new GpuResiliencePolicyOptions();

        // Assert
        options.CircuitBreakerOptions.FailureRatio.Should().BeInRange(0.0, 1.0);
        options.CircuitBreakerOptions.SamplingDuration.Should().BeGreaterThan(TimeSpan.Zero);
        options.CircuitBreakerOptions.MinimumThroughput.Should().BeGreaterThan(0);
        options.CircuitBreakerOptions.BreakDuration.Should().BeGreaterThan(TimeSpan.Zero);
    }

    [Fact]
    public void TimeoutOptions_ShouldHaveReasonableDefaults()
    {
        // Act
        var options = new GpuResiliencePolicyOptions();

        // Assert
        options.TimeoutOptions.KernelExecution.Should().BeGreaterThan(TimeSpan.Zero);
        options.TimeoutOptions.DeviceOperation.Should().BeGreaterThan(TimeSpan.Zero);
        options.TimeoutOptions.MemoryAllocation.Should().BeGreaterThan(TimeSpan.Zero);
        options.TimeoutOptions.KernelCompilation.Should().BeGreaterThan(TimeSpan.Zero);
    }

    [Fact]
    public void BulkheadOptions_ShouldHaveReasonableDefaults()
    {
        // Act
        var options = new GpuResiliencePolicyOptions();

        // Assert
        options.BulkheadOptions.MaxConcurrentOperations.Should().BeGreaterThan(0);
        options.BulkheadOptions.MaxQueuedOperations.Should().BeGreaterThanOrEqualTo(0);
    }

    [Fact]
    public void RateLimitingOptions_ShouldHaveReasonableDefaults()
    {
        // Act
        var options = new GpuResiliencePolicyOptions();

        // Assert
        options.RateLimitingOptions.MaxRequests.Should().BeGreaterThan(0);
        options.RateLimitingOptions.TimeWindow.Should().BeGreaterThan(TimeSpan.Zero);
        options.RateLimitingOptions.TokenRefillRate.Should().BeGreaterThan(0);
        options.RateLimitingOptions.MaxBurstSize.Should().BeGreaterThan(0);
    }

    [Fact]
    public void ChaosOptions_ShouldBeDisabledByDefault()
    {
        // Act
        var options = new GpuResiliencePolicyOptions();

        // Assert
        options.ChaosOptions.Enabled.Should().BeFalse();
    }

    [Theory]
    [InlineData(1)]
    [InlineData(3)]
    [InlineData(5)]
    [InlineData(10)]
    public void RetryOptions_MaxAttempts_ShouldBeSettable(int maxAttempts)
    {
        // Arrange
        var options = new GpuResiliencePolicyOptions
        {
            RetryOptions = { MaxAttempts = maxAttempts }
        };

        // Assert
        options.RetryOptions.MaxAttempts.Should().Be(maxAttempts);
    }

    [Theory]
    [InlineData(0.1)]
    [InlineData(0.5)]
    [InlineData(0.9)]
    public void CircuitBreakerOptions_FailureRatio_ShouldBeSettable(double ratio)
    {
        // Arrange
        var options = new GpuResiliencePolicyOptions
        {
            CircuitBreakerOptions = { FailureRatio = ratio }
        };

        // Assert
        options.CircuitBreakerOptions.FailureRatio.Should().Be(ratio);
    }
}

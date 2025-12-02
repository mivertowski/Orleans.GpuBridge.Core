// Orleans.GpuBridge - GPU-native distributed computing for Microsoft Orleans
// Copyright (c) 2025 Michael Ivertowski

using Orleans.GpuBridge.Resilience.Fallback;

namespace Orleans.GpuBridge.Resilience.Tests.Fallback;

/// <summary>
/// Tests for fallback types and options.
/// </summary>
public class FallbackTypesTests
{
    [Fact]
    public void FallbackLevel_ShouldHaveExpectedValues()
    {
        // Assert
        ((int)FallbackLevel.Optimal).Should().Be(0);
        ((int)FallbackLevel.Reduced).Should().Be(1);
        ((int)FallbackLevel.Degraded).Should().Be(2);
        ((int)FallbackLevel.Failed).Should().Be(3);
    }

    [Fact]
    public void FallbackChainOptions_SectionName_ShouldBeCorrect()
    {
        // Assert
        FallbackChainOptions.SectionName.Should().Be("FallbackChain");
    }

    [Fact]
    public void FallbackChainOptions_DefaultValues_ShouldBeReasonable()
    {
        // Act
        var options = new FallbackChainOptions();

        // Assert
        options.AutoDegradationEnabled.Should().BeTrue();
        options.AutoRecoveryEnabled.Should().BeTrue();
        options.DegradationErrorThreshold.Should().BeInRange(0.0, 1.0);
        options.RecoveryErrorThreshold.Should().BeInRange(0.0, 1.0);
        options.RecoveryErrorThreshold.Should().BeLessThan(options.DegradationErrorThreshold);
        options.MinimumRequestsForDegradation.Should().BeGreaterThan(0);
        options.MinimumRequestsForRecovery.Should().BeGreaterThan(0);
        options.MinimumRecoveryInterval.Should().BeGreaterThan(TimeSpan.Zero);
        options.ErrorRateWindow.Should().BeGreaterThan(TimeSpan.Zero);
    }

    [Fact]
    public void FallbackChainOptions_AllProperties_ShouldBeSettable()
    {
        // Arrange & Act
        var options = new FallbackChainOptions
        {
            AutoDegradationEnabled = false,
            AutoRecoveryEnabled = false,
            DegradationErrorThreshold = 0.8,
            RecoveryErrorThreshold = 0.2,
            MinimumRequestsForDegradation = 20,
            MinimumRequestsForRecovery = 10,
            MinimumRecoveryInterval = TimeSpan.FromMinutes(10),
            ErrorRateWindow = TimeSpan.FromMinutes(20)
        };

        // Assert
        options.AutoDegradationEnabled.Should().BeFalse();
        options.AutoRecoveryEnabled.Should().BeFalse();
        options.DegradationErrorThreshold.Should().Be(0.8);
        options.RecoveryErrorThreshold.Should().Be(0.2);
        options.MinimumRequestsForDegradation.Should().Be(20);
        options.MinimumRequestsForRecovery.Should().Be(10);
        options.MinimumRecoveryInterval.Should().Be(TimeSpan.FromMinutes(10));
        options.ErrorRateWindow.Should().Be(TimeSpan.FromMinutes(20));
    }

    [Fact]
    public void FallbackAttempt_ShouldStoreAllProperties()
    {
        // Arrange
        var exception = new InvalidOperationException("Test");
        var duration = TimeSpan.FromMilliseconds(100);

        // Act
        var attempt = new FallbackAttempt(
            Level: FallbackLevel.Degraded,
            ExecutorType: "TestExecutor",
            Success: false,
            Duration: duration,
            Exception: exception);

        // Assert
        attempt.Level.Should().Be(FallbackLevel.Degraded);
        attempt.ExecutorType.Should().Be("TestExecutor");
        attempt.Success.Should().BeFalse();
        attempt.Duration.Should().Be(duration);
        attempt.Exception.Should().Be(exception);
    }

    [Fact]
    public void FallbackAttempt_ShouldSupportSuccessfulAttempt()
    {
        // Act
        var attempt = new FallbackAttempt(
            Level: FallbackLevel.Optimal,
            ExecutorType: "GpuExecutor",
            Success: true,
            Duration: TimeSpan.FromMilliseconds(50),
            Exception: null);

        // Assert
        attempt.Success.Should().BeTrue();
        attempt.Exception.Should().BeNull();
    }

    [Fact]
    public void FallbackChainExhaustedException_ShouldContainAttempts()
    {
        // Arrange
        var attempts = new List<FallbackAttempt>
        {
            new(FallbackLevel.Optimal, "GpuExecutor", false, TimeSpan.FromMilliseconds(10), new Exception("GPU failed")),
            new(FallbackLevel.Degraded, "CpuExecutor", false, TimeSpan.FromMilliseconds(20), new Exception("CPU failed"))
        };

        // Act
        var exception = new FallbackChainExhaustedException("TestOperation", attempts, null);

        // Assert
        exception.Attempts.Should().HaveCount(2);
        exception.Message.Should().Contain("TestOperation");
    }

    [Fact]
    public void FallbackChainExhaustedException_ShouldHaveCorrectErrorCode()
    {
        // Act
        var exception = new FallbackChainExhaustedException("Op", new List<FallbackAttempt>(), null);

        // Assert
        exception.ErrorCode.Should().Be(FallbackChainExhaustedException.DefaultErrorCode);
    }
}

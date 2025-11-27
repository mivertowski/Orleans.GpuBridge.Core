// Orleans.GpuBridge - GPU-native distributed computing for Microsoft Orleans
// Copyright (c) 2025 Michael Ivertowski

using Microsoft.Extensions.Logging.Abstractions;
using Orleans.GpuBridge.Abstractions.Temporal;
using Orleans.GpuBridge.Runtime.Temporal.FaultTolerance;
using Xunit;

namespace Orleans.GpuBridge.Temporal.Tests;

/// <summary>
/// Tests for the TemporalFaultHandler fault detection and recovery system.
/// </summary>
public sealed class TemporalFaultHandlerTests : IDisposable
{
    private readonly TemporalFaultHandler _handler;

    public TemporalFaultHandlerTests()
    {
        var options = new TemporalFaultOptions
        {
            ForwardJumpThreshold = TimeSpan.FromMilliseconds(50),
            BackwardJumpThreshold = TimeSpan.FromMilliseconds(1),
            EnableHealthChecks = false, // Disable for unit tests
            DegradedJumpThreshold = 2,
            CriticalJumpThreshold = 5
        };

        _handler = new TemporalFaultHandler(
            NullLogger<TemporalFaultHandler>.Instance,
            options);
    }

    [Fact]
    public void InitialState_ShouldBeHealthy()
    {
        // Assert
        Assert.True(_handler.IsHealthy);
        Assert.Equal(ClockFaultState.Healthy, _handler.CurrentFaultState);
        Assert.Equal(0, _handler.TotalJumpsDetected);
    }

    [Fact]
    public void DetectClockJump_NormalProgression_ShouldNotDetectJump()
    {
        // Arrange
        var baseTime = HybridTimestamp.GetCurrentPhysicalTimeNanos();

        // Act - Small time progression (1ms = 1_000_000 ns)
        var result = _handler.DetectClockJump(baseTime + 1_000_000);

        // Assert
        Assert.Null(result);
        Assert.Equal(0, _handler.TotalJumpsDetected);
        Assert.True(_handler.IsHealthy);
    }

    [Fact]
    public void DetectClockJump_ForwardJump_ShouldDetectJump()
    {
        // Arrange
        var baseTime = HybridTimestamp.GetCurrentPhysicalTimeNanos();
        // Threshold is 50ms, so jump 100ms forward
        var jumpTime = baseTime + (100 * 1_000_000L);

        // Act
        var result = _handler.DetectClockJump(jumpTime);

        // Assert
        Assert.NotNull(result);
        Assert.Equal(ClockJumpType.Forward, result!.JumpType);
        Assert.True(result.JumpMagnitudeNanos > 50_000_000); // > 50ms
        Assert.Equal(1, _handler.TotalJumpsDetected);
    }

    [Fact]
    public void DetectClockJump_BackwardJump_ShouldDetectJump()
    {
        // Arrange
        var baseTime = HybridTimestamp.GetCurrentPhysicalTimeNanos();

        // First, advance time to establish baseline
        _handler.DetectClockJump(baseTime + 10_000_000); // +10ms

        // Now go backward by more than 1ms threshold
        var jumpTime = baseTime - 5_000_000; // -5ms (before initial baseline)

        // Act
        var result = _handler.DetectClockJump(jumpTime);

        // Assert
        Assert.NotNull(result);
        Assert.Equal(ClockJumpType.Backward, result!.JumpType);
        Assert.Equal(1, _handler.TotalJumpsDetected);
    }

    [Fact]
    public void HandleClockJump_ForwardJump_ShouldResetLogicalCounter()
    {
        // Arrange
        var jumpInfo = new ClockJumpInfo
        {
            JumpType = ClockJumpType.Forward,
            JumpMagnitudeNanos = 100_000_000,
            DetectedAt = DateTimeOffset.UtcNow,
            PreviousTimeNanos = 1000,
            CurrentTimeNanos = 100_000_001000
        };

        var currentTimestamp = new HybridTimestamp(1000, 42, nodeId: 1);

        // Act
        var recovered = _handler.HandleClockJump(jumpInfo, currentTimestamp);

        // Assert
        Assert.Equal(jumpInfo.CurrentTimeNanos, recovered.PhysicalTime);
        Assert.Equal(0, recovered.LogicalCounter); // Reset on forward jump
        Assert.Equal(1, recovered.NodeId);
        Assert.Equal(1, _handler.SuccessfulRecoveries);
    }

    [Fact]
    public void HandleClockJump_BackwardJump_ShouldKeepHigherTimeAndIncrementLogical()
    {
        // Arrange
        var jumpInfo = new ClockJumpInfo
        {
            JumpType = ClockJumpType.Backward,
            JumpMagnitudeNanos = 5_000_000,
            DetectedAt = DateTimeOffset.UtcNow,
            PreviousTimeNanos = 100_000_000,
            CurrentTimeNanos = 95_000_000
        };

        var currentTimestamp = new HybridTimestamp(100_000_000, 5, nodeId: 2);

        // Act
        var recovered = _handler.HandleClockJump(jumpInfo, currentTimestamp);

        // Assert
        Assert.Equal(jumpInfo.PreviousTimeNanos, recovered.PhysicalTime); // Keep higher time
        Assert.Equal(6, recovered.LogicalCounter); // Incremented from 5
        Assert.Equal(2, recovered.NodeId);
    }

    [Fact]
    public void ConsecutiveJumps_ShouldTransitionToDegradedState()
    {
        // Arrange
        var baseTime = HybridTimestamp.GetCurrentPhysicalTimeNanos();

        // Act - Create 2+ consecutive jumps (degraded threshold)
        for (int i = 0; i < 3; i++)
        {
            // Simulate large forward jump each time
            var jumpTime = baseTime + ((i + 2) * 100_000_000L);
            _handler.DetectClockJump(jumpTime);
        }

        // Assert
        Assert.Equal(ClockFaultState.Degraded, _handler.CurrentFaultState);
        Assert.True(_handler.IsHealthy); // Still operational but degraded
        Assert.Equal(3, _handler.TotalJumpsDetected);
    }

    [Fact]
    public void ManyConsecutiveJumps_ShouldTransitionToCriticalState()
    {
        // Arrange
        var baseTime = HybridTimestamp.GetCurrentPhysicalTimeNanos();

        // Act - Create 5+ consecutive jumps (critical threshold)
        for (int i = 0; i < 6; i++)
        {
            var jumpTime = baseTime + ((i + 2) * 100_000_000L);
            _handler.DetectClockJump(jumpTime);
        }

        // Assert
        Assert.Equal(ClockFaultState.Critical, _handler.CurrentFaultState);
        Assert.False(_handler.IsHealthy); // Not healthy in critical state
        Assert.Equal(6, _handler.TotalJumpsDetected);
    }

    [Fact]
    public void NormalOperation_ShouldResetConsecutiveJumpsCounter()
    {
        // Arrange
        var baseTime = HybridTimestamp.GetCurrentPhysicalTimeNanos();

        // Create one jump
        _handler.DetectClockJump(baseTime + 100_000_000);

        // Act - Normal progression (no jump)
        _handler.DetectClockJump(baseTime + 110_000_000);

        // Assert - Should be back to healthy
        Assert.Equal(ClockFaultState.Healthy, _handler.CurrentFaultState);
        Assert.True(_handler.IsHealthy);
    }

    [Fact]
    public void ClockJumpDetectedEvent_ShouldFire()
    {
        // Arrange
        var eventFired = false;
        ClockJumpEventArgs? receivedArgs = null;

        _handler.ClockJumpDetected += (sender, args) =>
        {
            eventFired = true;
            receivedArgs = args;
        };

        var baseTime = HybridTimestamp.GetCurrentPhysicalTimeNanos();

        // Act
        _handler.DetectClockJump(baseTime + 200_000_000); // 200ms jump

        // Assert
        Assert.True(eventFired);
        Assert.NotNull(receivedArgs);
        Assert.Equal(ClockJumpType.Forward, receivedArgs!.JumpInfo.JumpType);
        Assert.Equal(1, receivedArgs.TotalJumps);
    }

    [Fact]
    public void FaultStateChangedEvent_ShouldFire()
    {
        // Arrange
        var eventFired = false;
        FaultStateChangedEventArgs? receivedArgs = null;

        _handler.FaultStateChanged += (sender, args) =>
        {
            eventFired = true;
            receivedArgs = args;
        };

        var baseTime = HybridTimestamp.GetCurrentPhysicalTimeNanos();

        // Act - Create enough jumps to trigger degraded state
        for (int i = 0; i < 3; i++)
        {
            _handler.DetectClockJump(baseTime + ((i + 2) * 100_000_000L));
        }

        // Assert
        Assert.True(eventFired);
        Assert.NotNull(receivedArgs);
        Assert.Equal(ClockFaultState.Healthy, receivedArgs!.PreviousState);
        Assert.Equal(ClockFaultState.Degraded, receivedArgs.NewState);
    }

    [Fact]
    public void RecoveryAttemptedEvent_ShouldFire()
    {
        // Arrange
        var eventFired = false;
        RecoveryEventArgs? receivedArgs = null;

        _handler.RecoveryAttempted += (sender, args) =>
        {
            eventFired = true;
            receivedArgs = args;
        };

        var jumpInfo = new ClockJumpInfo
        {
            JumpType = ClockJumpType.Forward,
            JumpMagnitudeNanos = 100_000_000,
            DetectedAt = DateTimeOffset.UtcNow,
            PreviousTimeNanos = 1000,
            CurrentTimeNanos = 100_001_000
        };

        var currentTimestamp = new HybridTimestamp(1000, 0, nodeId: 1);

        // Act
        _handler.HandleClockJump(jumpInfo, currentTimestamp);

        // Assert
        Assert.True(eventFired);
        Assert.NotNull(receivedArgs);
        Assert.True(receivedArgs!.Success);
        Assert.Equal(ClockJumpType.Forward, receivedArgs.JumpType);
    }

    [Fact]
    public void GetStatistics_ShouldReturnCorrectValues()
    {
        // Arrange
        var baseTime = HybridTimestamp.GetCurrentPhysicalTimeNanos();

        // Create some jumps
        _handler.DetectClockJump(baseTime + 100_000_000);
        _handler.DetectClockJump(baseTime + 200_000_000);

        // Act
        var stats = _handler.GetStatistics();

        // Assert
        Assert.Equal(2, stats.TotalJumpsDetected);
        Assert.True(stats.IsHealthy || stats.CurrentFaultState != ClockFaultState.Critical);
        Assert.True(stats.RecoverySuccessRate >= 0 && stats.RecoverySuccessRate <= 1);
    }

    [Fact]
    public void TryRecoverClockSource_WithoutSelector_ShouldReturnFalse()
    {
        // Act - Handler created without ClockSourceSelector
        var result = _handler.TryRecoverClockSource();

        // Assert
        Assert.False(result);
    }

    [Fact]
    public void DefaultOptions_ShouldHaveReasonableDefaults()
    {
        // Act
        var options = TemporalFaultOptions.Default;

        // Assert
        Assert.Equal(TimeSpan.FromMilliseconds(100), options.ForwardJumpThreshold);
        Assert.Equal(TimeSpan.FromMilliseconds(1), options.BackwardJumpThreshold);
        Assert.True(options.EnableHealthChecks);
        Assert.Equal(TimeSpan.FromSeconds(5), options.HealthCheckInterval);
        Assert.Equal(3, options.DegradedJumpThreshold);
        Assert.Equal(10, options.CriticalJumpThreshold);
    }

    [Fact]
    public void StrictOptions_ShouldHaveLowerThresholds()
    {
        // Act
        var options = TemporalFaultOptions.Strict;

        // Assert
        Assert.Equal(TimeSpan.FromMilliseconds(10), options.ForwardJumpThreshold);
        Assert.Equal(TimeSpan.FromMicroseconds(100), options.BackwardJumpThreshold);
        Assert.Equal(TimeSpan.FromSeconds(1), options.HealthCheckInterval);
        Assert.Equal(2, options.DegradedJumpThreshold);
        Assert.Equal(5, options.CriticalJumpThreshold);
    }

    [Fact]
    public void ClockJumpInfo_JumpMagnitudeMs_ShouldConvertCorrectly()
    {
        // Arrange
        var jumpInfo = new ClockJumpInfo
        {
            JumpType = ClockJumpType.Forward,
            JumpMagnitudeNanos = 50_000_000, // 50ms
            DetectedAt = DateTimeOffset.UtcNow,
            PreviousTimeNanos = 0,
            CurrentTimeNanos = 50_000_000
        };

        // Act
        var ms = jumpInfo.JumpMagnitudeMs;

        // Assert
        Assert.Equal(50.0, ms);
    }

    [Fact]
    public void Statistics_RecoverySuccessRate_ShouldBeOneWhenNoAttempts()
    {
        // Act
        var stats = _handler.GetStatistics();

        // Assert - No recovery attempts means 100% success rate (no failures)
        Assert.Equal(1.0, stats.RecoverySuccessRate);
    }

    public void Dispose()
    {
        _handler.Dispose();
    }
}

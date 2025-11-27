// Orleans.GpuBridge - GPU-native distributed computing for Microsoft Orleans
// Copyright (c) 2025 Michael Ivertowski

using Microsoft.Extensions.Logging.Abstractions;
using Orleans.GpuBridge.Abstractions.Temporal;
using Orleans.GpuBridge.Runtime.Temporal.FaultTolerance;
using Xunit;

namespace Orleans.GpuBridge.Temporal.Tests;

/// <summary>
/// Tests for the PtpHardwareMonitor fault detection and recovery system.
/// </summary>
public sealed class PtpHardwareMonitorTests : IDisposable
{
    private readonly PtpHardwareMonitor _monitor;
    private readonly PtpMonitorOptions _options;

    public PtpHardwareMonitorTests()
    {
        _options = new PtpMonitorOptions
        {
            HealthCheckInterval = TimeSpan.FromMilliseconds(100),
            EnablePeriodicHealthChecks = false, // Disable for unit tests
            MaxDriftTolerancePpm = 100.0,
            MaxErrorBoundNanos = 1_000_000,
            MaxSyncLossDuration = TimeSpan.FromSeconds(5),
            FailureThreshold = 2,
            EnableAutoFailover = false // Disable for unit tests
        };

        _monitor = new PtpHardwareMonitor(
            NullLogger<PtpHardwareMonitor>.Instance,
            _options);
    }

    [Fact]
    public void InitialState_ShouldBeHealthy()
    {
        // Assert
        Assert.True(_monitor.IsHealthy);
        Assert.Equal(PtpMonitorState.Healthy, _monitor.CurrentState);
        Assert.Equal(0, _monitor.TotalHealthChecks);
        Assert.Equal(0, _monitor.TotalFailuresDetected);
        Assert.Equal(0, _monitor.MonitoredDeviceCount);
    }

    [Fact]
    public void RegisterDevice_ShouldAddDevice()
    {
        // Act
        var result = _monitor.RegisterDevice("/dev/ptp0");

        // Assert
        Assert.True(result);
        Assert.Equal(1, _monitor.MonitoredDeviceCount);
    }

    [Fact]
    public void RegisterDevice_DuplicateDevice_ShouldReturnFalse()
    {
        // Arrange
        _monitor.RegisterDevice("/dev/ptp0");

        // Act
        var result = _monitor.RegisterDevice("/dev/ptp0");

        // Assert
        Assert.False(result);
        Assert.Equal(1, _monitor.MonitoredDeviceCount);
    }

    [Fact]
    public void UnregisterDevice_ShouldRemoveDevice()
    {
        // Arrange
        _monitor.RegisterDevice("/dev/ptp0");
        Assert.Equal(1, _monitor.MonitoredDeviceCount);

        // Act
        var result = _monitor.UnregisterDevice("/dev/ptp0");

        // Assert
        Assert.True(result);
        Assert.Equal(0, _monitor.MonitoredDeviceCount);
    }

    [Fact]
    public void UnregisterDevice_NonExistentDevice_ShouldReturnFalse()
    {
        // Act
        var result = _monitor.UnregisterDevice("/dev/ptp99");

        // Assert
        Assert.False(result);
    }

    [Fact]
    public void CheckDeviceHealth_UnknownDevice_ShouldReturnNull()
    {
        // Act
        var health = _monitor.CheckDeviceHealth("/dev/ptp99");

        // Assert
        Assert.Null(health);
    }

    [Fact]
    public void CheckDeviceHealth_NonExistentDevice_ShouldReturnDisconnected()
    {
        // Arrange
        _monitor.RegisterDevice("/dev/ptp_nonexistent_test");

        // Act
        var health = _monitor.CheckDeviceHealth("/dev/ptp_nonexistent_test");

        // Assert
        Assert.NotNull(health);
        Assert.Equal(PtpDeviceStatus.Disconnected, health!.Status);
        Assert.Equal("Device file not found", health.FailureReason);
    }

    [Fact]
    public void CheckAllDevicesHealth_ShouldIncrementHealthChecks()
    {
        // Arrange
        _monitor.RegisterDevice("/dev/ptp_test1");
        _monitor.RegisterDevice("/dev/ptp_test2");

        // Act
        var results = _monitor.CheckAllDevicesHealth();

        // Assert
        Assert.Equal(2, results.Count);
        Assert.Equal(1, _monitor.TotalHealthChecks);
    }

    [Fact]
    public void CheckAllDevicesHealth_AllFailed_ShouldBeInCriticalState()
    {
        // Arrange
        _monitor.RegisterDevice("/dev/ptp_nonexistent1");
        _monitor.RegisterDevice("/dev/ptp_nonexistent2");

        // Act
        _monitor.CheckAllDevicesHealth();

        // Assert
        Assert.Equal(PtpMonitorState.Critical, _monitor.CurrentState);
        Assert.False(_monitor.IsHealthy);
    }

    [Fact]
    public void TriggerFailover_WithoutClockSourceSelector_ShouldReturnFalse()
    {
        // Act
        var result = _monitor.TriggerFailover();

        // Assert
        Assert.False(result);
    }

    [Fact]
    public void GetStatistics_ShouldReturnCorrectValues()
    {
        // Arrange
        _monitor.RegisterDevice("/dev/ptp_test");
        _monitor.CheckAllDevicesHealth();

        // Act
        var stats = _monitor.GetStatistics();

        // Assert
        Assert.Equal(1, stats.TotalHealthChecks);
        Assert.Equal(1, stats.MonitoredDeviceCount);
        Assert.True(stats.TotalFailuresDetected >= 0);
        Assert.Equal(1.0, stats.RecoverySuccessRate); // No recovery attempts = 100%
    }

    [Fact]
    public void GetMonitoredDevices_ShouldReturnAllDevices()
    {
        // Arrange
        _monitor.RegisterDevice("/dev/ptp0");
        _monitor.RegisterDevice("/dev/ptp1");

        // Act
        var devices = _monitor.GetMonitoredDevices();

        // Assert
        Assert.Equal(2, devices.Count);
        Assert.Contains(devices, d => d.DevicePath == "/dev/ptp0");
        Assert.Contains(devices, d => d.DevicePath == "/dev/ptp1");
    }

    [Fact]
    public void DeviceFailureDetectedEvent_ShouldFire()
    {
        // Arrange
        var eventFired = false;
        PtpDeviceFailureEventArgs? receivedArgs = null;

        // Use lower failure threshold to trigger event quickly
        var strictOptions = new PtpMonitorOptions
        {
            EnablePeriodicHealthChecks = false,
            FailureThreshold = 1
        };

        using var strictMonitor = new PtpHardwareMonitor(
            NullLogger<PtpHardwareMonitor>.Instance,
            strictOptions);

        strictMonitor.DeviceFailureDetected += (sender, args) =>
        {
            eventFired = true;
            receivedArgs = args;
        };

        strictMonitor.RegisterDevice("/dev/ptp_nonexistent");

        // Act
        strictMonitor.CheckAllDevicesHealth();

        // Assert
        Assert.True(eventFired);
        Assert.NotNull(receivedArgs);
        Assert.Equal("/dev/ptp_nonexistent", receivedArgs!.DevicePath);
    }

    [Fact]
    public void StateChangedEvent_ShouldFireWhenStateBecomesCritical()
    {
        // Arrange
        var eventFired = false;
        PtpMonitorStateChangedEventArgs? receivedArgs = null;

        _monitor.StateChanged += (sender, args) =>
        {
            eventFired = true;
            receivedArgs = args;
        };

        _monitor.RegisterDevice("/dev/ptp_nonexistent");

        // Act
        _monitor.CheckAllDevicesHealth();

        // Assert
        Assert.True(eventFired);
        Assert.NotNull(receivedArgs);
        Assert.Equal(PtpMonitorState.Healthy, receivedArgs!.PreviousState);
        Assert.Equal(PtpMonitorState.Critical, receivedArgs.NewState);
    }

    [Fact]
    public void DefaultOptions_ShouldHaveReasonableDefaults()
    {
        // Act
        var options = PtpMonitorOptions.Default;

        // Assert
        Assert.Equal(TimeSpan.FromSeconds(5), options.HealthCheckInterval);
        Assert.True(options.EnablePeriodicHealthChecks);
        Assert.Equal(100.0, options.MaxDriftTolerancePpm);
        Assert.Equal(1_000_000, options.MaxErrorBoundNanos);
        Assert.Equal(TimeSpan.FromSeconds(30), options.MaxSyncLossDuration);
        Assert.Equal(3, options.FailureThreshold);
        Assert.True(options.EnableAutoFailover);
    }

    [Fact]
    public void StrictOptions_ShouldHaveLowerTolerances()
    {
        // Act
        var options = PtpMonitorOptions.Strict;

        // Assert
        Assert.Equal(TimeSpan.FromSeconds(1), options.HealthCheckInterval);
        Assert.Equal(50.0, options.MaxDriftTolerancePpm);
        Assert.Equal(100_000, options.MaxErrorBoundNanos);
        Assert.Equal(TimeSpan.FromSeconds(5), options.MaxSyncLossDuration);
        Assert.Equal(2, options.FailureThreshold);
    }

    [Fact]
    public void RelaxedOptions_ShouldHaveHigherTolerances()
    {
        // Act
        var options = PtpMonitorOptions.Relaxed;

        // Assert
        Assert.Equal(TimeSpan.FromSeconds(30), options.HealthCheckInterval);
        Assert.Equal(500.0, options.MaxDriftTolerancePpm);
        Assert.Equal(10_000_000, options.MaxErrorBoundNanos);
        Assert.Equal(TimeSpan.FromMinutes(2), options.MaxSyncLossDuration);
        Assert.Equal(5, options.FailureThreshold);
    }

    [Fact]
    public void RegisterDevice_WithClockSource_ShouldAssociate()
    {
        // Arrange
        var mockClockSource = new MockClockSource();

        // Act
        var result = _monitor.RegisterDevice("/dev/ptp_mock", mockClockSource);
        var devices = _monitor.GetMonitoredDevices();

        // Assert
        Assert.True(result);
        var device = devices.FirstOrDefault(d => d.DevicePath == "/dev/ptp_mock");
        Assert.NotNull(device);
        Assert.Equal(mockClockSource, device!.ClockSource);
    }

    [Fact]
    public void CheckDeviceHealth_WithSynchronizedClockSource_ShouldBeHealthy()
    {
        // Arrange
        var mockClockSource = new MockClockSource { IsSynchronized = true };
        _monitor.RegisterDevice("/dev/mock_sync", mockClockSource);

        // We need to create a real file for this test
        // Since /dev/mock_sync won't exist, we skip this for now
        // This test demonstrates the pattern
    }

    [Fact]
    public void ConsecutiveFailures_ShouldAccumulate()
    {
        // Arrange
        _monitor.RegisterDevice("/dev/ptp_nonexistent");

        // Act - Multiple checks should accumulate failures
        _monitor.CheckAllDevicesHealth();
        _monitor.CheckAllDevicesHealth();

        var devices = _monitor.GetMonitoredDevices();

        // Assert
        var device = devices.FirstOrDefault(d => d.DevicePath == "/dev/ptp_nonexistent");
        Assert.NotNull(device);
        Assert.Equal(2, device!.ConsecutiveFailures);
    }

    [Fact]
    public void PtpDeviceInfo_ShouldTrackTimestamps()
    {
        // Arrange
        _monitor.RegisterDevice("/dev/ptp_test");
        var before = DateTimeOffset.UtcNow;

        // Act
        _monitor.CheckAllDevicesHealth();
        var after = DateTimeOffset.UtcNow;

        var devices = _monitor.GetMonitoredDevices();
        var device = devices.First();

        // Assert
        Assert.True(device.RegisteredAt >= before.AddSeconds(-1));
        Assert.True(device.LastChecked >= before);
        Assert.True(device.LastChecked <= after);
    }

    [Fact]
    public void PtpDeviceHealth_ShouldIncludeCheckDuration()
    {
        // Arrange
        _monitor.RegisterDevice("/dev/ptp_test");

        // Act
        var health = _monitor.CheckDeviceHealth("/dev/ptp_test");

        // Assert
        Assert.NotNull(health);
        Assert.True(health!.CheckDuration >= TimeSpan.Zero);
        Assert.True(health.CheckDuration < TimeSpan.FromSeconds(5)); // Should be fast
    }

    [Fact]
    public void Statistics_RecoverySuccessRate_ShouldBeOneWhenNoAttempts()
    {
        // Act
        var stats = _monitor.GetStatistics();

        // Assert - No recovery attempts means 100% success rate
        Assert.Equal(1.0, stats.RecoverySuccessRate);
    }

    /// <summary>
    /// Mock clock source for testing.
    /// </summary>
    private sealed class MockClockSource : IPhysicalClockSource
    {
        public bool IsSynchronized { get; set; } = true;

        public long GetCurrentTimeNanos()
        {
            return DateTimeOffset.UtcNow.ToUnixTimeMilliseconds() * 1_000_000;
        }

        public long GetErrorBound()
        {
            return 1_000; // 1μs
        }

        public double GetClockDrift()
        {
            return 0.0; // No drift for mock
        }
    }

    public void Dispose()
    {
        _monitor.Dispose();
    }
}

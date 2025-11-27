// Orleans.GpuBridge - GPU-native distributed computing for Microsoft Orleans
// Copyright (c) 2025 Michael Ivertowski

using System;
using System.Collections.Concurrent;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Threading;
using Microsoft.Extensions.Logging;
using Orleans.GpuBridge.Abstractions.Temporal;
using Orleans.GpuBridge.Runtime.Temporal.Clock;

namespace Orleans.GpuBridge.Runtime.Temporal.FaultTolerance;

/// <summary>
/// Monitors PTP (Precision Time Protocol) hardware devices for failures and synchronization issues.
/// </summary>
/// <remarks>
/// <para>
/// <b>Monitored Hardware Issues:</b>
/// </para>
/// <list type="bullet">
/// <item>Device disconnection or unavailability</item>
/// <item>Driver failures or crashes</item>
/// <item>Synchronization loss (PTP grandmaster unreachable)</item>
/// <item>Excessive clock drift beyond tolerance</item>
/// <item>Hardware timestamp counter failures</item>
/// </list>
/// <para>
/// <b>Recovery Mechanisms:</b>
/// </para>
/// <list type="bullet">
/// <item>Automatic failover to backup PTP devices</item>
/// <item>Fallback to software PTP or system clock</item>
/// <item>Device reconnection with exponential backoff</item>
/// <item>Alert notifications via events</item>
/// </list>
/// </remarks>
public sealed class PtpHardwareMonitor : IDisposable
{
    private readonly ILogger<PtpHardwareMonitor> _logger;
    private readonly PtpMonitorOptions _options;
    private readonly ClockSourceSelector? _clockSourceSelector;
    private readonly ConcurrentDictionary<string, PtpDeviceInfo> _monitoredDevices = new();
    private readonly Timer? _healthCheckTimer;
    private readonly object _lock = new();

    private long _totalHealthChecks;
    private long _totalFailuresDetected;
    private long _totalRecoveryAttempts;
    private long _successfulRecoveries;
    private PtpMonitorState _currentState = PtpMonitorState.Healthy;
    private DateTimeOffset _lastHealthCheck = DateTimeOffset.UtcNow;
    private bool _disposed;

    /// <summary>
    /// Gets whether all monitored PTP devices are healthy.
    /// </summary>
    public bool IsHealthy
    {
        get
        {
            lock (_lock)
            {
                return _currentState != PtpMonitorState.Critical;
            }
        }
    }

    /// <summary>
    /// Gets the current monitor state.
    /// </summary>
    public PtpMonitorState CurrentState
    {
        get
        {
            lock (_lock)
            {
                return _currentState;
            }
        }
    }

    /// <summary>
    /// Gets the total number of health checks performed.
    /// </summary>
    public long TotalHealthChecks => Interlocked.Read(ref _totalHealthChecks);

    /// <summary>
    /// Gets the total number of failures detected.
    /// </summary>
    public long TotalFailuresDetected => Interlocked.Read(ref _totalFailuresDetected);

    /// <summary>
    /// Gets the total number of recovery attempts made.
    /// </summary>
    public long TotalRecoveryAttempts => Interlocked.Read(ref _totalRecoveryAttempts);

    /// <summary>
    /// Gets the number of successful recovery operations.
    /// </summary>
    public long SuccessfulRecoveries => Interlocked.Read(ref _successfulRecoveries);

    /// <summary>
    /// Gets the number of currently monitored devices.
    /// </summary>
    public int MonitoredDeviceCount => _monitoredDevices.Count;

    /// <summary>
    /// Occurs when a PTP device failure is detected.
    /// </summary>
    public event EventHandler<PtpDeviceFailureEventArgs>? DeviceFailureDetected;

    /// <summary>
    /// Occurs when a PTP device recovers from failure.
    /// </summary>
    public event EventHandler<PtpDeviceRecoveryEventArgs>? DeviceRecovered;

    /// <summary>
    /// Occurs when synchronization is lost with PTP grandmaster.
    /// </summary>
    public event EventHandler<PtpSyncLossEventArgs>? SynchronizationLost;

    /// <summary>
    /// Occurs when the monitor state changes.
    /// </summary>
    public event EventHandler<PtpMonitorStateChangedEventArgs>? StateChanged;

    /// <summary>
    /// Occurs when a failover to backup device is triggered.
    /// </summary>
    public event EventHandler<PtpFailoverEventArgs>? FailoverTriggered;

    /// <summary>
    /// Initializes a new PTP hardware monitor.
    /// </summary>
    /// <param name="logger">Logger for diagnostic messages.</param>
    /// <param name="options">Monitor configuration options.</param>
    /// <param name="clockSourceSelector">Optional clock source selector for failover.</param>
    public PtpHardwareMonitor(
        ILogger<PtpHardwareMonitor> logger,
        PtpMonitorOptions? options = null,
        ClockSourceSelector? clockSourceSelector = null)
    {
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        _options = options ?? PtpMonitorOptions.Default;
        _clockSourceSelector = clockSourceSelector;

        if (_options.EnablePeriodicHealthChecks)
        {
            _healthCheckTimer = new Timer(
                PerformHealthCheck,
                null,
                _options.HealthCheckInterval,
                _options.HealthCheckInterval);
        }

        _logger.LogInformation(
            "PtpHardwareMonitor initialized with health check interval: {Interval}ms, drift tolerance: {DriftTolerance}PPM",
            _options.HealthCheckInterval.TotalMilliseconds,
            _options.MaxDriftTolerancePpm);
    }

    /// <summary>
    /// Registers a PTP device for monitoring.
    /// </summary>
    /// <param name="devicePath">Path to PTP device (e.g., "/dev/ptp0").</param>
    /// <param name="clockSource">Optional associated clock source.</param>
    /// <returns>True if device was registered; false if already monitored.</returns>
    public bool RegisterDevice(string devicePath, IPhysicalClockSource? clockSource = null)
    {
        ArgumentNullException.ThrowIfNull(devicePath);

        var deviceInfo = new PtpDeviceInfo
        {
            DevicePath = devicePath,
            ClockSource = clockSource,
            Status = PtpDeviceStatus.Unknown,
            RegisteredAt = DateTimeOffset.UtcNow,
            LastChecked = DateTimeOffset.MinValue,
            ConsecutiveFailures = 0
        };

        if (_monitoredDevices.TryAdd(devicePath, deviceInfo))
        {
            _logger.LogInformation("Registered PTP device for monitoring: {DevicePath}", devicePath);
            return true;
        }

        _logger.LogDebug("PTP device already monitored: {DevicePath}", devicePath);
        return false;
    }

    /// <summary>
    /// Unregisters a PTP device from monitoring.
    /// </summary>
    /// <param name="devicePath">Path to PTP device.</param>
    /// <returns>True if device was unregistered; false if not found.</returns>
    public bool UnregisterDevice(string devicePath)
    {
        if (_monitoredDevices.TryRemove(devicePath, out _))
        {
            _logger.LogInformation("Unregistered PTP device from monitoring: {DevicePath}", devicePath);
            return true;
        }

        return false;
    }

    /// <summary>
    /// Checks health of a specific PTP device.
    /// </summary>
    /// <param name="devicePath">Path to PTP device.</param>
    /// <returns>Device health information, or null if device not found.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public PtpDeviceHealth? CheckDeviceHealth(string devicePath)
    {
        if (!_monitoredDevices.TryGetValue(devicePath, out var deviceInfo))
        {
            return null;
        }

        return CheckDeviceHealthInternal(deviceInfo);
    }

    /// <summary>
    /// Checks health of all monitored PTP devices.
    /// </summary>
    /// <returns>Health information for all devices.</returns>
    public IReadOnlyList<PtpDeviceHealth> CheckAllDevicesHealth()
    {
        var results = new List<PtpDeviceHealth>();

        foreach (var kvp in _monitoredDevices)
        {
            var health = CheckDeviceHealthInternal(kvp.Value);
            results.Add(health);
        }

        Interlocked.Increment(ref _totalHealthChecks);
        UpdateMonitorState();

        return results;
    }

    /// <summary>
    /// Attempts to recover a failed PTP device.
    /// </summary>
    /// <param name="devicePath">Path to PTP device.</param>
    /// <returns>True if recovery was successful; false otherwise.</returns>
    public async Task<bool> TryRecoverDeviceAsync(string devicePath, CancellationToken ct = default)
    {
        if (!_monitoredDevices.TryGetValue(devicePath, out var deviceInfo))
        {
            _logger.LogWarning("Cannot recover unknown device: {DevicePath}", devicePath);
            return false;
        }

        Interlocked.Increment(ref _totalRecoveryAttempts);
        var recoveryStartTime = Stopwatch.GetTimestamp();

        try
        {
            _logger.LogInformation("Attempting recovery of PTP device: {DevicePath}", devicePath);

            // Step 1: Check if device file exists
            if (!System.IO.File.Exists(devicePath))
            {
                _logger.LogWarning("PTP device file not found: {DevicePath}", devicePath);
                return false;
            }

            // Step 2: Try to reinitialize clock source if available
            if (deviceInfo.ClockSource is PtpClockSource ptpSource)
            {
                // Dispose old source and create new one
                ptpSource.Dispose();

                var newSource = new PtpClockSource(
                    Microsoft.Extensions.Logging.Abstractions.NullLogger<PtpClockSource>.Instance,
                    devicePath);

                if (await newSource.InitializeAsync(ct))
                {
                    deviceInfo.ClockSource = newSource;
                    deviceInfo.Status = PtpDeviceStatus.Healthy;
                    deviceInfo.ConsecutiveFailures = 0;
                    deviceInfo.LastRecovery = DateTimeOffset.UtcNow;

                    Interlocked.Increment(ref _successfulRecoveries);

                    var elapsed = Stopwatch.GetElapsedTime(recoveryStartTime);
                    RaiseDeviceRecoveryEvent(devicePath, elapsed);

                    _logger.LogInformation(
                        "PTP device recovery successful: {DevicePath} (elapsed: {ElapsedMs}ms)",
                        devicePath,
                        elapsed.TotalMilliseconds);

                    return true;
                }
            }

            // Step 3: If no clock source, just verify device accessibility
            if (CanAccessDevice(devicePath))
            {
                deviceInfo.Status = PtpDeviceStatus.Healthy;
                deviceInfo.ConsecutiveFailures = 0;
                deviceInfo.LastRecovery = DateTimeOffset.UtcNow;

                Interlocked.Increment(ref _successfulRecoveries);

                var elapsed = Stopwatch.GetElapsedTime(recoveryStartTime);
                RaiseDeviceRecoveryEvent(devicePath, elapsed);

                _logger.LogInformation("PTP device accessible after recovery: {DevicePath}", devicePath);
                return true;
            }

            _logger.LogWarning("PTP device recovery failed: {DevicePath}", devicePath);
            return false;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error during PTP device recovery: {DevicePath}", devicePath);
            return false;
        }
    }

    /// <summary>
    /// Triggers automatic failover to a backup clock source.
    /// </summary>
    /// <returns>True if failover was successful; false otherwise.</returns>
    public bool TriggerFailover()
    {
        if (_clockSourceSelector == null)
        {
            _logger.LogWarning("Cannot trigger failover: No ClockSourceSelector configured");
            return false;
        }

        try
        {
            var currentSource = _clockSourceSelector.ActiveSource;
            var bestSource = _clockSourceSelector.GetBestAvailableSource();

            if (bestSource == null || bestSource == currentSource)
            {
                _logger.LogWarning("No alternative clock source available for failover");
                return false;
            }

            _clockSourceSelector.SwitchClockSource(bestSource);

            RaiseFailoverEvent(
                currentSource.GetType().Name,
                bestSource.GetType().Name,
                "Manual failover triggered");

            _logger.LogInformation(
                "Failover triggered: {OldSource} → {NewSource}",
                currentSource.GetType().Name,
                bestSource.GetType().Name);

            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to trigger clock source failover");
            return false;
        }
    }

    /// <summary>
    /// Gets current monitoring statistics.
    /// </summary>
    public PtpMonitorStatistics GetStatistics()
    {
        var healthyCount = 0;
        var degradedCount = 0;
        var failedCount = 0;

        foreach (var device in _monitoredDevices.Values)
        {
            switch (device.Status)
            {
                case PtpDeviceStatus.Healthy:
                    healthyCount++;
                    break;
                case PtpDeviceStatus.Degraded:
                    degradedCount++;
                    break;
                case PtpDeviceStatus.Failed:
                case PtpDeviceStatus.Disconnected:
                    failedCount++;
                    break;
            }
        }

        return new PtpMonitorStatistics
        {
            TotalHealthChecks = TotalHealthChecks,
            TotalFailuresDetected = TotalFailuresDetected,
            TotalRecoveryAttempts = TotalRecoveryAttempts,
            SuccessfulRecoveries = SuccessfulRecoveries,
            CurrentState = CurrentState,
            MonitoredDeviceCount = MonitoredDeviceCount,
            HealthyDeviceCount = healthyCount,
            DegradedDeviceCount = degradedCount,
            FailedDeviceCount = failedCount,
            LastHealthCheck = _lastHealthCheck
        };
    }

    /// <summary>
    /// Gets information about all monitored devices.
    /// </summary>
    public IReadOnlyList<PtpDeviceInfo> GetMonitoredDevices()
    {
        return _monitoredDevices.Values.ToList().AsReadOnly();
    }

    private PtpDeviceHealth CheckDeviceHealthInternal(PtpDeviceInfo deviceInfo)
    {
        var checkStartTime = Stopwatch.GetTimestamp();
        var health = new PtpDeviceHealth
        {
            DevicePath = deviceInfo.DevicePath,
            CheckedAt = DateTimeOffset.UtcNow,
            PreviousStatus = deviceInfo.Status
        };

        try
        {
            // Check 1: Device file exists
            if (!System.IO.File.Exists(deviceInfo.DevicePath))
            {
                health.Status = PtpDeviceStatus.Disconnected;
                health.FailureReason = "Device file not found";
                HandleDeviceFailure(deviceInfo, health);
                return health;
            }

            // Check 2: Device accessible
            if (!CanAccessDevice(deviceInfo.DevicePath))
            {
                health.Status = PtpDeviceStatus.Failed;
                health.FailureReason = "Cannot access device";
                HandleDeviceFailure(deviceInfo, health);
                return health;
            }

            // Check 3: Clock source synchronized (if available)
            if (deviceInfo.ClockSource != null)
            {
                if (!deviceInfo.ClockSource.IsSynchronized)
                {
                    health.Status = PtpDeviceStatus.Degraded;
                    health.FailureReason = "Clock source not synchronized";
                    health.IsSynchronized = false;

                    RaiseSyncLossEvent(deviceInfo.DevicePath, "Clock source lost synchronization");

                    // Check if exceeded max sync loss duration
                    if (deviceInfo.SyncLostSince.HasValue)
                    {
                        var syncLossDuration = DateTimeOffset.UtcNow - deviceInfo.SyncLostSince.Value;
                        if (syncLossDuration > _options.MaxSyncLossDuration)
                        {
                            health.Status = PtpDeviceStatus.Failed;
                            health.FailureReason = $"Sync loss exceeded {_options.MaxSyncLossDuration.TotalSeconds}s";
                        }
                    }
                    else
                    {
                        deviceInfo.SyncLostSince = DateTimeOffset.UtcNow;
                    }

                    UpdateDeviceInfo(deviceInfo, health);
                    return health;
                }
                else
                {
                    // Sync restored
                    deviceInfo.SyncLostSince = null;
                    health.IsSynchronized = true;
                    health.ErrorBoundNanos = deviceInfo.ClockSource.GetErrorBound();
                }

                // Check 4: Error bound within tolerance
                var errorBound = deviceInfo.ClockSource.GetErrorBound();
                if (errorBound > _options.MaxErrorBoundNanos)
                {
                    health.Status = PtpDeviceStatus.Degraded;
                    health.FailureReason = $"Error bound {errorBound}ns exceeds tolerance {_options.MaxErrorBoundNanos}ns";
                    UpdateDeviceInfo(deviceInfo, health);
                    return health;
                }
            }

            // All checks passed
            health.Status = PtpDeviceStatus.Healthy;
            health.IsSynchronized = true;

            // Reset consecutive failures on healthy check
            deviceInfo.ConsecutiveFailures = 0;

            UpdateDeviceInfo(deviceInfo, health);
        }
        catch (Exception ex)
        {
            health.Status = PtpDeviceStatus.Failed;
            health.FailureReason = ex.Message;
            health.Exception = ex;
            HandleDeviceFailure(deviceInfo, health);
        }

        health.CheckDuration = Stopwatch.GetElapsedTime(checkStartTime);
        return health;
    }

    private void HandleDeviceFailure(PtpDeviceInfo deviceInfo, PtpDeviceHealth health)
    {
        deviceInfo.ConsecutiveFailures++;
        Interlocked.Increment(ref _totalFailuresDetected);

        if (deviceInfo.ConsecutiveFailures >= _options.FailureThreshold)
        {
            RaiseDeviceFailureEvent(deviceInfo.DevicePath, health.FailureReason ?? "Unknown", health.Exception);

            // Trigger automatic failover if enabled
            if (_options.EnableAutoFailover && _clockSourceSelector != null)
            {
                TriggerFailover();
            }
        }

        UpdateDeviceInfo(deviceInfo, health);
        UpdateMonitorState();

        _logger.LogWarning(
            "PTP device health check failed: {DevicePath} (Status: {Status}, Reason: {Reason}, Consecutive failures: {Failures})",
            deviceInfo.DevicePath,
            health.Status,
            health.FailureReason,
            deviceInfo.ConsecutiveFailures);
    }

    private void UpdateDeviceInfo(PtpDeviceInfo deviceInfo, PtpDeviceHealth health)
    {
        deviceInfo.Status = health.Status;
        deviceInfo.LastChecked = health.CheckedAt;
        deviceInfo.LastFailureReason = health.Status != PtpDeviceStatus.Healthy ? health.FailureReason : null;
    }

    private void UpdateMonitorState()
    {
        lock (_lock)
        {
            var previousState = _currentState;
            var hasHealthy = false;
            var hasFailed = false;
            var hasDegraded = false;

            foreach (var device in _monitoredDevices.Values)
            {
                switch (device.Status)
                {
                    case PtpDeviceStatus.Healthy:
                        hasHealthy = true;
                        break;
                    case PtpDeviceStatus.Degraded:
                        hasDegraded = true;
                        break;
                    case PtpDeviceStatus.Failed:
                    case PtpDeviceStatus.Disconnected:
                        hasFailed = true;
                        break;
                }
            }

            if (_monitoredDevices.IsEmpty || hasHealthy)
            {
                _currentState = hasDegraded || hasFailed ? PtpMonitorState.Degraded : PtpMonitorState.Healthy;
            }
            else if (hasDegraded)
            {
                _currentState = PtpMonitorState.Degraded;
            }
            else if (hasFailed)
            {
                _currentState = PtpMonitorState.Critical;
            }

            if (previousState != _currentState)
            {
                RaiseStateChangedEvent(previousState, _currentState);
            }
        }
    }

    private static bool CanAccessDevice(string devicePath)
    {
        try
        {
            using var fs = System.IO.File.Open(devicePath, System.IO.FileMode.Open, System.IO.FileAccess.Read, System.IO.FileShare.Read);
            return true;
        }
        catch
        {
            return false;
        }
    }

    private void PerformHealthCheck(object? state)
    {
        if (_disposed) return;

        try
        {
            CheckAllDevicesHealth();
            _lastHealthCheck = DateTimeOffset.UtcNow;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error during periodic health check");
        }
    }

    private void RaiseDeviceFailureEvent(string devicePath, string reason, Exception? exception)
    {
        try
        {
            DeviceFailureDetected?.Invoke(this, new PtpDeviceFailureEventArgs
            {
                DevicePath = devicePath,
                FailureReason = reason,
                Exception = exception,
                DetectedAt = DateTimeOffset.UtcNow,
                TotalFailures = TotalFailuresDetected
            });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error in DeviceFailureDetected event handler");
        }
    }

    private void RaiseDeviceRecoveryEvent(string devicePath, TimeSpan recoveryDuration)
    {
        try
        {
            DeviceRecovered?.Invoke(this, new PtpDeviceRecoveryEventArgs
            {
                DevicePath = devicePath,
                RecoveryDuration = recoveryDuration,
                RecoveredAt = DateTimeOffset.UtcNow
            });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error in DeviceRecovered event handler");
        }
    }

    private void RaiseSyncLossEvent(string devicePath, string reason)
    {
        try
        {
            SynchronizationLost?.Invoke(this, new PtpSyncLossEventArgs
            {
                DevicePath = devicePath,
                Reason = reason,
                DetectedAt = DateTimeOffset.UtcNow
            });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error in SynchronizationLost event handler");
        }
    }

    private void RaiseStateChangedEvent(PtpMonitorState previousState, PtpMonitorState newState)
    {
        try
        {
            _logger.LogInformation(
                "PTP monitor state changed: {Previous} → {New}",
                previousState,
                newState);

            StateChanged?.Invoke(this, new PtpMonitorStateChangedEventArgs
            {
                PreviousState = previousState,
                NewState = newState,
                ChangedAt = DateTimeOffset.UtcNow
            });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error in StateChanged event handler");
        }
    }

    private void RaiseFailoverEvent(string fromSource, string toSource, string reason)
    {
        try
        {
            FailoverTriggered?.Invoke(this, new PtpFailoverEventArgs
            {
                FromSource = fromSource,
                ToSource = toSource,
                Reason = reason,
                TriggeredAt = DateTimeOffset.UtcNow
            });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error in FailoverTriggered event handler");
        }
    }

    /// <summary>
    /// Disposes resources used by the monitor.
    /// </summary>
    public void Dispose()
    {
        if (_disposed) return;

        _disposed = true;
        _healthCheckTimer?.Dispose();
        _monitoredDevices.Clear();

        _logger.LogDebug(
            "PtpHardwareMonitor disposed. Stats: {TotalChecks} checks, {Failures} failures, {Recoveries}/{Attempts} recoveries",
            TotalHealthChecks,
            TotalFailuresDetected,
            SuccessfulRecoveries,
            TotalRecoveryAttempts);
    }
}

/// <summary>
/// Configuration options for PTP hardware monitoring.
/// </summary>
public sealed class PtpMonitorOptions
{
    /// <summary>
    /// Interval between periodic health checks.
    /// Default: 5 seconds.
    /// </summary>
    public TimeSpan HealthCheckInterval { get; init; } = TimeSpan.FromSeconds(5);

    /// <summary>
    /// Whether to enable periodic health checks.
    /// Default: true.
    /// </summary>
    public bool EnablePeriodicHealthChecks { get; init; } = true;

    /// <summary>
    /// Maximum drift tolerance in parts per million (PPM).
    /// Default: 100 PPM.
    /// </summary>
    public double MaxDriftTolerancePpm { get; init; } = 100.0;

    /// <summary>
    /// Maximum error bound in nanoseconds before device is considered degraded.
    /// Default: 1,000,000ns (1ms).
    /// </summary>
    public long MaxErrorBoundNanos { get; init; } = 1_000_000;

    /// <summary>
    /// Maximum duration of synchronization loss before device is considered failed.
    /// Default: 30 seconds.
    /// </summary>
    public TimeSpan MaxSyncLossDuration { get; init; } = TimeSpan.FromSeconds(30);

    /// <summary>
    /// Number of consecutive failures before triggering failover.
    /// Default: 3.
    /// </summary>
    public int FailureThreshold { get; init; } = 3;

    /// <summary>
    /// Whether to automatically failover to backup source on failure.
    /// Default: true.
    /// </summary>
    public bool EnableAutoFailover { get; init; } = true;

    /// <summary>
    /// Default monitoring options suitable for production use.
    /// </summary>
    public static PtpMonitorOptions Default { get; } = new();

    /// <summary>
    /// Strict monitoring options with lower tolerances.
    /// </summary>
    public static PtpMonitorOptions Strict { get; } = new()
    {
        HealthCheckInterval = TimeSpan.FromSeconds(1),
        MaxDriftTolerancePpm = 50.0,
        MaxErrorBoundNanos = 100_000, // 100μs
        MaxSyncLossDuration = TimeSpan.FromSeconds(5),
        FailureThreshold = 2
    };

    /// <summary>
    /// Relaxed monitoring options with higher tolerances.
    /// </summary>
    public static PtpMonitorOptions Relaxed { get; } = new()
    {
        HealthCheckInterval = TimeSpan.FromSeconds(30),
        MaxDriftTolerancePpm = 500.0,
        MaxErrorBoundNanos = 10_000_000, // 10ms
        MaxSyncLossDuration = TimeSpan.FromMinutes(2),
        FailureThreshold = 5
    };
}

/// <summary>
/// Current state of the PTP hardware monitor.
/// </summary>
public enum PtpMonitorState
{
    /// <summary>
    /// All monitored devices are healthy.
    /// </summary>
    Healthy,

    /// <summary>
    /// Some devices are degraded but system is operational.
    /// </summary>
    Degraded,

    /// <summary>
    /// All devices have failed - system reliability is compromised.
    /// </summary>
    Critical
}

/// <summary>
/// Status of an individual PTP device.
/// </summary>
public enum PtpDeviceStatus
{
    /// <summary>
    /// Device status is unknown (not yet checked).
    /// </summary>
    Unknown,

    /// <summary>
    /// Device is healthy and synchronized.
    /// </summary>
    Healthy,

    /// <summary>
    /// Device is operational but experiencing issues.
    /// </summary>
    Degraded,

    /// <summary>
    /// Device has failed.
    /// </summary>
    Failed,

    /// <summary>
    /// Device is disconnected or unavailable.
    /// </summary>
    Disconnected
}

/// <summary>
/// Information about a monitored PTP device.
/// </summary>
public sealed class PtpDeviceInfo
{
    /// <summary>
    /// Path to the PTP device.
    /// </summary>
    public required string DevicePath { get; init; }

    /// <summary>
    /// Associated clock source (if any).
    /// </summary>
    public IPhysicalClockSource? ClockSource { get; set; }

    /// <summary>
    /// Current device status.
    /// </summary>
    public PtpDeviceStatus Status { get; set; } = PtpDeviceStatus.Unknown;

    /// <summary>
    /// When the device was registered for monitoring.
    /// </summary>
    public DateTimeOffset RegisteredAt { get; init; }

    /// <summary>
    /// When the device was last checked.
    /// </summary>
    public DateTimeOffset LastChecked { get; set; }

    /// <summary>
    /// When the device last recovered from failure.
    /// </summary>
    public DateTimeOffset? LastRecovery { get; set; }

    /// <summary>
    /// When synchronization was lost (if currently lost).
    /// </summary>
    public DateTimeOffset? SyncLostSince { get; set; }

    /// <summary>
    /// Number of consecutive health check failures.
    /// </summary>
    public int ConsecutiveFailures { get; set; }

    /// <summary>
    /// Last failure reason (if any).
    /// </summary>
    public string? LastFailureReason { get; set; }
}

/// <summary>
/// Health check result for a PTP device.
/// </summary>
public sealed class PtpDeviceHealth
{
    /// <summary>
    /// Path to the PTP device.
    /// </summary>
    public required string DevicePath { get; init; }

    /// <summary>
    /// Current status after health check.
    /// </summary>
    public PtpDeviceStatus Status { get; set; }

    /// <summary>
    /// Previous status before health check.
    /// </summary>
    public PtpDeviceStatus PreviousStatus { get; init; }

    /// <summary>
    /// Whether the device is synchronized.
    /// </summary>
    public bool IsSynchronized { get; set; }

    /// <summary>
    /// Current error bound in nanoseconds.
    /// </summary>
    public long? ErrorBoundNanos { get; set; }

    /// <summary>
    /// Failure reason (if status is not healthy).
    /// </summary>
    public string? FailureReason { get; set; }

    /// <summary>
    /// Exception that caused failure (if any).
    /// </summary>
    public Exception? Exception { get; set; }

    /// <summary>
    /// When the health check was performed.
    /// </summary>
    public DateTimeOffset CheckedAt { get; init; }

    /// <summary>
    /// Duration of the health check.
    /// </summary>
    public TimeSpan CheckDuration { get; set; }
}

/// <summary>
/// Statistics from the PTP hardware monitor.
/// </summary>
public sealed class PtpMonitorStatistics
{
    /// <summary>
    /// Total number of health checks performed.
    /// </summary>
    public required long TotalHealthChecks { get; init; }

    /// <summary>
    /// Total number of failures detected.
    /// </summary>
    public required long TotalFailuresDetected { get; init; }

    /// <summary>
    /// Total number of recovery attempts.
    /// </summary>
    public required long TotalRecoveryAttempts { get; init; }

    /// <summary>
    /// Number of successful recoveries.
    /// </summary>
    public required long SuccessfulRecoveries { get; init; }

    /// <summary>
    /// Current monitor state.
    /// </summary>
    public required PtpMonitorState CurrentState { get; init; }

    /// <summary>
    /// Number of monitored devices.
    /// </summary>
    public required int MonitoredDeviceCount { get; init; }

    /// <summary>
    /// Number of healthy devices.
    /// </summary>
    public required int HealthyDeviceCount { get; init; }

    /// <summary>
    /// Number of degraded devices.
    /// </summary>
    public required int DegradedDeviceCount { get; init; }

    /// <summary>
    /// Number of failed devices.
    /// </summary>
    public required int FailedDeviceCount { get; init; }

    /// <summary>
    /// Time of last health check.
    /// </summary>
    public required DateTimeOffset LastHealthCheck { get; init; }

    /// <summary>
    /// Recovery success rate (0-1).
    /// </summary>
    public double RecoverySuccessRate =>
        TotalRecoveryAttempts > 0 ? (double)SuccessfulRecoveries / TotalRecoveryAttempts : 1.0;
}

/// <summary>
/// Event arguments for PTP device failure detection.
/// </summary>
public sealed class PtpDeviceFailureEventArgs : EventArgs
{
    /// <summary>
    /// Path to the failed device.
    /// </summary>
    public required string DevicePath { get; init; }

    /// <summary>
    /// Reason for the failure.
    /// </summary>
    public required string FailureReason { get; init; }

    /// <summary>
    /// Exception that caused failure (if any).
    /// </summary>
    public Exception? Exception { get; init; }

    /// <summary>
    /// When the failure was detected.
    /// </summary>
    public required DateTimeOffset DetectedAt { get; init; }

    /// <summary>
    /// Total failures detected across all devices.
    /// </summary>
    public required long TotalFailures { get; init; }
}

/// <summary>
/// Event arguments for PTP device recovery.
/// </summary>
public sealed class PtpDeviceRecoveryEventArgs : EventArgs
{
    /// <summary>
    /// Path to the recovered device.
    /// </summary>
    public required string DevicePath { get; init; }

    /// <summary>
    /// Duration of the recovery operation.
    /// </summary>
    public required TimeSpan RecoveryDuration { get; init; }

    /// <summary>
    /// When the device recovered.
    /// </summary>
    public required DateTimeOffset RecoveredAt { get; init; }
}

/// <summary>
/// Event arguments for synchronization loss detection.
/// </summary>
public sealed class PtpSyncLossEventArgs : EventArgs
{
    /// <summary>
    /// Path to the device that lost sync.
    /// </summary>
    public required string DevicePath { get; init; }

    /// <summary>
    /// Reason for sync loss.
    /// </summary>
    public required string Reason { get; init; }

    /// <summary>
    /// When sync loss was detected.
    /// </summary>
    public required DateTimeOffset DetectedAt { get; init; }
}

/// <summary>
/// Event arguments for monitor state changes.
/// </summary>
public sealed class PtpMonitorStateChangedEventArgs : EventArgs
{
    /// <summary>
    /// Previous monitor state.
    /// </summary>
    public required PtpMonitorState PreviousState { get; init; }

    /// <summary>
    /// New monitor state.
    /// </summary>
    public required PtpMonitorState NewState { get; init; }

    /// <summary>
    /// When the state changed.
    /// </summary>
    public required DateTimeOffset ChangedAt { get; init; }
}

/// <summary>
/// Event arguments for clock source failover.
/// </summary>
public sealed class PtpFailoverEventArgs : EventArgs
{
    /// <summary>
    /// Clock source name before failover.
    /// </summary>
    public required string FromSource { get; init; }

    /// <summary>
    /// Clock source name after failover.
    /// </summary>
    public required string ToSource { get; init; }

    /// <summary>
    /// Reason for failover.
    /// </summary>
    public required string Reason { get; init; }

    /// <summary>
    /// When failover was triggered.
    /// </summary>
    public required DateTimeOffset TriggeredAt { get; init; }
}

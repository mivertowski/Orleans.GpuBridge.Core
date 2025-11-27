// Orleans.GpuBridge - GPU-native distributed computing for Microsoft Orleans
// Copyright (c) 2025 Michael Ivertowski

using System;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Threading;
using Microsoft.Extensions.Logging;
using Orleans.GpuBridge.Abstractions.Temporal;
using Orleans.GpuBridge.Runtime.Temporal.Clock;

namespace Orleans.GpuBridge.Runtime.Temporal.FaultTolerance;

/// <summary>
/// Detects and handles temporal faults in distributed clock systems.
/// </summary>
/// <remarks>
/// <para>
/// <b>Fault Types Detected:</b>
/// </para>
/// <list type="bullet">
/// <item>Clock jumps (forward or backward)</item>
/// <item>Clock source failures</item>
/// <item>Excessive clock drift</item>
/// <item>Synchronization failures</item>
/// </list>
/// <para>
/// <b>Recovery Mechanisms:</b>
/// </para>
/// <list type="bullet">
/// <item>Automatic clock source failover</item>
/// <item>Logical counter advancement for backward jumps</item>
/// <item>Gradual clock adjustment for forward jumps</item>
/// <item>Alert notifications via events</item>
/// </list>
/// </remarks>
public sealed class TemporalFaultHandler : IDisposable
{
    private readonly ILogger<TemporalFaultHandler> _logger;
    private readonly ClockSourceSelector? _clockSourceSelector;
    private readonly TemporalFaultOptions _options;
    private readonly Timer? _healthCheckTimer;
    private readonly object _lock = new();

    private long _lastKnownPhysicalTime;
    private long _consecutiveJumps;
    private long _totalJumpsDetected;
    private long _totalRecoveryAttempts;
    private long _successfulRecoveries;
    private bool _isHealthy = true;
    private DateTimeOffset _lastHealthCheck = DateTimeOffset.UtcNow;
    private ClockFaultState _currentFaultState = ClockFaultState.Healthy;
    private bool _disposed;

    /// <summary>
    /// Gets whether the temporal system is currently healthy.
    /// </summary>
    public bool IsHealthy
    {
        get
        {
            lock (_lock)
            {
                return _isHealthy;
            }
        }
    }

    /// <summary>
    /// Gets the current fault state.
    /// </summary>
    public ClockFaultState CurrentFaultState
    {
        get
        {
            lock (_lock)
            {
                return _currentFaultState;
            }
        }
    }

    /// <summary>
    /// Gets the total number of clock jumps detected since handler creation.
    /// </summary>
    public long TotalJumpsDetected => Interlocked.Read(ref _totalJumpsDetected);

    /// <summary>
    /// Gets the total number of recovery attempts made.
    /// </summary>
    public long TotalRecoveryAttempts => Interlocked.Read(ref _totalRecoveryAttempts);

    /// <summary>
    /// Gets the number of successful recovery operations.
    /// </summary>
    public long SuccessfulRecoveries => Interlocked.Read(ref _successfulRecoveries);

    /// <summary>
    /// Occurs when a clock jump is detected.
    /// </summary>
    public event EventHandler<ClockJumpEventArgs>? ClockJumpDetected;

    /// <summary>
    /// Occurs when a clock source failure is detected.
    /// </summary>
    public event EventHandler<ClockSourceFailureEventArgs>? ClockSourceFailed;

    /// <summary>
    /// Occurs when recovery is attempted.
    /// </summary>
    public event EventHandler<RecoveryEventArgs>? RecoveryAttempted;

    /// <summary>
    /// Occurs when the fault state changes.
    /// </summary>
    public event EventHandler<FaultStateChangedEventArgs>? FaultStateChanged;

    /// <summary>
    /// Initializes a new temporal fault handler.
    /// </summary>
    /// <param name="logger">Logger for diagnostic messages.</param>
    /// <param name="options">Fault handling configuration options.</param>
    /// <param name="clockSourceSelector">Optional clock source selector for failover.</param>
    public TemporalFaultHandler(
        ILogger<TemporalFaultHandler> logger,
        TemporalFaultOptions? options = null,
        ClockSourceSelector? clockSourceSelector = null)
    {
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        _options = options ?? TemporalFaultOptions.Default;
        _clockSourceSelector = clockSourceSelector;
        _lastKnownPhysicalTime = HybridTimestamp.GetCurrentPhysicalTimeNanos();

        if (_options.EnableHealthChecks)
        {
            _healthCheckTimer = new Timer(
                PerformHealthCheck,
                null,
                _options.HealthCheckInterval,
                _options.HealthCheckInterval);
        }

        _logger.LogInformation(
            "TemporalFaultHandler initialized with forward jump threshold: {ForwardMs}ms, backward threshold: {BackwardMs}ms",
            _options.ForwardJumpThreshold.TotalMilliseconds,
            _options.BackwardJumpThreshold.TotalMilliseconds);
    }

    /// <summary>
    /// Detects clock jumps by comparing current time against last known time.
    /// </summary>
    /// <param name="currentTimeNanos">Current physical time in nanoseconds.</param>
    /// <returns>Information about detected jump, or null if no jump detected.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public ClockJumpInfo? DetectClockJump(long currentTimeNanos)
    {
        long lastTime = Interlocked.Read(ref _lastKnownPhysicalTime);
        long delta = currentTimeNanos - lastTime;

        // Calculate thresholds in nanoseconds
        long forwardThresholdNanos = (long)_options.ForwardJumpThreshold.TotalMilliseconds * 1_000_000;
        long backwardThresholdNanos = (long)_options.BackwardJumpThreshold.TotalMilliseconds * 1_000_000;

        ClockJumpInfo? jumpInfo = null;

        if (delta > forwardThresholdNanos)
        {
            // Forward jump detected (clock advanced too fast)
            jumpInfo = new ClockJumpInfo
            {
                JumpType = ClockJumpType.Forward,
                JumpMagnitudeNanos = delta,
                DetectedAt = DateTimeOffset.UtcNow,
                PreviousTimeNanos = lastTime,
                CurrentTimeNanos = currentTimeNanos
            };

            Interlocked.Increment(ref _totalJumpsDetected);
            Interlocked.Increment(ref _consecutiveJumps);

            _logger.LogWarning(
                "Forward clock jump detected: {DeltaMs}ms (threshold: {ThresholdMs}ms)",
                delta / 1_000_000.0,
                _options.ForwardJumpThreshold.TotalMilliseconds);

            RaiseClockJumpEvent(jumpInfo);
        }
        else if (delta < -backwardThresholdNanos)
        {
            // Backward jump detected (clock went backwards)
            jumpInfo = new ClockJumpInfo
            {
                JumpType = ClockJumpType.Backward,
                JumpMagnitudeNanos = Math.Abs(delta),
                DetectedAt = DateTimeOffset.UtcNow,
                PreviousTimeNanos = lastTime,
                CurrentTimeNanos = currentTimeNanos
            };

            Interlocked.Increment(ref _totalJumpsDetected);
            Interlocked.Increment(ref _consecutiveJumps);

            _logger.LogWarning(
                "Backward clock jump detected: {DeltaMs}ms (threshold: {ThresholdMs}ms)",
                Math.Abs(delta) / 1_000_000.0,
                _options.BackwardJumpThreshold.TotalMilliseconds);

            RaiseClockJumpEvent(jumpInfo);
        }
        else
        {
            // No jump - reset consecutive counter
            Interlocked.Exchange(ref _consecutiveJumps, 0);
        }

        // Update fault state based on consecutive jumps
        UpdateFaultState();

        // Update last known time
        Interlocked.Exchange(ref _lastKnownPhysicalTime, currentTimeNanos);

        return jumpInfo;
    }

    /// <summary>
    /// Handles a detected clock jump with appropriate recovery action.
    /// </summary>
    /// <param name="jumpInfo">Information about the detected jump.</param>
    /// <param name="currentTimestamp">Current HLC timestamp to potentially adjust.</param>
    /// <returns>Recovered HLC timestamp.</returns>
    public HybridTimestamp HandleClockJump(ClockJumpInfo jumpInfo, HybridTimestamp currentTimestamp)
    {
        ArgumentNullException.ThrowIfNull(jumpInfo);

        Interlocked.Increment(ref _totalRecoveryAttempts);
        var recoveryStartTime = Stopwatch.GetTimestamp();

        try
        {
            HybridTimestamp recoveredTimestamp;

            switch (jumpInfo.JumpType)
            {
                case ClockJumpType.Forward:
                    recoveredTimestamp = HandleForwardJump(jumpInfo, currentTimestamp);
                    break;

                case ClockJumpType.Backward:
                    recoveredTimestamp = HandleBackwardJump(jumpInfo, currentTimestamp);
                    break;

                default:
                    recoveredTimestamp = currentTimestamp;
                    break;
            }

            Interlocked.Increment(ref _successfulRecoveries);

            var elapsed = Stopwatch.GetElapsedTime(recoveryStartTime);
            RaiseRecoveryEvent(true, jumpInfo.JumpType, elapsed);

            _logger.LogInformation(
                "Clock jump recovery successful: {JumpType}, elapsed: {ElapsedMicros}μs",
                jumpInfo.JumpType,
                elapsed.TotalMicroseconds);

            return recoveredTimestamp;
        }
        catch (Exception ex)
        {
            var elapsed = Stopwatch.GetElapsedTime(recoveryStartTime);
            RaiseRecoveryEvent(false, jumpInfo.JumpType, elapsed, ex);

            _logger.LogError(ex,
                "Clock jump recovery failed: {JumpType}",
                jumpInfo.JumpType);

            // Return current timestamp unchanged on failure
            return currentTimestamp;
        }
    }

    /// <summary>
    /// Attempts to recover from a clock source failure by switching to backup source.
    /// </summary>
    /// <returns>True if recovery was successful; otherwise, false.</returns>
    public bool TryRecoverClockSource()
    {
        if (_clockSourceSelector == null)
        {
            _logger.LogWarning("Cannot recover clock source: No ClockSourceSelector configured");
            return false;
        }

        Interlocked.Increment(ref _totalRecoveryAttempts);

        try
        {
            var bestSource = _clockSourceSelector.GetBestAvailableSource();
            if (bestSource == null)
            {
                _logger.LogError("No alternative clock source available for failover");
                return false;
            }

            if (bestSource == _clockSourceSelector.ActiveSource)
            {
                _logger.LogDebug("Best available source is already active");
                return true;
            }

            _clockSourceSelector.SwitchClockSource(bestSource);
            Interlocked.Increment(ref _successfulRecoveries);

            _logger.LogInformation(
                "Successfully switched to backup clock source: {SourceType}",
                bestSource.GetType().Name);

            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to recover clock source");
            return false;
        }
    }

    /// <summary>
    /// Gets current fault statistics.
    /// </summary>
    public TemporalFaultStatistics GetStatistics()
    {
        return new TemporalFaultStatistics
        {
            TotalJumpsDetected = TotalJumpsDetected,
            TotalRecoveryAttempts = TotalRecoveryAttempts,
            SuccessfulRecoveries = SuccessfulRecoveries,
            CurrentFaultState = CurrentFaultState,
            IsHealthy = IsHealthy,
            LastHealthCheck = _lastHealthCheck,
            ConsecutiveJumps = Interlocked.Read(ref _consecutiveJumps)
        };
    }

    private HybridTimestamp HandleForwardJump(ClockJumpInfo jumpInfo, HybridTimestamp currentTimestamp)
    {
        // For forward jumps, we accept the new time but log the anomaly
        // The HLC algorithm ensures monotonicity through the logical counter
        _logger.LogDebug(
            "Accepting forward clock jump of {DeltaMs}ms",
            jumpInfo.JumpMagnitudeNanos / 1_000_000.0);

        return new HybridTimestamp(
            jumpInfo.CurrentTimeNanos,
            0, // Reset logical counter for new physical time
            currentTimestamp.NodeId);
    }

    private HybridTimestamp HandleBackwardJump(ClockJumpInfo jumpInfo, HybridTimestamp currentTimestamp)
    {
        // For backward jumps, we keep the old (higher) physical time
        // and increment the logical counter to maintain monotonicity
        long newLogical = currentTimestamp.LogicalCounter + 1;

        // Check for logical counter overflow risk
        if (newLogical > _options.MaxLogicalCounterBeforeWarning)
        {
            _logger.LogWarning(
                "Logical counter is high ({Counter}) due to backward clock jumps. " +
                "Consider investigating clock synchronization.",
                newLogical);
        }

        _logger.LogDebug(
            "Handling backward clock jump by incrementing logical counter to {Counter}",
            newLogical);

        return new HybridTimestamp(
            jumpInfo.PreviousTimeNanos, // Keep the higher time
            newLogical,
            currentTimestamp.NodeId);
    }

    private void PerformHealthCheck(object? state)
    {
        if (_disposed) return;

        try
        {
            var currentTime = HybridTimestamp.GetCurrentPhysicalTimeNanos();
            var jumpInfo = DetectClockJump(currentTime);

            if (jumpInfo != null)
            {
                _logger.LogWarning(
                    "Health check detected clock jump: {Type}, magnitude: {MagnitudeMs}ms",
                    jumpInfo.JumpType,
                    jumpInfo.JumpMagnitudeNanos / 1_000_000.0);
            }

            // Check clock source health if available
            if (_clockSourceSelector?.IsInitialized == true)
            {
                var activeSource = _clockSourceSelector.ActiveSource;
                if (!activeSource.IsSynchronized)
                {
                    _logger.LogWarning(
                        "Active clock source {Source} lost synchronization",
                        activeSource.GetType().Name);

                    RaiseClockSourceFailedEvent(activeSource.GetType().Name, "Lost synchronization");
                    TryRecoverClockSource();
                }
            }

            _lastHealthCheck = DateTimeOffset.UtcNow;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Health check failed");
        }
    }

    private void UpdateFaultState()
    {
        lock (_lock)
        {
            var consecutiveJumps = Interlocked.Read(ref _consecutiveJumps);
            var previousState = _currentFaultState;

            if (consecutiveJumps >= _options.CriticalJumpThreshold)
            {
                _currentFaultState = ClockFaultState.Critical;
                _isHealthy = false;
            }
            else if (consecutiveJumps >= _options.DegradedJumpThreshold)
            {
                _currentFaultState = ClockFaultState.Degraded;
                _isHealthy = true; // Still operational but degraded
            }
            else
            {
                _currentFaultState = ClockFaultState.Healthy;
                _isHealthy = true;
            }

            if (previousState != _currentFaultState)
            {
                RaiseFaultStateChangedEvent(previousState, _currentFaultState);
            }
        }
    }

    private void RaiseClockJumpEvent(ClockJumpInfo jumpInfo)
    {
        try
        {
            ClockJumpDetected?.Invoke(this, new ClockJumpEventArgs
            {
                JumpInfo = jumpInfo,
                ConsecutiveJumps = Interlocked.Read(ref _consecutiveJumps),
                TotalJumps = TotalJumpsDetected
            });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error in ClockJumpDetected event handler");
        }
    }

    private void RaiseClockSourceFailedEvent(string sourceName, string reason)
    {
        try
        {
            ClockSourceFailed?.Invoke(this, new ClockSourceFailureEventArgs
            {
                SourceName = sourceName,
                FailureReason = reason,
                FailedAt = DateTimeOffset.UtcNow
            });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error in ClockSourceFailed event handler");
        }
    }

    private void RaiseRecoveryEvent(bool success, ClockJumpType jumpType, TimeSpan elapsed, Exception? exception = null)
    {
        try
        {
            RecoveryAttempted?.Invoke(this, new RecoveryEventArgs
            {
                Success = success,
                JumpType = jumpType,
                RecoveryDuration = elapsed,
                Exception = exception,
                AttemptedAt = DateTimeOffset.UtcNow
            });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error in RecoveryAttempted event handler");
        }
    }

    private void RaiseFaultStateChangedEvent(ClockFaultState previousState, ClockFaultState newState)
    {
        try
        {
            _logger.LogInformation(
                "Fault state changed: {Previous} → {New}",
                previousState,
                newState);

            FaultStateChanged?.Invoke(this, new FaultStateChangedEventArgs
            {
                PreviousState = previousState,
                NewState = newState,
                ChangedAt = DateTimeOffset.UtcNow
            });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error in FaultStateChanged event handler");
        }
    }

    /// <summary>
    /// Disposes resources used by the fault handler.
    /// </summary>
    public void Dispose()
    {
        if (_disposed) return;

        _disposed = true;
        _healthCheckTimer?.Dispose();

        _logger.LogDebug(
            "TemporalFaultHandler disposed. Stats: {TotalJumps} jumps, {Recoveries}/{Attempts} recoveries",
            TotalJumpsDetected,
            SuccessfulRecoveries,
            TotalRecoveryAttempts);
    }
}

/// <summary>
/// Configuration options for temporal fault handling.
/// </summary>
public sealed class TemporalFaultOptions
{
    /// <summary>
    /// Threshold for detecting forward clock jumps.
    /// Default: 100ms.
    /// </summary>
    public TimeSpan ForwardJumpThreshold { get; init; } = TimeSpan.FromMilliseconds(100);

    /// <summary>
    /// Threshold for detecting backward clock jumps.
    /// Default: 1ms.
    /// </summary>
    public TimeSpan BackwardJumpThreshold { get; init; } = TimeSpan.FromMilliseconds(1);

    /// <summary>
    /// Whether to enable periodic health checks.
    /// Default: true.
    /// </summary>
    public bool EnableHealthChecks { get; init; } = true;

    /// <summary>
    /// Interval between health checks.
    /// Default: 5 seconds.
    /// </summary>
    public TimeSpan HealthCheckInterval { get; init; } = TimeSpan.FromSeconds(5);

    /// <summary>
    /// Number of consecutive jumps before entering degraded state.
    /// Default: 3.
    /// </summary>
    public int DegradedJumpThreshold { get; init; } = 3;

    /// <summary>
    /// Number of consecutive jumps before entering critical state.
    /// Default: 10.
    /// </summary>
    public int CriticalJumpThreshold { get; init; } = 10;

    /// <summary>
    /// Maximum logical counter value before warning.
    /// Default: 10,000.
    /// </summary>
    public long MaxLogicalCounterBeforeWarning { get; init; } = 10_000;

    /// <summary>
    /// Default fault options suitable for production use.
    /// </summary>
    public static TemporalFaultOptions Default { get; } = new();

    /// <summary>
    /// Strict fault options with lower thresholds for testing.
    /// </summary>
    public static TemporalFaultOptions Strict { get; } = new()
    {
        ForwardJumpThreshold = TimeSpan.FromMilliseconds(10),
        BackwardJumpThreshold = TimeSpan.FromMicroseconds(100),
        HealthCheckInterval = TimeSpan.FromSeconds(1),
        DegradedJumpThreshold = 2,
        CriticalJumpThreshold = 5,
        MaxLogicalCounterBeforeWarning = 1_000
    };
}

/// <summary>
/// Represents the current fault state of the temporal system.
/// </summary>
public enum ClockFaultState
{
    /// <summary>
    /// System is healthy with no detected issues.
    /// </summary>
    Healthy,

    /// <summary>
    /// System is operational but experiencing intermittent issues.
    /// </summary>
    Degraded,

    /// <summary>
    /// System is in critical state and may produce unreliable timestamps.
    /// </summary>
    Critical
}

/// <summary>
/// Type of clock jump detected.
/// </summary>
public enum ClockJumpType
{
    /// <summary>
    /// Clock jumped forward (time advanced unexpectedly).
    /// </summary>
    Forward,

    /// <summary>
    /// Clock jumped backward (time went backwards).
    /// </summary>
    Backward
}

/// <summary>
/// Information about a detected clock jump.
/// </summary>
public sealed class ClockJumpInfo
{
    /// <summary>
    /// Type of jump detected.
    /// </summary>
    public required ClockJumpType JumpType { get; init; }

    /// <summary>
    /// Magnitude of the jump in nanoseconds.
    /// </summary>
    public required long JumpMagnitudeNanos { get; init; }

    /// <summary>
    /// When the jump was detected.
    /// </summary>
    public required DateTimeOffset DetectedAt { get; init; }

    /// <summary>
    /// Previous time reading in nanoseconds.
    /// </summary>
    public required long PreviousTimeNanos { get; init; }

    /// <summary>
    /// Current time reading in nanoseconds.
    /// </summary>
    public required long CurrentTimeNanos { get; init; }

    /// <summary>
    /// Gets the jump magnitude in milliseconds.
    /// </summary>
    public double JumpMagnitudeMs => JumpMagnitudeNanos / 1_000_000.0;
}

/// <summary>
/// Statistics about temporal fault handling.
/// </summary>
public sealed class TemporalFaultStatistics
{
    /// <summary>
    /// Total number of clock jumps detected.
    /// </summary>
    public required long TotalJumpsDetected { get; init; }

    /// <summary>
    /// Total number of recovery attempts.
    /// </summary>
    public required long TotalRecoveryAttempts { get; init; }

    /// <summary>
    /// Number of successful recoveries.
    /// </summary>
    public required long SuccessfulRecoveries { get; init; }

    /// <summary>
    /// Current fault state.
    /// </summary>
    public required ClockFaultState CurrentFaultState { get; init; }

    /// <summary>
    /// Whether the system is healthy.
    /// </summary>
    public required bool IsHealthy { get; init; }

    /// <summary>
    /// Time of last health check.
    /// </summary>
    public required DateTimeOffset LastHealthCheck { get; init; }

    /// <summary>
    /// Number of consecutive jumps (resets on normal operation).
    /// </summary>
    public required long ConsecutiveJumps { get; init; }

    /// <summary>
    /// Recovery success rate (0-1).
    /// </summary>
    public double RecoverySuccessRate =>
        TotalRecoveryAttempts > 0 ? (double)SuccessfulRecoveries / TotalRecoveryAttempts : 1.0;
}

/// <summary>
/// Event arguments for clock jump detection.
/// </summary>
public sealed class ClockJumpEventArgs : EventArgs
{
    /// <summary>
    /// Information about the detected jump.
    /// </summary>
    public required ClockJumpInfo JumpInfo { get; init; }

    /// <summary>
    /// Number of consecutive jumps.
    /// </summary>
    public required long ConsecutiveJumps { get; init; }

    /// <summary>
    /// Total jumps detected since handler creation.
    /// </summary>
    public required long TotalJumps { get; init; }
}

/// <summary>
/// Event arguments for clock source failure.
/// </summary>
public sealed class ClockSourceFailureEventArgs : EventArgs
{
    /// <summary>
    /// Name of the failed clock source.
    /// </summary>
    public required string SourceName { get; init; }

    /// <summary>
    /// Reason for the failure.
    /// </summary>
    public required string FailureReason { get; init; }

    /// <summary>
    /// When the failure was detected.
    /// </summary>
    public required DateTimeOffset FailedAt { get; init; }
}

/// <summary>
/// Event arguments for recovery attempts.
/// </summary>
public sealed class RecoveryEventArgs : EventArgs
{
    /// <summary>
    /// Whether recovery was successful.
    /// </summary>
    public required bool Success { get; init; }

    /// <summary>
    /// Type of jump that triggered recovery.
    /// </summary>
    public required ClockJumpType JumpType { get; init; }

    /// <summary>
    /// How long the recovery took.
    /// </summary>
    public required TimeSpan RecoveryDuration { get; init; }

    /// <summary>
    /// Exception if recovery failed.
    /// </summary>
    public Exception? Exception { get; init; }

    /// <summary>
    /// When the recovery was attempted.
    /// </summary>
    public required DateTimeOffset AttemptedAt { get; init; }
}

/// <summary>
/// Event arguments for fault state changes.
/// </summary>
public sealed class FaultStateChangedEventArgs : EventArgs
{
    /// <summary>
    /// Previous fault state.
    /// </summary>
    public required ClockFaultState PreviousState { get; init; }

    /// <summary>
    /// New fault state.
    /// </summary>
    public required ClockFaultState NewState { get; init; }

    /// <summary>
    /// When the state changed.
    /// </summary>
    public required DateTimeOffset ChangedAt { get; init; }
}

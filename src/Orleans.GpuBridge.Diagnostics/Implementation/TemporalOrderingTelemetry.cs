using System;
using System.Collections.Concurrent;
using System.Diagnostics;
using System.Diagnostics.Metrics;
using Microsoft.Extensions.Logging;

namespace Orleans.GpuBridge.Diagnostics.Implementation;

/// <summary>
/// OpenTelemetry metrics for GPU-native temporal ordering with Hybrid Logical Clocks (HLC).
/// Monitors clock drift, calibration, and causal ordering violations.
/// </summary>
/// <remarks>
/// Metrics Categories:
/// 1. Clock Health - Drift, calibration frequency, error bounds
/// 2. HLC Operations - Update latency, logical counter jumps
/// 3. Causal Ordering - Violations, late messages, reordering events
/// 4. Temporal Correctness - Happened-before relationships, concurrent events
///
/// Critical Alerts:
/// - Clock drift >100μs (potential causality violations)
/// - Causal order violations detected
/// - Calibration failures
/// - HLC update latency >50ns (degraded performance)
/// </remarks>
public sealed class TemporalOrderingTelemetry : IDisposable
{
    private readonly ILogger<TemporalOrderingTelemetry> _logger;
    private readonly Meter _meter;

    // Counters
    private readonly Counter<long> _hlcUpdatesTotal;
    private readonly Counter<long> _hlcLocalUpdatesTotal;
    private readonly Counter<long> _hlcRemoteUpdatesTotal;
    private readonly Counter<long> _clockRecalibrationsTotal;
    private readonly Counter<long> _causalViolationsDetected;
    private readonly Counter<long> _lateMessagesReordered;
    private readonly Counter<long> _calibrationFailures;

    // Histograms
    private readonly Histogram<double> _hlcUpdateLatency;
    private readonly Histogram<long> _clockDriftNanos;
    private readonly Histogram<double> _calibrationLatency;
    private readonly Histogram<long> _logicalCounterJumps;
    private readonly Histogram<long> _messageTimestampSkew;

    // Observable Gauges
    private readonly ObservableGauge<long> _currentClockDrift;
    private readonly ObservableGauge<double> _timeSinceLastCalibration;
    private readonly ObservableGauge<int> _activeHlcInstances;

    // State
    private readonly ConcurrentDictionary<Guid, HlcMetricState> _hlcStates;
    private long _lastCalibrationTime;
    private long _lastObservedDrift;
    private bool _disposed;

    public TemporalOrderingTelemetry(
        ILogger<TemporalOrderingTelemetry> logger,
        string meterName = "Orleans.GpuBridge.TemporalOrdering")
    {
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));

        _meter = new Meter(meterName, "1.0.0");
        _hlcStates = new ConcurrentDictionary<Guid, HlcMetricState>();

        // Initialize counters
        _hlcUpdatesTotal = _meter.CreateCounter<long>(
            "hlc_updates_total",
            description: "Total number of HLC updates (local + remote)");

        _hlcLocalUpdatesTotal = _meter.CreateCounter<long>(
            "hlc_local_updates_total",
            description: "Total number of HLC local event updates");

        _hlcRemoteUpdatesTotal = _meter.CreateCounter<long>(
            "hlc_remote_updates_total",
            description: "Total number of HLC remote event updates");

        _clockRecalibrationsTotal = _meter.CreateCounter<long>(
            "clock_recalibrations_total",
            description: "Total number of GPU/CPU clock recalibrations");

        _causalViolationsDetected = _meter.CreateCounter<long>(
            "causal_violations_detected_total",
            description: "Total number of detected causal ordering violations");

        _lateMessagesReordered = _meter.CreateCounter<long>(
            "late_messages_reordered_total",
            description: "Total number of late messages reordered by HLC");

        _calibrationFailures = _meter.CreateCounter<long>(
            "calibration_failures_total",
            description: "Total number of clock calibration failures");

        // Initialize histograms
        _hlcUpdateLatency = _meter.CreateHistogram<double>(
            "hlc_update_latency_nanoseconds",
            unit: "ns",
            description: "HLC update operation latency distribution");

        _clockDriftNanos = _meter.CreateHistogram<long>(
            "clock_drift_nanoseconds",
            unit: "ns",
            description: "GPU/CPU clock drift distribution");

        _calibrationLatency = _meter.CreateHistogram<double>(
            "calibration_latency_microseconds",
            unit: "μs",
            description: "Clock calibration operation latency distribution");

        _logicalCounterJumps = _meter.CreateHistogram<long>(
            "logical_counter_jumps",
            description: "HLC logical counter increment size distribution");

        _messageTimestampSkew = _meter.CreateHistogram<long>(
            "message_timestamp_skew_nanoseconds",
            unit: "ns",
            description: "Message timestamp skew from expected arrival time");

        // Initialize observable gauges
        _currentClockDrift = _meter.CreateObservableGauge(
            "clock_drift_current_nanoseconds",
            ObserveCurrentClockDrift,
            unit: "ns",
            description: "Current GPU/CPU clock drift");

        _timeSinceLastCalibration = _meter.CreateObservableGauge(
            "time_since_last_calibration_seconds",
            ObserveTimeSinceLastCalibration,
            unit: "s",
            description: "Time since last successful clock calibration");

        _activeHlcInstances = _meter.CreateObservableGauge(
            "hlc_instances_active",
            ObserveActiveHlcInstances,
            description: "Number of active HLC clock instances");

        _lastCalibrationTime = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();

        _logger.LogInformation(
            "Temporal ordering telemetry initialized - Meter: {MeterName}",
            meterName);
    }

    #region HLC Operations

    /// <summary>
    /// Records an HLC local update (actor generates event).
    /// </summary>
    public void RecordHlcLocalUpdate(
        Guid hlcInstanceId,
        double latencyNanoseconds,
        long physicalTime,
        int logicalCounter)
    {
        _hlcUpdatesTotal.Add(1);
        _hlcLocalUpdatesTotal.Add(1);
        _hlcUpdateLatency.Record(latencyNanoseconds);

        // Track logical counter jumps
        if (_hlcStates.TryGetValue(hlcInstanceId, out var state))
        {
            if (physicalTime == state.LastPhysicalTime)
            {
                // Physical time didn't advance - logical counter incremented
                var jump = logicalCounter - state.LastLogicalCounter;
                if (jump > 0)
                {
                    _logicalCounterJumps.Record(jump);
                }
            }

            state.LastPhysicalTime = physicalTime;
            state.LastLogicalCounter = logicalCounter;
        }
        else
        {
            _hlcStates[hlcInstanceId] = new HlcMetricState
            {
                LastPhysicalTime = physicalTime,
                LastLogicalCounter = logicalCounter
            };
        }
    }

    /// <summary>
    /// Records an HLC remote update (actor receives message with timestamp).
    /// </summary>
    public void RecordHlcRemoteUpdate(
        Guid hlcInstanceId,
        double latencyNanoseconds,
        long localPhysicalTime,
        long remotePhysicalTime,
        int remoteLogicalCounter)
    {
        _hlcUpdatesTotal.Add(1);
        _hlcRemoteUpdatesTotal.Add(1);
        _hlcUpdateLatency.Record(latencyNanoseconds);

        // Check for timestamp skew
        var skew = Math.Abs(remotePhysicalTime - localPhysicalTime);
        _messageTimestampSkew.Record(skew);

        // Detect late messages (remote timestamp is older)
        if (remotePhysicalTime < localPhysicalTime)
        {
            _lateMessagesReordered.Add(1);

            _logger.LogDebug(
                "Late message reordered - Local: {Local}ns, Remote: {Remote}ns, Skew: {Skew}ns",
                localPhysicalTime,
                remotePhysicalTime,
                skew);
        }
    }

    #endregion

    #region Clock Calibration

    /// <summary>
    /// Records a successful clock calibration.
    /// </summary>
    public void RecordClockCalibration(
        long driftNanoseconds,
        double calibrationLatencyMicroseconds,
        double errorBoundNanoseconds)
    {
        _clockRecalibrationsTotal.Add(1);
        _clockDriftNanos.Record(driftNanoseconds);
        _calibrationLatency.Record(calibrationLatencyMicroseconds);

        _lastCalibrationTime = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();
        _lastObservedDrift = driftNanoseconds;

        // Alert if drift is excessive
        if (Math.Abs(driftNanoseconds) > 100_000) // >100μs
        {
            _logger.LogWarning(
                "Excessive clock drift detected - Drift: {Drift}ns, Error bound: {ErrorBound}ns",
                driftNanoseconds,
                errorBoundNanoseconds);
        }

        _logger.LogDebug(
            "Clock calibration completed - Drift: {Drift}ns, Latency: {Latency:F2}μs",
            driftNanoseconds,
            calibrationLatencyMicroseconds);
    }

    /// <summary>
    /// Records a failed clock calibration attempt.
    /// </summary>
    public void RecordCalibrationFailure(string reason)
    {
        _calibrationFailures.Add(1);

        _logger.LogError(
            "Clock calibration failed - Reason: {Reason}",
            reason);
    }

    #endregion

    #region Causal Ordering

    /// <summary>
    /// Records a detected causal ordering violation.
    /// </summary>
    public void RecordCausalViolation(
        Guid sourceActorId,
        Guid targetActorId,
        long expectedTimestamp,
        long actualTimestamp)
    {
        _causalViolationsDetected.Add(1);

        var skew = actualTimestamp - expectedTimestamp;

        _logger.LogError(
            "Causal ordering violation detected - " +
            "Source: {SourceId}, Target: {TargetId}, " +
            "Expected: {Expected}ns, Actual: {Actual}ns, Skew: {Skew}ns",
            sourceActorId,
            targetActorId,
            expectedTimestamp,
            actualTimestamp,
            skew);
    }

    /// <summary>
    /// Registers an HLC instance for monitoring.
    /// </summary>
    public void RegisterHlcInstance(Guid hlcInstanceId)
    {
        _hlcStates.TryAdd(hlcInstanceId, new HlcMetricState
        {
            LastPhysicalTime = 0,
            LastLogicalCounter = 0
        });

        _logger.LogDebug(
            "HLC instance {InstanceId} registered with telemetry",
            hlcInstanceId);
    }

    /// <summary>
    /// Unregisters an HLC instance from monitoring.
    /// </summary>
    public void UnregisterHlcInstance(Guid hlcInstanceId)
    {
        _hlcStates.TryRemove(hlcInstanceId, out _);

        _logger.LogDebug(
            "HLC instance {InstanceId} unregistered from telemetry",
            hlcInstanceId);
    }

    #endregion

    #region Observable Gauge Callbacks

    private long ObserveCurrentClockDrift()
    {
        return _lastObservedDrift;
    }

    private double ObserveTimeSinceLastCalibration()
    {
        var now = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();
        return (now - _lastCalibrationTime) / 1000.0; // Convert to seconds
    }

    private int ObserveActiveHlcInstances()
    {
        return _hlcStates.Count;
    }

    #endregion

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        _meter?.Dispose();

        _logger.LogInformation("Temporal ordering telemetry disposed");
    }

    private sealed class HlcMetricState
    {
        public required long LastPhysicalTime { get; set; }
        public required int LastLogicalCounter { get; set; }
    }
}

// Orleans.GpuBridge - GPU-native distributed computing for Microsoft Orleans
// Copyright (c) 2025 Michael Ivertowski

using System.Diagnostics;
using System.Diagnostics.Metrics;
using System.Runtime.CompilerServices;

namespace Orleans.GpuBridge.Runtime.Temporal;

/// <summary>
/// OpenTelemetry metrics for temporal coordination components.
/// Provides comprehensive observability for HLC, Vector Clocks, and fault handling.
/// </summary>
/// <remarks>
/// <para>
/// This class implements the .NET Metrics API for OpenTelemetry integration,
/// enabling real-time monitoring of temporal coordination performance.
/// </para>
/// <para>
/// Supported exporters include Prometheus, OTLP, and console output.
/// Configure via MeterProviderBuilder.AddMeter("Orleans.GpuBridge.Temporal").
/// </para>
/// </remarks>
public sealed class TemporalMetrics : IDisposable
{
    /// <summary>
    /// Meter name for temporal metrics. Use this when configuring OpenTelemetry.
    /// </summary>
    public const string MeterName = "Orleans.GpuBridge.Temporal";

    /// <summary>
    /// Meter version following semantic versioning.
    /// </summary>
    public const string MeterVersion = "1.0.0";

    private readonly Meter _meter;
    private bool _disposed;

    // HLC Metrics
    private readonly Counter<long> _hlcNowCallsTotal;
    private readonly Counter<long> _hlcUpdateCallsTotal;
    private readonly Histogram<double> _hlcNowLatencyNs;
    private readonly Histogram<double> _hlcUpdateLatencyNs;
    private readonly Counter<long> _hlcLogicalCounterIncrements;
    private readonly ObservableGauge<long> _hlcCurrentPhysicalTime;
    private readonly ObservableGauge<long> _hlcCurrentLogicalCounter;

    // Clock Drift Metrics
    private readonly Histogram<double> _clockDriftNs;
    private readonly Counter<long> _clockJumpsDetected;
    private readonly Counter<long> _clockJumpsForward;
    private readonly Counter<long> _clockJumpsBackward;
    private readonly Histogram<double> _clockJumpMagnitudeMs;

    // Vector Clock Metrics
    private readonly Counter<long> _vectorClockIncrements;
    private readonly Counter<long> _vectorClockMerges;
    private readonly Histogram<double> _vectorClockMergeLatencyNs;
    private readonly Histogram<int> _vectorClockEntryCount;
    private readonly Counter<long> _vectorClockConcurrencyDetections;

    // Fault Handler Metrics
    private readonly Counter<long> _faultHandlerClockJumpsHandled;
    private readonly Counter<long> _faultHandlerRecoveryAttempts;
    private readonly Counter<long> _faultHandlerRecoverySuccesses;
    private readonly Counter<long> _faultHandlerRecoveryFailures;
    private readonly Histogram<double> _faultHandlerRecoveryTimeMs;

    // Network Retry Metrics
    private readonly Counter<long> _networkRetryAttempts;
    private readonly Counter<long> _networkRetrySuccesses;
    private readonly Counter<long> _networkRetryExhausted;
    private readonly Histogram<double> _networkRetryDelayMs;
    private readonly Histogram<int> _networkRetryAttemptsPerOperation;

    // PTP Hardware Monitor Metrics
    private readonly Counter<long> _ptpDeviceFailures;
    private readonly Counter<long> _ptpSynchronizationAttempts;
    private readonly Counter<long> _ptpSynchronizationSuccesses;
    private readonly Histogram<double> _ptpOffsetNs;

    // Message Queue Metrics
    private readonly Counter<long> _temporalMessagesEnqueued;
    private readonly Counter<long> _temporalMessagesDequeued;
    private readonly Counter<long> _temporalMessagesReordered;
    private readonly Histogram<double> _temporalMessageLatencyMs;
    private readonly ObservableGauge<int> _temporalQueueDepth;

    // Snapshot values for observable gauges
    private long _currentPhysicalTime;
    private long _currentLogicalCounter;
    private int _currentQueueDepth;

    /// <summary>
    /// Initializes a new instance of <see cref="TemporalMetrics"/>.
    /// </summary>
    /// <param name="meterFactory">Optional meter factory for creating the meter.</param>
    public TemporalMetrics(IMeterFactory? meterFactory = null)
    {
        _meter = meterFactory?.Create(MeterName, MeterVersion)
            ?? new Meter(MeterName, MeterVersion);

        // Initialize HLC metrics
        _hlcNowCallsTotal = _meter.CreateCounter<long>(
            "gpubridge.hlc.now.calls.total",
            unit: "{calls}",
            description: "Total number of HLC.Now() calls");

        _hlcUpdateCallsTotal = _meter.CreateCounter<long>(
            "gpubridge.hlc.update.calls.total",
            unit: "{calls}",
            description: "Total number of HLC.Update() calls");

        _hlcNowLatencyNs = _meter.CreateHistogram<double>(
            "gpubridge.hlc.now.latency.ns",
            unit: "ns",
            description: "Latency of HLC.Now() calls in nanoseconds");

        _hlcUpdateLatencyNs = _meter.CreateHistogram<double>(
            "gpubridge.hlc.update.latency.ns",
            unit: "ns",
            description: "Latency of HLC.Update() calls in nanoseconds");

        _hlcLogicalCounterIncrements = _meter.CreateCounter<long>(
            "gpubridge.hlc.logical_counter.increments.total",
            unit: "{increments}",
            description: "Total logical counter increments (indicates clock contention)");

        _hlcCurrentPhysicalTime = _meter.CreateObservableGauge(
            "gpubridge.hlc.physical_time.current",
            () => Volatile.Read(ref _currentPhysicalTime),
            unit: "ticks",
            description: "Current HLC physical time in ticks");

        _hlcCurrentLogicalCounter = _meter.CreateObservableGauge(
            "gpubridge.hlc.logical_counter.current",
            () => Volatile.Read(ref _currentLogicalCounter),
            unit: "{count}",
            description: "Current HLC logical counter value");

        // Initialize Clock Drift metrics
        _clockDriftNs = _meter.CreateHistogram<double>(
            "gpubridge.clock.drift.ns",
            unit: "ns",
            description: "Observed clock drift from reference time");

        _clockJumpsDetected = _meter.CreateCounter<long>(
            "gpubridge.clock.jumps.total",
            unit: "{jumps}",
            description: "Total clock jumps detected");

        _clockJumpsForward = _meter.CreateCounter<long>(
            "gpubridge.clock.jumps.forward.total",
            unit: "{jumps}",
            description: "Total forward clock jumps detected");

        _clockJumpsBackward = _meter.CreateCounter<long>(
            "gpubridge.clock.jumps.backward.total",
            unit: "{jumps}",
            description: "Total backward clock jumps detected");

        _clockJumpMagnitudeMs = _meter.CreateHistogram<double>(
            "gpubridge.clock.jump.magnitude.ms",
            unit: "ms",
            description: "Magnitude of detected clock jumps");

        // Initialize Vector Clock metrics
        _vectorClockIncrements = _meter.CreateCounter<long>(
            "gpubridge.vectorclock.increments.total",
            unit: "{increments}",
            description: "Total vector clock increments");

        _vectorClockMerges = _meter.CreateCounter<long>(
            "gpubridge.vectorclock.merges.total",
            unit: "{merges}",
            description: "Total vector clock merge operations");

        _vectorClockMergeLatencyNs = _meter.CreateHistogram<double>(
            "gpubridge.vectorclock.merge.latency.ns",
            unit: "ns",
            description: "Latency of vector clock merge operations");

        _vectorClockEntryCount = _meter.CreateHistogram<int>(
            "gpubridge.vectorclock.entries",
            unit: "{entries}",
            description: "Number of entries in vector clocks");

        _vectorClockConcurrencyDetections = _meter.CreateCounter<long>(
            "gpubridge.vectorclock.concurrency.detected.total",
            unit: "{detections}",
            description: "Concurrent event detections via vector clocks");

        // Initialize Fault Handler metrics
        _faultHandlerClockJumpsHandled = _meter.CreateCounter<long>(
            "gpubridge.faulthandler.clockjumps.handled.total",
            unit: "{jumps}",
            description: "Clock jumps handled by fault handler");

        _faultHandlerRecoveryAttempts = _meter.CreateCounter<long>(
            "gpubridge.faulthandler.recovery.attempts.total",
            unit: "{attempts}",
            description: "Recovery attempts by fault handler");

        _faultHandlerRecoverySuccesses = _meter.CreateCounter<long>(
            "gpubridge.faulthandler.recovery.successes.total",
            unit: "{successes}",
            description: "Successful fault recoveries");

        _faultHandlerRecoveryFailures = _meter.CreateCounter<long>(
            "gpubridge.faulthandler.recovery.failures.total",
            unit: "{failures}",
            description: "Failed fault recoveries");

        _faultHandlerRecoveryTimeMs = _meter.CreateHistogram<double>(
            "gpubridge.faulthandler.recovery.duration.ms",
            unit: "ms",
            description: "Duration of fault recovery operations");

        // Initialize Network Retry metrics
        _networkRetryAttempts = _meter.CreateCounter<long>(
            "gpubridge.network.retry.attempts.total",
            unit: "{attempts}",
            description: "Total network retry attempts");

        _networkRetrySuccesses = _meter.CreateCounter<long>(
            "gpubridge.network.retry.successes.total",
            unit: "{successes}",
            description: "Successful network retries");

        _networkRetryExhausted = _meter.CreateCounter<long>(
            "gpubridge.network.retry.exhausted.total",
            unit: "{exhausted}",
            description: "Operations that exhausted all retries");

        _networkRetryDelayMs = _meter.CreateHistogram<double>(
            "gpubridge.network.retry.delay.ms",
            unit: "ms",
            description: "Network retry delay durations");

        _networkRetryAttemptsPerOperation = _meter.CreateHistogram<int>(
            "gpubridge.network.retry.attempts_per_operation",
            unit: "{attempts}",
            description: "Number of retry attempts per operation");

        // Initialize PTP Hardware Monitor metrics
        _ptpDeviceFailures = _meter.CreateCounter<long>(
            "gpubridge.ptp.device.failures.total",
            unit: "{failures}",
            description: "PTP hardware device failures detected");

        _ptpSynchronizationAttempts = _meter.CreateCounter<long>(
            "gpubridge.ptp.sync.attempts.total",
            unit: "{attempts}",
            description: "PTP synchronization attempts");

        _ptpSynchronizationSuccesses = _meter.CreateCounter<long>(
            "gpubridge.ptp.sync.successes.total",
            unit: "{successes}",
            description: "Successful PTP synchronizations");

        _ptpOffsetNs = _meter.CreateHistogram<double>(
            "gpubridge.ptp.offset.ns",
            unit: "ns",
            description: "PTP offset from master clock");

        // Initialize Message Queue metrics
        _temporalMessagesEnqueued = _meter.CreateCounter<long>(
            "gpubridge.temporal.messages.enqueued.total",
            unit: "{messages}",
            description: "Messages enqueued to temporal queue");

        _temporalMessagesDequeued = _meter.CreateCounter<long>(
            "gpubridge.temporal.messages.dequeued.total",
            unit: "{messages}",
            description: "Messages dequeued from temporal queue");

        _temporalMessagesReordered = _meter.CreateCounter<long>(
            "gpubridge.temporal.messages.reordered.total",
            unit: "{messages}",
            description: "Messages reordered for causal consistency");

        _temporalMessageLatencyMs = _meter.CreateHistogram<double>(
            "gpubridge.temporal.message.latency.ms",
            unit: "ms",
            description: "End-to-end temporal message latency");

        _temporalQueueDepth = _meter.CreateObservableGauge(
            "gpubridge.temporal.queue.depth",
            () => Volatile.Read(ref _currentQueueDepth),
            unit: "{messages}",
            description: "Current temporal message queue depth");
    }

    #region HLC Metrics Recording

    /// <summary>
    /// Records a HLC.Now() call with its latency.
    /// </summary>
    /// <param name="latencyNs">Latency in nanoseconds.</param>
    /// <param name="physicalTime">Current physical time.</param>
    /// <param name="logicalCounter">Current logical counter.</param>
    /// <param name="logicalCounterIncremented">Whether the logical counter was incremented.</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void RecordHlcNow(double latencyNs, long physicalTime, long logicalCounter, bool logicalCounterIncremented)
    {
        _hlcNowCallsTotal.Add(1);
        _hlcNowLatencyNs.Record(latencyNs);

        if (logicalCounterIncremented)
        {
            _hlcLogicalCounterIncrements.Add(1);
        }

        Volatile.Write(ref _currentPhysicalTime, physicalTime);
        Volatile.Write(ref _currentLogicalCounter, logicalCounter);
    }

    /// <summary>
    /// Records a HLC.Update() call with its latency.
    /// </summary>
    /// <param name="latencyNs">Latency in nanoseconds.</param>
    /// <param name="physicalTime">Current physical time after update.</param>
    /// <param name="logicalCounter">Current logical counter after update.</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void RecordHlcUpdate(double latencyNs, long physicalTime, long logicalCounter)
    {
        _hlcUpdateCallsTotal.Add(1);
        _hlcUpdateLatencyNs.Record(latencyNs);

        Volatile.Write(ref _currentPhysicalTime, physicalTime);
        Volatile.Write(ref _currentLogicalCounter, logicalCounter);
    }

    #endregion

    #region Clock Drift Metrics Recording

    /// <summary>
    /// Records observed clock drift.
    /// </summary>
    /// <param name="driftNs">Clock drift in nanoseconds.</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void RecordClockDrift(double driftNs)
    {
        _clockDriftNs.Record(driftNs);
    }

    /// <summary>
    /// Records a detected clock jump.
    /// </summary>
    /// <param name="magnitudeMs">Jump magnitude in milliseconds (positive = forward, negative = backward).</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void RecordClockJump(double magnitudeMs)
    {
        _clockJumpsDetected.Add(1);
        _clockJumpMagnitudeMs.Record(Math.Abs(magnitudeMs));

        if (magnitudeMs > 0)
        {
            _clockJumpsForward.Add(1);
        }
        else
        {
            _clockJumpsBackward.Add(1);
        }
    }

    #endregion

    #region Vector Clock Metrics Recording

    /// <summary>
    /// Records a vector clock increment operation.
    /// </summary>
    /// <param name="entryCount">Current number of entries in the vector clock.</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void RecordVectorClockIncrement(int entryCount)
    {
        _vectorClockIncrements.Add(1);
        _vectorClockEntryCount.Record(entryCount);
    }

    /// <summary>
    /// Records a vector clock merge operation.
    /// </summary>
    /// <param name="latencyNs">Merge operation latency in nanoseconds.</param>
    /// <param name="resultingEntryCount">Number of entries after merge.</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void RecordVectorClockMerge(double latencyNs, int resultingEntryCount)
    {
        _vectorClockMerges.Add(1);
        _vectorClockMergeLatencyNs.Record(latencyNs);
        _vectorClockEntryCount.Record(resultingEntryCount);
    }

    /// <summary>
    /// Records detection of concurrent events via vector clock comparison.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void RecordVectorClockConcurrencyDetected()
    {
        _vectorClockConcurrencyDetections.Add(1);
    }

    #endregion

    #region Fault Handler Metrics Recording

    /// <summary>
    /// Records a clock jump handled by the fault handler.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void RecordFaultHandlerClockJump()
    {
        _faultHandlerClockJumpsHandled.Add(1);
    }

    /// <summary>
    /// Records a fault recovery attempt with its outcome.
    /// </summary>
    /// <param name="success">Whether recovery was successful.</param>
    /// <param name="durationMs">Recovery duration in milliseconds.</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void RecordFaultHandlerRecovery(bool success, double durationMs)
    {
        _faultHandlerRecoveryAttempts.Add(1);
        _faultHandlerRecoveryTimeMs.Record(durationMs);

        if (success)
        {
            _faultHandlerRecoverySuccesses.Add(1);
        }
        else
        {
            _faultHandlerRecoveryFailures.Add(1);
        }
    }

    #endregion

    #region Network Retry Metrics Recording

    /// <summary>
    /// Records a network retry attempt.
    /// </summary>
    /// <param name="delayMs">Delay before retry in milliseconds.</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void RecordNetworkRetryAttempt(double delayMs)
    {
        _networkRetryAttempts.Add(1);
        _networkRetryDelayMs.Record(delayMs);
    }

    /// <summary>
    /// Records completion of a retry sequence.
    /// </summary>
    /// <param name="success">Whether the operation eventually succeeded.</param>
    /// <param name="attemptCount">Total number of attempts.</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void RecordNetworkRetryCompletion(bool success, int attemptCount)
    {
        _networkRetryAttemptsPerOperation.Record(attemptCount);

        if (success)
        {
            _networkRetrySuccesses.Add(1);
        }
        else
        {
            _networkRetryExhausted.Add(1);
        }
    }

    #endregion

    #region PTP Hardware Metrics Recording

    /// <summary>
    /// Records a PTP hardware device failure.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void RecordPtpDeviceFailure()
    {
        _ptpDeviceFailures.Add(1);
    }

    /// <summary>
    /// Records a PTP synchronization attempt.
    /// </summary>
    /// <param name="success">Whether synchronization succeeded.</param>
    /// <param name="offsetNs">Offset from master clock in nanoseconds.</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void RecordPtpSynchronization(bool success, double offsetNs)
    {
        _ptpSynchronizationAttempts.Add(1);

        if (success)
        {
            _ptpSynchronizationSuccesses.Add(1);
            _ptpOffsetNs.Record(offsetNs);
        }
    }

    #endregion

    #region Temporal Message Queue Metrics Recording

    /// <summary>
    /// Records a message being enqueued to the temporal queue.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void RecordMessageEnqueued()
    {
        _temporalMessagesEnqueued.Add(1);
        Interlocked.Increment(ref _currentQueueDepth);
    }

    /// <summary>
    /// Records a message being dequeued from the temporal queue.
    /// </summary>
    /// <param name="latencyMs">End-to-end message latency in milliseconds.</param>
    /// <param name="wasReordered">Whether the message was reordered for causal consistency.</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void RecordMessageDequeued(double latencyMs, bool wasReordered)
    {
        _temporalMessagesDequeued.Add(1);
        _temporalMessageLatencyMs.Record(latencyMs);
        Interlocked.Decrement(ref _currentQueueDepth);

        if (wasReordered)
        {
            _temporalMessagesReordered.Add(1);
        }
    }

    /// <summary>
    /// Updates the current queue depth for observable gauge.
    /// </summary>
    /// <param name="depth">Current queue depth.</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void UpdateQueueDepth(int depth)
    {
        Volatile.Write(ref _currentQueueDepth, depth);
    }

    #endregion

    #region Convenience Methods

    /// <summary>
    /// Creates a scope for timing an HLC.Now() operation.
    /// </summary>
    /// <returns>A disposable timing scope.</returns>
    public HlcNowTimingScope StartHlcNowTiming() => new(this);

    /// <summary>
    /// Creates a scope for timing an HLC.Update() operation.
    /// </summary>
    /// <returns>A disposable timing scope.</returns>
    public HlcUpdateTimingScope StartHlcUpdateTiming() => new(this);

    /// <summary>
    /// Creates a scope for timing a vector clock merge operation.
    /// </summary>
    /// <returns>A disposable timing scope.</returns>
    public VectorClockMergeTimingScope StartVectorClockMergeTiming() => new(this);

    #endregion

    #region Timing Scopes

    /// <summary>
    /// Timing scope for HLC.Now() operations.
    /// </summary>
    public readonly struct HlcNowTimingScope : IDisposable
    {
        private readonly TemporalMetrics _metrics;
        private readonly long _startTicks;

        internal HlcNowTimingScope(TemporalMetrics metrics)
        {
            _metrics = metrics;
            _startTicks = Stopwatch.GetTimestamp();
        }

        /// <summary>
        /// Completes the timing and records the metric.
        /// </summary>
        /// <param name="physicalTime">Physical time from the operation.</param>
        /// <param name="logicalCounter">Logical counter from the operation.</param>
        /// <param name="logicalCounterIncremented">Whether counter was incremented.</param>
        public void Complete(long physicalTime, long logicalCounter, bool logicalCounterIncremented)
        {
            var elapsed = Stopwatch.GetTimestamp() - _startTicks;
            var elapsedNs = (double)elapsed / Stopwatch.Frequency * 1_000_000_000;
            _metrics.RecordHlcNow(elapsedNs, physicalTime, logicalCounter, logicalCounterIncremented);
        }

        /// <summary>
        /// Disposes the scope. Use <see cref="Complete"/> for full metrics.
        /// </summary>
        public void Dispose()
        {
            // No-op; use Complete() for metrics recording
        }
    }

    /// <summary>
    /// Timing scope for HLC.Update() operations.
    /// </summary>
    public readonly struct HlcUpdateTimingScope : IDisposable
    {
        private readonly TemporalMetrics _metrics;
        private readonly long _startTicks;

        internal HlcUpdateTimingScope(TemporalMetrics metrics)
        {
            _metrics = metrics;
            _startTicks = Stopwatch.GetTimestamp();
        }

        /// <summary>
        /// Completes the timing and records the metric.
        /// </summary>
        /// <param name="physicalTime">Physical time after update.</param>
        /// <param name="logicalCounter">Logical counter after update.</param>
        public void Complete(long physicalTime, long logicalCounter)
        {
            var elapsed = Stopwatch.GetTimestamp() - _startTicks;
            var elapsedNs = (double)elapsed / Stopwatch.Frequency * 1_000_000_000;
            _metrics.RecordHlcUpdate(elapsedNs, physicalTime, logicalCounter);
        }

        /// <summary>
        /// Disposes the scope. Use <see cref="Complete"/> for full metrics.
        /// </summary>
        public void Dispose()
        {
            // No-op; use Complete() for metrics recording
        }
    }

    /// <summary>
    /// Timing scope for vector clock merge operations.
    /// </summary>
    public readonly struct VectorClockMergeTimingScope : IDisposable
    {
        private readonly TemporalMetrics _metrics;
        private readonly long _startTicks;

        internal VectorClockMergeTimingScope(TemporalMetrics metrics)
        {
            _metrics = metrics;
            _startTicks = Stopwatch.GetTimestamp();
        }

        /// <summary>
        /// Completes the timing and records the metric.
        /// </summary>
        /// <param name="resultingEntryCount">Entry count after merge.</param>
        public void Complete(int resultingEntryCount)
        {
            var elapsed = Stopwatch.GetTimestamp() - _startTicks;
            var elapsedNs = (double)elapsed / Stopwatch.Frequency * 1_000_000_000;
            _metrics.RecordVectorClockMerge(elapsedNs, resultingEntryCount);
        }

        /// <summary>
        /// Disposes the scope. Use <see cref="Complete"/> for full metrics.
        /// </summary>
        public void Dispose()
        {
            // No-op; use Complete() for metrics recording
        }
    }

    #endregion

    /// <summary>
    /// Disposes the meter and releases resources.
    /// </summary>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _meter.Dispose();
    }
}

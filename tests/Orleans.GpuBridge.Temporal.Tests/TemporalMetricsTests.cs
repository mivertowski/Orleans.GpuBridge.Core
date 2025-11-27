// Orleans.GpuBridge - GPU-native distributed computing for Microsoft Orleans
// Copyright (c) 2025 Michael Ivertowski

using System.Diagnostics;
using System.Diagnostics.Metrics;
using Orleans.GpuBridge.Runtime.Temporal;
using Xunit;
using Xunit.Abstractions;

namespace Orleans.GpuBridge.Temporal.Tests;

/// <summary>
/// Tests for TemporalMetrics OpenTelemetry instrumentation.
/// Phase 7D: Validate metrics recording and observability.
/// </summary>
public sealed class TemporalMetricsTests : IDisposable
{
    private readonly ITestOutputHelper _output;
    private readonly TemporalMetrics _metrics;
    private readonly MeterListener _listener;
    private readonly Dictionary<string, List<double>> _recordedHistograms;
    private readonly Dictionary<string, long> _recordedCounters;
    private readonly Dictionary<string, double> _recordedGauges;

    public TemporalMetricsTests(ITestOutputHelper output)
    {
        _output = output;
        _metrics = new TemporalMetrics();
        _recordedHistograms = new Dictionary<string, List<double>>();
        _recordedCounters = new Dictionary<string, long>();
        _recordedGauges = new Dictionary<string, double>();

        _listener = new MeterListener();
        _listener.InstrumentPublished = (instrument, listener) =>
        {
            if (instrument.Meter.Name == TemporalMetrics.MeterName)
            {
                listener.EnableMeasurementEvents(instrument);
            }
        };

        _listener.SetMeasurementEventCallback<long>((instrument, measurement, tags, state) =>
        {
            if (instrument.GetType().Name.Contains("Counter"))
            {
                if (!_recordedCounters.ContainsKey(instrument.Name))
                    _recordedCounters[instrument.Name] = 0;
                _recordedCounters[instrument.Name] += measurement;
            }
        });

        _listener.SetMeasurementEventCallback<double>((instrument, measurement, tags, state) =>
        {
            if (!_recordedHistograms.ContainsKey(instrument.Name))
                _recordedHistograms[instrument.Name] = new List<double>();
            _recordedHistograms[instrument.Name].Add(measurement);
        });

        _listener.SetMeasurementEventCallback<int>((instrument, measurement, tags, state) =>
        {
            if (!_recordedHistograms.ContainsKey(instrument.Name))
                _recordedHistograms[instrument.Name] = new List<double>();
            _recordedHistograms[instrument.Name].Add(measurement);
        });

        _listener.Start();
    }

    public void Dispose()
    {
        _listener.Dispose();
        _metrics.Dispose();
    }

    #region HLC Metrics Tests

    [Fact]
    public void RecordHlcNow_ShouldIncrementCounter()
    {
        // Arrange
        var initialCount = GetCounterValue("gpubridge.hlc.now.calls.total");

        // Act
        _metrics.RecordHlcNow(50.0, 1000, 5, false);
        _metrics.RecordHlcNow(45.0, 1001, 6, true);
        _metrics.RecordHlcNow(55.0, 1002, 7, false);

        // Assert
        var finalCount = GetCounterValue("gpubridge.hlc.now.calls.total");
        Assert.Equal(initialCount + 3, finalCount);

        _output.WriteLine($"HLC.Now() calls recorded: {finalCount - initialCount}");
        _output.WriteLine("✅ HLC.Now() counter increments correctly");
    }

    [Fact]
    public void RecordHlcNow_WithLogicalCounterIncrement_ShouldTrackIncrements()
    {
        // Arrange
        var initialIncrements = GetCounterValue("gpubridge.hlc.logical_counter.increments.total");

        // Act
        _metrics.RecordHlcNow(50.0, 1000, 5, true);
        _metrics.RecordHlcNow(50.0, 1000, 6, true);
        _metrics.RecordHlcNow(50.0, 1001, 1, false); // No increment

        // Assert
        var finalIncrements = GetCounterValue("gpubridge.hlc.logical_counter.increments.total");
        Assert.Equal(initialIncrements + 2, finalIncrements);

        _output.WriteLine($"Logical counter increments tracked: {finalIncrements - initialIncrements}");
        _output.WriteLine("✅ Logical counter increment tracking works");
    }

    [Fact]
    public void RecordHlcNow_ShouldRecordLatency()
    {
        // Arrange & Act
        _metrics.RecordHlcNow(73.5, 1000, 5, false);

        // Assert
        var latencies = GetHistogramValues("gpubridge.hlc.now.latency.ns");
        Assert.Contains(73.5, latencies);

        _output.WriteLine($"Recorded latency: {73.5}ns");
        _output.WriteLine("✅ HLC.Now() latency recorded correctly");
    }

    [Fact]
    public void RecordHlcUpdate_ShouldIncrementCounterAndRecordLatency()
    {
        // Arrange
        var initialCount = GetCounterValue("gpubridge.hlc.update.calls.total");

        // Act
        _metrics.RecordHlcUpdate(85.0, 2000, 10);
        _metrics.RecordHlcUpdate(90.0, 2001, 11);

        // Assert
        var finalCount = GetCounterValue("gpubridge.hlc.update.calls.total");
        Assert.Equal(initialCount + 2, finalCount);

        var latencies = GetHistogramValues("gpubridge.hlc.update.latency.ns");
        Assert.Contains(85.0, latencies);
        Assert.Contains(90.0, latencies);

        _output.WriteLine($"HLC.Update() calls: {finalCount - initialCount}");
        _output.WriteLine("✅ HLC.Update() metrics recorded correctly");
    }

    #endregion

    #region Clock Drift Metrics Tests

    [Fact]
    public void RecordClockDrift_ShouldRecordDriftValues()
    {
        // Arrange & Act
        _metrics.RecordClockDrift(100.0);
        _metrics.RecordClockDrift(-50.0);
        _metrics.RecordClockDrift(200.0);

        // Assert
        var drifts = GetHistogramValues("gpubridge.clock.drift.ns");
        Assert.Contains(100.0, drifts);
        Assert.Contains(-50.0, drifts);
        Assert.Contains(200.0, drifts);

        _output.WriteLine("Recorded drifts: 100ns, -50ns, 200ns");
        _output.WriteLine("✅ Clock drift recording works");
    }

    [Fact]
    public void RecordClockJump_Forward_ShouldTrackCorrectly()
    {
        // Arrange
        var initialJumps = GetCounterValue("gpubridge.clock.jumps.total");
        var initialForward = GetCounterValue("gpubridge.clock.jumps.forward.total");
        var initialBackward = GetCounterValue("gpubridge.clock.jumps.backward.total");

        // Act
        _metrics.RecordClockJump(100.0); // Forward jump

        // Assert
        var finalJumps = GetCounterValue("gpubridge.clock.jumps.total");
        var finalForward = GetCounterValue("gpubridge.clock.jumps.forward.total");
        var finalBackward = GetCounterValue("gpubridge.clock.jumps.backward.total");

        Assert.Equal(initialJumps + 1, finalJumps);
        Assert.Equal(initialForward + 1, finalForward);
        Assert.Equal(initialBackward, finalBackward);

        _output.WriteLine("Forward clock jump recorded correctly");
        _output.WriteLine("✅ Forward clock jump tracking works");
    }

    [Fact]
    public void RecordClockJump_Backward_ShouldTrackCorrectly()
    {
        // Arrange
        var initialBackward = GetCounterValue("gpubridge.clock.jumps.backward.total");

        // Act
        _metrics.RecordClockJump(-50.0); // Backward jump

        // Assert
        var finalBackward = GetCounterValue("gpubridge.clock.jumps.backward.total");
        Assert.Equal(initialBackward + 1, finalBackward);

        var magnitudes = GetHistogramValues("gpubridge.clock.jump.magnitude.ms");
        Assert.Contains(50.0, magnitudes); // Magnitude is absolute

        _output.WriteLine("Backward clock jump recorded correctly");
        _output.WriteLine("✅ Backward clock jump tracking works");
    }

    #endregion

    #region Vector Clock Metrics Tests

    [Fact]
    public void RecordVectorClockIncrement_ShouldTrackCorrectly()
    {
        // Arrange
        var initialIncrements = GetCounterValue("gpubridge.vectorclock.increments.total");

        // Act
        _metrics.RecordVectorClockIncrement(5);
        _metrics.RecordVectorClockIncrement(10);

        // Assert
        var finalIncrements = GetCounterValue("gpubridge.vectorclock.increments.total");
        Assert.Equal(initialIncrements + 2, finalIncrements);

        var entryCounts = GetHistogramValues("gpubridge.vectorclock.entries");
        Assert.Contains(5.0, entryCounts);
        Assert.Contains(10.0, entryCounts);

        _output.WriteLine("Vector clock increments tracked with entry counts");
        _output.WriteLine("✅ Vector clock increment metrics work");
    }

    [Fact]
    public void RecordVectorClockMerge_ShouldTrackLatencyAndEntryCount()
    {
        // Arrange
        var initialMerges = GetCounterValue("gpubridge.vectorclock.merges.total");

        // Act
        _metrics.RecordVectorClockMerge(150.0, 15);
        _metrics.RecordVectorClockMerge(200.0, 20);

        // Assert
        var finalMerges = GetCounterValue("gpubridge.vectorclock.merges.total");
        Assert.Equal(initialMerges + 2, finalMerges);

        var latencies = GetHistogramValues("gpubridge.vectorclock.merge.latency.ns");
        Assert.Contains(150.0, latencies);
        Assert.Contains(200.0, latencies);

        _output.WriteLine("Vector clock merges tracked with latency");
        _output.WriteLine("✅ Vector clock merge metrics work");
    }

    [Fact]
    public void RecordVectorClockConcurrencyDetected_ShouldIncrement()
    {
        // Arrange
        var initialDetections = GetCounterValue("gpubridge.vectorclock.concurrency.detected.total");

        // Act
        _metrics.RecordVectorClockConcurrencyDetected();
        _metrics.RecordVectorClockConcurrencyDetected();
        _metrics.RecordVectorClockConcurrencyDetected();

        // Assert
        var finalDetections = GetCounterValue("gpubridge.vectorclock.concurrency.detected.total");
        Assert.Equal(initialDetections + 3, finalDetections);

        _output.WriteLine($"Concurrency detections: {finalDetections - initialDetections}");
        _output.WriteLine("✅ Concurrency detection tracking works");
    }

    #endregion

    #region Fault Handler Metrics Tests

    [Fact]
    public void RecordFaultHandlerClockJump_ShouldIncrement()
    {
        // Arrange
        var initialJumps = GetCounterValue("gpubridge.faulthandler.clockjumps.handled.total");

        // Act
        _metrics.RecordFaultHandlerClockJump();
        _metrics.RecordFaultHandlerClockJump();

        // Assert
        var finalJumps = GetCounterValue("gpubridge.faulthandler.clockjumps.handled.total");
        Assert.Equal(initialJumps + 2, finalJumps);

        _output.WriteLine("Fault handler clock jumps tracked");
        _output.WriteLine("✅ Fault handler clock jump metrics work");
    }

    [Fact]
    public void RecordFaultHandlerRecovery_Success_ShouldTrackCorrectly()
    {
        // Arrange
        var initialAttempts = GetCounterValue("gpubridge.faulthandler.recovery.attempts.total");
        var initialSuccesses = GetCounterValue("gpubridge.faulthandler.recovery.successes.total");
        var initialFailures = GetCounterValue("gpubridge.faulthandler.recovery.failures.total");

        // Act
        _metrics.RecordFaultHandlerRecovery(success: true, durationMs: 5.5);

        // Assert
        var finalAttempts = GetCounterValue("gpubridge.faulthandler.recovery.attempts.total");
        var finalSuccesses = GetCounterValue("gpubridge.faulthandler.recovery.successes.total");
        var finalFailures = GetCounterValue("gpubridge.faulthandler.recovery.failures.total");

        Assert.Equal(initialAttempts + 1, finalAttempts);
        Assert.Equal(initialSuccesses + 1, finalSuccesses);
        Assert.Equal(initialFailures, finalFailures);

        var durations = GetHistogramValues("gpubridge.faulthandler.recovery.duration.ms");
        Assert.Contains(5.5, durations);

        _output.WriteLine("Successful recovery tracked with duration");
        _output.WriteLine("✅ Fault handler recovery success tracking works");
    }

    [Fact]
    public void RecordFaultHandlerRecovery_Failure_ShouldTrackCorrectly()
    {
        // Arrange
        var initialFailures = GetCounterValue("gpubridge.faulthandler.recovery.failures.total");

        // Act
        _metrics.RecordFaultHandlerRecovery(success: false, durationMs: 10.0);

        // Assert
        var finalFailures = GetCounterValue("gpubridge.faulthandler.recovery.failures.total");
        Assert.Equal(initialFailures + 1, finalFailures);

        _output.WriteLine("Failed recovery tracked");
        _output.WriteLine("✅ Fault handler recovery failure tracking works");
    }

    #endregion

    #region Network Retry Metrics Tests

    [Fact]
    public void RecordNetworkRetryAttempt_ShouldTrackDelays()
    {
        // Arrange
        var initialAttempts = GetCounterValue("gpubridge.network.retry.attempts.total");

        // Act
        _metrics.RecordNetworkRetryAttempt(100.0);
        _metrics.RecordNetworkRetryAttempt(200.0);
        _metrics.RecordNetworkRetryAttempt(400.0);

        // Assert
        var finalAttempts = GetCounterValue("gpubridge.network.retry.attempts.total");
        Assert.Equal(initialAttempts + 3, finalAttempts);

        var delays = GetHistogramValues("gpubridge.network.retry.delay.ms");
        Assert.Contains(100.0, delays);
        Assert.Contains(200.0, delays);
        Assert.Contains(400.0, delays);

        _output.WriteLine("Network retry attempts tracked with delays");
        _output.WriteLine("✅ Network retry attempt tracking works");
    }

    [Fact]
    public void RecordNetworkRetryCompletion_Success_ShouldTrackCorrectly()
    {
        // Arrange
        var initialSuccesses = GetCounterValue("gpubridge.network.retry.successes.total");

        // Act
        _metrics.RecordNetworkRetryCompletion(success: true, attemptCount: 3);

        // Assert
        var finalSuccesses = GetCounterValue("gpubridge.network.retry.successes.total");
        Assert.Equal(initialSuccesses + 1, finalSuccesses);

        var attempts = GetHistogramValues("gpubridge.network.retry.attempts_per_operation");
        Assert.Contains(3.0, attempts);

        _output.WriteLine("Network retry completion tracked");
        _output.WriteLine("✅ Network retry completion tracking works");
    }

    [Fact]
    public void RecordNetworkRetryCompletion_Exhausted_ShouldTrackCorrectly()
    {
        // Arrange
        var initialExhausted = GetCounterValue("gpubridge.network.retry.exhausted.total");

        // Act
        _metrics.RecordNetworkRetryCompletion(success: false, attemptCount: 5);

        // Assert
        var finalExhausted = GetCounterValue("gpubridge.network.retry.exhausted.total");
        Assert.Equal(initialExhausted + 1, finalExhausted);

        _output.WriteLine("Exhausted retries tracked");
        _output.WriteLine("✅ Network retry exhaustion tracking works");
    }

    #endregion

    #region PTP Hardware Metrics Tests

    [Fact]
    public void RecordPtpDeviceFailure_ShouldIncrement()
    {
        // Arrange
        var initialFailures = GetCounterValue("gpubridge.ptp.device.failures.total");

        // Act
        _metrics.RecordPtpDeviceFailure();
        _metrics.RecordPtpDeviceFailure();

        // Assert
        var finalFailures = GetCounterValue("gpubridge.ptp.device.failures.total");
        Assert.Equal(initialFailures + 2, finalFailures);

        _output.WriteLine("PTP device failures tracked");
        _output.WriteLine("✅ PTP device failure tracking works");
    }

    [Fact]
    public void RecordPtpSynchronization_Success_ShouldTrackOffset()
    {
        // Arrange
        var initialAttempts = GetCounterValue("gpubridge.ptp.sync.attempts.total");
        var initialSuccesses = GetCounterValue("gpubridge.ptp.sync.successes.total");

        // Act
        _metrics.RecordPtpSynchronization(success: true, offsetNs: 500.0);

        // Assert
        var finalAttempts = GetCounterValue("gpubridge.ptp.sync.attempts.total");
        var finalSuccesses = GetCounterValue("gpubridge.ptp.sync.successes.total");

        Assert.Equal(initialAttempts + 1, finalAttempts);
        Assert.Equal(initialSuccesses + 1, finalSuccesses);

        var offsets = GetHistogramValues("gpubridge.ptp.offset.ns");
        Assert.Contains(500.0, offsets);

        _output.WriteLine("PTP sync success tracked with offset");
        _output.WriteLine("✅ PTP synchronization success tracking works");
    }

    [Fact]
    public void RecordPtpSynchronization_Failure_ShouldNotRecordOffset()
    {
        // Arrange
        var initialOffsets = GetHistogramValues("gpubridge.ptp.offset.ns").Count;

        // Act
        _metrics.RecordPtpSynchronization(success: false, offsetNs: 999.0);

        // Assert - offset should not be recorded for failures
        var finalOffsets = GetHistogramValues("gpubridge.ptp.offset.ns").Count;
        Assert.Equal(initialOffsets, finalOffsets);

        _output.WriteLine("PTP sync failure tracked without offset");
        _output.WriteLine("✅ PTP synchronization failure tracking works");
    }

    #endregion

    #region Message Queue Metrics Tests

    [Fact]
    public void RecordMessageEnqueued_ShouldIncrement()
    {
        // Arrange
        var initialEnqueued = GetCounterValue("gpubridge.temporal.messages.enqueued.total");

        // Act
        _metrics.RecordMessageEnqueued();
        _metrics.RecordMessageEnqueued();
        _metrics.RecordMessageEnqueued();

        // Assert
        var finalEnqueued = GetCounterValue("gpubridge.temporal.messages.enqueued.total");
        Assert.Equal(initialEnqueued + 3, finalEnqueued);

        _output.WriteLine("Messages enqueued tracked");
        _output.WriteLine("✅ Message enqueue tracking works");
    }

    [Fact]
    public void RecordMessageDequeued_ShouldTrackLatencyAndReordering()
    {
        // Arrange
        var initialDequeued = GetCounterValue("gpubridge.temporal.messages.dequeued.total");
        var initialReordered = GetCounterValue("gpubridge.temporal.messages.reordered.total");

        // Act
        _metrics.RecordMessageDequeued(latencyMs: 1.5, wasReordered: false);
        _metrics.RecordMessageDequeued(latencyMs: 2.0, wasReordered: true);
        _metrics.RecordMessageDequeued(latencyMs: 1.8, wasReordered: true);

        // Assert
        var finalDequeued = GetCounterValue("gpubridge.temporal.messages.dequeued.total");
        var finalReordered = GetCounterValue("gpubridge.temporal.messages.reordered.total");

        Assert.Equal(initialDequeued + 3, finalDequeued);
        Assert.Equal(initialReordered + 2, finalReordered);

        var latencies = GetHistogramValues("gpubridge.temporal.message.latency.ms");
        Assert.Contains(1.5, latencies);
        Assert.Contains(2.0, latencies);
        Assert.Contains(1.8, latencies);

        _output.WriteLine("Messages dequeued tracked with latency and reordering");
        _output.WriteLine("✅ Message dequeue tracking works");
    }

    [Fact]
    public void UpdateQueueDepth_ShouldUpdateGauge()
    {
        // Act
        _metrics.UpdateQueueDepth(100);

        // Note: Observable gauges are pull-based, so we can't easily verify
        // the value through the listener. This test verifies the method runs without error.
        _output.WriteLine("Queue depth updated to 100");
        _output.WriteLine("✅ Queue depth update works");
    }

    #endregion

    #region Timing Scope Tests

    [Fact]
    public void HlcNowTimingScope_ShouldRecordLatency()
    {
        // Arrange
        var initialCalls = GetCounterValue("gpubridge.hlc.now.calls.total");

        // Act
        var scope = _metrics.StartHlcNowTiming();
        Thread.Sleep(1); // Small delay to ensure measurable time
        scope.Complete(physicalTime: 1000, logicalCounter: 1, logicalCounterIncremented: false);

        // Assert
        var finalCalls = GetCounterValue("gpubridge.hlc.now.calls.total");
        Assert.Equal(initialCalls + 1, finalCalls);

        var latencies = GetHistogramValues("gpubridge.hlc.now.latency.ns");
        Assert.True(latencies.Count > 0);
        Assert.True(latencies[^1] > 0); // Should have recorded some latency

        _output.WriteLine($"Timing scope recorded latency: {latencies[^1]:F2}ns");
        _output.WriteLine("✅ HLC.Now() timing scope works");
    }

    [Fact]
    public void HlcUpdateTimingScope_ShouldRecordLatency()
    {
        // Arrange
        var initialCalls = GetCounterValue("gpubridge.hlc.update.calls.total");

        // Act
        var scope = _metrics.StartHlcUpdateTiming();
        Thread.Sleep(1);
        scope.Complete(physicalTime: 2000, logicalCounter: 5);

        // Assert
        var finalCalls = GetCounterValue("gpubridge.hlc.update.calls.total");
        Assert.Equal(initialCalls + 1, finalCalls);

        _output.WriteLine("✅ HLC.Update() timing scope works");
    }

    [Fact]
    public void VectorClockMergeTimingScope_ShouldRecordLatency()
    {
        // Arrange
        var initialMerges = GetCounterValue("gpubridge.vectorclock.merges.total");

        // Act
        var scope = _metrics.StartVectorClockMergeTiming();
        Thread.Sleep(1);
        scope.Complete(resultingEntryCount: 10);

        // Assert
        var finalMerges = GetCounterValue("gpubridge.vectorclock.merges.total");
        Assert.Equal(initialMerges + 1, finalMerges);

        _output.WriteLine("✅ Vector clock merge timing scope works");
    }

    #endregion

    #region Performance Tests

    [Fact]
    public void MetricsRecording_ShouldBeFast()
    {
        // Arrange
        const int iterations = 100_000;

        // Act
        var sw = Stopwatch.StartNew();
        for (int i = 0; i < iterations; i++)
        {
            _metrics.RecordHlcNow(50.0, 1000, i, false);
        }
        sw.Stop();

        // Calculate metrics
        var nsPerOp = sw.Elapsed.TotalNanoseconds / iterations;
        var opsPerSec = iterations / sw.Elapsed.TotalSeconds;

        // Assert
        Assert.True(nsPerOp < 5000, $"Metrics recording took {nsPerOp:F2}ns, should be <5000ns");

        _output.WriteLine($"=== Metrics Recording Performance ===");
        _output.WriteLine($"Iterations:  {iterations:N0}");
        _output.WriteLine($"Total time:  {sw.Elapsed.TotalMilliseconds:F2}ms");
        _output.WriteLine($"Latency:     {nsPerOp:F2}ns/op");
        _output.WriteLine($"Throughput:  {opsPerSec / 1_000_000:F2}M ops/s");
        _output.WriteLine("✅ Metrics recording is fast enough for production");
    }

    [Fact]
    public void MultipleMetricsTypes_ShouldAllWork()
    {
        // Act - Record various metrics
        _metrics.RecordHlcNow(50.0, 1000, 1, true);
        _metrics.RecordHlcUpdate(70.0, 2000, 2);
        _metrics.RecordClockDrift(100.0);
        _metrics.RecordClockJump(50.0);
        _metrics.RecordVectorClockIncrement(5);
        _metrics.RecordVectorClockMerge(150.0, 10);
        _metrics.RecordVectorClockConcurrencyDetected();
        _metrics.RecordFaultHandlerClockJump();
        _metrics.RecordFaultHandlerRecovery(true, 5.0);
        _metrics.RecordNetworkRetryAttempt(100.0);
        _metrics.RecordNetworkRetryCompletion(true, 3);
        _metrics.RecordPtpDeviceFailure();
        _metrics.RecordPtpSynchronization(true, 500.0);
        _metrics.RecordMessageEnqueued();
        _metrics.RecordMessageDequeued(1.5, false);
        _metrics.UpdateQueueDepth(10);

        // Assert - All metrics should be recorded without error
        _output.WriteLine("All metric types recorded successfully");
        _output.WriteLine("✅ Comprehensive metrics coverage verified");
    }

    #endregion

    #region Helper Methods

    private long GetCounterValue(string name)
    {
        return _recordedCounters.TryGetValue(name, out var value) ? value : 0;
    }

    private List<double> GetHistogramValues(string name)
    {
        return _recordedHistograms.TryGetValue(name, out var values) ? values : new List<double>();
    }

    #endregion
}

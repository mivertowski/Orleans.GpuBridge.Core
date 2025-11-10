using System;
using System.Collections.Concurrent;
using System.Diagnostics;
using System.Diagnostics.Metrics;
using Microsoft.Extensions.Logging;

namespace Orleans.GpuBridge.Diagnostics.Implementation;

/// <summary>
/// OpenTelemetry metrics for GPU-native ring kernels.
/// Provides comprehensive monitoring of ring kernel lifecycle, performance, and health.
/// </summary>
/// <remarks>
/// Metrics Categories:
/// 1. Lifecycle - Launches, stops, restarts, crashes
/// 2. Performance - Message latency, throughput, queue depth
/// 3. Health - Uptime, watchdog events, backpressure
/// 4. Resources - GPU memory, thread utilization
///
/// Integration:
/// - Prometheus: Automatic export via OpenTelemetry
/// - Grafana: Pre-built dashboards available
/// - Azure Monitor: Native integration
/// - Jaeger/Zipkin: Distributed tracing spans
/// </remarks>
public sealed class RingKernelTelemetry : IDisposable
{
    private readonly ILogger<RingKernelTelemetry> _logger;
    private readonly Meter _meter;
    private readonly ActivitySource _activitySource;

    // Counters
    private readonly Counter<long> _ringKernelLaunches;
    private readonly Counter<long> _ringKernelStops;
    private readonly Counter<long> _ringKernelRestarts;
    private readonly Counter<long> _ringKernelCrashes;
    private readonly Counter<long> _actorMessagesProcessed;
    private readonly Counter<long> _actorMessagesSent;
    private readonly Counter<long> _actorMessagesDropped;
    private readonly Counter<long> _watchdogRecoveries;
    private readonly Counter<long> _backpressureEvents;

    // Histograms
    private readonly Histogram<double> _ringKernelLaunchLatency;
    private readonly Histogram<double> _ringKernelUptimeSeconds;
    private readonly Histogram<double> _actorMessageLatency;
    private readonly Histogram<double> _actorMessageProcessingLatency;

    // Observable Gauges
    private readonly ObservableGauge<int> _activeRingKernels;
    private readonly ObservableGauge<long> _totalActiveActors;
    private readonly ObservableGauge<double> _averageQueueUtilization;
    private readonly ObservableGauge<int> _unhealthyKernels;

    // State for observable gauges (thread-safe)
    private readonly ConcurrentDictionary<Guid, RingKernelMetricState> _kernelStates;
    private bool _disposed;

    public RingKernelTelemetry(
        ILogger<RingKernelTelemetry> logger,
        string meterName = "Orleans.GpuBridge.RingKernels")
    {
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));

        _meter = new Meter(meterName, "1.0.0");
        _activitySource = new ActivitySource(meterName, "1.0.0");
        _kernelStates = new ConcurrentDictionary<Guid, RingKernelMetricState>();

        // Initialize counters
        _ringKernelLaunches = _meter.CreateCounter<long>(
            "ring_kernel_launches_total",
            description: "Total number of ring kernel launches");

        _ringKernelStops = _meter.CreateCounter<long>(
            "ring_kernel_stops_total",
            description: "Total number of ring kernel graceful stops");

        _ringKernelRestarts = _meter.CreateCounter<long>(
            "ring_kernel_restarts_total",
            description: "Total number of ring kernel restarts (watchdog or manual)");

        _ringKernelCrashes = _meter.CreateCounter<long>(
            "ring_kernel_crashes_total",
            description: "Total number of ring kernel crashes");

        _actorMessagesProcessed = _meter.CreateCounter<long>(
            "actor_messages_processed_total",
            description: "Total number of actor messages processed");

        _actorMessagesSent = _meter.CreateCounter<long>(
            "actor_messages_sent_total",
            description: "Total number of actor messages sent");

        _actorMessagesDropped = _meter.CreateCounter<long>(
            "actor_messages_dropped_total",
            description: "Total number of actor messages dropped (queue overflow)");

        _watchdogRecoveries = _meter.CreateCounter<long>(
            "ring_kernel_watchdog_recoveries_total",
            description: "Total number of watchdog-triggered kernel recoveries");

        _backpressureEvents = _meter.CreateCounter<long>(
            "ring_kernel_backpressure_events_total",
            description: "Total number of backpressure events");

        // Initialize histograms
        _ringKernelLaunchLatency = _meter.CreateHistogram<double>(
            "ring_kernel_launch_latency_microseconds",
            unit: "μs",
            description: "Ring kernel launch latency distribution");

        _ringKernelUptimeSeconds = _meter.CreateHistogram<double>(
            "ring_kernel_uptime_seconds",
            unit: "s",
            description: "Ring kernel uptime before stop/crash distribution");

        _actorMessageLatency = _meter.CreateHistogram<double>(
            "actor_message_latency_nanoseconds",
            unit: "ns",
            description: "Actor message end-to-end latency distribution");

        _actorMessageProcessingLatency = _meter.CreateHistogram<double>(
            "actor_message_processing_latency_nanoseconds",
            unit: "ns",
            description: "Actor message processing time distribution");

        // Initialize observable gauges
        _activeRingKernels = _meter.CreateObservableGauge(
            "ring_kernels_active",
            ObserveActiveRingKernels,
            description: "Number of currently running ring kernels");

        _totalActiveActors = _meter.CreateObservableGauge(
            "actors_active_total",
            ObserveTotalActiveActors,
            description: "Total number of active actors across all ring kernels");

        _averageQueueUtilization = _meter.CreateObservableGauge(
            "actor_queue_utilization_percent",
            ObserveAverageQueueUtilization,
            description: "Average queue utilization across all actors");

        _unhealthyKernels = _meter.CreateObservableGauge(
            "ring_kernels_unhealthy",
            ObserveUnhealthyKernels,
            description: "Number of ring kernels with health issues");

        _logger.LogInformation(
            "Ring kernel telemetry initialized - Meter: {MeterName}",
            meterName);
    }

    #region Ring Kernel Lifecycle

    /// <summary>
    /// Records a ring kernel launch event.
    /// </summary>
    public Activity? RecordRingKernelLaunch(
        Guid kernelId,
        int actorCount,
        int threadsPerActor)
    {
        _ringKernelLaunches.Add(1, new KeyValuePair<string, object?>("kernel_id", kernelId.ToString()));

        // Register kernel state for observability
        _kernelStates[kernelId] = new RingKernelMetricState
        {
            KernelId = kernelId,
            ActorCount = actorCount,
            ThreadsPerActor = threadsPerActor,
            LaunchTime = DateTimeOffset.UtcNow,
            IsHealthy = true,
            QueueUtilizationPercent = 0,
            MessagesProcessed = 0,
            MessagesSent = 0,
            MessagesDropped = 0
        };

        _logger.LogDebug(
            "Ring kernel {KernelId} launched - Actors: {ActorCount}, Threads: {Threads}",
            kernelId,
            actorCount,
            threadsPerActor);

        // Create distributed tracing span
        return _activitySource.StartActivity(
            "RingKernel.Launch",
            ActivityKind.Server,
            tags: new ActivityTagsCollection
            {
                { "kernel.id", kernelId.ToString() },
                { "actor.count", actorCount },
                { "threads.per.actor", threadsPerActor }
            });
    }

    /// <summary>
    /// Records ring kernel launch latency.
    /// </summary>
    public void RecordRingKernelLaunchLatency(
        Guid kernelId,
        double latencyMicroseconds)
    {
        _ringKernelLaunchLatency.Record(
            latencyMicroseconds,
            new KeyValuePair<string, object?>("kernel_id", kernelId.ToString()));

        _logger.LogDebug(
            "Ring kernel {KernelId} launch completed - Latency: {Latency:F2}μs",
            kernelId,
            latencyMicroseconds);
    }

    /// <summary>
    /// Records a ring kernel graceful stop.
    /// </summary>
    public void RecordRingKernelStop(
        Guid kernelId,
        TimeSpan uptime,
        long messagesProcessed)
    {
        _ringKernelStops.Add(1, new KeyValuePair<string, object?>("kernel_id", kernelId.ToString()));
        _ringKernelUptimeSeconds.Record(uptime.TotalSeconds);

        // Remove from active tracking
        _kernelStates.TryRemove(kernelId, out _);

        _logger.LogInformation(
            "Ring kernel {KernelId} stopped gracefully - " +
            "Uptime: {Uptime}, Messages: {Messages:N0}",
            kernelId,
            uptime,
            messagesProcessed);
    }

    /// <summary>
    /// Records a ring kernel restart (watchdog or manual).
    /// </summary>
    public void RecordRingKernelRestart(
        Guid kernelId,
        string reason,
        int restartCount)
    {
        _ringKernelRestarts.Add(
            1,
            new KeyValuePair<string, object?>("kernel_id", kernelId.ToString()),
            new KeyValuePair<string, object?>("reason", reason),
            new KeyValuePair<string, object?>("restart_count", restartCount));

        if (_kernelStates.TryGetValue(kernelId, out var state))
        {
            state.RestartCount = restartCount;
        }

        _logger.LogWarning(
            "Ring kernel {KernelId} restarted - Reason: {Reason}, Count: {Count}",
            kernelId,
            reason,
            restartCount);
    }

    /// <summary>
    /// Records a ring kernel crash.
    /// </summary>
    public void RecordRingKernelCrash(
        Guid kernelId,
        TimeSpan uptime,
        string? errorMessage = null)
    {
        _ringKernelCrashes.Add(1, new KeyValuePair<string, object?>("kernel_id", kernelId.ToString()));
        _ringKernelUptimeSeconds.Record(uptime.TotalSeconds);

        if (_kernelStates.TryGetValue(kernelId, out var state))
        {
            state.IsHealthy = false;
        }

        _logger.LogError(
            "Ring kernel {KernelId} crashed - Uptime: {Uptime}, Error: {Error}",
            kernelId,
            uptime,
            errorMessage ?? "Unknown");
    }

    #endregion

    #region Actor Message Metrics

    /// <summary>
    /// Records actor message processing.
    /// </summary>
    public void RecordActorMessageProcessed(
        Guid kernelId,
        Guid actorId,
        double latencyNanoseconds,
        double processingTimeNanoseconds)
    {
        _actorMessagesProcessed.Add(
            1,
            new KeyValuePair<string, object?>("kernel_id", kernelId.ToString()),
            new KeyValuePair<string, object?>("actor_id", actorId.ToString()));

        _actorMessageLatency.Record(latencyNanoseconds);
        _actorMessageProcessingLatency.Record(processingTimeNanoseconds);

        if (_kernelStates.TryGetValue(kernelId, out var state))
        {
            state.MessagesProcessed++;
        }
    }

    /// <summary>
    /// Records actor message sent.
    /// </summary>
    public void RecordActorMessageSent(
        Guid kernelId,
        Guid sourceActorId,
        Guid targetActorId)
    {
        _actorMessagesSent.Add(
            1,
            new KeyValuePair<string, object?>("kernel_id", kernelId.ToString()),
            new KeyValuePair<string, object?>("source_actor_id", sourceActorId.ToString()));

        if (_kernelStates.TryGetValue(kernelId, out var state))
        {
            state.MessagesSent++;
        }
    }

    /// <summary>
    /// Records actor message dropped (queue overflow).
    /// </summary>
    public void RecordActorMessageDropped(
        Guid kernelId,
        Guid actorId,
        string reason)
    {
        _actorMessagesDropped.Add(
            1,
            new KeyValuePair<string, object?>("kernel_id", kernelId.ToString()),
            new KeyValuePair<string, object?>("actor_id", actorId.ToString()),
            new KeyValuePair<string, object?>("reason", reason));

        if (_kernelStates.TryGetValue(kernelId, out var state))
        {
            state.MessagesDropped++;
        }

        _logger.LogWarning(
            "Actor message dropped - Kernel: {KernelId}, Actor: {ActorId}, Reason: {Reason}",
            kernelId,
            actorId,
            reason);
    }

    #endregion

    #region Health & Resilience

    /// <summary>
    /// Records a watchdog recovery event.
    /// </summary>
    public void RecordWatchdogRecovery(
        Guid kernelId,
        TimeSpan hungDuration)
    {
        _watchdogRecoveries.Add(1, new KeyValuePair<string, object?>("kernel_id", kernelId.ToString()));

        if (_kernelStates.TryGetValue(kernelId, out var state))
        {
            state.IsHealthy = true; // Recovered
        }

        _logger.LogWarning(
            "Watchdog recovered kernel {KernelId} - Hung for: {Duration}",
            kernelId,
            hungDuration);
    }

    /// <summary>
    /// Records a backpressure event.
    /// </summary>
    public void RecordBackpressureEvent(
        Guid kernelId,
        Guid actorId,
        string severity,
        double queueUtilizationPercent)
    {
        _backpressureEvents.Add(
            1,
            new KeyValuePair<string, object?>("kernel_id", kernelId.ToString()),
            new KeyValuePair<string, object?>("severity", severity));

        _logger.LogDebug(
            "Backpressure event - Kernel: {KernelId}, Actor: {ActorId}, " +
            "Severity: {Severity}, Queue: {Utilization:F1}%",
            kernelId,
            actorId,
            severity,
            queueUtilizationPercent);
    }

    /// <summary>
    /// Updates queue utilization for a kernel.
    /// </summary>
    public void UpdateQueueUtilization(
        Guid kernelId,
        double utilizationPercent)
    {
        if (_kernelStates.TryGetValue(kernelId, out var state))
        {
            state.QueueUtilizationPercent = utilizationPercent;
        }
    }

    /// <summary>
    /// Updates kernel health status.
    /// </summary>
    public void UpdateKernelHealth(
        Guid kernelId,
        bool isHealthy)
    {
        if (_kernelStates.TryGetValue(kernelId, out var state))
        {
            state.IsHealthy = isHealthy;
        }
    }

    #endregion

    #region Observable Gauge Callbacks

    private int ObserveActiveRingKernels()
    {
        return _kernelStates.Count;
    }

    private long ObserveTotalActiveActors()
    {
        long total = 0;
        foreach (var state in _kernelStates.Values)
        {
            total += state.ActorCount;
        }
        return total;
    }

    private double ObserveAverageQueueUtilization()
    {
        if (_kernelStates.IsEmpty)
            return 0;

        double sum = 0;
        foreach (var state in _kernelStates.Values)
        {
            sum += state.QueueUtilizationPercent;
        }
        return sum / _kernelStates.Count;
    }

    private int ObserveUnhealthyKernels()
    {
        int unhealthy = 0;
        foreach (var state in _kernelStates.Values)
        {
            if (!state.IsHealthy)
                unhealthy++;
        }
        return unhealthy;
    }

    #endregion

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        _meter?.Dispose();
        _activitySource?.Dispose();

        _logger.LogInformation("Ring kernel telemetry disposed");
    }

    private sealed class RingKernelMetricState
    {
        public required Guid KernelId { get; init; }
        public required int ActorCount { get; init; }
        public required int ThreadsPerActor { get; init; }
        public required DateTimeOffset LaunchTime { get; init; }
        public required bool IsHealthy { get; set; }
        public required double QueueUtilizationPercent { get; set; }
        public required long MessagesProcessed { get; set; }
        public required long MessagesSent { get; set; }
        public required long MessagesDropped { get; set; }
        public int RestartCount { get; set; }
    }
}

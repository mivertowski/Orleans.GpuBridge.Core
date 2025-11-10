using System;
using System.Collections.Concurrent;
using System.Diagnostics.Metrics;
using Microsoft.Extensions.Logging;

namespace Orleans.GpuBridge.Diagnostics.Implementation;

/// <summary>
/// OpenTelemetry metrics for GPU-resident message queues.
/// Monitors queue health, latency, throughput, and overflow events.
/// </summary>
/// <remarks>
/// Metrics Categories:
/// 1. Queue Operations - Enqueue/dequeue latency and success rates
/// 2. Queue Health - Depth, utilization, age of messages
/// 3. Overflow & Backpressure - Dropped messages, backpressure events
/// 4. Performance - Throughput, batch sizes, lock contention
///
/// Critical Alerts:
/// - Queue utilization >85% (backpressure threshold)
/// - Message drops detected (data loss)
/// - Enqueue/dequeue latency >1μs (performance degradation)
/// - Messages aging >1ms in queue (processing bottleneck)
/// </remarks>
public sealed class MessageQueueTelemetry : IDisposable
{
    private readonly ILogger<MessageQueueTelemetry> _logger;
    private readonly Meter _meter;

    // Counters
    private readonly Counter<long> _messagesEnqueuedTotal;
    private readonly Counter<long> _messagesDequeuedTotal;
    private readonly Counter<long> _messagesDroppedTotal;
    private readonly Counter<long> _enqueueFailuresTotal;
    private readonly Counter<long> _dequeueFailuresTotal;
    private readonly Counter<long> _queueOverflowsTotal;
    private readonly Counter<long> _backpressureActivationsTotal;

    // Histograms
    private readonly Histogram<double> _enqueueLatency;
    private readonly Histogram<double> _dequeueLatency;
    private readonly Histogram<double> _messageAgeMillis;
    private readonly Histogram<int> _enqueueBatchSize;
    private readonly Histogram<int> _dequeueBatchSize;

    // Observable Gauges
    private readonly ObservableGauge<int> _currentQueueDepth;
    private readonly ObservableGauge<double> _queueUtilizationPercent;
    private readonly ObservableGauge<int> _activeQueues;
    private readonly ObservableGauge<long> _oldestMessageAgeMillis;

    // State
    private readonly ConcurrentDictionary<Guid, QueueMetricState> _queueStates;
    private bool _disposed;

    public MessageQueueTelemetry(
        ILogger<MessageQueueTelemetry> logger,
        string meterName = "Orleans.GpuBridge.MessageQueues")
    {
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));

        _meter = new Meter(meterName, "1.0.0");
        _queueStates = new ConcurrentDictionary<Guid, QueueMetricState>();

        // Initialize counters
        _messagesEnqueuedTotal = _meter.CreateCounter<long>(
            "messages_enqueued_total",
            description: "Total number of messages successfully enqueued");

        _messagesDequeuedTotal = _meter.CreateCounter<long>(
            "messages_dequeued_total",
            description: "Total number of messages successfully dequeued");

        _messagesDroppedTotal = _meter.CreateCounter<long>(
            "messages_dropped_total",
            description: "Total number of messages dropped due to queue overflow");

        _enqueueFailuresTotal = _meter.CreateCounter<long>(
            "enqueue_failures_total",
            description: "Total number of enqueue operation failures");

        _dequeueFailuresTotal = _meter.CreateCounter<long>(
            "dequeue_failures_total",
            description: "Total number of dequeue operation failures");

        _queueOverflowsTotal = _meter.CreateCounter<long>(
            "queue_overflows_total",
            description: "Total number of queue overflow events");

        _backpressureActivationsTotal = _meter.CreateCounter<long>(
            "backpressure_activations_total",
            description: "Total number of backpressure activation events");

        // Initialize histograms
        _enqueueLatency = _meter.CreateHistogram<double>(
            "enqueue_latency_nanoseconds",
            unit: "ns",
            description: "Message enqueue operation latency distribution");

        _dequeueLatency = _meter.CreateHistogram<double>(
            "dequeue_latency_nanoseconds",
            unit: "ns",
            description: "Message dequeue operation latency distribution");

        _messageAgeMillis = _meter.CreateHistogram<double>(
            "message_age_milliseconds",
            unit: "ms",
            description: "Time messages spend in queue distribution");

        _enqueueBatchSize = _meter.CreateHistogram<int>(
            "enqueue_batch_size",
            description: "Number of messages enqueued per batch operation");

        _dequeueBatchSize = _meter.CreateHistogram<int>(
            "dequeue_batch_size",
            description: "Number of messages dequeued per batch operation");

        // Initialize observable gauges
        _currentQueueDepth = _meter.CreateObservableGauge(
            "queue_depth_current",
            ObserveCurrentQueueDepth,
            description: "Current number of messages in queue");

        _queueUtilizationPercent = _meter.CreateObservableGauge(
            "queue_utilization_percent",
            ObserveQueueUtilization,
            description: "Current queue utilization percentage");

        _activeQueues = _meter.CreateObservableGauge(
            "queues_active",
            ObserveActiveQueues,
            description: "Number of active message queues");

        _oldestMessageAgeMillis = _meter.CreateObservableGauge(
            "oldest_message_age_milliseconds",
            ObserveOldestMessageAge,
            unit: "ms",
            description: "Age of oldest message in queue");

        _logger.LogInformation(
            "Message queue telemetry initialized - Meter: {MeterName}",
            meterName);
    }

    #region Queue Operations

    /// <summary>
    /// Records a successful message enqueue operation.
    /// </summary>
    public void RecordMessageEnqueued(
        Guid queueId,
        double latencyNanoseconds,
        int currentDepth,
        int capacity)
    {
        _messagesEnqueuedTotal.Add(1,
            new KeyValuePair<string, object?>("queue_id", queueId.ToString()));

        _enqueueLatency.Record(latencyNanoseconds);

        UpdateQueueState(queueId, currentDepth, capacity);
    }

    /// <summary>
    /// Records a successful message dequeue operation.
    /// </summary>
    public void RecordMessageDequeued(
        Guid queueId,
        double latencyNanoseconds,
        double messageAgeMilliseconds,
        int currentDepth,
        int capacity)
    {
        _messagesDequeuedTotal.Add(1,
            new KeyValuePair<string, object?>("queue_id", queueId.ToString()));

        _dequeueLatency.Record(latencyNanoseconds);
        _messageAgeMillis.Record(messageAgeMilliseconds);

        UpdateQueueState(queueId, currentDepth, capacity);

        // Alert if message aged too long
        if (messageAgeMilliseconds > 1.0) // >1ms
        {
            _logger.LogWarning(
                "Message aged in queue - Queue: {QueueId}, Age: {Age:F2}ms",
                queueId,
                messageAgeMilliseconds);
        }
    }

    /// <summary>
    /// Records a dropped message due to queue overflow.
    /// </summary>
    public void RecordMessageDropped(
        Guid queueId,
        string reason)
    {
        _messagesDroppedTotal.Add(1,
            new KeyValuePair<string, object?>("queue_id", queueId.ToString()),
            new KeyValuePair<string, object?>("reason", reason));

        _logger.LogError(
            "Message dropped - Queue: {QueueId}, Reason: {Reason}",
            queueId,
            reason);
    }

    /// <summary>
    /// Records a queue overflow event.
    /// </summary>
    public void RecordQueueOverflow(
        Guid queueId,
        int attemptedDepth,
        int capacity)
    {
        _queueOverflowsTotal.Add(1,
            new KeyValuePair<string, object?>("queue_id", queueId.ToString()));

        _logger.LogError(
            "Queue overflow - Queue: {QueueId}, Attempted: {Attempted}, Capacity: {Capacity}",
            queueId,
            attemptedDepth,
            capacity);
    }

    /// <summary>
    /// Records a backpressure activation.
    /// </summary>
    public void RecordBackpressureActivation(
        Guid queueId,
        string severity,
        double utilizationPercent)
    {
        _backpressureActivationsTotal.Add(1,
            new KeyValuePair<string, object?>("queue_id", queueId.ToString()),
            new KeyValuePair<string, object?>("severity", severity));

        _logger.LogWarning(
            "Backpressure activated - Queue: {QueueId}, Severity: {Severity}, Utilization: {Utilization:F1}%",
            queueId,
            severity,
            utilizationPercent);
    }

    /// <summary>
    /// Records batch enqueue operation.
    /// </summary>
    public void RecordBatchEnqueue(
        Guid queueId,
        int batchSize,
        double totalLatencyNanoseconds)
    {
        _messagesEnqueuedTotal.Add(batchSize,
            new KeyValuePair<string, object?>("queue_id", queueId.ToString()));

        _enqueueBatchSize.Record(batchSize);
        _enqueueLatency.Record(totalLatencyNanoseconds / batchSize); // Per-message latency
    }

    /// <summary>
    /// Records batch dequeue operation.
    /// </summary>
    public void RecordBatchDequeue(
        Guid queueId,
        int batchSize,
        double totalLatencyNanoseconds)
    {
        _messagesDequeuedTotal.Add(batchSize,
            new KeyValuePair<string, object?>("queue_id", queueId.ToString()));

        _dequeueBatchSize.Record(batchSize);
        _dequeueLatency.Record(totalLatencyNanoseconds / batchSize); // Per-message latency
    }

    /// <summary>
    /// Records an enqueue failure.
    /// </summary>
    public void RecordEnqueueFailure(
        Guid queueId,
        string errorType)
    {
        _enqueueFailuresTotal.Add(1,
            new KeyValuePair<string, object?>("queue_id", queueId.ToString()),
            new KeyValuePair<string, object?>("error_type", errorType));

        _logger.LogError(
            "Enqueue failed - Queue: {QueueId}, Error: {ErrorType}",
            queueId,
            errorType);
    }

    /// <summary>
    /// Records a dequeue failure.
    /// </summary>
    public void RecordDequeueFailure(
        Guid queueId,
        string errorType)
    {
        _dequeueFailuresTotal.Add(1,
            new KeyValuePair<string, object?>("queue_id", queueId.ToString()),
            new KeyValuePair<string, object?>("error_type", errorType));

        _logger.LogError(
            "Dequeue failed - Queue: {QueueId}, Error: {ErrorType}",
            queueId,
            errorType);
    }

    #endregion

    #region Queue Management

    /// <summary>
    /// Registers a message queue for monitoring.
    /// </summary>
    public void RegisterQueue(
        Guid queueId,
        int capacity)
    {
        _queueStates[queueId] = new QueueMetricState
        {
            QueueId = queueId,
            CurrentDepth = 0,
            Capacity = capacity,
            LastUpdateTime = DateTimeOffset.UtcNow
        };

        _logger.LogDebug(
            "Message queue {QueueId} registered with telemetry - Capacity: {Capacity}",
            queueId,
            capacity);
    }

    /// <summary>
    /// Unregisters a message queue from monitoring.
    /// </summary>
    public void UnregisterQueue(Guid queueId)
    {
        _queueStates.TryRemove(queueId, out _);

        _logger.LogDebug(
            "Message queue {QueueId} unregistered from telemetry",
            queueId);
    }

    private void UpdateQueueState(
        Guid queueId,
        int currentDepth,
        int capacity)
    {
        if (_queueStates.TryGetValue(queueId, out var state))
        {
            state.CurrentDepth = currentDepth;
            state.Capacity = capacity;
            state.LastUpdateTime = DateTimeOffset.UtcNow;
        }
    }

    #endregion

    #region Observable Gauge Callbacks

    private int ObserveCurrentQueueDepth()
    {
        int totalDepth = 0;
        foreach (var state in _queueStates.Values)
        {
            totalDepth += state.CurrentDepth;
        }
        return totalDepth;
    }

    private double ObserveQueueUtilization()
    {
        if (_queueStates.IsEmpty)
            return 0;

        double totalUtilization = 0;
        foreach (var state in _queueStates.Values)
        {
            if (state.Capacity > 0)
            {
                totalUtilization += (state.CurrentDepth / (double)state.Capacity) * 100.0;
            }
        }
        return totalUtilization / _queueStates.Count;
    }

    private int ObserveActiveQueues()
    {
        return _queueStates.Count;
    }

    private long ObserveOldestMessageAge()
    {
        if (_queueStates.IsEmpty)
            return 0;

        var now = DateTimeOffset.UtcNow;
        long maxAge = 0;

        foreach (var state in _queueStates.Values)
        {
            if (state.CurrentDepth > 0)
            {
                var age = (now - state.LastUpdateTime).TotalMilliseconds;
                if (age > maxAge)
                {
                    maxAge = (long)age;
                }
            }
        }

        return maxAge;
    }

    #endregion

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        _meter?.Dispose();

        _logger.LogInformation("Message queue telemetry disposed");
    }

    private sealed class QueueMetricState
    {
        public required Guid QueueId { get; init; }
        public required int CurrentDepth { get; set; }
        public required int Capacity { get; set; }
        public required DateTimeOffset LastUpdateTime { get; set; }
    }
}

// Orleans.GpuBridge - GPU-native distributed computing for Microsoft Orleans
// Copyright (c) 2025 Michael Ivertowski

using System.Collections.Concurrent;
using DotCompute.Abstractions.RingKernels;
using Microsoft.Extensions.Logging;
using Orleans.GpuBridge.Abstractions.Placement;

namespace Orleans.GpuBridge.Runtime.Placement;

/// <summary>
/// Monitors queue depth across GPU ring kernels for intelligent placement decisions.
/// </summary>
/// <remarks>
/// <para>
/// This implementation integrates with DotCompute's <see cref="IRingKernelRuntime"/>
/// to provide real-time queue depth metrics for GPU-native actors.
/// </para>
/// <para>
/// <strong>Features:</strong>
/// <list type="bullet">
/// <item><description>Real-time queue utilization from ring kernel telemetry</description></item>
/// <item><description>Historical tracking for trend analysis</description></item>
/// <item><description>Threshold-based alerting</description></item>
/// <item><description>Predictive utilization estimates</description></item>
/// </list>
/// </para>
/// </remarks>
public sealed class QueueDepthMonitor : IQueueDepthMonitor, IDisposable
{
    private readonly ILogger<QueueDepthMonitor> _logger;
    private readonly IRingKernelRuntime _ringKernelRuntime;
    private readonly string _localSiloId;
    private readonly ConcurrentDictionary<string, QueueDepthHistory> _historyCache = new();
    private readonly ConcurrentDictionary<double, List<Action<QueueDepthAlert>>> _alertSubscriptions = new();
    private readonly Timer _historyCollectionTimer;
    private readonly ConcurrentQueue<QueueDepthSample> _recentSamples = new();
    private readonly object _historyLock = new();
    private bool _disposed;

    private const int MaxHistorySamples = 360; // 1 hour at 10-second intervals
    private const int HistoryCollectionIntervalMs = 10_000; // 10 seconds

    /// <summary>
    /// Initializes a new instance of the <see cref="QueueDepthMonitor"/> class.
    /// </summary>
    /// <param name="ringKernelRuntime">Ring kernel runtime for metrics access.</param>
    /// <param name="logger">Logger for diagnostic output.</param>
    /// <param name="localSiloId">Local silo identifier.</param>
    public QueueDepthMonitor(
        IRingKernelRuntime ringKernelRuntime,
        ILogger<QueueDepthMonitor> logger,
        string? localSiloId = null)
    {
        _ringKernelRuntime = ringKernelRuntime ?? throw new ArgumentNullException(nameof(ringKernelRuntime));
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        _localSiloId = localSiloId ?? Environment.MachineName;

        // Start periodic history collection
        _historyCollectionTimer = new Timer(
            CollectHistorySample,
            null,
            TimeSpan.FromSeconds(10),
            TimeSpan.FromMilliseconds(HistoryCollectionIntervalMs));

        _logger.LogInformation(
            "QueueDepthMonitor initialized for silo {SiloId}",
            _localSiloId);
    }

    /// <inheritdoc/>
    public async Task<QueueDepthSnapshot> GetQueueDepthAsync(
        string? siloId = null,
        int deviceIndex = 0,
        CancellationToken ct = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        var effectiveSiloId = siloId ?? _localSiloId;
        var timestampNanos = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds() * 1_000_000;

        try
        {
            var kernels = await _ringKernelRuntime.ListKernelsAsync();

            if (kernels.Count == 0)
            {
                // No active kernels - return empty snapshot
                return CreateEmptySnapshot(effectiveSiloId, deviceIndex, timestampNanos);
            }

            int totalInputDepth = 0;
            int totalOutputDepth = 0;
            int totalInputCapacity = 0;
            int totalOutputCapacity = 0;
            double totalThroughput = 0;
            double totalGpuUtil = 0;
            int validKernels = 0;

            foreach (var kernelId in kernels)
            {
                try
                {
                    var metrics = await _ringKernelRuntime.GetMetricsAsync(kernelId, ct);
                    var status = await _ringKernelRuntime.GetStatusAsync(kernelId, ct);

                    if (!status.IsLaunched) continue;

                    // Calculate queue depths from utilization (assuming standard queue sizes)
                    const int defaultQueueSize = 4096;
                    int inputDepth = (int)(metrics.InputQueueUtilization * defaultQueueSize);
                    int outputDepth = (int)(metrics.OutputQueueUtilization * defaultQueueSize);

                    totalInputDepth += inputDepth;
                    totalOutputDepth += outputDepth;
                    totalInputCapacity += defaultQueueSize;
                    totalOutputCapacity += defaultQueueSize;
                    totalThroughput += metrics.ThroughputMsgsPerSec;
                    totalGpuUtil += metrics.GpuUtilizationPercent / 100.0;
                    validKernels++;
                }
                catch (Exception ex)
                {
                    _logger.LogDebug(ex, "Failed to get metrics for kernel {KernelId}", kernelId);
                }
            }

            if (validKernels == 0)
            {
                return CreateEmptySnapshot(effectiveSiloId, deviceIndex, timestampNanos);
            }

            // Query memory info (placeholder - would come from device broker in production)
            const long totalMemoryBytes = 8L * 1024 * 1024 * 1024; // 8GB placeholder
            var usedMemoryRatio = totalGpuUtil / validKernels; // Approximate
            var availableMemoryBytes = (long)(totalMemoryBytes * (1.0 - usedMemoryRatio * 0.5));

            return new QueueDepthSnapshot
            {
                TimestampNanos = timestampNanos,
                SiloId = effectiveSiloId,
                DeviceIndex = deviceIndex,
                ActiveKernelCount = validKernels,
                TotalInputQueueDepth = totalInputDepth,
                TotalOutputQueueDepth = totalOutputDepth,
                TotalInputQueueCapacity = totalInputCapacity,
                TotalOutputQueueCapacity = totalOutputCapacity,
                ThroughputMsgsPerSec = totalThroughput,
                GpuUtilization = validKernels > 0 ? totalGpuUtil / validKernels : 0.0,
                AvailableMemoryBytes = availableMemoryBytes,
                TotalMemoryBytes = totalMemoryBytes
            };
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to get queue depth for silo {SiloId}, device {DeviceIndex}",
                effectiveSiloId, deviceIndex);
            return CreateEmptySnapshot(effectiveSiloId, deviceIndex, timestampNanos);
        }
    }

    /// <inheritdoc/>
    public async Task<AggregatedQueueMetrics> GetAggregatedMetricsAsync(
        string? siloId = null,
        int deviceIndex = 0,
        CancellationToken ct = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        var effectiveSiloId = siloId ?? _localSiloId;
        var timestampNanos = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds() * 1_000_000;

        try
        {
            var kernels = await _ringKernelRuntime.ListKernelsAsync();

            if (kernels.Count == 0)
            {
                return new AggregatedQueueMetrics
                {
                    TimestampNanos = timestampNanos,
                    SiloId = effectiveSiloId,
                    DeviceIndex = deviceIndex,
                    KernelCount = 0,
                    MinQueueUtilization = 0.0,
                    MaxQueueUtilization = 0.0,
                    AvgQueueUtilization = 0.0,
                    StdDevQueueUtilization = 0.0,
                    TotalThroughput = 0.0,
                    AvgProcessingLatencyNanos = 0.0,
                    P99ProcessingLatencyNanos = 0.0
                };
            }

            var utilizations = new List<double>();
            var latencies = new List<double>();
            double totalThroughput = 0;

            foreach (var kernelId in kernels)
            {
                try
                {
                    var metrics = await _ringKernelRuntime.GetMetricsAsync(kernelId, ct);
                    var util = (metrics.InputQueueUtilization + metrics.OutputQueueUtilization) / 2.0;
                    utilizations.Add(util);
                    latencies.Add(metrics.AvgProcessingTimeMs * 1_000_000); // Convert ms to ns
                    totalThroughput += metrics.ThroughputMsgsPerSec;
                }
                catch
                {
                    // Skip kernels with invalid metrics
                }
            }

            if (utilizations.Count == 0)
            {
                return new AggregatedQueueMetrics
                {
                    TimestampNanos = timestampNanos,
                    SiloId = effectiveSiloId,
                    DeviceIndex = deviceIndex,
                    KernelCount = 0,
                    MinQueueUtilization = 0.0,
                    MaxQueueUtilization = 0.0,
                    AvgQueueUtilization = 0.0,
                    StdDevQueueUtilization = 0.0,
                    TotalThroughput = 0.0,
                    AvgProcessingLatencyNanos = 0.0,
                    P99ProcessingLatencyNanos = 0.0
                };
            }

            var avgUtil = utilizations.Average();
            var stdDev = Math.Sqrt(utilizations.Sum(u => Math.Pow(u - avgUtil, 2)) / utilizations.Count);

            // Sort latencies for P99
            latencies.Sort();
            var p99Index = (int)(latencies.Count * 0.99);
            var p99Latency = latencies.Count > 0 ? latencies[Math.Min(p99Index, latencies.Count - 1)] : 0;

            return new AggregatedQueueMetrics
            {
                TimestampNanos = timestampNanos,
                SiloId = effectiveSiloId,
                DeviceIndex = deviceIndex,
                KernelCount = utilizations.Count,
                MinQueueUtilization = utilizations.Min(),
                MaxQueueUtilization = utilizations.Max(),
                AvgQueueUtilization = avgUtil,
                StdDevQueueUtilization = stdDev,
                TotalThroughput = totalThroughput,
                AvgProcessingLatencyNanos = latencies.Average(),
                P99ProcessingLatencyNanos = p99Latency
            };
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to get aggregated metrics");
            throw;
        }
    }

    /// <inheritdoc/>
    public Task<QueueDepthHistory> GetHistoryAsync(
        string? siloId = null,
        int deviceIndex = 0,
        TimeSpan? duration = null,
        CancellationToken ct = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        var effectiveSiloId = siloId ?? _localSiloId;
        var effectiveDuration = duration ?? TimeSpan.FromMinutes(5);
        var now = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds() * 1_000_000;
        var startTime = now - (long)(effectiveDuration.TotalSeconds * 1_000_000_000);

        // Get samples within duration
        var samples = _recentSamples
            .Where(s => s.TimestampNanos >= startTime)
            .OrderBy(s => s.TimestampNanos)
            .ToList();

        // Calculate trend
        int trendDirection = 0;
        double predictedUtil = 0.0;

        if (samples.Count >= 2)
        {
            var firstHalf = samples.Take(samples.Count / 2).Average(s => s.QueueUtilization);
            var secondHalf = samples.Skip(samples.Count / 2).Average(s => s.QueueUtilization);
            var diff = secondHalf - firstHalf;
            trendDirection = diff > 0.05 ? 1 : (diff < -0.05 ? -1 : 0);

            // Simple linear extrapolation for 1-minute prediction
            predictedUtil = Math.Clamp(secondHalf + diff, 0.0, 1.0);
        }

        var history = new QueueDepthHistory
        {
            SiloId = effectiveSiloId,
            DeviceIndex = deviceIndex,
            StartTimestampNanos = samples.Count > 0 ? samples[0].TimestampNanos : now,
            EndTimestampNanos = now,
            Samples = samples,
            TrendDirection = trendDirection,
            PredictedUtilization1Min = predictedUtil
        };

        return Task.FromResult(history);
    }

    /// <inheritdoc/>
    public async Task<bool> HasCapacityAsync(
        string? siloId = null,
        int deviceIndex = 0,
        double maxQueueUtilization = 0.8,
        CancellationToken ct = default)
    {
        var snapshot = await GetQueueDepthAsync(siloId, deviceIndex, ct);
        return snapshot.AverageQueueUtilization < maxQueueUtilization;
    }

    /// <inheritdoc/>
    public IDisposable SubscribeToAlerts(double threshold, Action<QueueDepthAlert> callback)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        if (!_alertSubscriptions.TryGetValue(threshold, out var callbacks))
        {
            callbacks = new List<Action<QueueDepthAlert>>();
            _alertSubscriptions[threshold] = callbacks;
        }

        lock (callbacks)
        {
            callbacks.Add(callback);
        }

        _logger.LogDebug("Alert subscription added for threshold {Threshold}", threshold);

        return new AlertSubscription(this, threshold, callback);
    }

    private void UnsubscribeFromAlerts(double threshold, Action<QueueDepthAlert> callback)
    {
        if (_alertSubscriptions.TryGetValue(threshold, out var callbacks))
        {
            lock (callbacks)
            {
                callbacks.Remove(callback);
            }
        }
    }

    private async void CollectHistorySample(object? state)
    {
        if (_disposed) return;

        try
        {
            var snapshot = await GetQueueDepthAsync();
            var sample = new QueueDepthSample
            {
                TimestampNanos = snapshot.TimestampNanos,
                QueueUtilization = snapshot.AverageQueueUtilization,
                Throughput = snapshot.ThroughputMsgsPerSec
            };

            _recentSamples.Enqueue(sample);

            // Trim old samples
            while (_recentSamples.Count > MaxHistorySamples)
            {
                _recentSamples.TryDequeue(out _);
            }

            // Check alert thresholds
            foreach (var (threshold, callbacks) in _alertSubscriptions)
            {
                if (snapshot.AverageQueueUtilization >= threshold)
                {
                    var severity = snapshot.AverageQueueUtilization >= 0.95
                        ? QueueAlertSeverity.Emergency
                        : snapshot.AverageQueueUtilization >= 0.9
                            ? QueueAlertSeverity.Critical
                            : snapshot.AverageQueueUtilization >= 0.8
                                ? QueueAlertSeverity.Warning
                                : QueueAlertSeverity.Info;

                    var alert = new QueueDepthAlert
                    {
                        TimestampNanos = snapshot.TimestampNanos,
                        SiloId = snapshot.SiloId,
                        DeviceIndex = snapshot.DeviceIndex,
                        CurrentUtilization = snapshot.AverageQueueUtilization,
                        Threshold = threshold,
                        Severity = severity,
                        RecommendedAction = GetRecommendedAction(severity)
                    };

                    lock (callbacks)
                    {
                        foreach (var callback in callbacks)
                        {
                            try
                            {
                                callback(alert);
                            }
                            catch (Exception ex)
                            {
                                _logger.LogError(ex, "Alert callback failed");
                            }
                        }
                    }
                }
            }
        }
        catch (Exception ex)
        {
            _logger.LogDebug(ex, "History sample collection failed");
        }
    }

    private static string GetRecommendedAction(QueueAlertSeverity severity) => severity switch
    {
        QueueAlertSeverity.Emergency => "Reject new grain activations immediately",
        QueueAlertSeverity.Critical => "Trigger immediate grain rebalancing",
        QueueAlertSeverity.Warning => "Consider load balancing to other devices",
        QueueAlertSeverity.Info => "Monitor queue depth trend",
        _ => "No action required"
    };

    private static QueueDepthSnapshot CreateEmptySnapshot(string siloId, int deviceIndex, long timestampNanos) =>
        new()
        {
            TimestampNanos = timestampNanos,
            SiloId = siloId,
            DeviceIndex = deviceIndex,
            ActiveKernelCount = 0,
            TotalInputQueueDepth = 0,
            TotalOutputQueueDepth = 0,
            TotalInputQueueCapacity = 1,
            TotalOutputQueueCapacity = 1,
            ThroughputMsgsPerSec = 0.0,
            GpuUtilization = 0.0,
            AvailableMemoryBytes = 8L * 1024 * 1024 * 1024,
            TotalMemoryBytes = 8L * 1024 * 1024 * 1024
        };

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _historyCollectionTimer.Dispose();
        _logger.LogDebug("QueueDepthMonitor disposed");
    }

    private sealed class AlertSubscription : IDisposable
    {
        private readonly QueueDepthMonitor _monitor;
        private readonly double _threshold;
        private readonly Action<QueueDepthAlert> _callback;

        public AlertSubscription(QueueDepthMonitor monitor, double threshold, Action<QueueDepthAlert> callback)
        {
            _monitor = monitor;
            _threshold = threshold;
            _callback = callback;
        }

        public void Dispose() => _monitor.UnsubscribeFromAlerts(_threshold, _callback);
    }
}

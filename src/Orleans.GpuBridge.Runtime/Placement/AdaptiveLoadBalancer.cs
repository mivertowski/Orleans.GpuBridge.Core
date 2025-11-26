// Orleans.GpuBridge - GPU-native distributed computing for Microsoft Orleans
// Copyright (c) 2025 Michael Ivertowski

using System.Collections.Concurrent;
using System.Diagnostics;
using Microsoft.Extensions.Logging;
using Orleans.GpuBridge.Abstractions.Placement;

namespace Orleans.GpuBridge.Runtime.Placement;

/// <summary>
/// Adaptive load balancer for GPU-native grains with queue-depth awareness.
/// </summary>
/// <remarks>
/// <para>
/// This load balancer implements multiple strategies for distributing grains
/// across GPU devices based on real-time metrics from the queue depth monitor.
/// </para>
/// <para>
/// <strong>Strategies:</strong>
/// <list type="bullet">
/// <item><description><strong>RoundRobin</strong>: Simple rotation through devices</description></item>
/// <item><description><strong>LeastLoaded</strong>: Route to device with lowest queue depth</description></item>
/// <item><description><strong>WeightedScore</strong>: Multi-factor scoring (queue, memory, compute)</description></item>
/// <item><description><strong>Adaptive</strong>: Uses trend prediction for proactive placement</description></item>
/// </list>
/// </para>
/// </remarks>
public sealed class AdaptiveLoadBalancer : IAdaptiveLoadBalancer, IDisposable
{
    private readonly ILogger<AdaptiveLoadBalancer> _logger;
    private readonly IQueueDepthMonitor _queueDepthMonitor;
    private readonly ConcurrentDictionary<int, BackpressureState> _backpressureStates = new();
    private readonly ConcurrentDictionary<string, int> _affinityDeviceMap = new();
    private readonly ConcurrentBag<Action<LoadBalancingEvent>> _eventSubscribers = new();
    private readonly List<DeviceInfo> _availableDevices;
    private readonly object _roundRobinLock = new();
    private int _roundRobinIndex;
    private long _totalDecisions;
    private long _fallbackCount;
    private long _rebalanceCount;
    private long _backpressureEvents;
    private long _totalDecisionTimeNanos;
    private bool _disposed;

    /// <summary>
    /// Initializes a new instance of the <see cref="AdaptiveLoadBalancer"/> class.
    /// </summary>
    /// <param name="queueDepthMonitor">Queue depth monitor for metrics.</param>
    /// <param name="logger">Logger for diagnostic output.</param>
    /// <param name="deviceCount">Number of GPU devices available.</param>
    public AdaptiveLoadBalancer(
        IQueueDepthMonitor queueDepthMonitor,
        ILogger<AdaptiveLoadBalancer> logger,
        int deviceCount = 1)
    {
        _queueDepthMonitor = queueDepthMonitor ?? throw new ArgumentNullException(nameof(queueDepthMonitor));
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));

        // Initialize device list (in production, would query actual GPU devices)
        _availableDevices = Enumerable.Range(0, Math.Max(1, deviceCount))
            .Select(i => new DeviceInfo(i, $"GPU-{i}"))
            .ToList();

        _logger.LogInformation(
            "AdaptiveLoadBalancer initialized with {DeviceCount} devices",
            _availableDevices.Count);
    }

    /// <inheritdoc/>
    public async Task<LoadBalancingResult> SelectDeviceAsync(
        LoadBalancingRequest request,
        CancellationToken ct = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        var stopwatch = Stopwatch.StartNew();
        Interlocked.Increment(ref _totalDecisions);

        try
        {
            // Check affinity group first
            if (!string.IsNullOrEmpty(request.AffinityGroup) &&
                _affinityDeviceMap.TryGetValue(request.AffinityGroup, out var affinityDevice))
            {
                var affinitySnapshot = await _queueDepthMonitor.GetQueueDepthAsync(
                    null, affinityDevice, ct);

                if (affinitySnapshot.AverageQueueUtilization < request.MaxQueueUtilization &&
                    !IsUnderBackpressure(affinityDevice))
                {
                    stopwatch.Stop();
                    return CreateResult(
                        affinityDevice,
                        affinitySnapshot,
                        1.0, // Perfect score for affinity match
                        false,
                        "Affinity group placement",
                        1,
                        stopwatch.ElapsedTicks * 100); // Convert to nanos
                }
            }

            // Select based on strategy
            var result = request.Strategy switch
            {
                LoadBalancingStrategy.RoundRobin => await SelectRoundRobinAsync(request, ct),
                LoadBalancingStrategy.LeastLoaded => await SelectLeastLoadedAsync(request, ct),
                LoadBalancingStrategy.WeightedScore => await SelectWeightedScoreAsync(request, ct),
                LoadBalancingStrategy.Adaptive => await SelectAdaptiveAsync(request, ct),
                LoadBalancingStrategy.AffinityFirst => await SelectAffinityFirstAsync(request, ct),
                _ => await SelectAdaptiveAsync(request, ct)
            };

            // Register affinity if specified
            if (!string.IsNullOrEmpty(request.AffinityGroup))
            {
                _affinityDeviceMap[request.AffinityGroup] = result.DeviceIndex;
            }

            stopwatch.Stop();
            Interlocked.Add(ref _totalDecisionTimeNanos, stopwatch.ElapsedTicks * 100);

            // Fire event
            NotifyEvent(new LoadBalancingEvent
            {
                TimestampNanos = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds() * 1_000_000,
                EventType = LoadBalancingEventType.PlacementDecision,
                DeviceIndex = result.DeviceIndex,
                Message = $"Placed grain {request.GrainIdentity} using {request.Strategy}"
            });

            return result;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Load balancing failed for grain {GrainId}", request.GrainIdentity);
            Interlocked.Increment(ref _fallbackCount);

            stopwatch.Stop();
            return new LoadBalancingResult
            {
                SiloId = Environment.MachineName,
                DeviceIndex = 0,
                PlacementScore = 0.0,
                IsFallback = true,
                SelectionReason = $"Fallback due to error: {ex.Message}",
                CurrentQueueUtilization = 0.0,
                AvailableMemoryBytes = 0,
                CandidatesEvaluated = 0,
                DecisionTimeNanos = stopwatch.ElapsedTicks * 100
            };
        }
    }

    /// <inheritdoc/>
    public async Task<IReadOnlyList<DeviceLoadStatus>> GetLoadStatusAsync(CancellationToken ct = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        var statuses = new List<DeviceLoadStatus>();

        foreach (var device in _availableDevices)
        {
            try
            {
                var snapshot = await _queueDepthMonitor.GetQueueDepthAsync(null, device.Index, ct);
                var isBackpressure = IsUnderBackpressure(device.Index);

                var status = new DeviceLoadStatus
                {
                    SiloId = Environment.MachineName,
                    DeviceIndex = device.Index,
                    DeviceName = device.Name,
                    ActiveGrainCount = snapshot.ActiveKernelCount * 100, // Approximate
                    ActiveKernelCount = snapshot.ActiveKernelCount,
                    QueueUtilization = snapshot.AverageQueueUtilization,
                    ComputeUtilization = snapshot.GpuUtilization,
                    MemoryUtilization = snapshot.MemoryUtilization,
                    AvailableMemoryBytes = snapshot.AvailableMemoryBytes,
                    CurrentThroughput = snapshot.ThroughputMsgsPerSec,
                    IsUnderBackpressure = isBackpressure,
                    HealthStatus = DetermineHealthStatus(snapshot, isBackpressure)
                };

                statuses.Add(status);
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Failed to get load status for device {DeviceIndex}", device.Index);
                statuses.Add(new DeviceLoadStatus
                {
                    SiloId = Environment.MachineName,
                    DeviceIndex = device.Index,
                    DeviceName = device.Name,
                    ActiveGrainCount = 0,
                    ActiveKernelCount = 0,
                    QueueUtilization = 0,
                    ComputeUtilization = 0,
                    MemoryUtilization = 0,
                    AvailableMemoryBytes = 0,
                    CurrentThroughput = 0,
                    IsUnderBackpressure = false,
                    HealthStatus = DeviceHealthStatus.Unhealthy
                });
            }
        }

        return statuses;
    }

    /// <inheritdoc/>
    public async Task<RebalanceRecommendation> EvaluateRebalanceAsync(CancellationToken ct = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        var statuses = await GetLoadStatusAsync(ct);
        var migrations = new List<MigrationSuggestion>();

        if (statuses.Count < 2)
        {
            return new RebalanceRecommendation
            {
                TimestampNanos = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds() * 1_000_000,
                ShouldRebalance = false,
                Urgency = RebalanceUrgency.None,
                Reason = "Insufficient devices for rebalancing",
                Migrations = migrations,
                ExpectedImprovement = 0.0,
                EstimatedMigrationTime = TimeSpan.Zero
            };
        }

        var loadScores = statuses.Select(s => s.LoadScore).ToList();
        var avgLoad = loadScores.Average();
        var maxLoad = loadScores.Max();
        var minLoad = loadScores.Min();
        var loadDiff = maxLoad - minLoad;

        // Calculate imbalance
        var stdDev = Math.Sqrt(loadScores.Sum(l => Math.Pow(l - avgLoad, 2)) / loadScores.Count);

        // Determine if rebalancing is needed
        RebalanceUrgency urgency;
        bool shouldRebalance;
        string reason;

        if (maxLoad > 0.9)
        {
            urgency = RebalanceUrgency.Critical;
            shouldRebalance = true;
            reason = $"Device overloaded at {maxLoad:P0} utilization";
        }
        else if (loadDiff > 0.4)
        {
            urgency = RebalanceUrgency.High;
            shouldRebalance = true;
            reason = $"Large load imbalance: {loadDiff:P0} difference between devices";
        }
        else if (stdDev > 0.15)
        {
            urgency = RebalanceUrgency.Medium;
            shouldRebalance = true;
            reason = $"Moderate load imbalance (stddev: {stdDev:F2})";
        }
        else if (stdDev > 0.05)
        {
            urgency = RebalanceUrgency.Low;
            shouldRebalance = false; // Optional optimization
            reason = "Minor load imbalance - optimization opportunity";
        }
        else
        {
            urgency = RebalanceUrgency.None;
            shouldRebalance = false;
            reason = "Load is balanced";
        }

        // Generate migration suggestions
        if (shouldRebalance && loadDiff > 0.2)
        {
            var overloadedDevices = statuses.Where(s => s.LoadScore > avgLoad + 0.1).OrderByDescending(s => s.LoadScore);
            var underloadedDevices = statuses.Where(s => s.LoadScore < avgLoad - 0.1).OrderBy(s => s.LoadScore);

            int priority = 1;
            foreach (var overloaded in overloadedDevices)
            {
                var target = underloadedDevices.FirstOrDefault();
                if (target.DeviceIndex != overloaded.DeviceIndex)
                {
                    migrations.Add(new MigrationSuggestion
                    {
                        GrainIdentity = $"High-load grain from device {overloaded.DeviceIndex}",
                        SourceDeviceIndex = overloaded.DeviceIndex,
                        TargetDeviceIndex = target.DeviceIndex,
                        Priority = priority++,
                        ExpectedLoadReduction = (overloaded.LoadScore - avgLoad) * 0.5
                    });
                }
            }
        }

        return new RebalanceRecommendation
        {
            TimestampNanos = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds() * 1_000_000,
            ShouldRebalance = shouldRebalance,
            Urgency = urgency,
            Reason = reason,
            Migrations = migrations,
            ExpectedImprovement = loadDiff * 0.5,
            EstimatedMigrationTime = TimeSpan.FromMilliseconds(migrations.Count * 100)
        };
    }

    /// <inheritdoc/>
    public Task ApplyBackpressureAsync(
        int deviceIndex,
        TimeSpan duration,
        CancellationToken ct = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        var endTime = DateTimeOffset.UtcNow.Add(duration);
        _backpressureStates[deviceIndex] = new BackpressureState(endTime);

        Interlocked.Increment(ref _backpressureEvents);

        _logger.LogWarning(
            "Backpressure applied to device {DeviceIndex} until {EndTime}",
            deviceIndex,
            endTime);

        NotifyEvent(new LoadBalancingEvent
        {
            TimestampNanos = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds() * 1_000_000,
            EventType = LoadBalancingEventType.BackpressureApplied,
            DeviceIndex = deviceIndex,
            Message = $"Backpressure applied for {duration.TotalSeconds}s"
        });

        return Task.CompletedTask;
    }

    /// <inheritdoc/>
    public IDisposable SubscribeToEvents(Action<LoadBalancingEvent> callback)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        _eventSubscribers.Add(callback);
        return new EventSubscription(this, callback);
    }

    /// <inheritdoc/>
    public Task<LoadBalancerMetrics> GetMetricsAsync(CancellationToken ct = default)
    {
        var metrics = new LoadBalancerMetrics
        {
            TotalDecisions = Interlocked.Read(ref _totalDecisions),
            AvgDecisionTimeNanos = _totalDecisions > 0
                ? (double)Interlocked.Read(ref _totalDecisionTimeNanos) / _totalDecisions
                : 0,
            FallbackCount = Interlocked.Read(ref _fallbackCount),
            RebalanceCount = Interlocked.Read(ref _rebalanceCount),
            LoadImbalanceScore = 0.0, // Would calculate from recent history
            BackpressureEvents = Interlocked.Read(ref _backpressureEvents)
        };

        return Task.FromResult(metrics);
    }

    private async Task<LoadBalancingResult> SelectRoundRobinAsync(
        LoadBalancingRequest request,
        CancellationToken ct)
    {
        int deviceIndex;
        lock (_roundRobinLock)
        {
            deviceIndex = _roundRobinIndex;
            _roundRobinIndex = (_roundRobinIndex + 1) % _availableDevices.Count;
        }

        // Skip devices under backpressure
        int attempts = 0;
        while (IsUnderBackpressure(deviceIndex) && attempts < _availableDevices.Count)
        {
            lock (_roundRobinLock)
            {
                deviceIndex = _roundRobinIndex;
                _roundRobinIndex = (_roundRobinIndex + 1) % _availableDevices.Count;
            }
            attempts++;
        }

        var snapshot = await _queueDepthMonitor.GetQueueDepthAsync(null, deviceIndex, ct);
        return CreateResult(deviceIndex, snapshot, 0.5, false, "Round-robin selection",
            _availableDevices.Count, 0);
    }

    private async Task<LoadBalancingResult> SelectLeastLoadedAsync(
        LoadBalancingRequest request,
        CancellationToken ct)
    {
        var candidates = new List<(int DeviceIndex, QueueDepthSnapshot Snapshot, double Score)>();

        foreach (var device in _availableDevices)
        {
            if (IsUnderBackpressure(device.Index)) continue;

            var snapshot = await _queueDepthMonitor.GetQueueDepthAsync(null, device.Index, ct);
            if (snapshot.AverageQueueUtilization < request.MaxQueueUtilization)
            {
                var score = 1.0 - snapshot.AverageQueueUtilization;
                candidates.Add((device.Index, snapshot, score));
            }
        }

        if (candidates.Count == 0)
        {
            Interlocked.Increment(ref _fallbackCount);
            var fallbackSnapshot = await _queueDepthMonitor.GetQueueDepthAsync(null, 0, ct);
            return CreateResult(0, fallbackSnapshot, 0.0, true,
                "No devices below utilization threshold", _availableDevices.Count, 0);
        }

        var best = candidates.OrderByDescending(c => c.Score).First();
        return CreateResult(best.DeviceIndex, best.Snapshot, best.Score, false,
            "Least loaded device", candidates.Count, 0);
    }

    private async Task<LoadBalancingResult> SelectWeightedScoreAsync(
        LoadBalancingRequest request,
        CancellationToken ct)
    {
        const double queueWeight = 0.4;
        const double memoryWeight = 0.3;
        const double computeWeight = 0.3;

        var candidates = new List<(int DeviceIndex, QueueDepthSnapshot Snapshot, double Score)>();

        foreach (var device in _availableDevices)
        {
            if (IsUnderBackpressure(device.Index)) continue;

            var snapshot = await _queueDepthMonitor.GetQueueDepthAsync(null, device.Index, ct);

            if (snapshot.AvailableMemoryBytes < request.MinimumMemoryBytes) continue;
            if (snapshot.AverageQueueUtilization > request.MaxQueueUtilization) continue;

            var score =
                (1.0 - snapshot.AverageQueueUtilization) * queueWeight +
                snapshot.AvailableMemoryRatio * memoryWeight +
                (1.0 - snapshot.GpuUtilization) * computeWeight;

            candidates.Add((device.Index, snapshot, score));
        }

        if (candidates.Count == 0)
        {
            Interlocked.Increment(ref _fallbackCount);
            var fallbackSnapshot = await _queueDepthMonitor.GetQueueDepthAsync(null, 0, ct);
            return CreateResult(0, fallbackSnapshot, 0.0, true,
                "No suitable devices found", _availableDevices.Count, 0);
        }

        var best = candidates.OrderByDescending(c => c.Score).First();
        return CreateResult(best.DeviceIndex, best.Snapshot, best.Score, false,
            "Weighted score selection", candidates.Count, 0);
    }

    private async Task<LoadBalancingResult> SelectAdaptiveAsync(
        LoadBalancingRequest request,
        CancellationToken ct)
    {
        var candidates = new List<(int DeviceIndex, QueueDepthSnapshot Snapshot, QueueDepthHistory History, double Score)>();

        foreach (var device in _availableDevices)
        {
            if (IsUnderBackpressure(device.Index)) continue;

            var snapshot = await _queueDepthMonitor.GetQueueDepthAsync(null, device.Index, ct);
            var history = await _queueDepthMonitor.GetHistoryAsync(null, device.Index, TimeSpan.FromMinutes(5), ct);

            if (snapshot.AvailableMemoryBytes < request.MinimumMemoryBytes) continue;

            // Use predicted utilization for adaptive scoring
            var predictedUtil = history.PredictedUtilization1Min;
            var currentUtil = snapshot.AverageQueueUtilization;

            // Penalize devices with increasing trend
            var trendPenalty = history.TrendDirection > 0 ? 0.1 : 0.0;

            // Skip if predicted to exceed threshold
            if (predictedUtil > request.MaxQueueUtilization) continue;

            var score =
                (1.0 - currentUtil) * 0.3 +
                (1.0 - predictedUtil) * 0.3 +
                snapshot.AvailableMemoryRatio * 0.2 +
                (1.0 - snapshot.GpuUtilization) * 0.2 -
                trendPenalty;

            candidates.Add((device.Index, snapshot, history, Math.Max(0, score)));
        }

        if (candidates.Count == 0)
        {
            Interlocked.Increment(ref _fallbackCount);
            var fallbackSnapshot = await _queueDepthMonitor.GetQueueDepthAsync(null, 0, ct);
            return CreateResult(0, fallbackSnapshot, 0.0, true,
                "No devices with favorable trends", _availableDevices.Count, 0);
        }

        var best = candidates.OrderByDescending(c => c.Score).First();
        return CreateResult(best.DeviceIndex, best.Snapshot, best.Score, false,
            $"Adaptive selection (trend: {best.History.TrendDirection})", candidates.Count, 0);
    }

    private async Task<LoadBalancingResult> SelectAffinityFirstAsync(
        LoadBalancingRequest request,
        CancellationToken ct)
    {
        // Affinity is already checked in SelectDeviceAsync, fall back to weighted
        return await SelectWeightedScoreAsync(request, ct);
    }

    private bool IsUnderBackpressure(int deviceIndex)
    {
        if (_backpressureStates.TryGetValue(deviceIndex, out var state))
        {
            if (DateTimeOffset.UtcNow < state.EndTime)
            {
                return true;
            }

            // Backpressure expired, remove it
            _backpressureStates.TryRemove(deviceIndex, out _);

            NotifyEvent(new LoadBalancingEvent
            {
                TimestampNanos = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds() * 1_000_000,
                EventType = LoadBalancingEventType.BackpressureReleased,
                DeviceIndex = deviceIndex,
                Message = "Backpressure expired"
            });
        }

        return false;
    }

    private static DeviceHealthStatus DetermineHealthStatus(QueueDepthSnapshot snapshot, bool isBackpressure)
    {
        if (isBackpressure) return DeviceHealthStatus.Overloaded;
        if (snapshot.AverageQueueUtilization > 0.95) return DeviceHealthStatus.Unhealthy;
        if (snapshot.AverageQueueUtilization > 0.85) return DeviceHealthStatus.Overloaded;
        if (snapshot.AverageQueueUtilization > 0.7) return DeviceHealthStatus.Degraded;
        return DeviceHealthStatus.Healthy;
    }

    private LoadBalancingResult CreateResult(
        int deviceIndex,
        QueueDepthSnapshot snapshot,
        double score,
        bool isFallback,
        string reason,
        int candidatesEvaluated,
        long decisionTimeNanos)
    {
        return new LoadBalancingResult
        {
            SiloId = snapshot.SiloId,
            DeviceIndex = deviceIndex,
            PlacementScore = score,
            IsFallback = isFallback,
            SelectionReason = reason,
            CurrentQueueUtilization = snapshot.AverageQueueUtilization,
            AvailableMemoryBytes = snapshot.AvailableMemoryBytes,
            CandidatesEvaluated = candidatesEvaluated,
            DecisionTimeNanos = decisionTimeNanos
        };
    }

    private void NotifyEvent(LoadBalancingEvent evt)
    {
        foreach (var subscriber in _eventSubscribers)
        {
            try
            {
                subscriber(evt);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Event subscriber threw exception");
            }
        }
    }

    public void Dispose()
    {
        _disposed = true;
    }

    private readonly record struct DeviceInfo(int Index, string Name);
    private readonly record struct BackpressureState(DateTimeOffset EndTime);

    private sealed class EventSubscription : IDisposable
    {
        private readonly AdaptiveLoadBalancer _balancer;
        private readonly Action<LoadBalancingEvent> _callback;

        public EventSubscription(AdaptiveLoadBalancer balancer, Action<LoadBalancingEvent> callback)
        {
            _balancer = balancer;
            _callback = callback;
        }

        public void Dispose()
        {
            // Note: ConcurrentBag doesn't support removal, would need different collection in production
        }
    }
}

// Orleans.GpuBridge - GPU-native distributed computing for Microsoft Orleans
// Copyright (c) 2025 Michael Ivertowski

namespace Orleans.GpuBridge.Abstractions.Placement;

/// <summary>
/// Adaptive load balancer for GPU-native grains with queue-depth awareness.
/// </summary>
/// <remarks>
/// <para>
/// This interface enables intelligent load distribution across GPU ring kernels
/// based on real-time queue depth, memory availability, and compute utilization.
/// </para>
/// <para>
/// <strong>Load Balancing Strategies:</strong>
/// <list type="bullet">
/// <item><description><strong>RoundRobin</strong>: Distribute evenly across GPUs</description></item>
/// <item><description><strong>LeastLoaded</strong>: Route to GPU with lowest queue depth</description></item>
/// <item><description><strong>WeightedScore</strong>: Multi-factor scoring (queue, memory, compute)</description></item>
/// <item><description><strong>Adaptive</strong>: ML-based prediction with trend analysis</description></item>
/// </list>
/// </para>
/// </remarks>
public interface IAdaptiveLoadBalancer
{
    /// <summary>
    /// Selects the optimal GPU device for a new grain activation.
    /// </summary>
    /// <param name="request">Load balancing request with requirements.</param>
    /// <param name="ct">Cancellation token.</param>
    /// <returns>Selected device with placement score.</returns>
    Task<LoadBalancingResult> SelectDeviceAsync(
        LoadBalancingRequest request,
        CancellationToken ct = default);

    /// <summary>
    /// Gets the current load status across all available devices.
    /// </summary>
    /// <param name="ct">Cancellation token.</param>
    /// <returns>Load status for each device.</returns>
    Task<IReadOnlyList<DeviceLoadStatus>> GetLoadStatusAsync(CancellationToken ct = default);

    /// <summary>
    /// Evaluates whether grains should be rebalanced across devices.
    /// </summary>
    /// <param name="ct">Cancellation token.</param>
    /// <returns>Rebalancing recommendation with migration suggestions.</returns>
    Task<RebalanceRecommendation> EvaluateRebalanceAsync(CancellationToken ct = default);

    /// <summary>
    /// Applies backpressure by temporarily reducing acceptance of new grains.
    /// </summary>
    /// <param name="deviceIndex">GPU device to apply backpressure.</param>
    /// <param name="duration">Duration to apply backpressure.</param>
    /// <param name="ct">Cancellation token.</param>
    Task ApplyBackpressureAsync(
        int deviceIndex,
        TimeSpan duration,
        CancellationToken ct = default);

    /// <summary>
    /// Registers a callback for load balancing events.
    /// </summary>
    /// <param name="callback">Event callback.</param>
    /// <returns>Subscription handle.</returns>
    IDisposable SubscribeToEvents(Action<LoadBalancingEvent> callback);

    /// <summary>
    /// Gets performance metrics for the load balancer.
    /// </summary>
    /// <param name="ct">Cancellation token.</param>
    /// <returns>Load balancer performance metrics.</returns>
    Task<LoadBalancerMetrics> GetMetricsAsync(CancellationToken ct = default);
}

/// <summary>
/// Request for load balancing decision.
/// </summary>
public sealed class LoadBalancingRequest
{
    /// <summary>
    /// Grain type being placed.
    /// </summary>
    public required string GrainType { get; init; }

    /// <summary>
    /// Grain identity for affinity considerations.
    /// </summary>
    public required string GrainIdentity { get; init; }

    /// <summary>
    /// Affinity group for colocation (optional).
    /// </summary>
    public string? AffinityGroup { get; init; }

    /// <summary>
    /// Minimum required GPU memory in bytes.
    /// </summary>
    public long MinimumMemoryBytes { get; init; } = 0;

    /// <summary>
    /// Maximum acceptable queue utilization (0.0-1.0).
    /// </summary>
    public double MaxQueueUtilization { get; init; } = 0.8;

    /// <summary>
    /// Preferred silo (for local placement preference).
    /// </summary>
    public string? PreferredSilo { get; init; }

    /// <summary>
    /// Load balancing strategy to use.
    /// </summary>
    public LoadBalancingStrategy Strategy { get; init; } = LoadBalancingStrategy.Adaptive;

    /// <summary>
    /// Expected message throughput (msgs/sec) for capacity planning.
    /// </summary>
    public double ExpectedThroughput { get; init; } = 1000.0;
}

/// <summary>
/// Result of load balancing decision.
/// </summary>
public readonly record struct LoadBalancingResult
{
    /// <summary>
    /// Selected silo identifier.
    /// </summary>
    public required string SiloId { get; init; }

    /// <summary>
    /// Selected GPU device index.
    /// </summary>
    public required int DeviceIndex { get; init; }

    /// <summary>
    /// Placement score (0.0-1.0, higher is better).
    /// </summary>
    public required double PlacementScore { get; init; }

    /// <summary>
    /// Indicates if this is a fallback selection.
    /// </summary>
    public required bool IsFallback { get; init; }

    /// <summary>
    /// Reason for selection.
    /// </summary>
    public required string SelectionReason { get; init; }

    /// <summary>
    /// Current queue utilization on selected device.
    /// </summary>
    public required double CurrentQueueUtilization { get; init; }

    /// <summary>
    /// Available memory on selected device (bytes).
    /// </summary>
    public required long AvailableMemoryBytes { get; init; }

    /// <summary>
    /// Number of candidate devices evaluated.
    /// </summary>
    public required int CandidatesEvaluated { get; init; }

    /// <summary>
    /// Time taken to make decision (nanoseconds).
    /// </summary>
    public required long DecisionTimeNanos { get; init; }
}

/// <summary>
/// Load status for a single GPU device.
/// </summary>
public readonly record struct DeviceLoadStatus
{
    /// <summary>
    /// Silo identifier.
    /// </summary>
    public required string SiloId { get; init; }

    /// <summary>
    /// GPU device index.
    /// </summary>
    public required int DeviceIndex { get; init; }

    /// <summary>
    /// Device name/model.
    /// </summary>
    public required string DeviceName { get; init; }

    /// <summary>
    /// Number of active grains on this device.
    /// </summary>
    public required int ActiveGrainCount { get; init; }

    /// <summary>
    /// Number of ring kernels running.
    /// </summary>
    public required int ActiveKernelCount { get; init; }

    /// <summary>
    /// Current queue utilization (0.0-1.0).
    /// </summary>
    public required double QueueUtilization { get; init; }

    /// <summary>
    /// Current compute utilization (0.0-1.0).
    /// </summary>
    public required double ComputeUtilization { get; init; }

    /// <summary>
    /// Current memory utilization (0.0-1.0).
    /// </summary>
    public required double MemoryUtilization { get; init; }

    /// <summary>
    /// Available memory in bytes.
    /// </summary>
    public required long AvailableMemoryBytes { get; init; }

    /// <summary>
    /// Current throughput (msgs/sec).
    /// </summary>
    public required double CurrentThroughput { get; init; }

    /// <summary>
    /// Whether backpressure is currently applied.
    /// </summary>
    public required bool IsUnderBackpressure { get; init; }

    /// <summary>
    /// Health status of the device.
    /// </summary>
    public required DeviceHealthStatus HealthStatus { get; init; }

    /// <summary>
    /// Overall load score (0.0-1.0, lower is less loaded).
    /// </summary>
    public double LoadScore =>
        (QueueUtilization * 0.4) + (ComputeUtilization * 0.3) + (MemoryUtilization * 0.3);
}

/// <summary>
/// Recommendation for rebalancing grains across devices.
/// </summary>
public sealed class RebalanceRecommendation
{
    /// <summary>
    /// Timestamp of recommendation.
    /// </summary>
    public required long TimestampNanos { get; init; }

    /// <summary>
    /// Whether rebalancing is recommended.
    /// </summary>
    public required bool ShouldRebalance { get; init; }

    /// <summary>
    /// Urgency level of rebalancing.
    /// </summary>
    public required RebalanceUrgency Urgency { get; init; }

    /// <summary>
    /// Reason for recommendation.
    /// </summary>
    public required string Reason { get; init; }

    /// <summary>
    /// Suggested grain migrations.
    /// </summary>
    public required IReadOnlyList<MigrationSuggestion> Migrations { get; init; }

    /// <summary>
    /// Expected improvement in load balance (0.0-1.0).
    /// </summary>
    public required double ExpectedImprovement { get; init; }

    /// <summary>
    /// Estimated cost of rebalancing (in terms of message latency spike).
    /// </summary>
    public required TimeSpan EstimatedMigrationTime { get; init; }
}

/// <summary>
/// Suggested migration for rebalancing.
/// </summary>
public readonly record struct MigrationSuggestion
{
    /// <summary>
    /// Grain identity to migrate.
    /// </summary>
    public required string GrainIdentity { get; init; }

    /// <summary>
    /// Source device index.
    /// </summary>
    public required int SourceDeviceIndex { get; init; }

    /// <summary>
    /// Target device index.
    /// </summary>
    public required int TargetDeviceIndex { get; init; }

    /// <summary>
    /// Priority of migration (higher = more important).
    /// </summary>
    public required int Priority { get; init; }

    /// <summary>
    /// Expected load reduction on source device.
    /// </summary>
    public required double ExpectedLoadReduction { get; init; }
}

/// <summary>
/// Event from load balancer.
/// </summary>
public readonly record struct LoadBalancingEvent
{
    /// <summary>
    /// Timestamp of event.
    /// </summary>
    public required long TimestampNanos { get; init; }

    /// <summary>
    /// Type of event.
    /// </summary>
    public required LoadBalancingEventType EventType { get; init; }

    /// <summary>
    /// Device index related to event.
    /// </summary>
    public required int DeviceIndex { get; init; }

    /// <summary>
    /// Event message.
    /// </summary>
    public required string Message { get; init; }

    /// <summary>
    /// Additional event data.
    /// </summary>
    public IReadOnlyDictionary<string, object>? Data { get; init; }
}

/// <summary>
/// Performance metrics for load balancer.
/// </summary>
public readonly record struct LoadBalancerMetrics
{
    /// <summary>
    /// Total placement decisions made.
    /// </summary>
    public required long TotalDecisions { get; init; }

    /// <summary>
    /// Average decision time (nanoseconds).
    /// </summary>
    public required double AvgDecisionTimeNanos { get; init; }

    /// <summary>
    /// Number of fallback selections.
    /// </summary>
    public required long FallbackCount { get; init; }

    /// <summary>
    /// Number of rebalancing operations performed.
    /// </summary>
    public required long RebalanceCount { get; init; }

    /// <summary>
    /// Current load imbalance score (0 = perfectly balanced).
    /// </summary>
    public required double LoadImbalanceScore { get; init; }

    /// <summary>
    /// Number of backpressure events.
    /// </summary>
    public required long BackpressureEvents { get; init; }
}

/// <summary>
/// Load balancing strategies.
/// </summary>
public enum LoadBalancingStrategy
{
    /// <summary>
    /// Round-robin distribution across devices.
    /// </summary>
    RoundRobin = 0,

    /// <summary>
    /// Route to device with lowest queue depth.
    /// </summary>
    LeastLoaded = 1,

    /// <summary>
    /// Multi-factor weighted scoring.
    /// </summary>
    WeightedScore = 2,

    /// <summary>
    /// ML-based adaptive with trend prediction.
    /// </summary>
    Adaptive = 3,

    /// <summary>
    /// Affinity-first (prioritize colocation).
    /// </summary>
    AffinityFirst = 4
}

/// <summary>
/// Rebalance urgency levels.
/// </summary>
public enum RebalanceUrgency
{
    /// <summary>
    /// No rebalancing needed.
    /// </summary>
    None = 0,

    /// <summary>
    /// Optional optimization.
    /// </summary>
    Low = 1,

    /// <summary>
    /// Recommended for performance.
    /// </summary>
    Medium = 2,

    /// <summary>
    /// Required to avoid degradation.
    /// </summary>
    High = 3,

    /// <summary>
    /// Critical - immediate action required.
    /// </summary>
    Critical = 4
}

/// <summary>
/// Load balancing event types.
/// </summary>
public enum LoadBalancingEventType
{
    /// <summary>
    /// New grain placement decision.
    /// </summary>
    PlacementDecision = 0,

    /// <summary>
    /// Backpressure applied.
    /// </summary>
    BackpressureApplied = 1,

    /// <summary>
    /// Backpressure released.
    /// </summary>
    BackpressureReleased = 2,

    /// <summary>
    /// Rebalancing triggered.
    /// </summary>
    RebalanceTriggered = 3,

    /// <summary>
    /// Device became overloaded.
    /// </summary>
    DeviceOverloaded = 4,

    /// <summary>
    /// Device recovered from overload.
    /// </summary>
    DeviceRecovered = 5,

    /// <summary>
    /// Device health changed.
    /// </summary>
    HealthChanged = 6
}

/// <summary>
/// Device health status.
/// </summary>
public enum DeviceHealthStatus
{
    /// <summary>
    /// Device is healthy and accepting work.
    /// </summary>
    Healthy = 0,

    /// <summary>
    /// Device is degraded but functional.
    /// </summary>
    Degraded = 1,

    /// <summary>
    /// Device is overloaded.
    /// </summary>
    Overloaded = 2,

    /// <summary>
    /// Device is unhealthy.
    /// </summary>
    Unhealthy = 3,

    /// <summary>
    /// Device is offline.
    /// </summary>
    Offline = 4
}

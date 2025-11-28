# Phase 7 Implementation Guide

**Queue-Depth Aware Placement for GPU-Native Actors**

## Overview

Phase 7 focuses on intelligent grain placement based on GPU ring kernel queue depths. This enables automatic load balancing and optimal resource utilization across GPU-enabled silos.

## Goals

1. **Monitor Queue Depths**: Real-time tracking of ring kernel message queue utilization
2. **Intelligent Placement**: Place new activations on silos with lowest queue pressure
3. **Dynamic Rebalancing**: Migrate actors when queues become unbalanced
4. **Overflow Prevention**: Detect and handle queue saturation before message loss

## Architecture

### Component Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Orleans Cluster                           │
│                                                              │
│  ┌──────────────────┐    ┌──────────────────┐              │
│  │     Silo A       │    │     Silo B       │              │
│  │                  │    │                  │              │
│  │  ┌────────────┐  │    │  ┌────────────┐  │              │
│  │  │ Queue      │  │    │  │ Queue      │  │              │
│  │  │ Monitor    │  │    │  │ Monitor    │  │              │
│  │  └─────┬──────┘  │    │  └─────┬──────┘  │              │
│  │        │         │    │        │         │              │
│  │  ┌─────▼──────┐  │    │  ┌─────▼──────┐  │              │
│  │  │ Ring       │  │    │  │ Ring       │  │              │
│  │  │ Kernels    │  │    │  │ Kernels    │  │              │
│  │  │            │  │    │  │            │  │              │
│  │  │ Q1: 45%    │  │    │  │ Q1: 82%    │  │              │
│  │  │ Q2: 23%    │  │    │  │ Q2: 15%    │  │              │
│  │  │ Q3: 67%    │  │    │  │ Q3: 91%    │  │              │
│  │  └────────────┘  │    │  └────────────┘  │              │
│  │                  │    │                  │              │
│  │  Avg: 45%        │    │  Avg: 63%        │              │
│  └──────────────────┘    └──────────────────┘              │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │            Placement Director                          │  │
│  │                                                        │  │
│  │  New GpuResident activation request                    │  │
│  │  → Query all silos for queue depths                    │  │
│  │  → Select silo with lowest average (Silo A: 45%)       │  │
│  │  → Activate on selected silo                           │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### New Components

#### 1. IQueueDepthMonitor

```csharp
/// <summary>
/// Monitors ring kernel queue depths for placement decisions.
/// </summary>
public interface IQueueDepthMonitor
{
    /// <summary>
    /// Get current queue utilization for all kernels on this silo.
    /// </summary>
    Task<QueueDepthSnapshot> GetSnapshotAsync();

    /// <summary>
    /// Get queue depth for a specific kernel.
    /// </summary>
    Task<float> GetKernelUtilizationAsync(string kernelId);

    /// <summary>
    /// Subscribe to queue depth changes.
    /// </summary>
    IAsyncEnumerable<QueueDepthChange> WatchAsync(
        float threshold = 0.8f,
        CancellationToken ct = default);
}

public record QueueDepthSnapshot(
    string SiloId,
    DateTimeOffset Timestamp,
    IReadOnlyDictionary<string, float> KernelUtilizations,
    float AverageUtilization,
    float MaxUtilization);

public record QueueDepthChange(
    string KernelId,
    float OldUtilization,
    float NewUtilization,
    bool IsOverThreshold);
```

#### 2. QueueDepthAwarePlacementDirector

```csharp
/// <summary>
/// Placement director that considers ring kernel queue depths.
/// </summary>
public class QueueDepthAwarePlacementDirector : IPlacementDirector
{
    private readonly IQueueDepthMonitor _monitor;
    private readonly ILogger<QueueDepthAwarePlacementDirector> _logger;

    public async Task<SiloAddress> OnAddActivation(
        PlacementStrategy strategy,
        PlacementTarget target,
        IPlacementContext context)
    {
        // Get queue depths from all GPU-enabled silos
        var candidates = await GetGpuEnabledSilosAsync(context);
        var snapshots = await Task.WhenAll(
            candidates.Select(s => GetQueueSnapshotAsync(s)));

        // Select silo with lowest average queue depth
        var selected = snapshots
            .Where(s => s.AverageUtilization < 0.8f)  // Filter overloaded
            .OrderBy(s => s.AverageUtilization)
            .FirstOrDefault();

        if (selected == null)
        {
            _logger.LogWarning(
                "All GPU silos above 80% queue utilization, " +
                "falling back to random placement");
            return candidates.RandomElement();
        }

        return selected.SiloAddress;
    }
}
```

#### 3. QueueOverflowHandler

```csharp
/// <summary>
/// Handles queue overflow scenarios to prevent message loss.
/// </summary>
public interface IQueueOverflowHandler
{
    /// <summary>
    /// Handle queue full condition.
    /// </summary>
    Task<OverflowResult> HandleOverflowAsync(
        string kernelId,
        GpuMessage message,
        OverflowStrategy strategy);
}

public enum OverflowStrategy
{
    /// <summary>
    /// Block until space available.
    /// </summary>
    Block,

    /// <summary>
    /// Drop oldest messages.
    /// </summary>
    DropOldest,

    /// <summary>
    /// Drop new message.
    /// </summary>
    DropNew,

    /// <summary>
    /// Redirect to CPU fallback.
    /// </summary>
    CpuFallback,

    /// <summary>
    /// Migrate actor to less loaded silo.
    /// </summary>
    Migrate
}
```

## Implementation Plan

### Step 1: Queue Depth Metrics

Add metrics collection to `DotComputeRingKernelRuntime`:

```csharp
public class DotComputeRingKernelRuntime : IRingKernelRuntime
{
    private readonly ConcurrentDictionary<string, QueueMetrics> _queueMetrics = new();

    public async Task<RingKernelMetrics> GetMetricsAsync(string kernelId)
    {
        var controlBlock = await ReadControlBlockAsync(kernelId);

        var metrics = new RingKernelMetrics
        {
            KernelId = kernelId,
            InputQueueUtilization = CalculateUtilization(
                controlBlock.InputQueueHead,
                controlBlock.InputQueueTail,
                _queueCapacity),
            OutputQueueUtilization = CalculateUtilization(
                controlBlock.OutputQueueHead,
                controlBlock.OutputQueueTail,
                _queueCapacity),
            MessagesProcessed = controlBlock.MessagesProcessed,
            ErrorsEncountered = controlBlock.ErrorsEncountered,
            LastActivityTime = controlBlock.LastActivityTicks
        };

        return metrics;
    }

    private float CalculateUtilization(long head, long tail, int capacity)
    {
        long count = head - tail;
        if (count < 0) count += capacity;
        return (float)count / capacity;
    }
}
```

### Step 2: Cross-Silo Metrics Collection

Implement distributed metrics aggregation:

```csharp
public class DistributedQueueMonitor : IQueueDepthMonitor
{
    private readonly IClusterClient _clusterClient;
    private readonly ILocalSiloDetails _siloDetails;

    public async Task<IReadOnlyList<QueueDepthSnapshot>> GetClusterSnapshotsAsync()
    {
        var silos = await GetGpuEnabledSilosAsync();

        var tasks = silos.Select(async silo =>
        {
            var grain = _clusterClient.GetGrain<IQueueMonitorGrain>(silo.Address);
            return await grain.GetSnapshotAsync();
        });

        return await Task.WhenAll(tasks);
    }
}

/// <summary>
/// Per-silo grain for queue monitoring.
/// </summary>
public interface IQueueMonitorGrain : IGrainWithStringKey
{
    Task<QueueDepthSnapshot> GetSnapshotAsync();
    Task<float> GetKernelUtilizationAsync(string kernelId);
}
```

### Step 3: Placement Integration

Register the placement director:

```csharp
public static class GpuPlacementExtensions
{
    public static ISiloBuilder UseQueueDepthAwarePlacement(
        this ISiloBuilder builder,
        Action<QueueDepthPlacementOptions>? configure = null)
    {
        var options = new QueueDepthPlacementOptions();
        configure?.Invoke(options);

        builder.ConfigureServices(services =>
        {
            services.AddSingleton(options);
            services.AddSingleton<IQueueDepthMonitor, DistributedQueueMonitor>();
            services.AddSingleton<IPlacementDirector, QueueDepthAwarePlacementDirector>();
        });

        return builder;
    }
}

public class QueueDepthPlacementOptions
{
    public float HighUtilizationThreshold { get; set; } = 0.8f;
    public float CriticalUtilizationThreshold { get; set; } = 0.95f;
    public TimeSpan MetricsRefreshInterval { get; set; } = TimeSpan.FromSeconds(1);
    public OverflowStrategy DefaultOverflowStrategy { get; set; } = OverflowStrategy.Block;
}
```

### Step 4: Grain Attribute

```csharp
/// <summary>
/// Marks a grain for queue-depth-aware placement.
/// </summary>
[AttributeUsage(AttributeTargets.Class)]
public class QueueDepthAwarePlacementAttribute : PlacementAttribute
{
    public QueueDepthAwarePlacementAttribute()
        : base(new QueueDepthAwarePlacementStrategy())
    {
    }

    public float PreferredMaxUtilization { get; set; } = 0.7f;
}
```

## Usage Example

```csharp
[GpuResident]
[QueueDepthAwarePlacement(PreferredMaxUtilization = 0.6f)]
public class HighThroughputVertexGrain : Grain, ITemporalGraphVertex
{
    private readonly IGpuResidentManager _residentManager;

    public override async Task OnActivateAsync(CancellationToken ct)
    {
        // Grain will be placed on silo with lowest queue depth
        _gpuHandle = await _residentManager.AllocateVertexAsync(
            (ulong)this.GetPrimaryKeyLong());

        await base.OnActivateAsync(ct);
    }

    public async Task AddEdgeAsync(ulong targetId, long timestamp)
    {
        // If queue is full, overflow handler kicks in
        await _residentManager.SendMessageAsync(_vertexIndex, new GpuMessage
        {
            Type = MessageType.AddEdge,
            Data0 = (long)targetId,
            Data1 = timestamp
        });
    }
}
```

## Testing Strategy

### Unit Tests

```csharp
[Fact]
public async Task QueueMonitor_ReturnsCorrectUtilization()
{
    // Arrange
    var runtime = CreateMockRuntime(queueCapacity: 1000, messagesInQueue: 450);
    var monitor = new LocalQueueMonitor(runtime);

    // Act
    var snapshot = await monitor.GetSnapshotAsync();

    // Assert
    Assert.Equal(0.45f, snapshot.AverageUtilization, precision: 2);
}

[Fact]
public async Task PlacementDirector_SelectsLowestUtilizationSilo()
{
    // Arrange
    var siloA = CreateSiloWithUtilization(0.45f);
    var siloB = CreateSiloWithUtilization(0.63f);
    var siloC = CreateSiloWithUtilization(0.82f);

    var director = new QueueDepthAwarePlacementDirector([siloA, siloB, siloC]);

    // Act
    var selected = await director.OnAddActivation(...);

    // Assert
    Assert.Equal(siloA.Address, selected);
}
```

### Integration Tests

```csharp
[Fact]
public async Task HighLoad_DistributesAcrossSilos()
{
    // Arrange
    var cluster = CreateTestCluster(silos: 3, gpuPerSilo: 1);

    // Act: Create 100 GPU-resident actors
    var tasks = Enumerable.Range(0, 100)
        .Select(i => cluster.GetGrain<IVertexGrain>(i).ActivateAsync());
    await Task.WhenAll(tasks);

    // Assert: Distribution should be roughly balanced
    var distribution = await GetActorDistributionAsync(cluster);
    Assert.All(distribution.Values, count =>
        Assert.InRange(count, 25, 45)); // ~33 ± 12 per silo
}
```

## Migration Path

### From v0.1.0 to v0.2.0

1. **No breaking changes**: Existing `[GpuResident]` grains continue to work
2. **Opt-in**: Add `[QueueDepthAwarePlacement]` attribute for new behavior
3. **Configuration**: Enable cluster-wide via `UseQueueDepthAwarePlacement()`

### Backward Compatibility

```csharp
// v0.1.0 style (continues to work)
[GpuResident]
public class LegacyGrain : Grain { }

// v0.2.0 style (opt-in to queue awareness)
[GpuResident]
[QueueDepthAwarePlacement]
public class SmartGrain : Grain { }
```

## Success Criteria

| Metric | Target | Measurement |
|--------|--------|-------------|
| Queue utilization variance | <15% | Across silos |
| Placement decision latency | <10ms | P99 |
| No message drops | 100% | Under normal load |
| Overflow handling | <100ms | Recovery time |

## See Also

- [Implementation Roadmap](IMPLEMENTATION-ROADMAP.md) - Phase overview
- [Ring Kernel Integration](../../architecture/RING-KERNEL-INTEGRATION.md) - Queue architecture
- [GPU Memory Operations](../../implementation/GPU-MEMORY-OPERATIONS.md) - Memory management

---

*Phase 7: Intelligent placement for optimal GPU utilization.*

**Target Version**: 0.2.0
**Last Updated**: 2025-11-28

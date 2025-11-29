// Copyright (c) 2025 Michael Ivertowski
// Licensed under the Apache License, Version 2.0. See LICENSE file for details.

using Orleans.Placement;
using Orleans.Runtime;
using Orleans.Runtime.Placement;

namespace Orleans.GpuBridge.Runtime.Placement;

/// <summary>
/// Advanced placement strategy for GPU-native grains with ring kernel awareness.
/// </summary>
/// <remarks>
/// <para>
/// This strategy extends basic GPU placement with ring kernel-specific optimizations:
/// - **Queue Depth Tracking**: Monitors ring kernel message queue utilization
/// - **Device Affinity**: Collocates related grains on the same GPU for P2P messaging
/// - **Compute Unit Awareness**: Tracks actual GPU compute utilization
/// - **Dynamic Load Balancing**: Rebalances grains when GPUs become overloaded
/// </para>
/// <para>
/// Placement scoring algorithm:
/// <code>
/// Score = (AvailableMemoryRatio × 0.3) +
///         ((1 - QueueUtilization) × 0.4) +
///         ((1 - ComputeUtilization) × 0.3)
/// </code>
/// Higher scores indicate better placement targets.
/// </para>
/// </remarks>
[Serializable]
[GenerateSerializer]
public sealed class GpuNativePlacementStrategy : PlacementStrategy
{
    /// <summary>
    /// Singleton instance for GPU-native placement.
    /// </summary>
    public static GpuNativePlacementStrategy Instance { get; } = new();

    /// <summary>
    /// Minimum GPU memory required (MB).
    /// </summary>
    [Id(0)]
    public int MinimumGpuMemoryMB { get; init; } = 512;

    /// <summary>
    /// Maximum ring kernel queue depth utilization (0.0 - 1.0).
    /// </summary>
    /// <remarks>
    /// Grains won't be placed on GPUs where queue utilization exceeds this threshold.
    /// Default: 0.8 (80% full).
    /// </remarks>
    [Id(1)]
    public double MaxQueueUtilization { get; init; } = 0.8;

    /// <summary>
    /// Maximum GPU compute utilization (0.0 - 1.0).
    /// </summary>
    /// <remarks>
    /// Grains won't be placed on GPUs where compute utilization exceeds this threshold.
    /// Default: 0.9 (90% busy).
    /// </remarks>
    [Id(2)]
    public double MaxComputeUtilization { get; init; } = 0.9;

    /// <summary>
    /// Device affinity group ID for collocating related grains.
    /// </summary>
    /// <remarks>
    /// Grains with the same affinity group will be placed on the same GPU when possible.
    /// This enables efficient P2P messaging without PCIe transfers.
    /// Default: null (no affinity).
    /// </remarks>
    [Id(3)]
    public string? AffinityGroupId { get; init; }

    /// <summary>
    /// Prefer local GPU even if remote GPUs have better scores.
    /// </summary>
    /// <remarks>
    /// Reduces network latency at the cost of potentially suboptimal GPU utilization.
    /// Default: false.
    /// </remarks>
    [Id(4)]
    public bool PreferLocalGpu { get; init; }

    /// <summary>
    /// Enable dynamic rebalancing when GPU load becomes skewed.
    /// </summary>
    /// <remarks>
    /// When enabled, grains may be migrated between GPUs to maintain balanced load.
    /// Default: true.
    /// </remarks>
    [Id(5)]
    public bool EnableDynamicRebalancing { get; init; } = true;

    /// <summary>
    /// Minimum placement score to consider a GPU viable (0.0 - 1.0).
    /// </summary>
    /// <remarks>
    /// GPUs with scores below this threshold will be excluded from placement.
    /// Default: 0.2 (20% score).
    /// </remarks>
    [Id(6)]
    public double MinimumPlacementScore { get; init; } = 0.2;

    /// <summary>
    /// Weight for memory availability in placement score.
    /// </summary>
    [Id(7)]
    public double MemoryWeight { get; init; } = 0.3;

    /// <summary>
    /// Weight for queue utilization in placement score.
    /// </summary>
    [Id(8)]
    public double QueueWeight { get; init; } = 0.4;

    /// <summary>
    /// Weight for compute utilization in placement score.
    /// </summary>
    [Id(9)]
    public double ComputeWeight { get; init; } = 0.3;
}

/// <summary>
/// Attribute to mark grains for GPU-native placement with ring kernels.
/// </summary>
/// <example>
/// <code>
/// [GpuNativePlacement(AffinityGroupId = "physics-simulation", MinMemoryMB = 1024)]
/// public class PhysicsActorGrain : GpuNativeGrain, IPhysicsActor
/// {
///     // ...
/// }
/// </code>
/// </example>
[AttributeUsage(AttributeTargets.Class, AllowMultiple = false)]
public sealed class GpuNativePlacementAttribute : PlacementAttribute
{
    private int _minMemoryMB = 512;
    private double _maxQueueUtilization = 0.8;
    private string? _affinityGroupId;
    private bool _preferLocalGpu = false;

    /// <summary>
    /// Initializes a new instance of the GPU-native placement attribute.
    /// </summary>
    public GpuNativePlacementAttribute()
        : base(CreateStrategy())
    {
    }

    /// <summary>
    /// Minimum GPU memory required (MB).
    /// </summary>
    public int MinMemoryMB
    {
        get => _minMemoryMB;
        set
        {
            _minMemoryMB = value;
            UpdateStrategy();
        }
    }

    /// <summary>
    /// Maximum queue utilization threshold (0.0 - 1.0).
    /// </summary>
    public double MaxQueueUtilization
    {
        get => _maxQueueUtilization;
        set
        {
            _maxQueueUtilization = value;
            UpdateStrategy();
        }
    }

    /// <summary>
    /// Device affinity group ID for collocating related grains.
    /// </summary>
    public string? AffinityGroupId
    {
        get => _affinityGroupId;
        set
        {
            _affinityGroupId = value;
            UpdateStrategy();
        }
    }

    /// <summary>
    /// Prefer local GPU for placement.
    /// </summary>
    public bool PreferLocalGpu
    {
        get => _preferLocalGpu;
        set
        {
            _preferLocalGpu = value;
            UpdateStrategy();
        }
    }

    private static GpuNativePlacementStrategy CreateStrategy() =>
        new GpuNativePlacementStrategy();

    private void UpdateStrategy()
    {
        // Properties are set after construction via attribute syntax
        // The strategy is created in the constructor and cannot be modified
        // This is just for tracking values that would be used in a custom strategy instance
    }
}

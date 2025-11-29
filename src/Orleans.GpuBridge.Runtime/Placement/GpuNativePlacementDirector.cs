// Copyright (c) 2025 Michael Ivertowski
// Licensed under the Apache License, Version 2.0. See LICENSE file for details.

using Microsoft.Extensions.Logging;
using Orleans.Runtime;
using Orleans.Runtime.Placement;
using DotCompute.Abstractions.RingKernels;

namespace Orleans.GpuBridge.Runtime.Placement;

/// <summary>
/// Placement director for GPU-native grains with ring kernel awareness.
/// </summary>
/// <remarks>
/// <para>
/// Implements intelligent placement decisions based on:
/// - GPU memory availability
/// - Ring kernel queue depth
/// - GPU compute unit utilization
/// - Device affinity groups
/// - Dynamic load balancing
/// </para>
/// <para>
/// The director queries ring kernel runtime for real-time GPU metrics
/// and selects the optimal GPU for each grain activation.
/// </para>
/// </remarks>
public sealed class GpuNativePlacementDirector : IPlacementDirector
{
    private readonly ILogger<GpuNativePlacementDirector> _logger;
    private readonly IRingKernelRuntime _ringKernelRuntime;
    private readonly Dictionary<string, List<SiloAddress>> _affinityGroups = new();
    private readonly object _affinityLock = new();

    /// <summary>
    /// Initializes a new instance of the <see cref="GpuNativePlacementDirector"/> class.
    /// </summary>
    /// <param name="logger">Logger for diagnostic output.</param>
    /// <param name="ringKernelRuntime">Ring kernel runtime for GPU metrics.</param>
    /// <exception cref="ArgumentNullException">Thrown when logger or ringKernelRuntime is null.</exception>
    public GpuNativePlacementDirector(
        ILogger<GpuNativePlacementDirector> logger,
        IRingKernelRuntime ringKernelRuntime)
    {
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        _ringKernelRuntime = ringKernelRuntime ?? throw new ArgumentNullException(nameof(ringKernelRuntime));
    }

    /// <summary>
    /// Selects optimal GPU-equipped silo for grain activation.
    /// </summary>
    public async Task<SiloAddress> OnAddActivation(
        PlacementStrategy strategy,
        PlacementTarget target,
        IPlacementContext context)
    {
        if (strategy is not GpuNativePlacementStrategy gpuStrategy)
        {
            _logger.LogWarning(
                "GpuNativePlacementDirector invoked with incompatible strategy {StrategyType} for grain {GrainId}",
                strategy.GetType().Name,
                target.GrainIdentity);
            return SelectFallbackSilo(context, target);
        }

        _logger.LogDebug(
            "Selecting GPU-native placement for {GrainId} (AffinityGroup: {AffinityGroup}, MinMemory: {MinMemory}MB)",
            target.GrainIdentity,
            gpuStrategy.AffinityGroupId ?? "none",
            gpuStrategy.MinimumGpuMemoryMB);

        try
        {
            // Check affinity group for colocated placement
            if (!string.IsNullOrEmpty(gpuStrategy.AffinityGroupId))
            {
                var affinitySilo = GetAffinityGroupSilo(gpuStrategy.AffinityGroupId, target, context);
                if (affinitySilo != null)
                {
                    _logger.LogInformation(
                        "Placed grain {GrainId} on affinity silo {Silo} (group: {AffinityGroup})",
                        target.GrainIdentity,
                        affinitySilo.ToParsableString(),
                        gpuStrategy.AffinityGroupId);
                    return affinitySilo;
                }
            }

            // Get all compatible silos
            var compatibleSilos = context.GetCompatibleSilos(target).ToList();
            if (compatibleSilos.Count == 0)
            {
                throw new InvalidOperationException(
                    $"No compatible silos found for grain {target.GrainIdentity}");
            }

            // Prefer local GPU if requested
            if (gpuStrategy.PreferLocalGpu && compatibleSilos.Contains(context.LocalSilo))
            {
                var localScore = await CalculatePlacementScore(context.LocalSilo, gpuStrategy);
                if (localScore >= gpuStrategy.MinimumPlacementScore)
                {
                    _logger.LogInformation(
                        "Placed grain {GrainId} on local GPU silo {Silo} (score: {Score:F3})",
                        target.GrainIdentity,
                        context.LocalSilo.ToParsableString(),
                        localScore);

                    RegisterAffinityPlacement(gpuStrategy.AffinityGroupId, context.LocalSilo);
                    return context.LocalSilo;
                }
            }

            // Calculate scores for all silos
            var scoredSilos = new List<(SiloAddress Silo, double Score)>();
            foreach (var silo in compatibleSilos)
            {
                var score = await CalculatePlacementScore(silo, gpuStrategy);
                if (score >= gpuStrategy.MinimumPlacementScore)
                {
                    scoredSilos.Add((silo, score));
                }
            }

            if (scoredSilos.Count == 0)
            {
                _logger.LogWarning(
                    "No silos meet minimum placement score {MinScore:F2} for grain {GrainId}, using fallback",
                    gpuStrategy.MinimumPlacementScore,
                    target.GrainIdentity);
                return SelectFallbackSilo(context, target);
            }

            // Select best silo by score
            var bestSilo = scoredSilos.OrderByDescending(s => s.Score).First();

            _logger.LogInformation(
                "Placed grain {GrainId} on GPU silo {Silo} (score: {Score:F3}, {NumCandidates} candidates evaluated)",
                target.GrainIdentity,
                bestSilo.Silo.ToParsableString(),
                bestSilo.Score,
                scoredSilos.Count);

            RegisterAffinityPlacement(gpuStrategy.AffinityGroupId, bestSilo.Silo);
            return bestSilo.Silo;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex,
                "Error during GPU-native placement for grain {GrainId}, using fallback",
                target.GrainIdentity);
            return SelectFallbackSilo(context, target);
        }
    }

    /// <summary>
    /// Calculates placement score for a silo based on GPU metrics.
    /// </summary>
    /// <remarks>
    /// Score formula:
    /// <code>
    /// Score = (MemoryRatio × MemoryWeight) +
    ///         ((1 - QueueUtil) × QueueWeight) +
    ///         ((1 - ComputeUtil) × ComputeWeight)
    /// </code>
    /// </remarks>
    private async Task<double> CalculatePlacementScore(
        SiloAddress silo,
        GpuNativePlacementStrategy strategy)
    {
        try
        {
            // Get ring kernel metrics for this silo's GPU
            // Note: In production, this would query per-silo GPU metrics
            // For now, we use global ring kernel metrics as approximation
            var kernels = await _ringKernelRuntime.ListKernelsAsync();

            if (kernels.Count == 0)
            {
                // No active kernels, GPU is available
                return 1.0;
            }

            // Calculate average metrics across all kernels on this silo
            double totalQueueUtil = 0;
            double totalComputeUtil = 0;
            int validMetrics = 0;

            foreach (var kernelId in kernels)
            {
                try
                {
                    var metrics = await _ringKernelRuntime.GetMetricsAsync(kernelId);
                    totalQueueUtil += (metrics.InputQueueUtilization + metrics.OutputQueueUtilization) / 2.0;

                    // Approximate compute utilization from throughput
                    // (2M msgs/sec is theoretical max for GPU-native actors)
                    totalComputeUtil += Math.Min(1.0, metrics.ThroughputMsgsPerSec / 2_000_000.0);
                    validMetrics++;
                }
                catch
                {
                    // Skip kernels with invalid metrics
                    continue;
                }
            }

            if (validMetrics == 0)
            {
                return 1.0; // No metrics available, assume available
            }

            var avgQueueUtil = totalQueueUtil / validMetrics;
            var avgComputeUtil = totalComputeUtil / validMetrics;

            // Check hard limits
            if (avgQueueUtil > strategy.MaxQueueUtilization)
            {
                _logger.LogDebug(
                    "Silo {Silo} queue utilization {QueueUtil:F2} exceeds limit {MaxQueue:F2}",
                    silo.ToParsableString(),
                    avgQueueUtil,
                    strategy.MaxQueueUtilization);
                return 0.0; // Exceeds queue limit
            }

            if (avgComputeUtil > strategy.MaxComputeUtilization)
            {
                _logger.LogDebug(
                    "Silo {Silo} compute utilization {ComputeUtil:F2} exceeds limit {MaxCompute:F2}",
                    silo.ToParsableString(),
                    avgComputeUtil,
                    strategy.MaxComputeUtilization);
                return 0.0; // Exceeds compute limit
            }

            // Calculate weighted score
            // Note: Memory ratio would come from GPU device query in production
            var memoryRatio = 0.8; // Placeholder: 80% available

            var score = (memoryRatio * strategy.MemoryWeight) +
                       ((1.0 - avgQueueUtil) * strategy.QueueWeight) +
                       ((1.0 - avgComputeUtil) * strategy.ComputeWeight);

            _logger.LogTrace(
                "Placement score for silo {Silo}: {Score:F3} (mem={MemRatio:F2}, queue={QueueUtil:F2}, compute={CompUtil:F2})",
                silo.ToParsableString(),
                score,
                memoryRatio,
                avgQueueUtil,
                avgComputeUtil);

            return score;
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex,
                "Failed to calculate placement score for silo {Silo}, returning 0",
                silo.ToParsableString());
            return 0.0;
        }
    }

    /// <summary>
    /// Gets the preferred silo for an affinity group.
    /// </summary>
    private SiloAddress? GetAffinityGroupSilo(string affinityGroupId, PlacementTarget target, IPlacementContext context)
    {
        lock (_affinityLock)
        {
            if (_affinityGroups.TryGetValue(affinityGroupId, out var silos))
            {
                // Return first silo in affinity group that's still compatible
                var compatibleSilos = context.GetCompatibleSilos(target).ToList();
                return silos.FirstOrDefault(s => compatibleSilos.Contains(s));
            }
        }
        return null;
    }

    /// <summary>
    /// Registers a silo for an affinity group.
    /// </summary>
    private void RegisterAffinityPlacement(string? affinityGroupId, SiloAddress silo)
    {
        if (string.IsNullOrEmpty(affinityGroupId))
            return;

        lock (_affinityLock)
        {
            if (!_affinityGroups.ContainsKey(affinityGroupId))
            {
                _affinityGroups[affinityGroupId] = new List<SiloAddress>();
            }

            if (!_affinityGroups[affinityGroupId].Contains(silo))
            {
                _affinityGroups[affinityGroupId].Add(silo);
                _logger.LogDebug(
                    "Registered silo {Silo} for affinity group {AffinityGroup}",
                    silo.ToParsableString(),
                    affinityGroupId);
            }
        }
    }

    /// <summary>
    /// Selects a fallback silo when no GPU-capable silo is available.
    /// </summary>
    private SiloAddress SelectFallbackSilo(IPlacementContext context, PlacementTarget target)
    {
        var compatibleSilos = context.GetCompatibleSilos(target).OrderBy(_ => Guid.NewGuid()).ToList();
        var chosen = compatibleSilos.FirstOrDefault();

        if (chosen == null)
        {
            _logger.LogError(
                "No compatible silos found for grain {GrainId}",
                target.GrainIdentity);
            throw new InvalidOperationException(
                $"No compatible silos found for grain {target.GrainIdentity}");
        }

        _logger.LogDebug(
            "Using fallback silo {Silo} for grain {GrainId}",
            chosen.ToParsableString(),
            target.GrainIdentity);

        return chosen;
    }
}

using System;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Orleans;
using Orleans.GpuBridge.Abstractions.Capacity;
using Orleans.Runtime;
using Orleans.Runtime.Placement;

namespace Orleans.GpuBridge.Runtime;

/// <summary>
/// GPU-aware placement director that routes grains to silos with available GPU capacity
/// </summary>
public sealed class GpuPlacementDirector : IPlacementDirector
{
    private readonly ILogger<GpuPlacementDirector> _logger;
    private readonly IGrainFactory _grainFactory;

    /// <summary>
    /// Initializes a new instance of the GPU placement director
    /// </summary>
    public GpuPlacementDirector(
        ILogger<GpuPlacementDirector> logger,
        IGrainFactory grainFactory)
    {
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        _grainFactory = grainFactory ?? throw new ArgumentNullException(nameof(grainFactory));
    }

    /// <summary>
    /// Selects a silo for grain activation based on GPU capacity
    /// </summary>
    public async Task<SiloAddress> OnAddActivation(
        PlacementStrategy strategy,
        PlacementTarget target,
        IPlacementContext context)
    {
        if (strategy is not GpuPlacementStrategy gpuStrategy)
        {
            _logger.LogWarning(
                "GpuPlacementDirector called with non-GPU strategy {StrategyType} for grain {GrainIdentity}",
                strategy.GetType().Name,
                target.GrainIdentity);
            return SelectFallbackSilo(context, target);
        }

        _logger.LogDebug(
            "Selecting GPU-capable silo for grain {GrainIdentity} (MinMemory: {MinMemory}MB, PreferLocal: {PreferLocal})",
            target.GrainIdentity,
            gpuStrategy.MinimumGpuMemoryMB,
            gpuStrategy.PreferLocalPlacement);

        try
        {
            // Query the GPU capacity grain for best silo
            var capacityGrain = _grainFactory.GetGrain<IGpuCapacityGrain>(0);
            var bestSilo = await capacityGrain.GetBestSiloForPlacementAsync(
                gpuStrategy.MinimumGpuMemoryMB);

            if (bestSilo != null)
            {
                // Check if we should prefer local placement
                if (gpuStrategy.PreferLocalPlacement)
                {
                    var localSilo = context.LocalSilo;
                    var gpuSilos = await capacityGrain.GetGpuCapableSilosAsync();

                    // If local silo has GPU and meets minimum requirements, use it
                    var localGpuSilo = gpuSilos.FirstOrDefault(s => s.SiloAddress.Equals(localSilo));
                    if (localGpuSilo != null &&
                        localGpuSilo.AvailableMemoryMB >= gpuStrategy.MinimumGpuMemoryMB)
                    {
                        _logger.LogInformation(
                            "Placed grain {GrainIdentity} on local GPU-capable silo {SiloAddress} (Score: {Score:F2}, Memory: {AvailableMemory}/{TotalMemory}MB, Queue: {QueueDepth})",
                            target.GrainIdentity,
                            localSilo.ToParsableString(),
                            localGpuSilo.GetPlacementScore(),
                            localGpuSilo.AvailableMemoryMB,
                            localGpuSilo.TotalMemoryMB,
                            localGpuSilo.QueueDepth);

                        return localSilo;
                    }
                }

                // Use the best silo selected by capacity grain
                _logger.LogInformation(
                    "Placed grain {GrainIdentity} on GPU-capable silo {SiloAddress} (Score: {Score:F2}, Memory: {AvailableMemory}/{TotalMemory}MB, Queue: {QueueDepth})",
                    target.GrainIdentity,
                    bestSilo.SiloAddress.ToParsableString(),
                    bestSilo.GetPlacementScore(),
                    bestSilo.AvailableMemoryMB,
                    bestSilo.TotalMemoryMB,
                    bestSilo.QueueDepth);

                return bestSilo.SiloAddress;
            }

            // No GPU-capable silo found, fall back to any compatible silo
            _logger.LogWarning(
                "No GPU-capable silo found with at least {MinMemory}MB for grain {GrainIdentity}, falling back to CPU",
                gpuStrategy.MinimumGpuMemoryMB,
                target.GrainIdentity);

            return SelectFallbackSilo(context, target);
        }
        catch (Exception ex)
        {
            _logger.LogError(
                ex,
                "Error selecting GPU-capable silo for grain {GrainIdentity}, falling back to first compatible",
                target.GrainIdentity);

            return SelectFallbackSilo(context, target);
        }
    }

    /// <summary>
    /// Selects a fallback silo when GPU placement fails
    /// </summary>
    private SiloAddress SelectFallbackSilo(IPlacementContext context, PlacementTarget target)
    {
        var compatibleSilos = context.GetCompatibleSilos(target).OrderBy(x => Guid.NewGuid()).ToList();
        var chosen = compatibleSilos.FirstOrDefault();

        if (chosen == null)
        {
            _logger.LogError(
                "No compatible silos found for grain {GrainIdentity}",
                target.GrainIdentity);
            throw new InvalidOperationException(
                $"No compatible silos found for grain {target.GrainIdentity}");
        }

        _logger.LogDebug(
            "Placed grain {GrainIdentity} on fallback silo {SiloAddress}",
            target.GrainIdentity,
            chosen.ToParsableString());

        return chosen;
    }
}

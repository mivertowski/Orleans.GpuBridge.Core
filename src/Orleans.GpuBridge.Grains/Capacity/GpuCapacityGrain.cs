using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Orleans;
using Orleans.Concurrency;
using Orleans.GpuBridge.Abstractions.Capacity;
using Orleans.Runtime;

namespace Orleans.GpuBridge.Grains.Capacity;

/// <summary>
/// Central grain for tracking GPU capacity across the Orleans cluster
/// </summary>
/// <remarks>
/// This grain maintains a registry of GPU resources for all silos in the cluster.
/// It is used by the GpuPlacementDirector to make intelligent placement decisions.
/// The grain is reentrant to allow concurrent updates from multiple silos.
/// </remarks>
[Reentrant]
[KeepAlive]
public sealed class GpuCapacityGrain : Grain, IGpuCapacityGrain
{
    private readonly ILogger<GpuCapacityGrain> _logger;
    private readonly Dictionary<SiloAddress, GpuCapacity> _capacities = new();

    public GpuCapacityGrain(ILogger<GpuCapacityGrain> logger)
    {
        _logger = logger;
    }

    public override Task OnActivateAsync(CancellationToken cancellationToken)
    {
        _logger.LogInformation("GPU Capacity Grain activated");
        return base.OnActivateAsync(cancellationToken);
    }

    /// <inheritdoc/>
    public Task RegisterSiloAsync(SiloAddress silo, GpuCapacity capacity)
    {
        if (silo == null)
        {
            throw new ArgumentNullException(nameof(silo));
        }

        if (capacity == null)
        {
            throw new ArgumentNullException(nameof(capacity));
        }

        _capacities[silo] = capacity;

        _logger.LogInformation(
            "Registered GPU silo {Silo}: {Devices} devices, {Memory}MB total memory, Backend: {Backend}",
            silo.ToParsableString(),
            capacity.DeviceCount,
            capacity.TotalMemoryMB,
            capacity.Backend);

        return Task.CompletedTask;
    }

    /// <inheritdoc/>
    public Task UnregisterSiloAsync(SiloAddress silo)
    {
        if (silo == null)
        {
            throw new ArgumentNullException(nameof(silo));
        }

        if (_capacities.Remove(silo))
        {
            _logger.LogInformation(
                "Unregistered GPU silo {Silo}",
                silo.ToParsableString());
        }
        else
        {
            _logger.LogWarning(
                "Attempted to unregister unknown silo {Silo}",
                silo.ToParsableString());
        }

        return Task.CompletedTask;
    }

    /// <inheritdoc/>
    public Task UpdateCapacityAsync(SiloAddress silo, GpuCapacity capacity)
    {
        if (silo == null)
        {
            throw new ArgumentNullException(nameof(silo));
        }

        if (capacity == null)
        {
            throw new ArgumentNullException(nameof(capacity));
        }

        if (_capacities.ContainsKey(silo))
        {
            _capacities[silo] = capacity;

            _logger.LogDebug(
                "Updated capacity for silo {Silo}: {AvailableMemory}MB available, Queue depth: {QueueDepth}",
                silo.ToParsableString(),
                capacity.AvailableMemoryMB,
                capacity.QueueDepth);
        }
        else
        {
            _logger.LogWarning(
                "Attempted to update capacity for unregistered silo {Silo}",
                silo.ToParsableString());

            // Auto-register if update comes from unknown silo
            _capacities[silo] = capacity;

            _logger.LogInformation(
                "Auto-registered silo {Silo} during capacity update",
                silo.ToParsableString());
        }

        return Task.CompletedTask;
    }

    /// <inheritdoc/>
    public Task<List<SiloGpuCapacity>> GetGpuCapableSilosAsync()
    {
        // Remove stale entries (older than 5 minutes)
        var staleThreshold = DateTime.UtcNow - TimeSpan.FromMinutes(5);
        var staleSilos = _capacities
            .Where(kvp => kvp.Value.LastUpdated < staleThreshold)
            .Select(kvp => kvp.Key)
            .ToList();

        foreach (var staleSilo in staleSilos)
        {
            _capacities.Remove(staleSilo);
            _logger.LogWarning(
                "Removed stale GPU capacity entry for silo {Silo}",
                staleSilo.ToParsableString());
        }

        // Return all GPU-capable silos
        var result = _capacities
            .Where(kvp => kvp.Value.HasCapacity)
            .Select(kvp => new SiloGpuCapacity(kvp.Key, kvp.Value))
            .OrderByDescending(s => s.GetPlacementScore())
            .ToList();

        _logger.LogDebug(
            "Retrieved {Count} GPU-capable silos",
            result.Count);

        return Task.FromResult(result);
    }

    /// <inheritdoc/>
    public Task<GpuCapacity?> GetSiloCapacityAsync(SiloAddress silo)
    {
        if (silo == null)
        {
            throw new ArgumentNullException(nameof(silo));
        }

        if (_capacities.TryGetValue(silo, out var capacity))
        {
            // Check if stale
            if (capacity.IsStale)
            {
                _logger.LogWarning(
                    "Capacity data for silo {Silo} is stale (last updated: {LastUpdated})",
                    silo.ToParsableString(),
                    capacity.LastUpdated);
            }

            return Task.FromResult<GpuCapacity?>(capacity);
        }

        _logger.LogDebug(
            "No capacity information found for silo {Silo}",
            silo.ToParsableString());

        return Task.FromResult<GpuCapacity?>(null);
    }

    /// <inheritdoc/>
    public Task<SiloGpuCapacity?> GetBestSiloForPlacementAsync(int minimumMemoryMB = 0)
    {
        var gpuSilos = _capacities
            .Where(kvp => kvp.Value.HasCapacity)
            .Where(kvp => !kvp.Value.IsStale)
            .Where(kvp => kvp.Value.AvailableMemoryMB >= minimumMemoryMB)
            .Select(kvp => new SiloGpuCapacity(kvp.Key, kvp.Value))
            .ToList();

        if (gpuSilos.Count == 0)
        {
            _logger.LogWarning(
                "No GPU-capable silos found with at least {MinMemory}MB available",
                minimumMemoryMB);

            return Task.FromResult<SiloGpuCapacity?>(null);
        }

        // Select silo with best placement score
        var bestSilo = gpuSilos
            .OrderByDescending(s => s.GetPlacementScore())
            .ThenBy(s => s.QueueDepth)
            .First();

        _logger.LogDebug(
            "Selected best silo {Silo} for placement (score: {Score:F2}, available: {AvailableMemory}MB, queue: {QueueDepth})",
            bestSilo.SiloAddress.ToParsableString(),
            bestSilo.GetPlacementScore(),
            bestSilo.AvailableMemoryMB,
            bestSilo.QueueDepth);

        return Task.FromResult<SiloGpuCapacity?>(bestSilo);
    }

    /// <inheritdoc/>
    public Task<ClusterGpuStats> GetClusterStatsAsync()
    {
        var validCapacities = _capacities.Values
            .Where(c => !c.IsStale)
            .ToList();

        var stats = new ClusterGpuStats(
            TotalSilos: _capacities.Count,
            GpuCapableSilos: validCapacities.Count(c => c.HasCapacity),
            TotalDevices: validCapacities.Sum(c => c.DeviceCount),
            TotalMemoryMB: validCapacities.Sum(c => c.TotalMemoryMB),
            AvailableMemoryMB: validCapacities.Sum(c => c.AvailableMemoryMB),
            TotalQueueDepth: validCapacities.Sum(c => c.QueueDepth));

        _logger.LogDebug(
            "Cluster GPU stats: {GpuSilos}/{TotalSilos} silos, {Devices} devices, " +
            "{AvailableMemory}/{TotalMemory}MB memory ({UsagePercent:F1}% used)",
            stats.GpuCapableSilos,
            stats.TotalSilos,
            stats.TotalDevices,
            stats.AvailableMemoryMB,
            stats.TotalMemoryMB,
            stats.MemoryUsagePercent);

        return Task.FromResult(stats);
    }
}

using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Orleans;
using Orleans.Configuration;
using Orleans.GpuBridge.Runtime;
using Orleans.Hosting;
using Orleans.Placement;
using Orleans.Runtime;
using Orleans.Runtime.Placement;

namespace Orleans.GpuBridge.Grains;

/// <summary>
/// Custom placement strategy for GPU grains
/// </summary>
[Serializable]
[GenerateSerializer]
public sealed class GpuPlacementStrategy : PlacementStrategy
{
    [Id(0)]
    public bool PreferLocalGpu { get; set; }
    
    [Id(1)]
    public int? PreferredDeviceIndex { get; set; }
}

/// <summary>
/// Attribute to mark grains for GPU placement
/// </summary>
[AttributeUsage(AttributeTargets.Class, AllowMultiple = false)]
public sealed class GpuPlacementAttribute : PlacementAttribute
{
    public bool PreferLocalGpu { get; set; } = true;
    public int PreferredDeviceIndex { get; set; } = -1;
    
    public GpuPlacementAttribute() : base(
        new GpuPlacementStrategy 
        { 
            PreferLocalGpu = true,
            PreferredDeviceIndex = null
        })
    {
    }
}

/// <summary>
/// Director that makes GPU-aware placement decisions
/// </summary>
public sealed class GpuPlacementDirector : IPlacementDirector
{
    private readonly ILogger<GpuPlacementDirector> _logger;
    
    public GpuPlacementDirector(ILogger<GpuPlacementDirector> logger)
    {
        _logger = logger;
    }
    
    public async Task<SiloAddress> OnAddActivation(
        PlacementStrategy strategy,
        PlacementTarget target,
        IPlacementContext context)
    {
        var gpuStrategy = strategy as GpuPlacementStrategy;
        if (gpuStrategy == null)
        {
            throw new ArgumentException("Expected GpuPlacementStrategy", nameof(strategy));
        }
        
        var silos = context.GetCompatibleSilos(target).ToList();
        if (silos.Count == 0)
        {
            throw new OrleansException("No compatible silos found for GPU placement");
        }
        
        // Get GPU capabilities from silos
        var siloCapabilities = new Dictionary<SiloAddress, GpuCapabilities>();
        
        foreach (var silo in silos)
        {
            // Query silo capabilities - would need silo metadata in real implementation
            var capabilities = await GetSiloGpuCapabilitiesAsync(silo, context);
            if (capabilities != null)
            {
                siloCapabilities[silo] = capabilities;
            }
        }
        
        // Select best silo based on GPU availability
        var selectedSilo = SelectBestSilo(
            siloCapabilities,
            gpuStrategy,
            target);
        
        if (selectedSilo == null)
        {
            // Fallback to random selection if no GPU-capable silo found
            selectedSilo = silos[Random.Shared.Next(silos.Count)];
            _logger.LogWarning(
                "No GPU-capable silo found for {Grain}, using fallback placement",
                target.GrainIdentity);
        }
        else
        {
            _logger.LogDebug(
                "Placing {Grain} on silo {Silo} with GPU capabilities",
                target.GrainIdentity, selectedSilo);
        }
        
        return selectedSilo;
    }
    
    private async Task<GpuCapabilities?> GetSiloGpuCapabilitiesAsync(
        SiloAddress silo,
        IPlacementContext context)
    {
        // In a real implementation, this would query silo metadata
        // For now, simulate that all silos have basic GPU capabilities
        await Task.CompletedTask;
        
        return new GpuCapabilities
        {
            DeviceCount = 1,
            TotalMemoryBytes = 8L * 1024 * 1024 * 1024, // 8GB
            AvailableMemoryBytes = 6L * 1024 * 1024 * 1024, // 6GB
            CurrentLoad = Random.Shared.NextDouble() * 0.5 // 0-50% load
        };
    }
    
    private SiloAddress? SelectBestSilo(
        Dictionary<SiloAddress, GpuCapabilities> siloCapabilities,
        GpuPlacementStrategy strategy,
        PlacementTarget target)
    {
        if (siloCapabilities.Count == 0)
        {
            return null;
        }
        
        // Score each silo based on GPU capabilities
        var scores = new Dictionary<SiloAddress, double>();
        
        foreach (var (silo, capabilities) in siloCapabilities)
        {
            double score = 0;
            
            // Factor in device count
            score += capabilities.DeviceCount * 100;
            
            // Factor in available memory (GB)
            score += (capabilities.AvailableMemoryBytes / (1024.0 * 1024 * 1024)) * 10;
            
            // Factor in current load (prefer less loaded)
            score += (1.0 - capabilities.CurrentLoad) * 50;
            
            // Bonus for preferred device index match
            if (strategy.PreferredDeviceIndex.HasValue &&
                capabilities.DeviceCount > strategy.PreferredDeviceIndex.Value)
            {
                score += 200;
            }
            
            scores[silo] = score;
        }
        
        // Select silo with highest score
        var bestSilo = scores
            .OrderByDescending(kvp => kvp.Value)
            .Select(kvp => kvp.Key)
            .FirstOrDefault();
        
        return bestSilo;
    }
    
    /// <summary>
    /// GPU capabilities for a silo
    /// </summary>
    private sealed class GpuCapabilities
    {
        public int DeviceCount { get; set; }
        public long TotalMemoryBytes { get; set; }
        public long AvailableMemoryBytes { get; set; }
        public double CurrentLoad { get; set; }
    }
}

/// <summary>
/// Extension methods for configuring GPU placement
/// </summary>
public static class GpuPlacementExtensions
{
    /// <summary>
    /// Configures GPU placement strategy in the silo
    /// </summary>
    public static ISiloBuilder UseGpuPlacement(this ISiloBuilder builder)
    {
        return builder.ConfigureServices(services =>
        {
            services.AddSingleton<IPlacementDirector, GpuPlacementDirector>();
            services.AddSingleton<PlacementStrategy, GpuPlacementStrategy>();
        });
    }
    
    /// <summary>
    /// Configures GPU placement strategy in the client
    /// </summary>
    public static IClientBuilder UseGpuPlacement(this IClientBuilder builder)
    {
        return builder.ConfigureServices(services =>
        {
            services.AddSingleton<PlacementStrategy, GpuPlacementStrategy>();
        });
    }
}
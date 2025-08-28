using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.DependencyInjection.Extensions;
using Orleans.GpuBridge.Abstractions;
using Orleans.GpuBridge.Abstractions.Memory;
using Orleans.GpuBridge.Abstractions.Providers;
using Orleans.GpuBridge.Runtime.Builders;
using Orleans.GpuBridge.Runtime.Providers;

namespace Orleans.GpuBridge.Runtime.Extensions;

/// <summary>
/// Extension methods for configuring GPU Bridge services
/// </summary>
public static class ServiceCollectionExtensions
{
    /// <summary>
    /// Adds GPU Bridge services to the service collection
    /// </summary>
    public static IGpuBridgeBuilder AddGpuBridge(
        this IServiceCollection services, 
        Action<GpuBridgeOptions>? configure = null)
    {
        // Configure options
        if (configure != null)
        {
            services.Configure(configure);
        }
        else
        {
            services.Configure<GpuBridgeOptions>(_ => { });
        }
        
        // Backend provider system
        services.TryAddSingleton<IGpuBackendRegistry, GpuBackendRegistry>();
        services.TryAddSingleton<GpuBridgeProviderSelector>();
        
        // Core services
        services.TryAddSingleton<IGpuBridge, GpuBridge>();
        services.TryAddSingleton<KernelCatalog>();
        services.TryAddSingleton<DeviceBroker>();
        services.TryAddSingleton<PersistentKernelHost>();
        services.TryAddSingleton<GpuDiagnostics>();
        
        // Hosted service
        services.AddHostedService<GpuHostFeature>();
        
        // Memory management (placeholder for now)
        services.TryAddSingleton(typeof(IGpuMemoryPool<>), typeof(CpuMemoryPool<>));
        
        return new GpuBridgeBuilder(services);
    }
}
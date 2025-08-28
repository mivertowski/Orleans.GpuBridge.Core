using Microsoft.Extensions.DependencyInjection;
using Orleans.GpuBridge.Abstractions.Enums;
using Orleans.GpuBridge.Abstractions.Providers;
using Orleans.GpuBridge.Runtime.Providers;
using Orleans.GpuBridge.Runtime.Builders;
using Orleans.GpuBridge.Runtime.Configuration;

namespace Orleans.GpuBridge.Runtime.Extensions;

/// <summary>
/// Extension methods for adding specific backend providers
/// </summary>
public static class BackendProviderExtensions
{
    /// <summary>
    /// Adds the CPU fallback backend provider
    /// </summary>
    public static IGpuBridgeBuilder AddCpuFallbackBackend(this IGpuBridgeBuilder builder)
    {
        return builder.AddBackendProvider<CpuFallbackProvider>();
    }

    /// <summary>
    /// Adds the ILGPU backend provider (requires Orleans.GpuBridge.Backends.ILGPU package)
    /// </summary>
    public static IGpuBridgeBuilder AddILGPUBackend(this IGpuBridgeBuilder builder)
    {
        // This would be in the ILGPU package assembly
        var ilgpuProviderType = TryGetILGPUProviderType();
        if (ilgpuProviderType != null)
        {
            builder.Services.AddSingleton(typeof(IGpuBackendProvider), ilgpuProviderType);
            return builder;
        }
        
        throw new InvalidOperationException(
            "ILGPU backend provider not found. Ensure Orleans.GpuBridge.Backends.ILGPU package is installed.");
    }

    /// <summary>
    /// Adds the DotCompute backend provider (requires Orleans.GpuBridge.Backends.DotCompute package)
    /// </summary>
    public static IGpuBridgeBuilder AddDotComputeBackend(this IGpuBridgeBuilder builder)
    {
        // This would be in the DotCompute package assembly
        var dotComputeProviderType = TryGetDotComputeProviderType();
        if (dotComputeProviderType != null)
        {
            builder.Services.AddSingleton(typeof(IGpuBackendProvider), dotComputeProviderType);
            return builder;
        }
        
        throw new InvalidOperationException(
            "DotCompute backend provider not found. Ensure Orleans.GpuBridge.Backends.DotCompute package is installed.");
    }

    /// <summary>
    /// Adds all available backend providers by scanning loaded assemblies
    /// </summary>
    public static IGpuBridgeBuilder AddAllAvailableBackends(this IGpuBridgeBuilder builder)
    {
        var providerTypes = ScanForBackendProviders();
        
        foreach (var providerType in providerTypes)
        {
            builder.Services.AddSingleton(typeof(IGpuBackendProvider), providerType);
        }
        
        return builder;
    }

    /// <summary>
    /// Configures backend selection preferences
    /// </summary>
    public static IGpuBridgeBuilder ConfigureBackendSelection(
        this IGpuBridgeBuilder builder,
        Action<ProviderSelectionOptions> configure)
    {
        builder.Services.Configure(configure);
        return builder;
    }

    private static Type? TryGetILGPUProviderType()
    {
        try
        {
            // Try to find ILGPU provider in loaded assemblies
            var assemblies = System.AppDomain.CurrentDomain.GetAssemblies();
            foreach (var assembly in assemblies)
            {
                var providerType = assembly.GetTypes()
                    .FirstOrDefault(t => 
                        t.Name == "ILGPUBackendProvider" && 
                        t.IsAssignableTo(typeof(IGpuBackendProvider)));
                        
                if (providerType != null)
                    return providerType;
            }
            
            return null;
        }
        catch
        {
            return null;
        }
    }

    private static Type? TryGetDotComputeProviderType()
    {
        try
        {
            // Try to find DotCompute provider in loaded assemblies
            var assemblies = System.AppDomain.CurrentDomain.GetAssemblies();
            foreach (var assembly in assemblies)
            {
                var providerType = assembly.GetTypes()
                    .FirstOrDefault(t => 
                        t.Name == "DotComputeBackendProvider" && 
                        t.IsAssignableTo(typeof(IGpuBackendProvider)));
                        
                if (providerType != null)
                    return providerType;
            }
            
            return null;
        }
        catch
        {
            return null;
        }
    }

    private static IEnumerable<Type> ScanForBackendProviders()
    {
        var providerTypes = new List<Type>();
        
        try
        {
            var assemblies = System.AppDomain.CurrentDomain.GetAssemblies();
            foreach (var assembly in assemblies)
            {
                try
                {
                    var types = assembly.GetTypes()
                        .Where(t => 
                            t.IsClass && 
                            !t.IsAbstract && 
                            t.IsAssignableTo(typeof(IGpuBackendProvider)))
                        .ToList();
                    
                    providerTypes.AddRange(types);
                }
                catch
                {
                    // Skip assemblies we can't scan
                    continue;
                }
            }
        }
        catch
        {
            // If assembly scanning fails, just return CPU fallback
            providerTypes.Add(typeof(CpuFallbackProvider));
        }
        
        return providerTypes;
    }
}
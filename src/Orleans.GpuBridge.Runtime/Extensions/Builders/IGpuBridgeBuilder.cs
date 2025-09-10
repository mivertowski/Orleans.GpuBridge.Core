using System.Diagnostics.CodeAnalysis;
using Microsoft.Extensions.DependencyInjection;
using Orleans.GpuBridge.Abstractions;
using Orleans.GpuBridge.Abstractions.Providers;

namespace Orleans.GpuBridge.Runtime.Builders;

/// <summary>
/// Builder for configuring GPU Bridge
/// </summary>
public interface IGpuBridgeBuilder
{
    /// <summary>
    /// Adds a kernel to the catalog
    /// </summary>
    IGpuBridgeBuilder AddKernel(Action<KernelDescriptorBuilder> configure);
    
    /// <summary>
    /// Adds a kernel type to the catalog
    /// </summary>
    IGpuBridgeBuilder AddKernel<[DynamicallyAccessedMembers(DynamicallyAccessedMemberTypes.PublicConstructors)] TKernel>() where TKernel : class;
    
    /// <summary>
    /// Configures GPU Bridge options
    /// </summary>
    IGpuBridgeBuilder ConfigureOptions(Action<GpuBridgeOptions> configure);
    
    /// <summary>
    /// Adds a backend provider to the registry
    /// </summary>
    IGpuBridgeBuilder AddBackendProvider<[DynamicallyAccessedMembers(DynamicallyAccessedMemberTypes.PublicConstructors)] TProvider>() where TProvider : class, IGpuBackendProvider;
    
    /// <summary>
    /// Adds a backend provider instance to the registry
    /// </summary>
    IGpuBridgeBuilder AddBackendProvider<TProvider>(TProvider provider) where TProvider : class, IGpuBackendProvider;
    
    /// <summary>
    /// Adds a backend provider factory to the registry
    /// </summary>
    IGpuBridgeBuilder AddBackendProvider<TProvider>(Func<IServiceProvider, TProvider> factory) where TProvider : class, IGpuBackendProvider;
    
    /// <summary>
    /// Returns the service collection
    /// </summary>
    IServiceCollection Services { get; }
}
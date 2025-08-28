using System.Diagnostics.CodeAnalysis;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Options;
using Orleans.GpuBridge.Abstractions;
using Orleans.GpuBridge.Abstractions.Providers;

namespace Orleans.GpuBridge.Runtime.Builders;

/// <summary>
/// Implementation of GPU Bridge builder
/// </summary>
internal class GpuBridgeBuilder : IGpuBridgeBuilder
{
    public IServiceCollection Services { get; }
    
    public GpuBridgeBuilder(IServiceCollection services)
    {
        Services = services;
    }
    
    public IGpuBridgeBuilder AddKernel(Action<KernelDescriptorBuilder> configure)
    {
        var builder = new KernelDescriptorBuilder();
        configure(builder);
        var descriptor = builder.Build();
        
        Services.PostConfigure<KernelCatalogOptions>(options =>
            options.Descriptors.Add(descriptor));
        
        return this;
    }
    
    public IGpuBridgeBuilder AddKernel<[DynamicallyAccessedMembers(DynamicallyAccessedMemberTypes.PublicConstructors)] TKernel>() where TKernel : class
    {
        Services.AddTransient<TKernel>();
        // TODO: Auto-registration logic based on attributes
        return this;
    }
    
    public IGpuBridgeBuilder ConfigureOptions(Action<GpuBridgeOptions> configure)
    {
        Services.Configure(configure);
        return this;
    }
    
    public IGpuBridgeBuilder AddBackendProvider<TProvider>() where TProvider : class, IGpuBackendProvider
    {
        Services.AddSingleton<TProvider>();
        Services.AddSingleton<IGpuBackendProvider>(sp => sp.GetRequiredService<TProvider>());
        return this;
    }
    
    public IGpuBridgeBuilder AddBackendProvider<TProvider>(TProvider provider) where TProvider : class, IGpuBackendProvider
    {
        Services.AddSingleton<IGpuBackendProvider>(provider);
        return this;
    }
    
    public IGpuBridgeBuilder AddBackendProvider<TProvider>(Func<IServiceProvider, TProvider> factory) where TProvider : class, IGpuBackendProvider
    {
        Services.AddSingleton<IGpuBackendProvider>(factory);
        return this;
    }
}
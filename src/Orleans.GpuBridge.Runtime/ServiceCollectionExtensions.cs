using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.DependencyInjection.Extensions;
using Microsoft.Extensions.Options;
using Orleans.GpuBridge.Abstractions;

namespace Orleans.GpuBridge.Runtime;

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
    IGpuBridgeBuilder AddKernel<TKernel>() where TKernel : class;
    
    /// <summary>
    /// Configures GPU Bridge options
    /// </summary>
    IGpuBridgeBuilder ConfigureOptions(Action<GpuBridgeOptions> configure);
    
    /// <summary>
    /// Returns the service collection
    /// </summary>
    IServiceCollection Services { get; }
}

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
    
    public IGpuBridgeBuilder AddKernel<TKernel>() where TKernel : class
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
}

/// <summary>
/// Builder for kernel descriptors
/// </summary>
public class KernelDescriptorBuilder
{
    private readonly KernelDescriptor _descriptor = new();
    
    public KernelDescriptorBuilder Id(string id)
    {
        _descriptor.Id = new KernelId(id);
        return this;
    }
    
    public KernelDescriptorBuilder Input<TIn>() where TIn : notnull
    {
        _descriptor.InType = typeof(TIn);
        return this;
    }
    
    public KernelDescriptorBuilder Output<TOut>() where TOut : notnull
    {
        _descriptor.OutType = typeof(TOut);
        return this;
    }
    
    public KernelDescriptorBuilder WithFactory<TKernel>(Func<IServiceProvider, TKernel> factory)
        where TKernel : class
    {
        _descriptor.Factory = sp => factory(sp);
        return this;
    }
    
    public KernelDescriptorBuilder WithBatchSize(int size)
    {
        // TODO: Store batch size in descriptor
        return this;
    }
    
    public KernelDescriptor Build() => _descriptor;
}
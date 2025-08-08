using Microsoft.Extensions.DependencyInjection; using Microsoft.Extensions.DependencyInjection.Extensions;
using Orleans.GpuBridge.Abstractions;
namespace Orleans.GpuBridge.Runtime;
public static class ServiceCollectionExtensions
{
    public static IServiceCollection AddGpuBridge(this IServiceCollection s, System.Action<GpuBridgeOptions>? cfg=null)
    {
        if(cfg!=null) s.Configure(cfg);
        s.TryAddSingleton<DeviceBroker>(); s.TryAddSingleton<PersistentKernelHost>(); s.TryAddSingleton<GpuDiagnostics>(); s.TryAddSingleton<KernelCatalog>();
        s.AddHostedService<GpuHostFeature>();
        return s;
    }
    public static IServiceCollection AddKernel(this IServiceCollection s, System.Action<KernelDescriptor> b)
    { s.PostConfigure<KernelCatalogOptions>(o=>o.Descriptors.Add(KernelDescriptor.Build(b))); return s; }
}

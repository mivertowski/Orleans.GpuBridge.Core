using System.Reflection;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using Orleans.GpuBridge.Abstractions;
using Orleans.GpuBridge.Abstractions.Enums;

namespace Orleans.GpuBridge.Runtime;

/// <summary>
/// Main implementation of the GPU bridge
/// </summary>
public sealed class GpuBridge : IGpuBridge
{
    private readonly ILogger<GpuBridge> _logger;
    private readonly KernelCatalog _kernelCatalog;
    private readonly DeviceBroker _deviceBroker;
    private readonly GpuBridgeOptions _options;
    private readonly IServiceProvider _serviceProvider;
    
    public GpuBridge(
        ILogger<GpuBridge> logger,
        KernelCatalog kernelCatalog,
        DeviceBroker deviceBroker,
        IOptions<GpuBridgeOptions> options,
        IServiceProvider serviceProvider)
    {
        _logger = logger;
        _kernelCatalog = kernelCatalog;
        _deviceBroker = deviceBroker;
        _options = options.Value;
        _serviceProvider = serviceProvider;
    }
    
    public ValueTask<GpuBridgeInfo> GetInfoAsync(CancellationToken ct = default)
    {
        var assembly = Assembly.GetExecutingAssembly();
        var version = assembly.GetName().Version?.ToString() ?? "1.0.0";
        
        var info = new GpuBridgeInfo(
            Version: version,
            DeviceCount: _deviceBroker.DeviceCount,
            TotalMemoryBytes: _deviceBroker.TotalMemoryBytes,
            Backend: _options.PreferGpu ? GpuBackend.CUDA : GpuBackend.CPU,
            IsGpuAvailable: _deviceBroker.DeviceCount > 0,
            Metadata: new Dictionary<string, object>
            {
                ["MaxConcurrentKernels"] = _options.MaxConcurrentKernels,
                ["MemoryPoolSizeMB"] = _options.MemoryPoolSizeMB,
                ["EnableProfiling"] = _options.EnableProfiling
            });
        
        return new ValueTask<GpuBridgeInfo>(info);
    }
    
    public async ValueTask<IGpuKernel<TIn, TOut>> GetKernelAsync<TIn, TOut>(
        KernelId kernelId,
        CancellationToken ct = default)
        where TIn : notnull
        where TOut : notnull
    {
        _logger.LogDebug("Getting kernel {KernelId}", kernelId);
        
        var kernel = await _kernelCatalog.ResolveAsync<TIn, TOut>(kernelId, _serviceProvider);
        
        if (kernel == null)
        {
            _logger.LogWarning("Kernel {KernelId} not found, using CPU passthrough", kernelId);
            kernel = new CpuPassthroughKernel<TIn, TOut>();
        }
        
        return kernel;
    }
    
    public ValueTask<IReadOnlyList<GpuDevice>> GetDevicesAsync(CancellationToken ct = default)
    {
        var devices = _deviceBroker.GetDevices();
        return new ValueTask<IReadOnlyList<GpuDevice>>(devices);
    }
}
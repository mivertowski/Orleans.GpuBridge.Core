using System.Threading; using System.Threading.Tasks; using Microsoft.Extensions.Logging;
namespace Orleans.GpuBridge.Runtime;
public sealed class PersistentKernelHost{ private readonly Microsoft.Extensions.Logging.ILogger<PersistentKernelHost> _log;
    public PersistentKernelHost(Microsoft.Extensions.Logging.ILogger<PersistentKernelHost> log)=>_log=log;
    public Task StartAsync(CancellationToken ct){_log.LogInformation("Start persistent kernels (stub)"); return Task.CompletedTask;}
    public Task StopAsync(CancellationToken ct){_log.LogInformation("Stop persistent kernels (stub)"); return Task.CompletedTask;}
}

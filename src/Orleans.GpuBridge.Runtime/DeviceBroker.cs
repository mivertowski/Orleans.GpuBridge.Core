using System.Threading; using System.Threading.Tasks; using Microsoft.Extensions.Logging;
namespace Orleans.GpuBridge.Runtime;
public sealed class DeviceBroker{ private readonly ILogger<DeviceBroker> _log; public DeviceBroker(ILogger<DeviceBroker> log)=>_log=log;
    public Task InitializeAsync(CancellationToken ct){_log.LogInformation("DeviceBroker init (stub)"); return Task.CompletedTask;}
    public Task ShutdownAsync(CancellationToken ct){_log.LogInformation("DeviceBroker shutdown (stub)"); return Task.CompletedTask;}
    public int CurrentQueueDepth=>0;
}

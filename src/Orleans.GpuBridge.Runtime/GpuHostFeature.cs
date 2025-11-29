using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using System.Threading;
using System.Threading.Tasks;
namespace Orleans.GpuBridge.Runtime;
public sealed class GpuHostFeature : IHostedService
{
    private readonly ILogger<GpuHostFeature> _log; private readonly DeviceBroker _broker; private readonly PersistentKernelHost _kernels;
    public GpuHostFeature(ILogger<GpuHostFeature> log, DeviceBroker broker, PersistentKernelHost kernels) { _log = log; _broker = broker; _kernels = kernels; }
    public async Task StartAsync(CancellationToken ct) { _log.LogInformation("GpuHostFeature starting"); await _broker.InitializeAsync(ct); await _kernels.StartAsync(ct); }
    public async Task StopAsync(CancellationToken ct) { _log.LogInformation("GpuHostFeature stopping"); await _kernels.StopAsync(ct); await _broker.ShutdownAsync(ct); }
}

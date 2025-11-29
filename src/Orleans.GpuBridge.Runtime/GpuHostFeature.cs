using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using System.Threading;
using System.Threading.Tasks;

namespace Orleans.GpuBridge.Runtime;

/// <summary>
/// Orleans host feature that integrates GPU bridge lifecycle with the Orleans runtime
/// </summary>
public sealed class GpuHostFeature : IHostedService
{
    private readonly ILogger<GpuHostFeature> _log;
    private readonly DeviceBroker _broker;
    private readonly PersistentKernelHost _kernels;

    /// <summary>
    /// Initializes a new instance of the <see cref="GpuHostFeature"/> class
    /// </summary>
    /// <param name="log">Logger instance</param>
    /// <param name="broker">Device broker for GPU management</param>
    /// <param name="kernels">Persistent kernel host</param>
    public GpuHostFeature(ILogger<GpuHostFeature> log, DeviceBroker broker, PersistentKernelHost kernels)
    {
        _log = log;
        _broker = broker;
        _kernels = kernels;
    }

    /// <summary>
    /// Starts the GPU host feature and initializes GPU resources
    /// </summary>
    /// <param name="ct">Cancellation token</param>
    public async Task StartAsync(CancellationToken ct)
    {
        _log.LogInformation("GpuHostFeature starting");
        await _broker.InitializeAsync(ct);
        await _kernels.StartAsync(ct);
    }

    /// <summary>
    /// Stops the GPU host feature and releases GPU resources
    /// </summary>
    /// <param name="ct">Cancellation token</param>
    public async Task StopAsync(CancellationToken ct)
    {
        _log.LogInformation("GpuHostFeature stopping");
        await _kernels.StopAsync(ct);
        await _broker.ShutdownAsync(ct);
    }
}

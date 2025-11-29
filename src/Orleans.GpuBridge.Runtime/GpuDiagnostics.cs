using Microsoft.Extensions.Logging;
namespace Orleans.GpuBridge.Runtime;
public sealed class GpuDiagnostics
{
    private readonly ILogger<GpuDiagnostics> _log; public GpuDiagnostics(ILogger<GpuDiagnostics> log) => _log = log;
    public void Submit(string kernelId, int count) => _log.LogDebug("Submit {kernel} x{count}", kernelId, count);
}

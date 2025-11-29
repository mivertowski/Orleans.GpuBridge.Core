using Microsoft.Extensions.Logging;

namespace Orleans.GpuBridge.Runtime;

/// <summary>
/// GPU diagnostics and telemetry service
/// </summary>
public sealed class GpuDiagnostics
{
    private readonly ILogger<GpuDiagnostics> _log;

    /// <summary>
    /// Initializes a new instance of the <see cref="GpuDiagnostics"/> class
    /// </summary>
    /// <param name="log">Logger instance</param>
    public GpuDiagnostics(ILogger<GpuDiagnostics> log) => _log = log;

    /// <summary>
    /// Submits a diagnostic event for kernel execution
    /// </summary>
    /// <param name="kernelId">Kernel identifier</param>
    /// <param name="count">Number of executions</param>
    public void Submit(string kernelId, int count) => _log.LogDebug("Submit {kernel} x{count}", kernelId, count);
}

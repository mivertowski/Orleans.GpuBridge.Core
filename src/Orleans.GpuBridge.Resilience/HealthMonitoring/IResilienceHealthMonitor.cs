using System.Threading;
using System.Threading.Tasks;

namespace Orleans.GpuBridge.Resilience;

/// <summary>
/// Health monitor for resilience patterns
/// </summary>
public interface IResilienceHealthMonitor
{
    /// <summary>
    /// Gets overall health status
    /// </summary>
    Task<HealthStatus> GetHealthStatusAsync(CancellationToken cancellationToken = default);

    /// <summary>
    /// Gets detailed health report
    /// </summary>
    Task<HealthReport> GetHealthReportAsync(CancellationToken cancellationToken = default);
}

using System;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Orleans.GpuBridge.Resilience.Policies;
using Orleans.GpuBridge.Resilience.Telemetry;

namespace Orleans.GpuBridge.Resilience;

/// <summary>
/// Implementation of resilience health monitor
/// </summary>
internal sealed class ResilienceHealthMonitor : IResilienceHealthMonitor
{
    private readonly ResilienceTelemetryCollector _telemetryCollector;
    private readonly GpuResiliencePolicy _resiliencePolicy;
    private readonly ILogger<ResilienceHealthMonitor> _logger;

    public ResilienceHealthMonitor(
        ResilienceTelemetryCollector telemetryCollector,
        GpuResiliencePolicy resiliencePolicy,
        ILogger<ResilienceHealthMonitor> logger)
    {
        _telemetryCollector = telemetryCollector ?? throw new ArgumentNullException(nameof(telemetryCollector));
        _resiliencePolicy = resiliencePolicy ?? throw new ArgumentNullException(nameof(resiliencePolicy));
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
    }

    public async Task<HealthStatus> GetHealthStatusAsync(CancellationToken cancellationToken = default)
    {
        try
        {
            var metrics = _telemetryCollector.GetMetrics();
            var bulkheadMetrics = _resiliencePolicy.GetBulkheadMetrics();

            // Determine overall health based on various factors
            if (bulkheadMetrics.UtilizationPercentage > 90)
                return HealthStatus.Unhealthy;

            var unhealthyComponents = 0;
            var degradedComponents = 0;

            foreach (var health in metrics.ComponentHealth.Values)
            {
                switch (health)
                {
                    case ComponentHealth.Unhealthy:
                        unhealthyComponents++;
                        break;
                    case ComponentHealth.Degraded:
                        degradedComponents++;
                        break;
                }
            }

            if (unhealthyComponents > 0)
                return HealthStatus.Unhealthy;

            if (degradedComponents > 0 || bulkheadMetrics.UtilizationPercentage > 70)
                return HealthStatus.Degraded;

            return HealthStatus.Healthy;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error determining health status");
            return HealthStatus.Unhealthy;
        }
    }

    public async Task<HealthReport> GetHealthReportAsync(CancellationToken cancellationToken = default)
    {
        var metrics = _telemetryCollector.GetMetrics();
        var bulkheadMetrics = _resiliencePolicy.GetBulkheadMetrics();
        var overallStatus = await GetHealthStatusAsync(cancellationToken).ConfigureAwait(false);

        return new HealthReport(
            OverallStatus: overallStatus,
            ComponentHealth: metrics.ComponentHealth,
            BulkheadUtilization: bulkheadMetrics.UtilizationPercentage,
            CircuitBreakerStates: metrics.CircuitBreakerStates,
            FallbackLevels: metrics.FallbackLevels,
            RecentEvents: metrics.RecentHealthEvents,
            Timestamp: DateTimeOffset.UtcNow);
    }
}

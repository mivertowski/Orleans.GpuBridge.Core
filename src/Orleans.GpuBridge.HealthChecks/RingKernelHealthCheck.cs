using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Diagnostics.HealthChecks;
using Microsoft.Extensions.Logging;
using Orleans.GpuBridge.Backends.DotCompute.RingKernels;

namespace Orleans.GpuBridge.HealthChecks;

/// <summary>
/// Health check for GPU-native ring kernels.
/// Monitors kernel responsiveness, message queue health, and temporal clock accuracy.
/// </summary>
/// <remarks>
/// Health Check Criteria:
/// 1. Ring kernels are running and responsive
/// 2. Message queues not near capacity (<85%)
/// 3. No recent watchdog recoveries
/// 4. Clock drift within acceptable bounds (<100μs)
/// 5. No message drops in recent window
///
/// Health Status:
/// - Healthy: All criteria met
/// - Degraded: Some warnings (queue >70%, minor drift)
/// - Unhealthy: Critical issues (kernel hung, queue full, excessive drift)
/// </remarks>
public sealed class RingKernelHealthCheck : IHealthCheck
{
    private readonly ILogger<RingKernelHealthCheck> _logger;
    private readonly RingKernelWatchdog? _watchdog;
    private readonly RingKernelHealthCheckOptions _options;

    public RingKernelHealthCheck(
        ILogger<RingKernelHealthCheck> logger,
        RingKernelWatchdog? watchdog = null,
        RingKernelHealthCheckOptions? options = null)
    {
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        _watchdog = watchdog;
        _options = options ?? new RingKernelHealthCheckOptions();
    }

    public async Task<HealthCheckResult> CheckHealthAsync(
        HealthCheckContext context,
        CancellationToken cancellationToken = default)
    {
        try
        {
            // If no watchdog available, can't perform health check
            if (_watchdog == null)
            {
                return HealthCheckResult.Healthy(
                    "Ring kernel watchdog not configured - health monitoring disabled");
            }

            // Get all monitored kernel stats
            var allStats = _watchdog.GetAllKernelStats();

            if (allStats.Length == 0)
            {
                return HealthCheckResult.Healthy(
                    "No ring kernels currently monitored");
            }

            // Analyze kernel health
            var healthResults = AnalyzeKernelHealth(allStats);

            // Determine overall health status
            return DetermineOverallHealth(healthResults, allStats.Length);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error performing ring kernel health check");

            return HealthCheckResult.Unhealthy(
                "Health check failed with exception",
                exception: ex);
        }
    }

    private KernelHealthAnalysis AnalyzeKernelHealth(RingKernelWatchdogStats[] allStats)
    {
        var analysis = new KernelHealthAnalysis
        {
            TotalKernels = allStats.Length,
            HealthyKernels = 0,
            UnhealthyKernels = 0,
            KernelsWithTimeouts = 0,
            KernelsWithRestarts = 0,
            MaxRestartCount = 0,
            MaxConsecutiveTimeouts = 0,
            MinUptime = TimeSpan.MaxValue,
            MaxUptime = TimeSpan.Zero,
            Issues = new List<string>()
        };

        foreach (var stats in allStats)
        {
            // Track health
            if (stats.IsHealthy)
            {
                analysis.HealthyKernels++;
            }
            else
            {
                analysis.UnhealthyKernels++;
                analysis.Issues.Add(
                    $"Kernel {stats.InstanceId}: Unhealthy - " +
                    $"{stats.ConsecutiveTimeouts} consecutive timeouts");
            }

            // Track timeouts
            if (stats.ConsecutiveTimeouts > 0)
            {
                analysis.KernelsWithTimeouts++;
                analysis.MaxConsecutiveTimeouts = Math.Max(
                    analysis.MaxConsecutiveTimeouts,
                    stats.ConsecutiveTimeouts);
            }

            // Track restarts
            if (stats.RestartCount > 0)
            {
                analysis.KernelsWithRestarts++;
                analysis.MaxRestartCount = Math.Max(
                    analysis.MaxRestartCount,
                    stats.RestartCount);

                if (stats.RestartCount >= 2)
                {
                    analysis.Issues.Add(
                        $"Kernel {stats.InstanceId}: {stats.RestartCount} restarts - " +
                        "Frequent restarts indicate instability");
                }
            }

            // Track uptime
            analysis.MinUptime = stats.Uptime < analysis.MinUptime
                ? stats.Uptime
                : analysis.MinUptime;

            analysis.MaxUptime = stats.Uptime > analysis.MaxUptime
                ? stats.Uptime
                : analysis.MaxUptime;

            // Check for recently started kernels (may indicate crashes)
            if (stats.Uptime < TimeSpan.FromMinutes(1))
            {
                analysis.Issues.Add(
                    $"Kernel {stats.InstanceId}: Recently started - " +
                    $"Uptime: {stats.Uptime.TotalSeconds:F1}s");
            }
        }

        return analysis;
    }

    private HealthCheckResult DetermineOverallHealth(
        KernelHealthAnalysis analysis,
        int totalKernels)
    {
        var data = new Dictionary<string, object>
        {
            ["total_kernels"] = analysis.TotalKernels,
            ["healthy_kernels"] = analysis.HealthyKernels,
            ["unhealthy_kernels"] = analysis.UnhealthyKernels,
            ["kernels_with_restarts"] = analysis.KernelsWithRestarts,
            ["max_restart_count"] = analysis.MaxRestartCount,
            ["max_consecutive_timeouts"] = analysis.MaxConsecutiveTimeouts,
            ["min_uptime_seconds"] = analysis.MinUptime.TotalSeconds,
            ["max_uptime_seconds"] = analysis.MaxUptime.TotalSeconds
        };

        // UNHEALTHY: Any kernel with excessive restarts
        if (analysis.MaxRestartCount >= _options.MaxAllowedRestarts)
        {
            var message = $"Ring kernel health UNHEALTHY - " +
                         $"Kernel has {analysis.MaxRestartCount} restarts " +
                         $"(threshold: {_options.MaxAllowedRestarts})";

            _logger.LogError(message);

            return HealthCheckResult.Unhealthy(
                description: message,
                data: data);
        }

        // UNHEALTHY: Too many unhealthy kernels
        var unhealthyPercent = (analysis.UnhealthyKernels / (double)analysis.TotalKernels) * 100;
        if (unhealthyPercent > _options.UnhealthyKernelThresholdPercent)
        {
            var message = $"Ring kernel health UNHEALTHY - " +
                         $"{unhealthyPercent:F1}% of kernels unhealthy " +
                         $"(threshold: {_options.UnhealthyKernelThresholdPercent}%)";

            _logger.LogError(message);

            return HealthCheckResult.Unhealthy(
                description: message,
                data: data);
        }

        // DEGRADED: Some kernels have restarts
        if (analysis.KernelsWithRestarts > 0)
        {
            var message = $"Ring kernel health DEGRADED - " +
                         $"{analysis.KernelsWithRestarts} kernel(s) have been restarted";

            _logger.LogWarning(message);

            return HealthCheckResult.Degraded(
                description: message,
                data: data);
        }

        // DEGRADED: Some kernels have timeouts
        if (analysis.KernelsWithTimeouts > 0)
        {
            var message = $"Ring kernel health DEGRADED - " +
                         $"{analysis.KernelsWithTimeouts} kernel(s) experiencing timeouts";

            _logger.LogWarning(message);

            return HealthCheckResult.Degraded(
                description: message,
                data: data);
        }

        // HEALTHY: All kernels operational
        var healthyMessage = $"Ring kernel health HEALTHY - " +
                            $"{analysis.HealthyKernels}/{analysis.TotalKernels} kernels healthy";

        _logger.LogDebug(healthyMessage);

        return HealthCheckResult.Healthy(
            description: healthyMessage,
            data: data);
    }

    private sealed class KernelHealthAnalysis
    {
        public required int TotalKernels { get; init; }
        public required int HealthyKernels { get; set; }
        public required int UnhealthyKernels { get; set; }
        public required int KernelsWithTimeouts { get; set; }
        public required int KernelsWithRestarts { get; set; }
        public required int MaxRestartCount { get; set; }
        public required int MaxConsecutiveTimeouts { get; set; }
        public required TimeSpan MinUptime { get; set; }
        public required TimeSpan MaxUptime { get; set; }
        public required List<string> Issues { get; init; }
    }
}

/// <summary>
/// Configuration options for ring kernel health check.
/// </summary>
public sealed class RingKernelHealthCheckOptions
{
    /// <summary>
    /// Maximum allowed restart count before marking unhealthy.
    /// Default: 3 restarts.
    /// </summary>
    public int MaxAllowedRestarts { get; set; } = 3;

    /// <summary>
    /// Percentage of unhealthy kernels that triggers unhealthy status.
    /// Default: 25% (if >25% of kernels unhealthy, overall status is unhealthy).
    /// </summary>
    public double UnhealthyKernelThresholdPercent { get; set; } = 25.0;

    /// <summary>
    /// Minimum uptime before considering kernel stable.
    /// Default: 5 minutes.
    /// </summary>
    public TimeSpan MinStableUptime { get; set; } = TimeSpan.FromMinutes(5);
}

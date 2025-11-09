using System;
using System.Collections.Generic;
using Orleans.GpuBridge.Resilience.Fallback;
using Orleans.GpuBridge.Resilience.Policies;
using Orleans.GpuBridge.Resilience.Telemetry;

namespace Orleans.GpuBridge.Resilience;

/// <summary>
/// Comprehensive health report
/// </summary>
public readonly record struct HealthReport(
    HealthStatus OverallStatus,
    IReadOnlyDictionary<string, ComponentHealth> ComponentHealth,
    double BulkheadUtilization,
    IReadOnlyDictionary<string, CircuitBreakerState> CircuitBreakerStates,
    IReadOnlyDictionary<string, FallbackLevel> FallbackLevels,
    IReadOnlyList<HealthEvent> RecentEvents,
    DateTimeOffset Timestamp);

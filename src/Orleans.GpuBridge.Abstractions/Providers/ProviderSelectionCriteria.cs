using System.Collections.Generic;
using Orleans.GpuBridge.Abstractions.Enums;

namespace Orleans.GpuBridge.Abstractions.Providers;

/// <summary>
/// Criteria for selecting a backend provider
/// </summary>
public sealed record ProviderSelectionCriteria(
    string? PreferredProviderId = null,
    GpuBackend? PreferredBackend = null,
    IReadOnlyList<string>? RequiredCapabilities = null,
    bool RequireJitCompilation = false,
    bool RequireUnifiedMemory = false,
    bool RequireProfiling = false,
    bool RequireCpuDebugging = false,
    IReadOnlyList<string>? ExcludeProviders = null,
    bool PreferGpu = true)
{
    public static ProviderSelectionCriteria Default => new();

    public static ProviderSelectionCriteria PreferDotCompute => new(
        PreferredProviderId: "DotCompute",
        RequireJitCompilation: true);
}
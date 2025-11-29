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
    /// <summary>
    /// Gets the default provider selection criteria with GPU preference enabled.
    /// </summary>
    public static ProviderSelectionCriteria Default => new();

    /// <summary>
    /// Gets provider selection criteria that prefers the DotCompute backend with JIT compilation.
    /// </summary>
    public static ProviderSelectionCriteria PreferDotCompute => new(
        PreferredProviderId: "DotCompute",
        RequireJitCompilation: true);
}
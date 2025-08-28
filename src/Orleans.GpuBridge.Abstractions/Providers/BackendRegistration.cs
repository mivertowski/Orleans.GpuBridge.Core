using System;

namespace Orleans.GpuBridge.Abstractions.Providers;

/// <summary>
/// Registration information for a backend provider
/// </summary>
public sealed record BackendRegistration(
    string ProviderId,
    string DisplayName,
    Type? ProviderType = null,
    Func<IServiceProvider, IGpuBackendProvider>? Factory = null,
    int Priority = 100);
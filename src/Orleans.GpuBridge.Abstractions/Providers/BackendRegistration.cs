using System;
using System.Diagnostics.CodeAnalysis;

namespace Orleans.GpuBridge.Abstractions.Providers;

/// <summary>
/// Registration information for a backend provider
/// </summary>
public sealed record BackendRegistration(
    string ProviderId,
    string DisplayName,
    [DynamicallyAccessedMembers(DynamicallyAccessedMemberTypes.PublicConstructors)] Type? ProviderType = null,
    Func<IServiceProvider, IGpuBackendProvider>? Factory = null,
    int Priority = 100);
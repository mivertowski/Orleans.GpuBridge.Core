using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using System.Diagnostics.CodeAnalysis;

namespace Orleans.GpuBridge.Abstractions.Providers;

/// <summary>
/// Registry interface for managing GPU backend providers
/// </summary>
public interface IGpuBackendRegistry : IDisposable
{
    /// <summary>
    /// Initializes the backend registry
    /// </summary>
    Task InitializeAsync(CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Registers a backend provider
    /// </summary>
    void RegisterProvider([NotNull] BackendRegistration registration);
    
    /// <summary>
    /// Discovers available GPU backend providers
    /// </summary>
    [RequiresUnreferencedCode("Discovers provider types from assemblies which may be trimmed.")]
    Task DiscoverProvidersAsync(CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Gets a specific backend provider by ID
    /// </summary>
    [RequiresUnreferencedCode("Creates provider instances using reflection which may not work with trimming.")]
    Task<IGpuBackendProvider?> GetProviderAsync(
        [NotNull] string providerId,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Gets a specific backend provider by ID (alias for GetProviderAsync)
    /// </summary>
    [RequiresUnreferencedCode("Creates provider instances using reflection which may not work with trimming.")]
    Task<IGpuBackendProvider?> GetProviderByIdAsync(
        [NotNull] string providerId,
        CancellationToken cancellationToken = default)
    {
        return GetProviderAsync(providerId, cancellationToken);
    }

    /// <summary>
    /// Gets list of registered provider IDs
    /// </summary>
    IReadOnlyList<string> GetRegisteredProviders();
    
    /// <summary>
    /// Gets all available backend providers
    /// </summary>
    Task<IReadOnlyList<IGpuBackendProvider>> GetAvailableProvidersAsync(
        CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Selects the best provider based on criteria
    /// </summary>
    Task<IGpuBackendProvider?> SelectProviderAsync(
        [NotNull] ProviderSelectionCriteria criteria,
        CancellationToken cancellationToken = default);
}
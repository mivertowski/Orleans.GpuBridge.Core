using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;

namespace Orleans.GpuBridge.Abstractions.Providers;

/// <summary>
/// Registry interface for managing GPU backend providers
/// </summary>
public interface IGpuBackendRegistry : IDisposable
{
    /// <summary>
    /// Registers a backend provider
    /// </summary>
    void RegisterProvider(BackendRegistration registration);
    
    /// <summary>
    /// Discovers available GPU backend providers
    /// </summary>
    Task DiscoverProvidersAsync(CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Gets a specific backend provider by ID
    /// </summary>
    Task<IGpuBackendProvider?> GetProviderAsync(
        string providerId, 
        CancellationToken cancellationToken = default);
    
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
        ProviderSelectionCriteria criteria,
        CancellationToken cancellationToken = default);
}
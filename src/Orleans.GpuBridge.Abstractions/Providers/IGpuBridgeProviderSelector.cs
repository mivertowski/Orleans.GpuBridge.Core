using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using Orleans.GpuBridge.Abstractions;

namespace Orleans.GpuBridge.Abstractions.Providers;

/// <summary>
/// Interface for selecting appropriate GPU backend providers
/// </summary>
public interface IGpuBridgeProviderSelector
{
    /// <summary>
    /// Initializes the provider selector
    /// </summary>
    Task InitializeAsync(CancellationToken cancellationToken = default);

    /// <summary>
    /// Selects the best available provider for execution
    /// </summary>
    Task<IGpuBackendProvider> SelectProviderAsync(
        ProviderSelectionCriteria criteria,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Gets a provider by name
    /// </summary>
    Task<IGpuBackendProvider?> GetProviderByNameAsync(
        string providerName,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Gets all available providers
    /// </summary>
    Task<IReadOnlyList<IGpuBackendProvider>> GetAvailableProvidersAsync(
        CancellationToken cancellationToken = default);
}
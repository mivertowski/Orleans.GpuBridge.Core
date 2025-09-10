using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Threading;
using System.Threading.Tasks;
using System.Diagnostics.CodeAnalysis;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Orleans.GpuBridge.Abstractions.Enums;
using Orleans.GpuBridge.Abstractions.Providers;

namespace Orleans.GpuBridge.Runtime.Providers;

/// <summary>
/// Default implementation of GPU backend registry
/// </summary>
public sealed class GpuBackendRegistry : IGpuBackendRegistry
{
    private readonly IServiceProvider _serviceProvider;
    private readonly ILogger<GpuBackendRegistry> _logger;
    private readonly ConcurrentDictionary<string, BackendRegistration> _registrations = new();
    private readonly ConcurrentDictionary<string, IGpuBackendProvider> _providers = new();
    private readonly SemaphoreSlim _discoveryLock = new(1, 1);
    private volatile bool _discoveryCompleted;
    private volatile bool _disposed;

    public GpuBackendRegistry(
        IServiceProvider serviceProvider,
        ILogger<GpuBackendRegistry> logger)
    {
        _serviceProvider = serviceProvider ?? throw new ArgumentNullException(nameof(serviceProvider));
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
    }

    /// <inheritdoc/>
    public async Task InitializeAsync(CancellationToken cancellationToken = default)
    {
        ThrowIfDisposed();
        
        if (_discoveryCompleted)
            return;
            
        await _discoveryLock.WaitAsync(cancellationToken);
        try
        {
            if (!_discoveryCompleted)
            {
                await DiscoverProvidersAsync(cancellationToken);
                _discoveryCompleted = true;
                _logger.LogInformation("GPU backend registry initialized with {Count} providers", _registrations.Count);
            }
        }
        finally
        {
            _discoveryLock.Release();
        }
    }

    /// <inheritdoc/>
    public void RegisterProvider([NotNull] BackendRegistration registration)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(registration);

        if (string.IsNullOrWhiteSpace(registration.ProviderId))
            throw new ArgumentException("Provider ID cannot be null or empty", nameof(registration));

        if (registration.ProviderType is null && registration.Factory is null)
            throw new ArgumentException("Either ProviderType or Factory must be specified", nameof(registration));

        _registrations.AddOrUpdate(registration.ProviderId, registration, (_, _) => registration);
        _logger.LogDebug("Registered GPU backend provider: {ProviderId}", registration.ProviderId);
    }

    /// <inheritdoc/>
    [RequiresUnreferencedCode("Discovers provider types from assemblies which may be trimmed.")]
    public async Task DiscoverProvidersAsync(CancellationToken cancellationToken = default)
    {
        ThrowIfDisposed();

        if (_discoveryCompleted)
            return;

        await _discoveryLock.WaitAsync(cancellationToken);
        try
        {
            if (_discoveryCompleted)
                return;

            _logger.LogInformation("Starting GPU backend provider discovery...");

            // Discover from loaded assemblies
            await DiscoverFromAssembliesAsync(cancellationToken);

            _discoveryCompleted = true;
            _logger.LogInformation("GPU backend provider discovery completed. Found {Count} providers",
                _registrations.Count);
        }
        finally
        {
            _discoveryLock.Release();
        }
    }

    /// <inheritdoc/>
    [RequiresUnreferencedCode("Creates provider instances using reflection which may not work with trimming.")]
    public async Task<IGpuBackendProvider?> GetProviderAsync(
        [NotNull] string providerId,
        CancellationToken cancellationToken = default)
    {
        ThrowIfDisposed();
        ArgumentException.ThrowIfNullOrWhiteSpace(providerId);

        // Ensure discovery has been completed
        if (!_discoveryCompleted)
            await DiscoverProvidersAsync(cancellationToken);

        // Check if already instantiated
        if (_providers.TryGetValue(providerId, out var cachedProvider))
            return cachedProvider;

        // Check if registration exists
        if (!_registrations.TryGetValue(providerId, out var registration))
            return null;

        try
        {
            var provider = await CreateProviderAsync(registration, cancellationToken);
            if (provider != null)
            {
                _providers.TryAdd(providerId, provider);
                return provider;
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to create provider {ProviderId}", providerId);
        }

        return null;
    }

    /// <inheritdoc/>
    public IReadOnlyList<string> GetRegisteredProviders()
    {
        ThrowIfDisposed();
        return _registrations.Keys.ToList().AsReadOnly();
    }

    /// <inheritdoc/>
    public async Task<IReadOnlyList<IGpuBackendProvider>> GetAvailableProvidersAsync(
        CancellationToken cancellationToken = default)
    {
        ThrowIfDisposed();

        // Ensure discovery has been completed
        if (!_discoveryCompleted)
            await DiscoverProvidersAsync(cancellationToken);

        var providers = new List<IGpuBackendProvider>();
        
        foreach (var (providerId, registration) in _registrations.OrderBy(x => x.Value.Priority))
        {
            try
            {
                var provider = await GetProviderAsync(providerId, cancellationToken);
                if (provider != null)
                    providers.Add(provider);
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Failed to get provider {ProviderId}", providerId);
            }
        }

        return providers.AsReadOnly();
    }

    /// <inheritdoc/>
    public async Task<IGpuBackendProvider?> SelectProviderAsync(
        [NotNull] ProviderSelectionCriteria criteria,
        CancellationToken cancellationToken = default)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(criteria);

        // If specific provider requested, try that first
        if (!string.IsNullOrEmpty(criteria.PreferredProviderId))
        {
            var preferredProvider = await GetProviderAsync(criteria.PreferredProviderId, cancellationToken);
            if (preferredProvider != null && await IsProviderCompatibleAsync(preferredProvider, criteria))
                return preferredProvider;
        }

        // Get all available providers and find the best match
        var availableProviders = await GetAvailableProvidersAsync(cancellationToken);
        
        // Filter by exclusions
        if (criteria.ExcludeProviders?.Count > 0)
        {
            availableProviders = availableProviders
                .Where(p => !criteria.ExcludeProviders.Contains(p.ProviderId))
                .ToList()
                .AsReadOnly();
        }

        // Find compatible providers
        var compatibleProviders = new List<IGpuBackendProvider>();
        foreach (var provider in availableProviders)
        {
            if (await IsProviderCompatibleAsync(provider, criteria))
                compatibleProviders.Add(provider);
        }

        if (compatibleProviders.Count == 0)
        {
            _logger.LogWarning("No compatible GPU backend providers found for criteria");
            return null;
        }

        // Select best provider based on priority and preferences
        var selectedProvider = SelectBestProvider(compatibleProviders, criteria);
        
        if (selectedProvider != null)
        {
            _logger.LogInformation("Selected GPU backend provider: {ProviderId}", selectedProvider.ProviderId);
        }

        return selectedProvider;
    }

    [RequiresUnreferencedCode("Discovers provider types from assemblies which may be trimmed.")]
    private async Task DiscoverFromAssembliesAsync(CancellationToken cancellationToken)
    {
        var assemblies = AppDomain.CurrentDomain.GetAssemblies()
            .Where(a => !a.IsDynamic && !string.IsNullOrEmpty(AppContext.BaseDirectory)) // Use AppContext.BaseDirectory instead of Assembly.Location
            .ToList();

        foreach (var assembly in assemblies)
        {
            try
            {
                await DiscoverFromAssemblyAsync(assembly, cancellationToken);
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Failed to discover providers from assembly {AssemblyName}", 
                    assembly.FullName);
            }
        }
    }

    [RequiresUnreferencedCode("Uses Assembly.GetTypes() which is not compatible with trimming.")]
    private async Task DiscoverFromAssemblyAsync(Assembly assembly, CancellationToken cancellationToken)
    {
        var providerTypes = assembly.GetTypes()
            .Where(t => t is { IsClass: true, IsAbstract: false } &&
                       typeof(IGpuBackendProvider).IsAssignableFrom(t))
            .ToList();

        foreach (var providerType in providerTypes)
        {
            try
            {
                // Try to get provider ID from attribute or use type name
                var providerId = GetProviderIdFromType(providerType);
                
                var registration = new BackendRegistration(
                    ProviderId: providerId,
                    DisplayName: providerType.Name,
                    ProviderType: providerType);

                RegisterProvider(registration);
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Failed to register provider type {TypeName}", providerType.FullName);
            }
        }

        await Task.CompletedTask; // Make method async for future extensibility
    }

    private static string GetProviderIdFromType(Type providerType)
    {
        // Could check for custom attributes here in the future
        return providerType.Name.Replace("Provider", "").Replace("Backend", "");
    }

    [RequiresUnreferencedCode("Creates provider instances using reflection which may not work with trimming.")]
    private async Task<IGpuBackendProvider?> CreateProviderAsync(
        BackendRegistration registration,
        CancellationToken cancellationToken)
    {
        try
        {
            if (registration.Factory != null)
            {
                var provider = registration.Factory(_serviceProvider);
                if (provider != null)
                {
                    await provider.InitializeAsync(new BackendConfiguration(), cancellationToken);
                    return provider;
                }
            }
            else if (registration.ProviderType != null)
            {
                var provider = (IGpuBackendProvider?)ActivatorUtilities.CreateInstance(
                    _serviceProvider, registration.ProviderType);
                    
                if (provider != null)
                {
                    await provider.InitializeAsync(new BackendConfiguration(), cancellationToken);
                    return provider;
                }
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to create provider {ProviderId}", registration.ProviderId);
        }

        return null;
    }

    private static Task<bool> IsProviderCompatibleAsync(
        IGpuBackendProvider provider,
        ProviderSelectionCriteria criteria)
    {
        var capabilities = provider.Capabilities;

        // Check required capabilities
        if (criteria.RequiredCapabilities?.Count > 0)
        {
            var providerCapabilities = capabilities.SupportedKernelLanguages;
            if (!criteria.RequiredCapabilities.All(req => providerCapabilities.Contains(req)))
                return Task.FromResult(false);
        }

        // Check specific requirements
        if (criteria.RequireJitCompilation && !capabilities.SupportsJitCompilation)
            return Task.FromResult(false);

        if (criteria.RequireUnifiedMemory && !capabilities.SupportsUnifiedMemory)
            return Task.FromResult(false);

        if (criteria.RequireProfiling && !capabilities.SupportsProfiling)
            return Task.FromResult(false);

        if (criteria.RequireCpuDebugging && !capabilities.SupportsCpuDebugging)
            return Task.FromResult(false);

        // Check preferred backend
        if (criteria.PreferredBackend.HasValue && 
            !capabilities.SupportedBackends.Contains(criteria.PreferredBackend.Value))
            return Task.FromResult(false);

        return Task.FromResult(true);
    }

    private static IGpuBackendProvider? SelectBestProvider(
        IReadOnlyList<IGpuBackendProvider> compatibleProviders,
        ProviderSelectionCriteria criteria)
    {
        // For now, select the first compatible provider
        // In the future, could implement more sophisticated selection logic
        return compatibleProviders.FirstOrDefault();
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(nameof(GpuBackendRegistry));
    }

    public void Dispose()
    {
        if (_disposed)
            return;

        _disposed = true;

        foreach (var provider in _providers.Values)
        {
            try
            {
                provider.Dispose();
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Failed to dispose provider {ProviderId}", provider.ProviderId);
            }
        }

        _providers.Clear();
        _registrations.Clear();
        _discoveryLock.Dispose();

        GC.SuppressFinalize(this);
    }
}
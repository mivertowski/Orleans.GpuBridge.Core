using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using Orleans.GpuBridge.Abstractions;
using Orleans.GpuBridge.Abstractions.Enums;
using Orleans.GpuBridge.Abstractions.Providers;

namespace Orleans.GpuBridge.Runtime.Providers;

/// <summary>
/// Selects the appropriate GPU backend provider for kernel execution
/// </summary>
public sealed class GpuBridgeProviderSelector : IGpuBridgeProviderSelector
{
    private readonly ILogger<GpuBridgeProviderSelector> _logger;
    private readonly IGpuBackendRegistry _registry;
    private readonly GpuBridgeOptions _options;
    private readonly IServiceProvider _serviceProvider;
    private readonly Dictionary<string, IGpuBackendProvider> _providerCache;
    private readonly SemaphoreSlim _cacheLock;
    private IGpuBackendProvider? _defaultProvider;

    public GpuBridgeProviderSelector(
        ILogger<GpuBridgeProviderSelector> logger,
        IGpuBackendRegistry registry,
        IOptions<GpuBridgeOptions> options,
        IServiceProvider serviceProvider)
    {
        _logger = logger;
        _registry = registry;
        _options = options.Value;
        _serviceProvider = serviceProvider;
        _providerCache = new Dictionary<string, IGpuBackendProvider>(StringComparer.OrdinalIgnoreCase);
        _cacheLock = new SemaphoreSlim(1, 1);
    }

    /// <summary>
    /// Initializes the provider selector and discovers available providers
    /// </summary>
    [UnconditionalSuppressMessage("Trimming", "IL2026:Members annotated with 'RequiresUnreferencedCodeAttribute' require dynamic access otherwise can break functionality when trimming application code",
        Justification = "Provider discovery uses reflection to find GPU backend providers. When using AOT/trimming, providers must be registered explicitly via RegisterProvider() instead of relying on automatic discovery. This is acceptable as provider discovery is opt-in via EnableProviderDiscovery option.")]
    public async Task InitializeAsync(CancellationToken cancellationToken = default)
    {
        _logger.LogInformation("Initializing GPU backend provider selector");

        // Discover providers if enabled
        if (_options.EnableProviderDiscovery)
        {
            await _registry.DiscoverProvidersAsync(cancellationToken);
        }

        // Register built-in providers
        RegisterBuiltInProviders();

        // Initialize default provider
        await InitializeDefaultProviderAsync(cancellationToken);

        _logger.LogInformation(
            "GPU backend provider selector initialized with {Count} providers",
            _registry.GetRegisteredProviders().Count);
    }

    /// <summary>
    /// Selects a provider for kernel execution based on criteria
    /// </summary>
    public async Task<IGpuBackendProvider> SelectProviderAsync(
        ProviderSelectionCriteria criteria,
        CancellationToken cancellationToken = default)
    {
        // Check if specific provider is requested
        if (!string.IsNullOrEmpty(criteria.PreferredProviderId))
        {
            var provider = await GetProviderByNameAsync(criteria.PreferredProviderId, cancellationToken);
            if (provider != null)
            {
                return provider;
            }

            _logger.LogWarning(
                "Requested GPU backend provider {ProviderId} not available, falling back",
                criteria.PreferredProviderId);
        }

        // Try to select based on criteria from registry
        var selectedProvider = await _registry.SelectProviderAsync(criteria, cancellationToken);

        if (selectedProvider != null)
        {
            return selectedProvider;
        }

        // Fall back to default provider
        if (_defaultProvider != null)
        {
            _logger.LogDebug("Using default GPU backend provider: {ProviderId}", _defaultProvider.ProviderId);
            return _defaultProvider;
        }

        // Last resort: CPU fallback
        var cpuProvider = await GetOrCreateCpuProviderAsync(cancellationToken);
        _logger.LogWarning("No GPU backend available, using CPU fallback");
        return cpuProvider;
    }

    /// <summary>
    /// Gets a provider by name
    /// </summary>
    [UnconditionalSuppressMessage("Trimming", "IL2026:Members annotated with 'RequiresUnreferencedCodeAttribute' require dynamic access otherwise can break functionality when trimming application code",
        Justification = "GetProviderAsync uses reflection to instantiate providers. When using AOT/trimming, providers must be registered with factory delegates via RegisterProvider() to avoid reflection-based instantiation. Built-in providers (CPU, DotCompute) are registered with factories in RegisterBuiltInProviders().")]
    public async Task<IGpuBackendProvider?> GetProviderByNameAsync(
        string providerName,
        CancellationToken cancellationToken = default)
    {
        await _cacheLock.WaitAsync(cancellationToken);
        try
        {
            if (_providerCache.TryGetValue(providerName, out var cachedProvider))
            {
                return cachedProvider;
            }

            var provider = await _registry.GetProviderAsync(providerName, cancellationToken);
            if (provider != null)
            {
                _providerCache[providerName] = provider;
            }

            return provider;
        }
        finally
        {
            _cacheLock.Release();
        }
    }

    /// <summary>
    /// Gets all available providers
    /// </summary>
    public async Task<IReadOnlyList<IGpuBackendProvider>> GetAvailableProvidersAsync(
        CancellationToken cancellationToken = default)
    {
        return await _registry.GetAvailableProvidersAsync(cancellationToken);
    }

    private void RegisterBuiltInProviders()
    {
        // Register CPU fallback provider
        _registry.RegisterProvider(new BackendRegistration(
            ProviderId: "CPU",
            DisplayName: "CPU Fallback Provider",
            Factory: sp => new CpuFallbackProvider(
                sp.GetService<ILogger<CpuFallbackProvider>>()!,
                sp.GetService<ILoggerFactory>()!),
            Priority: 10));

        // Note: DotCompute providers will be registered via discovery
        // or explicit registration when their assemblies are loaded
    }

    private async Task InitializeDefaultProviderAsync(CancellationToken cancellationToken)
    {
        // Try to initialize based on configuration
        if (!string.IsNullOrEmpty(_options.DefaultBackend))
        {
            _defaultProvider = await GetProviderByNameAsync(_options.DefaultBackend, cancellationToken);
            if (_defaultProvider != null)
            {
                _logger.LogInformation(
                    "Default GPU backend provider set to: {ProviderId}",
                    _defaultProvider.ProviderId);
                return;
            }
        }

        // Try fallback chain
        if (_options.FallbackChain?.Any() == true)
        {
            foreach (var providerId in _options.FallbackChain)
            {
                var provider = await GetProviderByNameAsync(providerId, cancellationToken);
                if (provider != null && await provider.IsAvailableAsync(cancellationToken))
                {
                    _defaultProvider = provider;
                    _logger.LogInformation(
                        "Default GPU backend provider set to: {ProviderId} (from fallback chain)",
                        _defaultProvider.ProviderId);
                    return;
                }
            }
        }

        // Auto-select best available
        var autoBackend = _options.PreferGpu ? GpuBackend.CUDA : GpuBackend.CPU;
        var criteria = new ProviderSelectionCriteria(PreferredBackend: autoBackend);

        _defaultProvider = await _registry.SelectProviderAsync(criteria, cancellationToken);

        if (_defaultProvider != null)
        {
            _logger.LogInformation(
                "Auto-selected default GPU backend provider: {ProviderId}",
                _defaultProvider.ProviderId);
        }
        else
        {
            _logger.LogWarning("No default GPU backend provider could be initialized");
        }
    }


    private async Task<IGpuBackendProvider> GetOrCreateCpuProviderAsync(CancellationToken cancellationToken)
    {
        var cpuProvider = await GetProviderByNameAsync("CPU", cancellationToken);
        if (cpuProvider == null)
        {
            // Create CPU provider directly as last resort
            var loggerFactory = _serviceProvider.GetService<ILoggerFactory>();
            var logger = loggerFactory?.CreateLogger<CpuFallbackProvider>() ??
                        Microsoft.Extensions.Logging.Abstractions.NullLogger<CpuFallbackProvider>.Instance;

            cpuProvider = new CpuFallbackProvider(logger, loggerFactory!);

            await cpuProvider.InitializeAsync(
                new BackendConfiguration(),
                cancellationToken);
        }

        return cpuProvider;
    }
}

/// <summary>
/// Extensions for GpuBridgeOptions
/// </summary>
public static class GpuBridgeOptionsExtensions
{
    public static GpuBridgeOptions WithBackendProvider(this GpuBridgeOptions options, string backend)
    {
        options.DefaultBackend = backend;
        return options;
    }

    public static GpuBridgeOptions WithFallbackChain(this GpuBridgeOptions options, params string[] backends)
    {
        options.FallbackChain = backends;
        return options;
    }

    public static GpuBridgeOptions EnableProviderDiscovery(this GpuBridgeOptions options, bool enable = true)
    {
        options.EnableProviderDiscovery = enable;
        return options;
    }
}
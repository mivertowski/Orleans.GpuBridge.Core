using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Orleans.GpuBridge.Abstractions.Providers;
using Orleans.GpuBridge.Runtime.Providers;
using System;
using System.Threading;
using System.Threading.Tasks;

namespace Orleans.GpuBridge.Runtime.Infrastructure;

/// <summary>
/// Factory for creating and managing backend providers.
/// Handles discovery and initialization of GPU backends with CPU fallback support.
/// </summary>
public class BackendProviderFactory
{
    private readonly IServiceProvider _serviceProvider;
    private readonly ILoggerFactory _loggerFactory;
    private readonly ILogger<BackendProviderFactory> _logger;
    private IGpuBackendProvider? _primaryProvider;
    private bool _initialized;

    /// <summary>
    /// Creates a new backend provider factory.
    /// </summary>
    /// <param name="serviceProvider">Service provider for dependency resolution.</param>
    /// <param name="loggerFactory">Logger factory for creating loggers.</param>
    /// <param name="logger">Logger for this factory.</param>
    public BackendProviderFactory(
        IServiceProvider serviceProvider,
        ILoggerFactory loggerFactory,
        ILogger<BackendProviderFactory> logger)
    {
        _serviceProvider = serviceProvider ?? throw new ArgumentNullException(nameof(serviceProvider));
        _loggerFactory = loggerFactory ?? throw new ArgumentNullException(nameof(loggerFactory));
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
    }

    /// <summary>
    /// Initialize the factory and discover providers.
    /// First attempts to discover GPU providers (DotCompute), then falls back to CPU.
    /// </summary>
    public void Initialize()
    {
        if (_initialized)
        {
            _logger.LogDebug("Backend provider factory already initialized");
            return;
        }

        _logger.LogInformation("Initializing backend provider factory");

        // Try to get DotCompute provider from DI first
        _primaryProvider = _serviceProvider.GetService<IGpuBackendProvider>();

        if (_primaryProvider != null)
        {
            _logger.LogInformation(
                "Found GPU backend provider: {ProviderId} ({DisplayName})",
                _primaryProvider.ProviderId,
                _primaryProvider.DisplayName);

            // Initialize the provider asynchronously (blocking call here is intentional for startup)
            try
            {
                _primaryProvider.InitializeAsync(new BackendConfiguration(), CancellationToken.None)
                    .GetAwaiter().GetResult();
                _logger.LogInformation("GPU backend provider initialized successfully");
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "GPU backend provider initialization failed, falling back to CPU");
                _primaryProvider = null;
            }
        }

        // Fall back to CPU provider if no GPU provider available
        if (_primaryProvider == null)
        {
            _logger.LogInformation("No GPU backend available, initializing CPU fallback provider");
            _primaryProvider = CreateCpuFallbackProvider();

            _primaryProvider.InitializeAsync(new BackendConfiguration(), CancellationToken.None)
                .GetAwaiter().GetResult();
            _logger.LogInformation("CPU fallback provider initialized");
        }

        _initialized = true;
    }

    /// <summary>
    /// Initialize the factory asynchronously.
    /// </summary>
    /// <param name="cancellationToken">Cancellation token.</param>
    public async Task InitializeAsync(CancellationToken cancellationToken = default)
    {
        if (_initialized)
        {
            _logger.LogDebug("Backend provider factory already initialized");
            return;
        }

        _logger.LogInformation("Initializing backend provider factory (async)");

        // Try to get DotCompute provider from DI first
        _primaryProvider = _serviceProvider.GetService<IGpuBackendProvider>();

        if (_primaryProvider != null)
        {
            _logger.LogInformation(
                "Found GPU backend provider: {ProviderId} ({DisplayName})",
                _primaryProvider.ProviderId,
                _primaryProvider.DisplayName);

            try
            {
                await _primaryProvider.InitializeAsync(new BackendConfiguration(), cancellationToken);
                _logger.LogInformation("GPU backend provider initialized successfully");
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "GPU backend provider initialization failed, falling back to CPU");
                _primaryProvider = null;
            }
        }

        // Fall back to CPU provider if no GPU provider available
        if (_primaryProvider == null)
        {
            _logger.LogInformation("No GPU backend available, initializing CPU fallback provider");
            _primaryProvider = CreateCpuFallbackProvider();

            await _primaryProvider.InitializeAsync(new BackendConfiguration(), cancellationToken);
            _logger.LogInformation("CPU fallback provider initialized");
        }

        _initialized = true;
    }

    /// <summary>
    /// Get the primary backend provider.
    /// </summary>
    /// <returns>The initialized primary backend provider.</returns>
    /// <exception cref="InvalidOperationException">Thrown if Initialize() has not been called.</exception>
    public IGpuBackendProvider GetPrimaryProvider()
    {
        if (_primaryProvider == null)
        {
            throw new InvalidOperationException("No primary provider available. Call Initialize() first.");
        }

        return _primaryProvider;
    }

    /// <summary>
    /// Gets a value indicating whether a GPU provider is available.
    /// </summary>
    public bool HasGpuProvider => _primaryProvider != null && _primaryProvider.ProviderId != "CPU";

    /// <summary>
    /// Gets a value indicating whether the factory has been initialized.
    /// </summary>
    public bool IsInitialized => _initialized;

    /// <summary>
    /// Creates a CPU fallback provider with proper dependencies.
    /// </summary>
    private CpuFallbackProvider CreateCpuFallbackProvider()
    {
        var logger = _loggerFactory.CreateLogger<CpuFallbackProvider>();
        return new CpuFallbackProvider(logger, _loggerFactory);
    }
}
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Orleans.GpuBridge.Abstractions.Providers;
using Orleans.GpuBridge.Abstractions.Providers.Execution.Interfaces;
using System;

namespace Orleans.GpuBridge.Runtime.Infrastructure;

/// <summary>
/// Factory for creating and managing backend providers
/// </summary>
public class BackendProviderFactory
{
    private readonly IServiceProvider _serviceProvider;
    private readonly ILogger<BackendProviderFactory> _logger;
    private IGpuBackendProvider? _primaryProvider;

    public BackendProviderFactory(IServiceProvider serviceProvider, ILogger<BackendProviderFactory> logger)
    {
        _serviceProvider = serviceProvider ?? throw new ArgumentNullException(nameof(serviceProvider));
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
    }

    /// <summary>
    /// Initialize the factory and discover providers
    /// </summary>
    public void Initialize()
    {
        _logger.LogInformation("Initializing backend provider factory");
        
        // Try to get a CPU fallback provider as primary
        try
        {
            _primaryProvider = new CpuFallbackProvider();
            _logger.LogInformation("Initialized with CPU fallback provider");
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Failed to initialize primary provider");
        }
    }

    /// <summary>
    /// Get the primary backend provider
    /// </summary>
    public IGpuBackendProvider GetPrimaryProvider()
    {
        if (_primaryProvider == null)
        {
            throw new InvalidOperationException("No primary provider available. Call Initialize() first.");
        }

        return _primaryProvider;
    }
}

/// <summary>
/// Simple CPU fallback provider for testing
/// </summary>
internal class CpuFallbackProvider : IGpuBackendProvider
{
    public string ProviderId => "CpuFallback";
    public string DisplayName => "CPU Fallback Provider";
    public Version Version => new Version(1, 0, 0);
    public BackendCapabilities Capabilities => BackendCapabilities.CreateCpuFallback();

    public bool IsAvailable() => true;

    public Task<bool> IsAvailableAsync(CancellationToken cancellationToken = default) => Task.FromResult(true);

    public Task<object> CreateContext(int deviceIndex = 0) => Task.FromResult<object>(new DisposableStub());

    public void Dispose() { }

    private class DisposableStub : IDisposable
    {
        public void Dispose() { }
    }

    public Task<IReadOnlyDictionary<string, object>> GetMetricsAsync(CancellationToken cancellationToken = default)
    {
        var metrics = new Dictionary<string, object>
        {
            ["IsAvailable"] = true,
            ["Type"] = "CPU"
        };
        return Task.FromResult<IReadOnlyDictionary<string, object>>(metrics);
    }

    public Task InitializeAsync(BackendConfiguration configuration, CancellationToken cancellationToken = default)
    {
        return Task.CompletedTask;
    }

    public Orleans.GpuBridge.Abstractions.Providers.IDeviceManager GetDeviceManager() => throw new NotImplementedException();
    public Orleans.GpuBridge.Abstractions.Providers.IKernelCompiler GetKernelCompiler() => throw new NotImplementedException();
    public Orleans.GpuBridge.Abstractions.Providers.Memory.Allocators.IMemoryAllocator GetMemoryAllocator() => throw new NotImplementedException();
    public Orleans.GpuBridge.Abstractions.Providers.Execution.Interfaces.IKernelExecutor GetKernelExecutor() => throw new NotImplementedException();

    public Task<HealthCheckResult> CheckHealthAsync(CancellationToken cancellationToken = default)
    {
        return Task.FromResult(new HealthCheckResult(true, "CPU fallback available"));
    }
}
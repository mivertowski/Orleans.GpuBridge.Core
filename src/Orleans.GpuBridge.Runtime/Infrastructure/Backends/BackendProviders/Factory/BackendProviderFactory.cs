using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Orleans.GpuBridge.Abstractions.Enums;
using Orleans.GpuBridge.Runtime.BackendProviders.Implementations;
using Orleans.GpuBridge.Runtime.BackendProviders.Interfaces;

namespace Orleans.GpuBridge.Runtime.BackendProviders.Factory;

/// <summary>
/// Factory for creating and managing GPU backend providers based on runtime detection
/// </summary>
/// <remarks>
/// This factory automatically detects available GPU compute backends at runtime and creates
/// appropriate provider instances. It supports multiple backend types including CUDA, OpenCL,
/// DirectCompute, Metal, and Vulkan, with automatic fallback to CPU execution when no GPU
/// backends are available. The factory maintains provider priority ordering and provides
/// access to both primary and specific backend providers.
/// </remarks>
public sealed class BackendProviderFactory
{
    private readonly IServiceProvider _serviceProvider;
    private readonly ILogger<BackendProviderFactory> _logger;
    private readonly List<IBackendProvider> _availableProviders;
    private IBackendProvider? _primaryProvider;

    /// <summary>
    /// Initializes a new instance of the <see cref="BackendProviderFactory"/> class
    /// </summary>
    /// <param name="serviceProvider">The service provider for dependency resolution</param>
    /// <param name="logger">The logger for diagnostic output</param>
    public BackendProviderFactory(
        IServiceProvider serviceProvider,
        ILogger<BackendProviderFactory> logger)
    {
        _serviceProvider = serviceProvider;
        _logger = logger;
        _availableProviders = new List<IBackendProvider>();
    }

    /// <summary>
    /// Initializes and detects available backend providers
    /// </summary>
    /// <remarks>
    /// This method uses DotCompute as the unified GPU backend, which provides support for
    /// CUDA, OpenCL, Vulkan, Metal, and DirectCompute through a single abstraction.
    /// CPU fallback is always added as the last option to ensure functionality when
    /// no GPU backends are available.
    /// </remarks>
    public void Initialize()
    {
        _logger.LogInformation("Detecting available GPU backend providers");

        // Try DotCompute unified backend (supports CUDA, OpenCL, Vulkan, Metal, DirectCompute)
        // DotCompute automatically detects and uses the best available GPU backend
        if (TryInitializeDotCompute())
        {
            _logger.LogInformation("DotCompute unified backend initialized (CUDA/OpenCL/Vulkan/Metal/DirectCompute)");
        }

        // Always add CPU fallback
        var cpuProvider = new CpuBackendProvider(_serviceProvider, _logger);
        cpuProvider.Initialize();
        _availableProviders.Add(cpuProvider);
        _logger.LogInformation("CPU backend provider initialized");

        // Select primary provider based on priority
        _primaryProvider = SelectPrimaryProvider();

        _logger.LogInformation(
            "Backend provider initialization complete. {Count} providers available, primary: {Primary}",
            _availableProviders.Count,
            _primaryProvider?.Name ?? "None");
    }

    /// <summary>
    /// Attempts to initialize DotCompute unified backend provider.
    /// </summary>
    /// <returns>true if initialization was successful; otherwise, false.</returns>
    private bool TryInitializeDotCompute()
    {
        try
        {
            var dotComputeAdapter = new DotComputeBackendAdapter(_serviceProvider, _logger);
            if (dotComputeAdapter.Initialize())
            {
                _availableProviders.Add(dotComputeAdapter);
                return true;
            }
        }
        catch (Exception ex)
        {
            _logger.LogDebug(ex, "Failed to initialize DotCompute backend");
        }
        return false;
    }

    private IBackendProvider? SelectPrimaryProvider()
    {
        // Priority order: CUDA > Vulkan > OpenCL > DirectCompute > Metal > CPU
        var priorityOrder = new[]
        {
            GpuBackend.CUDA,
            GpuBackend.Vulkan,
            GpuBackend.OpenCL,
            GpuBackend.DirectCompute,
            GpuBackend.Metal,
            GpuBackend.CPU
        };

        foreach (var type in priorityOrder)
        {
            var provider = _availableProviders.FirstOrDefault(p => p.Type == type);
            if (provider != null)
                return provider;
        }

        return _availableProviders.FirstOrDefault();
    }

    /// <summary>
    /// Gets the primary backend provider based on priority ordering
    /// </summary>
    /// <returns>The primary backend provider</returns>
    /// <exception cref="InvalidOperationException">Thrown when no backend provider is available</exception>
    /// <remarks>
    /// The primary provider is selected based on a priority order: CUDA > Vulkan > OpenCL > DirectCompute > Metal > CPU.
    /// This method must be called after <see cref="Initialize"/> has been executed.
    /// </remarks>
    public IBackendProvider? GetPrimaryProvider()
    {
        if (_primaryProvider == null)
            throw new InvalidOperationException("No backend provider available");

        return _primaryProvider;
    }

    /// <summary>
    /// Gets a specific backend provider by type
    /// </summary>
    /// <param name="type">The backend type to retrieve</param>
    /// <returns>The backend provider for the specified type, or null if not available</returns>
    /// <remarks>
    /// This method allows direct access to a specific backend provider regardless of priority ordering.
    /// Returns null if the requested backend type is not available on the current system.
    /// </remarks>
    public IBackendProvider? GetProvider(GpuBackend type)
    {
        return _availableProviders.FirstOrDefault(p => p.Type == type);
    }

    /// <summary>
    /// Gets all available backend providers
    /// </summary>
    /// <returns>A read-only list of all available backend providers</returns>
    /// <remarks>
    /// Returns all backend providers that were successfully initialized during the <see cref="Initialize"/> call.
    /// The list includes the CPU fallback provider and any GPU backends that are available on the system.
    /// </remarks>
    public IReadOnlyList<IBackendProvider> GetAvailableProviders()
    {
        return _availableProviders.AsReadOnly();
    }

}
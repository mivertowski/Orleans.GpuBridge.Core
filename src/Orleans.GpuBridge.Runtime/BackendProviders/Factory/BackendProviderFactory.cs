using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Orleans.GpuBridge.Runtime.BackendProviders.Enums;
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
    /// This method performs runtime detection of available GPU backends by attempting to load
    /// their respective runtime libraries. It creates provider instances for each detected backend
    /// and establishes a priority order for backend selection. CPU fallback is always added as
    /// the last option to ensure functionality when no GPU backends are available.
    /// </remarks>
    public void Initialize()
    {
        _logger.LogInformation("Detecting available GPU backend providers");

        // Detect CUDA
        if (TryLoadCuda())
        {
            var cudaProvider = new CudaBackendProvider(_serviceProvider, _logger);
            if (cudaProvider.Initialize())
            {
                _availableProviders.Add(cudaProvider);
                _logger.LogInformation("CUDA backend provider initialized");
            }
        }

        // Detect OpenCL
        if (TryLoadOpenCL())
        {
            var openClProvider = new OpenCLBackendProvider(_serviceProvider, _logger);
            if (openClProvider.Initialize())
            {
                _availableProviders.Add(openClProvider);
                _logger.LogInformation("OpenCL backend provider initialized");
            }
        }

        // Detect DirectCompute (Windows)
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows) && TryLoadDirectCompute())
        {
            var directComputeProvider = new DirectComputeBackendProvider(_serviceProvider, _logger);
            if (directComputeProvider.Initialize())
            {
                _availableProviders.Add(directComputeProvider);
                _logger.LogInformation("DirectCompute backend provider initialized");
            }
        }

        // Detect Metal (macOS)
        if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX) && TryLoadMetal())
        {
            var metalProvider = new MetalBackendProvider(_serviceProvider, _logger);
            if (metalProvider.Initialize())
            {
                _availableProviders.Add(metalProvider);
                _logger.LogInformation("Metal backend provider initialized");
            }
        }

        // Detect Vulkan
        if (TryLoadVulkan())
        {
            var vulkanProvider = new VulkanBackendProvider(_serviceProvider, _logger);
            if (vulkanProvider.Initialize())
            {
                _availableProviders.Add(vulkanProvider);
                _logger.LogInformation("Vulkan backend provider initialized");
            }
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

    private IBackendProvider? SelectPrimaryProvider()
    {
        // Priority order: CUDA > Vulkan > OpenCL > DirectCompute > Metal > CPU
        var priorityOrder = new[]
        {
            BackendType.Cuda,
            BackendType.Vulkan,
            BackendType.OpenCL,
            BackendType.DirectCompute,
            BackendType.Metal,
            BackendType.Cpu
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
    public IBackendProvider GetPrimaryProvider()
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
    public IBackendProvider? GetProvider(BackendType type)
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

    private bool TryLoadCuda()
    {
        try
        {
            // Try to load CUDA runtime library
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                return NativeLibrary.TryLoad("cudart64_12.dll", out _) ||
                       NativeLibrary.TryLoad("cudart64_11.dll", out _) ||
                       NativeLibrary.TryLoad("cudart64_10.dll", out _);
            }
            else if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
            {
                return NativeLibrary.TryLoad("libcudart.so.12", out _) ||
                       NativeLibrary.TryLoad("libcudart.so.11", out _) ||
                       NativeLibrary.TryLoad("libcudart.so.10", out _) ||
                       NativeLibrary.TryLoad("libcudart.so", out _);
            }
            else if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
            {
                return NativeLibrary.TryLoad("libcudart.dylib", out _);
            }
        }
        catch (Exception ex)
        {
            _logger.LogDebug(ex, "Failed to load CUDA runtime");
        }
        return false;
    }

    private bool TryLoadOpenCL()
    {
        try
        {
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                return NativeLibrary.TryLoad("OpenCL.dll", out _);
            }
            else if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
            {
                return NativeLibrary.TryLoad("libOpenCL.so.1", out _) ||
                       NativeLibrary.TryLoad("libOpenCL.so", out _);
            }
            else if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
            {
                // OpenCL is built into macOS
                return true;
            }
        }
        catch (Exception ex)
        {
            _logger.LogDebug(ex, "Failed to load OpenCL runtime");
        }
        return false;
    }

    private bool TryLoadDirectCompute()
    {
        try
        {
            // DirectCompute is part of DirectX 11/12
            return NativeLibrary.TryLoad("d3d11.dll", out _) ||
                   NativeLibrary.TryLoad("d3d12.dll", out _);
        }
        catch (Exception ex)
        {
            _logger.LogDebug(ex, "Failed to load DirectCompute");
        }
        return false;
    }

    private bool TryLoadMetal()
    {
        try
        {
            // Metal is built into macOS
            return RuntimeInformation.IsOSPlatform(OSPlatform.OSX);
        }
        catch (Exception ex)
        {
            _logger.LogDebug(ex, "Failed to detect Metal");
        }
        return false;
    }

    private bool TryLoadVulkan()
    {
        try
        {
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                return NativeLibrary.TryLoad("vulkan-1.dll", out _);
            }
            else if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
            {
                return NativeLibrary.TryLoad("libvulkan.so.1", out _) ||
                       NativeLibrary.TryLoad("libvulkan.so", out _);
            }
            else if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
            {
                return NativeLibrary.TryLoad("libvulkan.dylib", out _) ||
                       NativeLibrary.TryLoad("libMoltenVK.dylib", out _);
            }
        }
        catch (Exception ex)
        {
            _logger.LogDebug(ex, "Failed to load Vulkan runtime");
        }
        return false;
    }
}
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Orleans.GpuBridge.Abstractions;

namespace Orleans.GpuBridge.Runtime.BackendProviders;

/// <summary>
/// Factory for creating GPU backend providers based on runtime detection
/// </summary>
public sealed class BackendProviderFactory
{
    private readonly IServiceProvider _serviceProvider;
    private readonly ILogger<BackendProviderFactory> _logger;
    private readonly List<IBackendProvider> _availableProviders;
    private IBackendProvider? _primaryProvider;

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

    public IBackendProvider GetPrimaryProvider()
    {
        if (_primaryProvider == null)
            throw new InvalidOperationException("No backend provider available");
        
        return _primaryProvider;
    }

    public IBackendProvider? GetProvider(BackendType type)
    {
        return _availableProviders.FirstOrDefault(p => p.Type == type);
    }

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

/// <summary>
/// Backend provider interface
/// </summary>
public interface IBackendProvider
{
    string Name { get; }
    BackendType Type { get; }
    bool IsAvailable { get; }
    int DeviceCount { get; }
    
    bool Initialize();
    void Shutdown();
    IComputeContext CreateContext(int deviceIndex = 0);
    IReadOnlyList<DeviceInfo> GetDevices();
}

/// <summary>
/// Backend type enumeration
/// </summary>
public enum BackendType
{
    Cpu,
    Cuda,
    OpenCL,
    DirectCompute,
    Metal,
    Vulkan
}

/// <summary>
/// Device information
/// </summary>
public sealed record DeviceInfo(
    int Index,
    string Name,
    BackendType Backend,
    long TotalMemory,
    int ComputeUnits,
    string[] Extensions);

/// <summary>
/// Compute context interface
/// </summary>
public interface IComputeContext : IDisposable
{
    BackendType Backend { get; }
    int DeviceIndex { get; }
    
    IComputeBuffer<T> CreateBuffer<T>(int size, BufferUsage usage) where T : unmanaged;
    IComputeKernel CompileKernel(string source, string entryPoint);
    void Execute(IComputeKernel kernel, int workSize);
    void Synchronize();
}

/// <summary>
/// Compute buffer interface
/// </summary>
public interface IComputeBuffer<T> : IDisposable where T : unmanaged
{
    int Size { get; }
    BufferUsage Usage { get; }
    
    void Write(ReadOnlySpan<T> data);
    void Read(Span<T> data);
    void CopyTo(IComputeBuffer<T> destination);
}

/// <summary>
/// Compute kernel interface
/// </summary>
public interface IComputeKernel : IDisposable
{
    string Name { get; }
    
    void SetArgument(int index, IComputeBuffer<float> buffer);
    void SetArgument(int index, IComputeBuffer<double> buffer);
    void SetArgument(int index, IComputeBuffer<int> buffer);
    void SetArgument(int index, float value);
    void SetArgument(int index, double value);
    void SetArgument(int index, int value);
}
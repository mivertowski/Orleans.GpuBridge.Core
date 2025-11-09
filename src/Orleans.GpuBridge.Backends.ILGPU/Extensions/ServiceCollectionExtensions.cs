using Microsoft.Extensions.DependencyInjection;
using Orleans.GpuBridge.Abstractions.Enums;
using Orleans.GpuBridge.Abstractions.Providers;
using Orleans.GpuBridge.Runtime.Builders;
using ILGPU;
using ILGPU.Runtime;

namespace Orleans.GpuBridge.Backends.ILGPU.Extensions;

/// <summary>
/// Extension methods for adding ILGPU backend provider
/// </summary>
public static class ServiceCollectionExtensions
{
    /// <summary>
    /// Adds the ILGPU backend provider to the GPU Bridge
    /// </summary>
    public static IGpuBridgeBuilder AddILGPUBackend(this IGpuBridgeBuilder builder)
    {
        return builder.AddBackendProvider<ILGPUBackendProvider>();
    }

    /// <summary>
    /// Adds the ILGPU backend provider with configuration
    /// </summary>
    public static IGpuBridgeBuilder AddILGPUBackend(
        this IGpuBridgeBuilder builder,
        Action<ILGPUBackendConfiguration> configure)
    {
        builder.Services.Configure(configure);
        return builder.AddBackendProvider<ILGPUBackendProvider>();
    }

    /// <summary>
    /// Adds the ILGPU backend provider with a custom factory
    /// </summary>
    public static IGpuBridgeBuilder AddILGPUBackend(
        this IGpuBridgeBuilder builder,
        Func<IServiceProvider, ILGPUBackendProvider> factory)
    {
        return builder.AddBackendProvider(factory);
    }
}

/// <summary>
/// Configuration options for ILGPU backend
/// </summary>
public class ILGPUBackendConfiguration
{
    /// <summary>
    /// Enable debug mode for ILGPU kernels
    /// </summary>
    public bool EnableDebugMode { get; set; } = false;

    /// <summary>
    /// Enable aggressive optimizations
    /// </summary>
    public bool EnableOptimizations { get; set; } = true;

    /// <summary>
    /// Maximum number of concurrent kernel compilations
    /// </summary>
    public int MaxConcurrentCompilations { get; set; } = Environment.ProcessorCount;

    /// <summary>
    /// Cache compiled kernels to disk
    /// </summary>
    public bool EnableDiskCache { get; set; } = true;

    /// <summary>
    /// Directory for caching compiled kernels
    /// </summary>
    public string? CacheDirectory { get; set; }

    /// <summary>
    /// Prefer specific accelerator types
    /// </summary>
    public List<AcceleratorType> PreferredAcceleratorTypes { get; set; } = new()
    {
        AcceleratorType.CUDA,
        AcceleratorType.OpenCL,
        AcceleratorType.CPU
    };

    /// <summary>
    /// Device selection filter
    /// </summary>
    public Func<Accelerator, bool>? DeviceFilter { get; set; }

    /// <summary>
    /// Custom memory allocator settings
    /// </summary>
    public ILGPUMemorySettings MemorySettings { get; set; } = new();
}

/// <summary>
/// Memory-specific settings for ILGPU
/// </summary>
public class ILGPUMemorySettings
{
    /// <summary>
    /// Enable memory pooling
    /// </summary>
    public bool EnableMemoryPooling { get; set; } = true;

    /// <summary>
    /// Initial memory pool size per device (in bytes)
    /// </summary>
    public long InitialPoolSize { get; set; } = 256 * 1024 * 1024; // 256 MB

    /// <summary>
    /// Maximum memory pool size per device (in bytes)
    /// </summary>
    public long MaxPoolSize { get; set; } = 2L * 1024 * 1024 * 1024; // 2 GB

    /// <summary>
    /// Memory allocation alignment (in bytes)
    /// </summary>
    public int AllocationAlignment { get; set; } = 256;

    /// <summary>
    /// Enable automatic defragmentation
    /// </summary>
    public bool EnableDefragmentation { get; set; } = true;

    /// <summary>
    /// Defragmentation threshold (percentage of fragmented memory)
    /// </summary>
    public double DefragmentationThreshold { get; set; } = 0.3; // 30%
}
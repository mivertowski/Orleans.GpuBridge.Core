using Microsoft.Extensions.DependencyInjection;
using Orleans.GpuBridge.Abstractions.Providers;
using Orleans.GpuBridge.Runtime.Builders;

namespace Orleans.GpuBridge.Backends.DotCompute.Extensions;

/// <summary>
/// Extension methods for adding DotCompute backend provider
/// </summary>
public static class ServiceCollectionExtensions
{
    /// <summary>
    /// Adds the DotCompute backend provider to the GPU Bridge
    /// </summary>
    public static IGpuBridgeBuilder AddDotComputeBackend(this IGpuBridgeBuilder builder)
    {
        return builder.AddBackendProvider<DotComputeBackendProvider>();
    }

    /// <summary>
    /// Adds the DotCompute backend provider with configuration
    /// </summary>
    public static IGpuBridgeBuilder AddDotComputeBackend(
        this IGpuBridgeBuilder builder,
        Action<DotComputeBackendConfiguration> configure)
    {
        builder.Services.Configure(configure);
        return builder.AddBackendProvider<DotComputeBackendProvider>();
    }

    /// <summary>
    /// Adds the DotCompute backend provider with a custom factory
    /// </summary>
    public static IGpuBridgeBuilder AddDotComputeBackend(
        this IGpuBridgeBuilder builder,
        Func<IServiceProvider, DotComputeBackendProvider> factory)
    {
        return builder.AddBackendProvider(factory);
    }
}

/// <summary>
/// Configuration options for DotCompute backend
/// </summary>
public class DotComputeBackendConfiguration
{
    /// <summary>
    /// Enable debug mode for DotCompute kernels
    /// </summary>
    public bool EnableDebugMode { get; set; } = false;

    /// <summary>
    /// Enable aggressive optimizations
    /// </summary>
    public bool EnableOptimizations { get; set; } = true;

    /// <summary>
    /// Optimization level for kernel compilation
    /// </summary>
    public OptimizationLevel OptimizationLevel { get; set; } = OptimizationLevel.Release;

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
    /// Prefer specific compute platforms
    /// </summary>
    public List<ComputePlatform> PreferredPlatforms { get; set; } = new()
    {
        ComputePlatform.Cuda,
        ComputePlatform.OpenCL,
        ComputePlatform.DirectCompute,
        ComputePlatform.Metal,
        ComputePlatform.Vulkan
    };

    /// <summary>
    /// Device selection filter
    /// </summary>
    public Func<object, bool>? DeviceFilter { get; set; } // object would be DotCompute device type

    /// <summary>
    /// Custom memory allocator settings
    /// </summary>
    public DotComputeMemorySettings MemorySettings { get; set; } = new();

    /// <summary>
    /// Kernel language preferences
    /// </summary>
    public KernelLanguageSettings LanguageSettings { get; set; } = new();

    /// <summary>
    /// Platform-specific settings
    /// </summary>
    public Dictionary<ComputePlatform, object> PlatformSpecificSettings { get; set; } = new();
}

/// <summary>
/// Memory-specific settings for DotCompute
/// </summary>
public class DotComputeMemorySettings
{
    /// <summary>
    /// Enable memory pooling
    /// </summary>
    public bool EnableMemoryPooling { get; set; } = true;

    /// <summary>
    /// Initial memory pool size per device (in bytes)
    /// </summary>
    public long InitialPoolSize { get; set; } = 512 * 1024 * 1024; // 512 MB

    /// <summary>
    /// Maximum memory pool size per device (in bytes)
    /// </summary>
    public long MaxPoolSize { get; set; } = 4L * 1024 * 1024 * 1024; // 4 GB

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
    public double DefragmentationThreshold { get; set; } = 0.25; // 25%

    /// <summary>
    /// Enable unified memory when available
    /// </summary>
    public bool PreferUnifiedMemory { get; set; } = true;

    /// <summary>
    /// Enable pinned memory for transfers
    /// </summary>
    public bool UsePinnedMemory { get; set; } = true;
}

/// <summary>
/// Kernel language configuration for DotCompute
/// </summary>
public class KernelLanguageSettings
{
    /// <summary>
    /// Preferred kernel language for each platform
    /// </summary>
    public Dictionary<ComputePlatform, KernelLanguage> PreferredLanguages { get; set; } = new()
    {
        { ComputePlatform.Cuda, KernelLanguage.CUDA },
        { ComputePlatform.OpenCL, KernelLanguage.OpenCL },
        { ComputePlatform.DirectCompute, KernelLanguage.HLSL },
        { ComputePlatform.Metal, KernelLanguage.MSL },
        { ComputePlatform.Vulkan, KernelLanguage.HLSL }
    };

    /// <summary>
    /// Enable automatic language translation
    /// </summary>
    public bool EnableLanguageTranslation { get; set; } = true;

    /// <summary>
    /// Custom preprocessor definitions
    /// </summary>
    public Dictionary<string, string> PreprocessorDefines { get; set; } = new();

    /// <summary>
    /// Include directories for kernel compilation
    /// </summary>
    public List<string> IncludeDirectories { get; set; } = new();
}

/// <summary>
/// Compute platform enumeration for DotCompute
/// </summary>
public enum ComputePlatform
{
    Cuda,
    OpenCL,
    DirectCompute,
    Metal,
    Vulkan
}

/// <summary>
/// Kernel language enumeration
/// </summary>
public enum KernelLanguage
{
    CSharp,
    CUDA,
    OpenCL,
    HLSL,
    MSL
}

/// <summary>
/// Optimization level for kernel compilation
/// </summary>
public enum OptimizationLevel
{
    Debug,
    Release,
    Aggressive
}
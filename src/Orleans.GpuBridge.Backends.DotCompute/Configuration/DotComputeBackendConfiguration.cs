using Orleans.GpuBridge.Abstractions.Enums;
using Orleans.GpuBridge.Abstractions.Enums.Compilation;

namespace Orleans.GpuBridge.Backends.DotCompute.Configuration;

/// <summary>
/// Configuration options for the DotCompute backend provider.
/// </summary>
/// <remarks>
/// This class provides comprehensive configuration options for the DotCompute GPU backend,
/// allowing fine-tuned control over compilation, execution, memory management, and platform
/// preferences. The configuration supports multiple GPU vendors and compute platforms.
/// 
/// <para>
/// The DotCompute backend serves as a unified abstraction over multiple GPU compute APIs
/// including CUDA, OpenCL, DirectCompute, Metal, and Vulkan. Configuration settings allow
/// you to optimize for specific hardware configurations and use cases.
/// </para>
/// 
/// <para>
/// Default settings are optimized for production use with balanced performance and resource
/// usage. Adjust specific settings based on your deployment environment and performance
/// requirements.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Basic configuration for production use
/// services.AddGpuBridge(options => options.PreferGpu = true)
///         .AddDotGpuBackend(config =>
///         {
///             config.EnableOptimizations = true;
///             config.OptimizationLevel = OptimizationLevel.O2;
///             config.PreferredPlatforms.Clear();
///             config.PreferredPlatforms.Add(GpuBackend.CUDA);
///             config.PreferredPlatforms.Add(GpuBackend.OpenCL);
///         });
/// 
/// // Debug configuration for development
/// services.AddGpuBridge(options => options.PreferGpu = true)
///         .AddDotGpuBackend(config =>
///         {
///             config.EnableDebugMode = true;
///             config.EnableOptimizations = false;
///             config.OptimizationLevel = OptimizationLevel.O0;
///             config.EnableDiskCache = false;
///         });
/// 
/// // High-performance configuration for compute-intensive workloads
/// services.AddGpuBridge(options => options.PreferGpu = true)
///         .AddDotGpuBackend(config =>
///         {
///             config.OptimizationLevel = OptimizationLevel.O3;
///             config.MaxConcurrentCompilations = Environment.ProcessorCount * 2;
///             config.MemorySettings.InitialPoolSize = 1024 * 1024 * 1024; // 1 GB
///             config.MemorySettings.MaxPoolSize = 8L * 1024 * 1024 * 1024; // 8 GB
///         });
/// </code>
/// </example>
public class DotGpuBackendConfiguration
{
    /// <summary>
    /// Gets or sets whether to enable debug mode for DotCompute kernel compilation and execution.
    /// </summary>
    /// <value>
    /// <c>true</c> to enable debug mode; otherwise, <c>false</c>. Default is <c>false</c>.
    /// </value>
    /// <remarks>
    /// Debug mode enables additional runtime checks, detailed error reporting, and debugging
    /// symbols in compiled kernels. This significantly impacts performance and should only
    /// be used during development.
    /// 
    /// <para>
    /// When enabled, debug mode provides:
    /// - Detailed kernel compilation errors and warnings
    /// - Runtime bounds checking and validation
    /// - Debugging symbols for GPU debuggers
    /// - Verbose logging of GPU operations
    /// - Slower kernel execution due to additional checks
    /// </para>
    /// 
    /// <para>
    /// Recommended settings:
    /// - Development: <c>true</c>
    /// - Staging: <c>false</c>
    /// - Production: <c>false</c>
    /// </para>
    /// </remarks>
    public bool EnableDebugMode { get; set; } = false;

    /// <summary>
    /// Gets or sets whether to enable compiler optimizations for kernel compilation.
    /// </summary>
    /// <value>
    /// <c>true</c> to enable optimizations; otherwise, <c>false</c>. Default is <c>true</c>.
    /// </value>
    /// <remarks>
    /// This setting controls whether the DotCompute compiler applies general optimizations
    /// during kernel compilation. Works in conjunction with <see cref="OptimizationLevel"/>
    /// to control the specific level of optimization applied.
    /// 
    /// <para>
    /// When enabled, optimizations can include:
    /// - Dead code elimination
    /// - Constant folding and propagation
    /// - Loop optimizations and unrolling
    /// - Instruction scheduling and reordering
    /// - Memory access pattern optimization
    /// </para>
    /// 
    /// <para>
    /// Disabling optimizations is primarily useful for debugging scenarios where you need
    /// predictable code generation and line-by-line correspondence with source code.
    /// </para>
    /// </remarks>
    public bool EnableOptimizations { get; set; } = true;

    /// <summary>
    /// Gets or sets the optimization level for kernel compilation.
    /// </summary>
    /// <value>
    /// An <see cref="Abstractions.Enums.Compilation.OptimizationLevel"/> value specifying the
    /// optimization level. Default is <see cref="Abstractions.Enums.Compilation.OptimizationLevel.O2"/>.
    /// </value>
    /// <remarks>
    /// The optimization level controls the trade-off between compilation time and runtime
    /// performance. Higher levels generally produce faster code but take longer to compile.
    /// 
    /// <para>
    /// Recommended levels by use case:
    /// - Development and debugging: <see cref="Abstractions.Enums.Compilation.OptimizationLevel.O0"/> or <see cref="Abstractions.Enums.Compilation.OptimizationLevel.O1"/>
    /// - Production deployments: <see cref="Abstractions.Enums.Compilation.OptimizationLevel.O2"/>
    /// - Performance-critical applications: <see cref="Abstractions.Enums.Compilation.OptimizationLevel.O3"/>
    /// </para>
    /// 
    /// <para>
    /// This setting is only effective when <see cref="EnableOptimizations"/> is <c>true</c>.
    /// </para>
    /// </remarks>
    public OptimizationLevel OptimizationLevel { get; set; } = OptimizationLevel.O2;

    /// <summary>
    /// Gets or sets the maximum number of concurrent kernel compilations allowed.
    /// </summary>
    /// <value>
    /// The maximum number of concurrent compilations. Default is the processor count.
    /// Must be a positive integer.
    /// </value>
    /// <remarks>
    /// This setting controls the parallelism level for kernel compilation operations.
    /// Higher values can reduce overall compilation time when compiling multiple kernels
    /// simultaneously, but may increase memory usage and system load.
    /// 
    /// <para>
    /// Considerations for setting this value:
    /// - CPU cores: Generally set to processor count or 2x processor count
    /// - Memory: Each compilation uses significant memory (100MB-1GB per compilation)
    /// - I/O: Concurrent disk operations for cache and temporary files
    /// - GPU driver limits: Some drivers have limits on concurrent compilation contexts
    /// </para>
    /// 
    /// <para>
    /// Recommended values:
    /// - Development machines: 2-4
    /// - CI/Build servers: <see cref="Environment.ProcessorCount"/>
    /// - Production servers: <see cref="Environment.ProcessorCount"/> * 2 (if memory permits)
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentOutOfRangeException">
    /// Thrown when the value is less than or equal to zero.
    /// </exception>
    public int MaxConcurrentCompilations { get; set; } = Environment.ProcessorCount;

    /// <summary>
    /// Gets or sets whether to cache compiled kernels to disk for faster subsequent loads.
    /// </summary>
    /// <value>
    /// <c>true</c> to enable disk caching; otherwise, <c>false</c>. Default is <c>true</c>.
    /// </value>
    /// <remarks>
    /// Disk caching stores compiled kernel binaries to persistent storage, significantly
    /// reducing application startup time and kernel compilation overhead on subsequent runs.
    /// 
    /// <para>
    /// Cache benefits:
    /// - Faster application startup (avoids recompilation)
    /// - Reduced CPU usage for repeated kernel compilations
    /// - Consistent performance across application restarts
    /// - Shared cache across application instances
    /// </para>
    /// 
    /// <para>
    /// Cache considerations:
    /// - Disk space usage (compiled kernels can be 1MB-100MB each)
    /// - Cache invalidation when kernel source or compiler versions change
    /// - File system permissions for cache directory access
    /// - Concurrent access from multiple application instances
    /// </para>
    /// 
    /// <para>
    /// The cache directory is specified by <see cref="CacheDirectory"/> or uses a
    /// system-appropriate default location if not specified.
    /// </para>
    /// </remarks>
    public bool EnableDiskCache { get; set; } = true;

    /// <summary>
    /// Gets or sets the directory path for caching compiled kernels.
    /// </summary>
    /// <value>
    /// The directory path for kernel cache, or <c>null</c> to use the default system cache directory.
    /// </value>
    /// <remarks>
    /// Specifies the filesystem location where compiled kernel binaries are stored for
    /// caching. If <c>null</c> or empty, the backend will use a system-appropriate default
    /// cache directory.
    /// 
    /// <para>
    /// Default cache locations by platform:
    /// - Windows: <c>%LOCALAPPDATA%\Orleans.GpuBridge\DotCompute\Cache</c>
    /// - Linux: <c>~/.cache/orleans-gpubridge/dotcompute</c>
    /// - macOS: <c>~/Library/Caches/Orleans.GpuBridge.DotCompute</c>
    /// </para>
    /// 
    /// <para>
    /// Directory requirements:
    /// - Must be writable by the application process
    /// - Should have sufficient disk space (recommend 1GB minimum)
    /// - Can be shared across application instances for efficiency
    /// - Should be on a fast storage device (SSD preferred)
    /// </para>
    /// 
    /// <para>
    /// This setting is only effective when <see cref="EnableDiskCache"/> is <c>true</c>.
    /// </para>
    /// </remarks>
    /// <example>
    /// <code>
    /// // Use custom cache directory
    /// config.CacheDirectory = "/opt/app/gpu-cache";
    /// 
    /// // Use default system directory
    /// config.CacheDirectory = null;
    /// </code>
    /// </example>
    public string? CacheDirectory { get; set; }

    /// <summary>
    /// Gets or sets the list of preferred GPU compute platforms in priority order.
    /// </summary>
    /// <value>
    /// A list of <see cref="GpuBackend"/> values representing preferred platforms.
    /// Default includes all supported platforms in optimal order.
    /// </value>
    /// <remarks>
    /// This list defines the platform selection priority when multiple GPU backends are
    /// available on the system. The DotCompute backend will attempt to use platforms
    /// in the specified order until a compatible one is found.
    /// 
    /// <para>
    /// Platform characteristics:
    /// - <see cref="GpuBackend.CUDA"/>: NVIDIA GPUs only, excellent performance
    /// - <see cref="GpuBackend.OpenCL"/>: Cross-vendor compatibility, good portability
    /// - <see cref="GpuBackend.DirectCompute"/>: Windows DirectX, good Windows integration
    /// - <see cref="GpuBackend.Metal"/>: Apple platforms only, optimized for Apple hardware
    /// - <see cref="GpuBackend.Vulkan"/>: Modern cross-platform API, excellent performance
    /// </para>
    /// 
    /// <para>
    /// Selection considerations:
    /// - Hardware availability (vendor-specific platforms)
    /// - Performance requirements (vendor-specific vs. cross-platform)
    /// - Portability needs (single platform vs. multiple platforms)
    /// - Development tools and debugging support
    /// </para>
    /// </remarks>
    /// <example>
    /// <code>
    /// // NVIDIA-first configuration
    /// config.PreferredPlatforms = new List&lt;GpuBackend&gt;
    /// {
    ///     GpuBackend.CUDA,
    ///     GpuBackend.Vulkan,
    ///     GpuBackend.OpenCL
    /// };
    /// 
    /// // Cross-platform configuration
    /// config.PreferredPlatforms = new List&lt;GpuBackend&gt;
    /// {
    ///     GpuBackend.OpenCL,
    ///     GpuBackend.Vulkan
    /// };
    /// </code>
    /// </example>
    public List<GpuBackend> PreferredPlatforms { get; set; } = new()
    {
        GpuBackend.CUDA,
        GpuBackend.OpenCL,
        GpuBackend.DirectCompute,
        GpuBackend.Metal,
        GpuBackend.Vulkan
    };

    /// <summary>
    /// Gets or sets the device selection filter function for choosing specific GPU devices.
    /// </summary>
    /// <value>
    /// A function that takes a device object and returns <c>true</c> if the device should be used,
    /// or <c>null</c> to use all available devices.
    /// </value>
    /// <remarks>
    /// This filter allows fine-grained control over which GPU devices are selected for
    /// kernel execution. The function receives a platform-specific device object and
    /// should return <c>true</c> for devices that meet your criteria.
    /// 
    /// <para>
    /// The device object type depends on the underlying platform:
    /// - CUDA: CudaDevice with properties like ComputeCapability, TotalMemory
    /// - OpenCL: OpenCLDevice with properties like DeviceType, MaxWorkGroupSize
    /// - DirectCompute: DirectComputeDevice with DirectX-specific properties
    /// - Metal: MetalDevice with Apple-specific properties
    /// - Vulkan: VulkanDevice with Vulkan API properties
    /// </para>
    /// 
    /// <para>
    /// Common filtering criteria:
    /// - Memory capacity (minimum required GPU memory)
    /// - Compute capability (minimum hardware feature support)
    /// - Device type (discrete vs. integrated GPUs)
    /// - Vendor preference (specific GPU manufacturers)
    /// - Performance characteristics (memory bandwidth, compute units)
    /// </para>
    /// </remarks>
    /// <example>
    /// <code>
    /// // Filter for high-memory devices only
    /// config.DeviceFilter = device =>
    /// {
    ///     return device switch
    ///     {
    ///         CudaDevice cuda => cuda.TotalMemory > 8L * 1024 * 1024 * 1024, // 8GB+
    ///         OpenCLDevice opencl => opencl.GlobalMemorySize > 8L * 1024 * 1024 * 1024,
    ///         _ => true // Accept other device types
    ///     };
    /// };
    /// 
    /// // Filter for discrete GPUs only
    /// config.DeviceFilter = device =>
    /// {
    ///     return device switch
    ///     {
    ///         OpenCLDevice opencl => opencl.DeviceType == DeviceType.GPU,
    ///         _ => true
    ///     };
    /// };
    /// </code>
    /// </example>
    public Func<object, bool>? DeviceFilter { get; set; }

    /// <summary>
    /// Gets or sets the memory management settings for the DotCompute backend.
    /// </summary>
    /// <value>
    /// A <see cref="DotComputeMemorySettings"/> instance containing memory configuration.
    /// Never <c>null</c>.
    /// </value>
    /// <remarks>
    /// These settings control how the DotCompute backend manages GPU memory allocation,
    /// pooling, and optimization. Proper memory configuration is crucial for achieving
    /// optimal performance in GPU compute workloads.
    /// 
    /// <para>
    /// Memory management affects:
    /// - Allocation performance and fragmentation
    /// - Memory usage efficiency and peak consumption
    /// - Transfer performance between CPU and GPU
    /// - Multi-kernel memory sharing and reuse
    /// </para>
    /// 
    /// <para>
    /// The default settings are optimized for typical workloads, but may need adjustment
    /// based on your specific memory usage patterns and hardware configuration.
    /// </para>
    /// </remarks>
    public DotComputeMemorySettings MemorySettings { get; set; } = new();

    /// <summary>
    /// Gets or sets the kernel language compilation settings.
    /// </summary>
    /// <value>
    /// A <see cref="KernelLanguageSettings"/> instance containing language configuration.
    /// Never <c>null</c>.
    /// </value>
    /// <remarks>
    /// These settings control how the DotCompute backend handles different kernel
    /// programming languages and their compilation. The backend supports multiple
    /// source languages and can automatically translate between them when needed.
    /// 
    /// <para>
    /// Language settings affect:
    /// - Source language preference for each platform
    /// - Automatic language translation capabilities
    /// - Preprocessor definitions and include paths
    /// - Compilation flags and optimization settings
    /// </para>
    /// 
    /// <para>
    /// The default settings provide good cross-platform compatibility while optimizing
    /// for platform-specific languages when available.
    /// </para>
    /// </remarks>
    public KernelLanguageSettings LanguageSettings { get; set; } = new();

    /// <summary>
    /// Gets or sets platform-specific configuration settings.
    /// </summary>
    /// <value>
    /// A dictionary mapping <see cref="GpuBackend"/> values to platform-specific configuration objects.
    /// </value>
    /// <remarks>
    /// This dictionary allows you to specify platform-specific configuration that cannot
    /// be expressed through the common configuration properties. Each platform may have
    /// unique settings that are not applicable to other platforms.
    /// 
    /// <para>
    /// Platform-specific settings examples:
    /// - CUDA: Stream priorities, memory pool configurations, P2P settings
    /// - OpenCL: Context properties, queue properties, extension usage
    /// - DirectCompute: Device debug layer settings, feature level requirements
    /// - Metal: Command queue configurations, heap settings
    /// - Vulkan: Instance extensions, device extensions, queue families
    /// </para>
    /// 
    /// <para>
    /// The configuration objects are platform-specific and should match the expected
    /// types for each backend. Consult the DotCompute documentation for detailed
    /// information about supported platform-specific options.
    /// </para>
    /// </remarks>
    /// <example>
    /// <code>
    /// // CUDA-specific settings
    /// config.PlatformSpecificSettings[GpuBackend.CUDA] = new CudaSettings
    /// {
    ///     EnableP2P = true,
    ///     StreamPriority = CudaStreamPriority.High
    /// };
    /// 
    /// // OpenCL-specific settings
    /// config.PlatformSpecificSettings[GpuBackend.OpenCL] = new OpenCLSettings
    /// {
    ///     EnableImageSupport = true,
    ///     PreferredWorkGroupSize = 256
    /// };
    /// </code>
    /// </example>
    public Dictionary<GpuBackend, object> PlatformSpecificSettings { get; set; } = new();
}
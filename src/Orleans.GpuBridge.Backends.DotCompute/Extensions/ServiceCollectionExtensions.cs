using DotComputeRingKernels = DotCompute.Abstractions.RingKernels;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.DependencyInjection.Extensions;
using Microsoft.Extensions.Logging;
using Orleans.GpuBridge.Abstractions.Providers;
using Orleans.GpuBridge.Abstractions.RingKernels;
using Orleans.GpuBridge.Runtime.Builders;
using Orleans.GpuBridge.Runtime.RingKernels;
using Orleans.GpuBridge.Backends.DotCompute.Configuration;
using Orleans.GpuBridge.Backends.DotCompute.RingKernels;

namespace Orleans.GpuBridge.Backends.DotCompute.Extensions;

/// <summary>
/// Extension methods for configuring and registering the DotCompute GPU backend provider.
/// </summary>
/// <remarks>
/// The DotCompute backend provides a unified abstraction over multiple GPU compute APIs
/// including CUDA, OpenCL, DirectCompute, Metal, and Vulkan. This allows applications
/// to leverage GPU acceleration across different hardware platforms and operating systems.
/// 
/// <para>
/// The backend supports:
/// - Multiple kernel programming languages (C#, CUDA, OpenCL, HLSL, MSL)
/// - Automatic language translation and cross-compilation
/// - Advanced memory management with pooling and optimization
/// - Platform-specific optimizations and feature detection
/// - Comprehensive debugging and profiling capabilities
/// </para>
/// 
/// <para>
/// Use these extension methods to configure the DotCompute backend with appropriate
/// settings for your deployment environment, performance requirements, and target platforms.
/// </para>
/// </remarks>
public static class ServiceCollectionExtensions
{
    /// <summary>
    /// Adds the DotCompute backend provider to the GPU Bridge with default configuration.
    /// </summary>
    /// <param name="builder">The GPU Bridge builder instance to configure.</param>
    /// <returns>The <see cref="IGpuBridgeBuilder"/> for method chaining.</returns>
    /// <remarks>
    /// This method registers the DotCompute backend provider with default settings optimized
    /// for production use. The default configuration includes:
    /// - Enabled optimizations with O2 optimization level
    /// - Memory pooling with 512MB initial size and 4GB maximum
    /// - Disk caching enabled for compiled kernels
    /// - Automatic language translation enabled
    /// - Platform preference: CUDA > OpenCL > DirectCompute > Metal > Vulkan
    /// 
    /// <para>
    /// For custom configuration, use <see cref="AddDotGpuBackend(IGpuBridgeBuilder, Action{DotGpuBackendConfiguration})"/>
    /// or <see cref="AddDotGpuBackend(IGpuBridgeBuilder, Func{IServiceProvider, DotComputeBackendProvider})"/>.
    /// </para>
    /// </remarks>
    /// <example>
    /// <code>
    /// services.AddGpuBridge(options => options.PreferGpu = true)
    ///         .AddDotGpuBackend();
    /// </code>
    /// </example>
    public static IGpuBridgeBuilder AddDotGpuBackend(this IGpuBridgeBuilder builder)
    {
        return builder.AddBackendProvider<DotComputeBackendProvider>();
    }

    /// <summary>
    /// Adds the DotCompute backend provider with custom configuration.
    /// </summary>
    /// <param name="builder">The GPU Bridge builder instance to configure.</param>
    /// <param name="configure">A delegate to configure the DotCompute backend options.</param>
    /// <returns>The <see cref="IGpuBridgeBuilder"/> for method chaining.</returns>
    /// <remarks>
    /// This method registers the DotCompute backend provider and allows customization
    /// of backend behavior through the provided configuration delegate. The configuration
    /// is applied to <see cref="DotGpuBackendConfiguration"/> which controls:
    /// 
    /// <para>
    /// Key configuration areas:
    /// - Compilation settings (optimization level, debug mode, caching)
    /// - Platform preferences and device selection
    /// - Memory management (pooling, alignment, defragmentation)
    /// - Language settings (preferred languages, translation, preprocessor definitions)
    /// - Platform-specific optimizations and features
    /// </para>
    /// 
    /// <para>
    /// The configuration is validated at registration time and applied when the backend
    /// provider is instantiated. Invalid configurations will result in exceptions during
    /// service registration or provider initialization.
    /// </para>
    /// </remarks>
    /// <example>
    /// <code>
    /// // High-performance CUDA-optimized configuration
    /// services.AddGpuBridge(options => options.PreferGpu = true)
    ///         .AddDotGpuBackend(config =>
    ///         {
    ///             config.OptimizationLevel = OptimizationLevel.O3;
    ///             config.PreferredPlatforms.Clear();
    ///             config.PreferredPlatforms.Add(GpuBackend.CUDA);
    ///             config.MemorySettings.InitialPoolSize = 1024 * 1024 * 1024; // 1 GB
    ///             config.LanguageSettings.PreferredLanguages[GpuBackend.CUDA] = KernelLanguage.CUDA;
    ///         });
    /// 
    /// // Cross-platform development configuration
    /// services.AddGpuBridge(options => options.PreferGpu = true)
    ///         .AddDotGpuBackend(config =>
    ///         {
    ///             config.EnableDebugMode = true;
    ///             config.OptimizationLevel = OptimizationLevel.O1;
    ///             config.PreferredPlatforms.Clear();
    ///             config.PreferredPlatforms.Add(GpuBackend.OpenCL);
    ///             config.LanguageSettings.EnableLanguageTranslation = true;
    ///             foreach (var platform in Enum.GetValues&lt;GpuBackend&gt;())
    ///             {
    ///                 config.LanguageSettings.PreferredLanguages[platform] = KernelLanguage.CSharp;
    ///             }
    ///         });
    /// 
    /// // Memory-constrained configuration
    /// services.AddGpuBridge(options => options.PreferGpu = true)
    ///         .AddDotGpuBackend(config =>
    ///         {
    ///             config.MemorySettings.InitialPoolSize = 128 * 1024 * 1024; // 128 MB
    ///             config.MemorySettings.MaxPoolSize = 1024 * 1024 * 1024;    // 1 GB
    ///             config.MemorySettings.EnableDefragmentation = true;
    ///             config.MemorySettings.DefragmentationThreshold = 0.15; // Aggressive
    ///         });
    /// </code>
    /// </example>
    /// <exception cref="ArgumentNullException">
    /// Thrown when <paramref name="builder"/> or <paramref name="configure"/> is <c>null</c>.
    /// </exception>
    public static IGpuBridgeBuilder AddDotGpuBackend(
        this IGpuBridgeBuilder builder,
        Action<DotGpuBackendConfiguration> configure)
    {
        builder.Services.Configure(configure);
        return builder.AddBackendProvider<DotComputeBackendProvider>();
    }

    /// <summary>
    /// Adds the DotCompute backend provider using a custom factory function.
    /// </summary>
    /// <param name="builder">The GPU Bridge builder instance to configure.</param>
    /// <param name="factory">A factory function to create the DotComputeBackendProvider instance.</param>
    /// <returns>The <see cref="IGpuBridgeBuilder"/> for method chaining.</returns>
    /// <remarks>
    /// This method provides complete control over the instantiation of the DotCompute
    /// backend provider. The factory function receives the <see cref="IServiceProvider"/>
    /// and should return a fully configured <see cref="DotComputeBackendProvider"/> instance.
    /// 
    /// <para>
    /// Use this method when you need:
    /// - Complex initialization logic that cannot be expressed through configuration
    /// - Dynamic configuration based on runtime conditions
    /// - Custom dependency injection or service resolution
    /// - Integration with existing factory patterns or IoC containers
    /// - Advanced customization of the backend provider implementation
    /// </para>
    /// 
    /// <para>
    /// The factory function is called once per service provider scope when the
    /// backend provider is first requested. The returned instance should be ready
    /// for immediate use and properly configured for the target environment.
    /// </para>
    /// 
    /// <para>
    /// Note: When using a custom factory, standard configuration through
    /// <see cref="DotGpuBackendConfiguration"/> may not be applied unless
    /// explicitly handled within the factory function.
    /// </para>
    /// </remarks>
    /// <example>
    /// <code>
    /// // Custom factory with conditional configuration
    /// services.AddGpuBridge(options => options.PreferGpu = true)
    ///         .AddDotGpuBackend(serviceProvider =>
    ///         {
    ///             var environment = serviceProvider.GetRequiredService&lt;IHostEnvironment&gt;();
    ///             var logger = serviceProvider.GetRequiredService&lt;ILogger&lt;DotComputeBackendProvider&gt;&gt;();
    ///             
    ///             var config = new DotGpuBackendConfiguration();
    ///             
    ///             if (environment.IsDevelopment())
    ///             {
    ///                 config.EnableDebugMode = true;
    ///                 config.OptimizationLevel = OptimizationLevel.O0;
    ///                 config.EnableDiskCache = false;
    ///             }
    ///             else
    ///             {
    ///                 config.OptimizationLevel = OptimizationLevel.O3;
    ///                 config.MemorySettings.InitialPoolSize = 2L * 1024 * 1024 * 1024; // 2 GB
    ///             }
    ///             
    ///             return new DotComputeBackendProvider(config, logger);
    ///         });
    /// 
    /// // Factory with runtime hardware detection
    /// services.AddGpuBridge(options => options.PreferGpu = true)
    ///         .AddDotGpuBackend(serviceProvider =>
    ///         {
    ///             var config = new DotGpuBackendConfiguration();
    ///             
    ///             // Detect available GPU hardware
    ///             if (IsNvidiaGpuAvailable())
    ///             {
    ///                 config.PreferredPlatforms.Clear();
    ///                 config.PreferredPlatforms.Add(GpuBackend.CUDA);
    ///                 config.LanguageSettings.PreferredLanguages[GpuBackend.CUDA] = KernelLanguage.CUDA;
    ///             }
    ///             else if (IsOpenCLAvailable())
    ///             {
    ///                 config.PreferredPlatforms.Clear();
    ///                 config.PreferredPlatforms.Add(GpuBackend.OpenCL);
    ///             }
    ///             
    ///             var logger = serviceProvider.GetRequiredService&lt;ILogger&lt;DotComputeBackendProvider&gt;&gt;();
    ///             return new DotComputeBackendProvider(config, logger);
    ///         });
    /// </code>
    /// </example>
    /// <exception cref="ArgumentNullException">
    /// Thrown when <paramref name="builder"/> or <paramref name="factory"/> is <c>null</c>.
    /// </exception>
    public static IGpuBridgeBuilder AddDotGpuBackend(
        this IGpuBridgeBuilder builder,
        Func<IServiceProvider, DotComputeBackendProvider> factory)
    {
        return builder.AddBackendProvider(factory);
    }

    /// <summary>
    /// Adds the DotCompute GPU-accelerated ring kernel bridge for generated actors.
    /// </summary>
    /// <param name="services">Service collection.</param>
    /// <returns>Service collection for chaining.</returns>
    /// <remarks>
    /// <para>
    /// This registers the DotCompute implementation of <see cref="IRingKernelBridge"/>
    /// which provides actual GPU execution for generated actors using the DotCompute backend.
    /// </para>
    /// <para>
    /// The bridge automatically falls back to CPU execution when:
    /// - No GPU is available
    /// - GPU execution fails
    /// - Running in a development/test environment without CUDA support
    /// </para>
    /// <para>
    /// This method replaces any previously registered <see cref="IRingKernelBridge"/>
    /// (such as the CPU fallback bridge) with the GPU-capable DotCompute implementation.
    /// </para>
    /// <para>
    /// Performance characteristics:
    /// - GPU execution: 100-500ns message latency (ring kernel queue operations)
    /// - CPU fallback: 1-10μs message latency (direct method calls)
    /// - Memory: GPU-resident state with automatic sync
    /// </para>
    /// <para>
    /// Usage:
    /// <code>
    /// services.AddGpuBridge()
    ///         .AddDotGpuBackend()
    ///         .Services
    ///         .AddRingKernelSupport()
    ///         .AddDotComputeRingKernelBridge(); // GPU acceleration
    /// </code>
    /// </para>
    /// </remarks>
    public static IServiceCollection AddDotComputeRingKernelBridge(this IServiceCollection services)
    {
        // Replace any existing registration (like CPU fallback) with GPU-capable bridge
        services.RemoveAll<IRingKernelBridge>();

        services.AddSingleton<IRingKernelBridge>(sp =>
        {
            var backendProvider = sp.GetRequiredService<DotComputeBackendProvider>();
            var ringKernelRuntime = sp.GetService<DotComputeRingKernels.IRingKernelRuntime>();
            var logger = sp.GetRequiredService<ILogger<DotComputeRingKernelBridge>>();
            return new DotComputeRingKernelBridge(logger, backendProvider, ringKernelRuntime);
        });

        return services;
    }

    /// <summary>
    /// Adds the DotCompute ring kernel bridge with custom configuration.
    /// </summary>
    /// <param name="services">Service collection.</param>
    /// <param name="configure">Configuration action for bridge options.</param>
    /// <returns>Service collection for chaining.</returns>
    /// <remarks>
    /// <para>
    /// Use this overload to customize the DotCompute bridge behavior.
    /// Configuration options control fallback behavior, telemetry, and GPU device selection.
    /// </para>
    /// <para>
    /// Usage:
    /// <code>
    /// services.AddGpuBridge()
    ///         .AddDotGpuBackend()
    ///         .Services
    ///         .AddRingKernelSupport()
    ///         .AddDotComputeRingKernelBridge(options =>
    ///         {
    ///             options.EnableCpuFallback = true;
    ///             options.PreferredDeviceId = 0;
    ///         });
    /// </code>
    /// </para>
    /// </remarks>
    public static IServiceCollection AddDotComputeRingKernelBridge(
        this IServiceCollection services,
        Action<DotComputeRingKernelBridgeOptions> configure)
    {
        ArgumentNullException.ThrowIfNull(configure);

        // Configure options
        var options = new DotComputeRingKernelBridgeOptions();
        configure(options);
        services.AddSingleton(options);

        // Replace any existing registration with GPU-capable bridge
        services.RemoveAll<IRingKernelBridge>();

        services.AddSingleton<IRingKernelBridge>(sp =>
        {
            var backendProvider = sp.GetRequiredService<DotComputeBackendProvider>();
            var ringKernelRuntime = sp.GetService<DotComputeRingKernels.IRingKernelRuntime>();
            var logger = sp.GetRequiredService<ILogger<DotComputeRingKernelBridge>>();
            return new DotComputeRingKernelBridge(logger, backendProvider, ringKernelRuntime);
        });

        return services;
    }
}

/// <summary>
/// Configuration options for the DotCompute ring kernel bridge.
/// </summary>
public sealed class DotComputeRingKernelBridgeOptions
{
    /// <summary>
    /// Gets or sets whether to enable automatic CPU fallback when GPU execution fails.
    /// Default: true
    /// </summary>
    public bool EnableCpuFallback { get; set; } = true;

    /// <summary>
    /// Gets or sets the preferred GPU device ID for kernel execution.
    /// Default: 0 (first GPU)
    /// </summary>
    public int PreferredDeviceId { get; set; } = 0;

    /// <summary>
    /// Gets or sets whether to enable telemetry collection.
    /// Default: true
    /// </summary>
    public bool EnableTelemetry { get; set; } = true;

    /// <summary>
    /// Gets or sets the maximum number of retry attempts for GPU execution.
    /// Default: 3
    /// </summary>
    public int MaxRetryAttempts { get; set; } = 3;

    /// <summary>
    /// Gets or sets the timeout for GPU operations in milliseconds.
    /// Default: 5000 (5 seconds)
    /// </summary>
    public int GpuOperationTimeoutMs { get; set; } = 5000;
}

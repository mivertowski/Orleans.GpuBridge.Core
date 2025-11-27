using DotCompute.Abstractions.RingKernels;
using DotCompute.Backends.CUDA.Compilation;
using DotCompute.Backends.CUDA.RingKernels;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.DependencyInjection.Extensions;
using Microsoft.Extensions.Logging;
using Orleans.GpuBridge.Abstractions;
using Orleans.GpuBridge.Abstractions.Memory;
using Orleans.GpuBridge.Abstractions.Providers;
using Orleans.GpuBridge.Abstractions.RingKernels;
using Orleans.GpuBridge.Runtime.Builders;
using Orleans.GpuBridge.Runtime.K2K;
using Orleans.GpuBridge.Runtime.Providers;
using Orleans.GpuBridge.Runtime.RingKernels;
using Orleans.GpuBridge.Runtime.Routing;

namespace Orleans.GpuBridge.Runtime.Extensions;

/// <summary>
/// Extension methods for configuring GPU Bridge services
/// </summary>
public static class ServiceCollectionExtensions
{
    /// <summary>
    /// Adds GPU Bridge services to the service collection
    /// </summary>
    public static IGpuBridgeBuilder AddGpuBridge(
        this IServiceCollection services, 
        Action<GpuBridgeOptions>? configure = null)
    {
        // Configure options
        if (configure != null)
        {
            services.Configure(configure);
        }
        else
        {
            services.Configure<GpuBridgeOptions>(_ => { });
        }
        
        // Backend provider system
        services.TryAddSingleton<IGpuBackendRegistry, GpuBackendRegistry>();
        services.TryAddSingleton<GpuBridgeProviderSelector>();
        
        // Core services
        services.TryAddSingleton<IGpuBridge, GpuBridge>();
        services.TryAddSingleton<KernelCatalog>();
        services.TryAddSingleton<DeviceBroker>();
        services.TryAddSingleton<PersistentKernelHost>();
        services.TryAddSingleton<GpuDiagnostics>();
        
        // Hosted service
        services.AddHostedService<GpuHostFeature>();
        
        // Memory management (placeholder for now)
        services.TryAddSingleton(typeof(IGpuMemoryPool<>), typeof(CpuMemoryPool<>));

        return new GpuBridgeBuilder(services);
    }

    /// <summary>
    /// Adds GPU-native actor support via DotCompute ring kernels.
    /// </summary>
    /// <param name="services">Service collection.</param>
    /// <param name="configure">Optional configuration action.</param>
    /// <returns>Service collection for chaining.</returns>
    /// <remarks>
    /// <para>
    /// This registers the DotCompute ring kernel runtime for GPU-native actors.
    /// Ring kernels are persistent GPU threads that process messages with sub-microsecond latency.
    /// </para>
    /// <para>
    /// Performance characteristics:
    /// - Message latency: 100-500ns (GPU queue operations only)
    /// - Zero kernel launch overhead (persistent threads)
    /// - Lock-free atomic queue operations
    /// - 20-200× faster than CPU actors
    /// </para>
    /// <para>
    /// Usage:
    /// <code>
    /// services.AddGpuBridge()
    ///         .AddRingKernelSupport();
    /// </code>
    /// </para>
    /// </remarks>
    public static IServiceCollection AddRingKernelSupport(
        this IServiceCollection services,
        Action<RingKernelOptions>? configure = null)
    {
        // Configure ring kernel options
        if (configure != null)
        {
            services.Configure(configure);
        }
        else
        {
            services.Configure<RingKernelOptions>(_ => { });
        }

        // Register DotCompute ring kernel infrastructure
        services.TryAddSingleton<RingKernelDiscovery>(sp =>
        {
            var logger = sp.GetRequiredService<ILogger<RingKernelDiscovery>>();
            return new RingKernelDiscovery(logger);
        });

        services.TryAddSingleton<CudaRingKernelStubGenerator>(sp =>
        {
            var logger = sp.GetRequiredService<ILogger<CudaRingKernelStubGenerator>>();
            return new CudaRingKernelStubGenerator(logger);
        });

        services.TryAddSingleton<CudaMemoryPackSerializerGenerator>(sp =>
        {
            var logger = sp.GetRequiredService<ILogger<CudaMemoryPackSerializerGenerator>>();
            return new CudaMemoryPackSerializerGenerator(logger);
        });

        services.TryAddSingleton<CudaRingKernelCompiler>(sp =>
        {
            var logger = sp.GetRequiredService<ILogger<CudaRingKernelCompiler>>();
            var kernelDiscovery = sp.GetRequiredService<RingKernelDiscovery>();
            var stubGenerator = sp.GetRequiredService<CudaRingKernelStubGenerator>();
            var serializerGenerator = sp.GetRequiredService<CudaMemoryPackSerializerGenerator>();
            return new CudaRingKernelCompiler(logger, kernelDiscovery, stubGenerator, serializerGenerator);
        });

        // Register DotCompute message queue registry (required for named queues)
        services.TryAddSingleton<DotCompute.Core.Messaging.MessageQueueRegistry>();

        services.TryAddSingleton<CudaRingKernelRuntime>(sp =>
        {
            var logger = sp.GetRequiredService<ILogger<CudaRingKernelRuntime>>();
            var compiler = sp.GetRequiredService<CudaRingKernelCompiler>();
            var registry = sp.GetRequiredService<DotCompute.Core.Messaging.MessageQueueRegistry>();
            return new CudaRingKernelRuntime(logger, compiler, registry);
        });

        // Register Orleans integration wrapper
        services.TryAddSingleton<IRingKernelRuntime>(sp =>
        {
            var cudaRuntime = sp.GetRequiredService<CudaRingKernelRuntime>();
            var logger = sp.GetRequiredService<ILogger<DotComputeRingKernelRuntime>>();
            return new DotComputeRingKernelRuntime(cudaRuntime, logger);
        });

        return services;
    }

    /// <summary>
    /// Adds K2K (Kernel-to-Kernel) messaging support for GPU-native actor communication.
    /// </summary>
    /// <param name="services">Service collection.</param>
    /// <returns>Service collection for chaining.</returns>
    /// <remarks>
    /// <para>
    /// K2K messaging enables sub-microsecond communication between GPU-resident actors
    /// without CPU involvement. Messages are dispatched directly through GPU memory queues.
    /// </para>
    /// <para>
    /// Supported routing strategies:
    /// - Direct: Point-to-point messaging (100-500ns latency)
    /// - Broadcast: One-to-many messaging
    /// - Ring: Circular topology for consensus protocols
    /// - HashRouted: Consistent hashing for load distribution
    /// </para>
    /// <para>
    /// Usage:
    /// <code>
    /// services.AddGpuBridge()
    ///         .AddRingKernelSupport()
    ///         .AddK2KSupport();
    /// </code>
    /// </para>
    /// </remarks>
    public static IServiceCollection AddK2KSupport(this IServiceCollection services)
    {
        // K2K dispatcher - handles GPU-to-GPU message routing
        services.TryAddSingleton<K2KDispatcher>();
        services.TryAddSingleton<IK2KDispatcher>(sp => sp.GetRequiredService<K2KDispatcher>());

        // K2K target registry - stores K2K target configurations
        services.TryAddSingleton<K2KTargetRegistry>();

        // K2K router factory - creates routers based on strategy
        services.TryAddSingleton<K2KRouterFactory>();

        // Register individual routers with keyed services
        services.TryAddKeyedSingleton<IK2KRouter>(
            K2KRoutingStrategy.Direct,
            (sp, _) => new DirectRouter(sp.GetRequiredService<IK2KDispatcher>()));

        services.TryAddKeyedSingleton<IK2KRouter>(
            K2KRoutingStrategy.Broadcast,
            (sp, _) => new BroadcastRouter(sp.GetRequiredService<K2KDispatcher>()));

        services.TryAddKeyedSingleton<IK2KRouter>(
            K2KRoutingStrategy.Ring,
            (sp, _) => new RingRouter(sp.GetRequiredService<K2KDispatcher>()));

        services.TryAddKeyedSingleton<IK2KRouter>(
            K2KRoutingStrategy.HashRouted,
            (sp, _) => new HashRoutedRouter(sp.GetRequiredService<K2KDispatcher>()));

        return services;
    }

    /// <summary>
    /// Adds the CPU fallback ring kernel bridge for generated actors.
    /// </summary>
    /// <param name="services">Service collection.</param>
    /// <returns>Service collection for chaining.</returns>
    /// <remarks>
    /// <para>
    /// This registers the CPU fallback implementation of <see cref="IRingKernelBridge"/>.
    /// Use this for development, testing, or environments without GPU hardware.
    /// </para>
    /// <para>
    /// For GPU acceleration, use the DotCompute backend's <c>AddDotComputeRingKernelBridge()</c>
    /// extension method instead, which will replace this registration with GPU-capable implementation.
    /// </para>
    /// <para>
    /// Usage:
    /// <code>
    /// services.AddGpuBridge()
    ///         .AddRingKernelSupport()
    ///         .AddRingKernelBridge(); // CPU fallback
    /// </code>
    /// </para>
    /// </remarks>
    public static IServiceCollection AddRingKernelBridge(this IServiceCollection services)
    {
        // Register CPU fallback as default (can be replaced by GPU implementation)
        services.TryAddSingleton<IRingKernelBridge>(sp =>
        {
            var logger = sp.GetRequiredService<ILogger<CpuFallbackRingKernelBridge>>();
            return new CpuFallbackRingKernelBridge(logger);
        });

        return services;
    }

    /// <summary>
    /// Adds a custom ring kernel bridge implementation.
    /// </summary>
    /// <typeparam name="TBridge">The bridge implementation type.</typeparam>
    /// <param name="services">Service collection.</param>
    /// <returns>Service collection for chaining.</returns>
    /// <remarks>
    /// Use this to register a custom <see cref="IRingKernelBridge"/> implementation.
    /// This replaces any previously registered bridge.
    /// </remarks>
    public static IServiceCollection AddRingKernelBridge<TBridge>(this IServiceCollection services)
        where TBridge : class, IRingKernelBridge
    {
        // Replace any existing registration
        services.RemoveAll<IRingKernelBridge>();
        services.AddSingleton<IRingKernelBridge, TBridge>();
        return services;
    }

    /// <summary>
    /// Adds a ring kernel bridge using a factory function.
    /// </summary>
    /// <param name="services">Service collection.</param>
    /// <param name="factory">Factory function to create the bridge.</param>
    /// <returns>Service collection for chaining.</returns>
    /// <remarks>
    /// Use this for custom bridge initialization with DI resolution.
    /// This replaces any previously registered bridge.
    /// </remarks>
    public static IServiceCollection AddRingKernelBridge(
        this IServiceCollection services,
        Func<IServiceProvider, IRingKernelBridge> factory)
    {
        ArgumentNullException.ThrowIfNull(factory);

        // Replace any existing registration
        services.RemoveAll<IRingKernelBridge>();
        services.AddSingleton(factory);
        return services;
    }
}

/// <summary>
/// Configuration options for ring kernel support.
/// </summary>
public sealed class RingKernelOptions
{
    /// <summary>
    /// Gets or sets the default grid size for ring kernel launches.
    /// </summary>
    /// <remarks>
    /// Default: 1 (single block, optimal for single-actor workloads).
    /// </remarks>
    public int DefaultGridSize { get; set; } = 1;

    /// <summary>
    /// Gets or sets the default block size for ring kernel launches.
    /// </summary>
    /// <remarks>
    /// Default: 256 (optimal for most GPU architectures).
    /// </remarks>
    public int DefaultBlockSize { get; set; } = 256;

    /// <summary>
    /// Gets or sets the default message queue capacity.
    /// </summary>
    /// <remarks>
    /// Default: 256 (must be power of 2).
    /// </remarks>
    public int DefaultQueueCapacity { get; set; } = 256;

    /// <summary>
    /// Gets or sets whether to enable GPU kernel compilation caching.
    /// </summary>
    /// <remarks>
    /// Default: true (compiled kernels cached to disk for faster restarts).
    /// </remarks>
    public bool EnableKernelCaching { get; set; } = true;

    /// <summary>
    /// Gets or sets the GPU device index to use.
    /// </summary>
    /// <remarks>
    /// Default: 0 (first GPU).
    /// </remarks>
    public int DeviceIndex { get; set; } = 0;
}
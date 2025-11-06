using System;
using Microsoft.Extensions.DependencyInjection;
using Orleans.GpuBridge.Runtime.Infrastructure;
using Orleans.Hosting;
using Orleans.Placement;
using Orleans.Runtime.Placement;

namespace Orleans.GpuBridge.Runtime.Extensions;

/// <summary>
/// Attribute to mark grains for GPU-aware placement
/// </summary>
/// <remarks>
/// Apply this attribute to grain classes that require GPU acceleration.
/// The placement director will automatically route these grains to silos with available GPU capacity.
/// </remarks>
[AttributeUsage(AttributeTargets.Class, AllowMultiple = false)]
public sealed class GpuPlacementAttribute : PlacementAttribute
{
    /// <summary>
    /// Whether to prefer local silo if it has GPU capacity
    /// </summary>
    public bool PreferLocalPlacement { get; set; }

    /// <summary>
    /// Minimum GPU memory required in MB
    /// </summary>
    public int MinimumGpuMemoryMB { get; set; }

    /// <summary>
    /// Initializes a new instance of the GPU placement attribute
    /// </summary>
    public GpuPlacementAttribute() : base(GpuPlacementStrategy.Instance)
    {
        PreferLocalPlacement = false;
        MinimumGpuMemoryMB = 0;
    }

    /// <summary>
    /// Initializes a new instance with specific requirements
    /// </summary>
    /// <param name="preferLocalPlacement">Whether to prefer local placement</param>
    /// <param name="minimumGpuMemoryMB">Minimum GPU memory in MB</param>
    public GpuPlacementAttribute(bool preferLocalPlacement, int minimumGpuMemoryMB = 0)
        : base(new GpuPlacementStrategy
        {
            PreferLocalPlacement = preferLocalPlacement,
            MinimumGpuMemoryMB = minimumGpuMemoryMB
        })
    {
        PreferLocalPlacement = preferLocalPlacement;
        MinimumGpuMemoryMB = minimumGpuMemoryMB;
    }
}

/// <summary>
/// Extension methods for configuring GPU-aware placement
/// </summary>
public static class GpuPlacementExtensions
{
    /// <summary>
    /// Configures GPU-aware placement strategy in the silo
    /// </summary>
    /// <remarks>
    /// This registers:
    /// - GpuPlacementDirector: Queries IGpuCapacityGrain for intelligent placement
    /// - GpuSiloLifecycleParticipant: Automatically registers and updates GPU capacity
    ///
    /// The lifecycle participant will:
    /// - Register GPU capacity on silo startup
    /// - Update capacity every 30 seconds
    /// - Unregister on graceful shutdown
    /// </remarks>
    public static ISiloBuilder AddGpuPlacement(this ISiloBuilder builder)
    {
        return builder.ConfigureServices(services =>
        {
            // Placement director for GPU-aware grain placement
            services.AddSingleton<IPlacementDirector, GpuPlacementDirector>();
            services.AddSingleton<PlacementStrategy, GpuPlacementStrategy>();

            // Lifecycle participant for automatic capacity registration
            // Orleans automatically discovers ILifecycleParticipant<ISiloLifecycle> implementations
            services.AddSingleton<ILifecycleParticipant<ISiloLifecycle>, GpuSiloLifecycleParticipant>();
        });
    }

    /// <summary>
    /// Configures GPU placement strategy in the client
    /// </summary>
    /// <remarks>
    /// Clients need to be aware of the GPU placement strategy to properly
    /// communicate grain placement requirements to silos.
    /// </remarks>
    public static IClientBuilder AddGpuPlacement(this IClientBuilder builder)
    {
        return builder.ConfigureServices(services =>
        {
            services.AddSingleton<PlacementStrategy, GpuPlacementStrategy>();
        });
    }
}

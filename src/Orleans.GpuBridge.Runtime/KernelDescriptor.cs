using System;
using System.Collections.Generic;
using Orleans.GpuBridge.Abstractions;
using Orleans.GpuBridge.Abstractions.Kernels;

namespace Orleans.GpuBridge.Runtime;

/// <summary>
/// Configuration options for the kernel catalog.
/// </summary>
public sealed class KernelCatalogOptions
{
    /// <summary>
    /// Gets the list of kernel descriptors to register.
    /// </summary>
    public List<KernelDescriptor> Descriptors { get; } = new();
}

/// <summary>
/// Describes a kernel registration with its input/output types and factory.
/// </summary>
public sealed class KernelDescriptor
{
    /// <summary>
    /// Gets or sets the kernel identifier.
    /// </summary>
    public KernelId Id { get; set; } = new("unset");

    /// <summary>
    /// Gets or sets the input type for the kernel.
    /// </summary>
    public Type InType { get; set; } = typeof(object);

    /// <summary>
    /// Gets or sets the output type for the kernel.
    /// </summary>
    public Type OutType { get; set; } = typeof(object);

    /// <summary>
    /// Gets or sets the factory function for creating kernel instances.
    /// </summary>
    public Func<IServiceProvider, object>? Factory { get; set; }

    /// <summary>
    /// Builds a kernel descriptor using a configuration action.
    /// </summary>
    /// <param name="cfg">Configuration action to apply to the descriptor.</param>
    /// <returns>A configured kernel descriptor.</returns>
    public static KernelDescriptor Build(Action<KernelDescriptor> cfg)
    {
        var d = new KernelDescriptor();
        cfg(d);
        return d;
    }

    /// <summary>
    /// Sets the kernel identifier.
    /// </summary>
    /// <param name="id">The kernel identifier string.</param>
    /// <returns>The descriptor for fluent chaining.</returns>
    public KernelDescriptor SetId(string id)
    {
        Id = new(id);
        return this;
    }

    /// <summary>
    /// Sets the input type for the kernel.
    /// </summary>
    /// <typeparam name="T">The input type.</typeparam>
    /// <returns>The descriptor for fluent chaining.</returns>
    public KernelDescriptor In<T>()
    {
        InType = typeof(T);
        return this;
    }

    /// <summary>
    /// Sets the output type for the kernel.
    /// </summary>
    /// <typeparam name="T">The output type.</typeparam>
    /// <returns>The descriptor for fluent chaining.</returns>
    public KernelDescriptor Out<T>()
    {
        OutType = typeof(T);
        return this;
    }

    /// <summary>
    /// Sets the factory function for creating kernel instances.
    /// </summary>
    /// <param name="f">The factory function.</param>
    /// <returns>The descriptor for fluent chaining.</returns>
    public KernelDescriptor FromFactory(Func<IServiceProvider, object> f)
    {
        Factory = f;
        return this;
    }
}

using System;

namespace Orleans.GpuBridge.Abstractions;

/// <summary>
/// Marks a class or method as GPU-accelerated, associating it with a specific kernel.
/// </summary>
/// <remarks>
/// This attribute is used to indicate that a grain or method should use GPU acceleration
/// through the specified kernel ID. The kernel must be registered with the KernelCatalog
/// before the grain is activated.
/// </remarks>
[AttributeUsage(AttributeTargets.Class | AttributeTargets.Method, AllowMultiple = false)]
public sealed class GpuAcceleratedAttribute : Attribute
{
    /// <summary>
    /// Gets the kernel identifier to use for GPU acceleration.
    /// </summary>
    public string KernelId { get; }

    /// <summary>
    /// Initializes a new instance of the <see cref="GpuAcceleratedAttribute"/> class.
    /// </summary>
    /// <param name="id">The kernel identifier. Must match a registered kernel in the KernelCatalog.</param>
    public GpuAcceleratedAttribute(string id) => KernelId = id;
}

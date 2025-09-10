using System;

namespace Orleans.GpuBridge.Backends.DotCompute.Attributes;

/// <summary>
/// Marks a method as a GPU kernel for the DotCompute backend
/// </summary>
[AttributeUsage(AttributeTargets.Method, AllowMultiple = false)]
public sealed class KernelAttribute : Attribute
{
    /// <summary>
    /// Gets the kernel name/identifier
    /// </summary>
    public string Name { get; }

    /// <summary>
    /// Gets the preferred work group size for GPU execution
    /// </summary>
    public int PreferredWorkGroupSize { get; init; } = 256;

    /// <summary>
    /// Gets whether this kernel requires shared memory
    /// </summary>
    public bool RequiresSharedMemory { get; init; } = false;

    /// <summary>
    /// Gets the minimum required compute capability
    /// </summary>
    public string? MinComputeCapability { get; init; }

    /// <summary>
    /// Initializes a new instance of the KernelAttribute
    /// </summary>
    /// <param name="name">The kernel name/identifier</param>
    public KernelAttribute(string name)
    {
        Name = name ?? throw new ArgumentNullException(nameof(name));
    }
}
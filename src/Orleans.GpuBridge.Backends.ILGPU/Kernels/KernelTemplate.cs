using System;
using System.Reflection;

namespace Orleans.GpuBridge.Backends.ILGPU.Kernels;

/// <summary>
/// Represents a kernel template
/// </summary>
public class KernelTemplate
{
    public required string Name { get; init; }
    public required Type Type { get; init; }
    public required MethodInfo Method { get; init; }
    public required KernelCategory Category { get; init; }
}

using System;
using ILGPU.Runtime;

namespace Orleans.GpuBridge.Backends.ILGPU.Kernels;

/// <summary>
/// Represents compiled kernel information
/// </summary>
public class CompiledKernelInfo
{
    public required KernelTemplate Template { get; init; }
    public required object Kernel { get; init; }
    public required Accelerator Accelerator { get; init; }
    public required TimeSpan CompilationTime { get; init; }
}

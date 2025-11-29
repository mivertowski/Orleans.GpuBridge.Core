using System;
using System.Collections.Generic;
using Orleans.GpuBridge.Abstractions.Providers.Memory.Interfaces;
using Orleans.GpuBridge.Abstractions.Providers;

namespace Orleans.GpuBridge.Abstractions.Providers.Execution.Parameters;

/// <summary>
/// Parameters for kernel execution
/// </summary>
public sealed class KernelExecutionParameters
{
    /// <summary>
    /// Gets or initializes the global work size dimensions for kernel execution.
    /// </summary>
    public int[] GlobalWorkSize { get; init; } = Array.Empty<int>();

    /// <summary>
    /// Gets or initializes the local work size dimensions for kernel execution.
    /// If null, the backend will choose an optimal local work size.
    /// </summary>
    public int[]? LocalWorkSize { get; init; }

    /// <summary>
    /// Gets or initializes the memory arguments passed to the kernel.
    /// Maps argument names to device memory buffers.
    /// </summary>
    public IReadOnlyDictionary<string, IDeviceMemory> MemoryArguments { get; init; } = new Dictionary<string, IDeviceMemory>();

    /// <summary>
    /// Gets or initializes the scalar arguments passed to the kernel.
    /// Maps argument names to scalar values.
    /// </summary>
    public IReadOnlyDictionary<string, object> ScalarArguments { get; init; } = new Dictionary<string, object>();

    /// <summary>
    /// Gets or initializes the size of dynamic shared memory in bytes.
    /// </summary>
    public int DynamicSharedMemoryBytes { get; init; }

    /// <summary>
    /// Gets or initializes the preferred command queue for execution.
    /// If null, the default queue will be used.
    /// </summary>
    public ICommandQueue? PreferredQueue { get; init; }

    /// <summary>
    /// Gets or initializes the execution priority (higher values = higher priority).
    /// </summary>
    public int Priority { get; init; }
}
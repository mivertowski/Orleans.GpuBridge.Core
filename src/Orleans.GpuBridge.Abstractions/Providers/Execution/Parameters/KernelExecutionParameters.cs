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
    public int[] GlobalWorkSize { get; init; } = Array.Empty<int>();
    public int[]? LocalWorkSize { get; init; }
    public IReadOnlyDictionary<string, IDeviceMemory> MemoryArguments { get; init; } = new Dictionary<string, IDeviceMemory>();
    public IReadOnlyDictionary<string, object> ScalarArguments { get; init; } = new Dictionary<string, object>();
    public int DynamicSharedMemoryBytes { get; init; }
    public ICommandQueue? PreferredQueue { get; init; }
    public int Priority { get; init; }
}
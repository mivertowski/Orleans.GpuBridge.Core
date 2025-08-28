using Orleans.GpuBridge.Abstractions.Enums;
using Orleans.GpuBridge.DotCompute.Enums;
using Orleans.GpuBridge.DotCompute.Memory;
using Orleans.GpuBridge.DotCompute.Compilation;
using Orleans.GpuBridge.DotCompute.Execution;

namespace Orleans.GpuBridge.DotCompute.Devices;

/// <summary>
/// Compute device abstraction for DotCompute
/// </summary>
public interface IComputeDevice
{
    string Name { get; }
    DeviceType Type { get; }
    int Index { get; }
    long TotalMemory { get; }
    long AvailableMemory { get; }
    int ComputeUnits { get; }
    bool IsAvailable { get; }
    
    Task<IUnifiedBuffer<T>> AllocateBufferAsync<T>(
        int size,
        BufferFlags flags = BufferFlags.ReadWrite,
        CancellationToken ct = default) where T : unmanaged;
    
    Task<ICompiledKernel> CompileKernelAsync(
        string code,
        string entryPoint,
        CompilationOptions? options = null,
        CancellationToken ct = default);
    
    Task<IKernelExecution> LaunchKernelAsync(
        ICompiledKernel kernel,
        KernelLaunchParams launchParams,
        CancellationToken ct = default);
}
using Orleans.GpuBridge.Abstractions.Enums;
using Orleans.GpuBridge.Abstractions.Memory;

namespace Orleans.GpuBridge.Runtime.BackendProviders.Interfaces;

/// <summary>
/// Compute context interface
/// </summary>
public interface IComputeContext : IDisposable
{
    GpuBackend Backend { get; }
    int DeviceIndex { get; }
    
    IComputeBuffer<T> CreateBuffer<T>(int size, BufferUsage usage) where T : unmanaged;
    IComputeKernel CompileKernel(string source, string entryPoint);
    void Execute(IComputeKernel kernel, int workSize);
    void Synchronize();
}
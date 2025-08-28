using Orleans.GpuBridge.DotCompute.Devices;
using Orleans.GpuBridge.DotCompute.Memory;

namespace Orleans.GpuBridge.DotCompute.Compilation;

/// <summary>
/// Compiled kernel ready for execution
/// </summary>
public interface ICompiledKernel : IDisposable
{
    string Name { get; }
    IComputeDevice Device { get; }
    
    void SetBuffer(int index, IUnifiedBuffer<byte> buffer);
    void SetConstant<T>(string name, T value) where T : unmanaged;
}
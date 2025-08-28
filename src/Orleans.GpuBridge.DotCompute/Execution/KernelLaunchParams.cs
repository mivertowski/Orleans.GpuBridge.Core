using Orleans.GpuBridge.DotCompute.Memory;

namespace Orleans.GpuBridge.DotCompute.Execution;

/// <summary>
/// Kernel launch parameters
/// </summary>
public sealed class KernelLaunchParams
{
    public int GlobalWorkSize { get; set; }
    public int LocalWorkSize { get; set; } = 256;
    public int SharedMemoryBytes { get; set; }
    public Dictionary<int, IUnifiedBuffer<byte>> Buffers { get; set; } = new();
    public Dictionary<string, object> Constants { get; set; } = new();
}
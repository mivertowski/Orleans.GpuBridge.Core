using System.Threading;
using System.Threading.Tasks;

namespace Orleans.GpuBridge.Abstractions;

/// <summary>
/// Main interface for GPU bridge operations
/// </summary>
public interface IGpuBridge
{
    /// <summary>
    /// Gets information about the GPU bridge and available resources
    /// </summary>
    ValueTask<GpuBridgeInfo> GetInfoAsync(CancellationToken ct = default);
    
    /// <summary>
    /// Gets a kernel executor for the specified kernel ID
    /// </summary>
    ValueTask<IGpuKernel<TIn, TOut>> GetKernelAsync<TIn, TOut>(
        KernelId kernelId, 
        CancellationToken ct = default) 
        where TIn : notnull 
        where TOut : notnull;
    
    /// <summary>
    /// Gets the list of available GPU devices
    /// </summary>
    ValueTask<IReadOnlyList<GpuDevice>> GetDevicesAsync(CancellationToken ct = default);
}
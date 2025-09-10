using System.Runtime.CompilerServices;
using System.Threading;
using System.Threading.Tasks;
using Orleans.GpuBridge.Abstractions.Kernels;

namespace Orleans.GpuBridge.Abstractions.Application.Interfaces;

/// <summary>
/// Interface for GPU kernel execution
/// </summary>
public interface IGpuKernel<TIn, TOut> 
    where TIn : notnull 
    where TOut : notnull
{
    /// <summary>
    /// Submits a batch of items for GPU processing
    /// </summary>
    ValueTask<KernelHandle> SubmitBatchAsync(
        IReadOnlyList<TIn> items,
        GpuExecutionHints? hints = null,
        CancellationToken ct = default);
    
    /// <summary>
    /// Reads results from a submitted batch
    /// </summary>
    IAsyncEnumerable<TOut> ReadResultsAsync(
        KernelHandle handle,
        CancellationToken ct = default);
    
    /// <summary>
    /// Gets information about this kernel
    /// </summary>
    ValueTask<KernelInfo> GetInfoAsync(CancellationToken ct = default);
}
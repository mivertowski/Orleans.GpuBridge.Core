using System.Collections.Generic;
using System.Threading.Tasks;
using Orleans;
using Orleans.GpuBridge.Abstractions;

namespace Orleans.GpuBridge.Grains.Batch;

/// <summary>
/// Grain interface for GPU batch processing
/// </summary>
public interface IGpuBatchGrain<TIn, TOut> : IGrainWithStringKey 
    where TIn : notnull 
    where TOut : notnull
{
    /// <summary>
    /// Executes a batch of items on the GPU
    /// </summary>
    Task<GpuBatchResult<TOut>> ExecuteAsync(
        IReadOnlyList<TIn> batch, 
        GpuExecutionHints? hints = null);
    
    /// <summary>
    /// Executes a batch with a callback observer
    /// </summary>
    Task<GpuBatchResult<TOut>> ExecuteWithCallbackAsync(
        IReadOnlyList<TIn> batch,
        IGpuResultObserver<TOut> observer,
        GpuExecutionHints? hints = null);
}
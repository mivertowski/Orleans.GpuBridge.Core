using System;
using System.Threading;
using System.Threading.Tasks;
using Orleans.GpuBridge.Abstractions;
using Orleans.GpuBridge.BridgeFX.Pipeline.Core;

namespace Orleans.GpuBridge.BridgeFX.Pipeline.Stages;

/// <summary>
/// Kernel processing stage
/// </summary>
internal sealed class KernelStage<TIn, TOut> : IPipelineStage
    where TIn : notnull
    where TOut : notnull
{
    private readonly KernelId _kernelId;
    private readonly IGpuBridge _bridge;
    private readonly Func<TIn, bool>? _filter;
    private IGpuKernel<TIn, TOut>? _kernel;
    
    public Type InputType => typeof(TIn);
    public Type OutputType => typeof(TOut);
    
    public KernelStage(KernelId kernelId, IGpuBridge bridge, Func<TIn, bool>? filter)
    {
        _kernelId = kernelId;
        _bridge = bridge;
        _filter = filter;
    }
    
    public async Task<object?> ProcessAsync(object input, CancellationToken ct)
    {
        if (input is not TIn typedInput)
        {
            throw new ArgumentException($"Expected {typeof(TIn)}, got {input.GetType()}");
        }
        
        // Apply filter if specified
        if (_filter != null && !_filter(typedInput))
        {
            return null;
        }
        
        // Get kernel lazily
        _kernel ??= await _bridge.GetKernelAsync<TIn, TOut>(_kernelId, ct);
        
        // Process single item
        var handle = await _kernel.SubmitBatchAsync(new[] { typedInput }, null, ct);
        
        await foreach (var result in _kernel.ReadResultsAsync(handle, ct))
        {
            return result;
        }
        
        return null;
    }
}
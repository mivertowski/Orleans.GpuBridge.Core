using System;
using System.Threading;
using System.Threading.Tasks;
using Orleans.GpuBridge.BridgeFX.Pipeline.Core;

namespace Orleans.GpuBridge.BridgeFX.Pipeline.Stages;

/// <summary>
/// Transform stage
/// </summary>
internal sealed class TransformStage<TIn, TOut> : IPipelineStage
    where TIn : notnull
    where TOut : notnull
{
    private readonly Func<TIn, TOut> _transform;
    
    public Type InputType => typeof(TIn);
    public Type OutputType => typeof(TOut);
    
    public TransformStage(Func<TIn, TOut> transform)
    {
        _transform = transform;
    }
    
    public Task<object?> ProcessAsync(object input, CancellationToken ct)
    {
        if (input is not TIn typedInput)
        {
            throw new ArgumentException($"Expected {typeof(TIn)}, got {input.GetType()}");
        }
        
        var result = _transform(typedInput);
        return Task.FromResult<object?>(result);
    }
}
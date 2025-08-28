using System;
using System.Threading;
using System.Threading.Tasks;
using Orleans.GpuBridge.BridgeFX.Pipeline.Core;

namespace Orleans.GpuBridge.BridgeFX.Pipeline.Stages;

/// <summary>
/// Tap stage for side effects
/// </summary>
internal sealed class TapStage<T> : IPipelineStage
    where T : notnull
{
    private readonly Action<T> _action;
    
    public Type InputType => typeof(T);
    public Type OutputType => typeof(T);
    
    public TapStage(Action<T> action)
    {
        _action = action;
    }
    
    public Task<object?> ProcessAsync(object input, CancellationToken ct)
    {
        if (input is not T typedInput)
        {
            throw new ArgumentException($"Expected {typeof(T)}, got {input.GetType()}");
        }
        
        _action(typedInput);
        return Task.FromResult<object?>(typedInput);
    }
}
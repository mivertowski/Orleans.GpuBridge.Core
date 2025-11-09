using System;
using System.Threading;
using System.Threading.Tasks;
using Orleans.GpuBridge.BridgeFX.Pipeline.Core;

namespace Orleans.GpuBridge.BridgeFX.Pipeline.Stages;

/// <summary>
/// Async transform stage for pipeline processing
/// </summary>
internal sealed class AsyncTransformStage<TIn, TOut> : IPipelineStage
    where TIn : notnull
    where TOut : notnull
{
    private readonly Func<TIn, Task<TOut>> _transform;

    public Type InputType => typeof(TIn);
    public Type OutputType => typeof(TOut);

    public AsyncTransformStage(Func<TIn, Task<TOut>> transform)
    {
        _transform = transform;
    }

    public async Task<object?> ProcessAsync(object input, CancellationToken ct)
    {
        if (input is not TIn typedInput)
        {
            throw new ArgumentException($"Expected {typeof(TIn)}, got {input.GetType()}");
        }

        // Check for cancellation before processing
        ct.ThrowIfCancellationRequested();

        var result = await _transform(typedInput);
        return result;
    }
}

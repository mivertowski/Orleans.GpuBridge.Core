using System;
using System.Threading;
using System.Threading.Tasks;
using Orleans.GpuBridge.BridgeFX.Pipeline.Core;

namespace Orleans.GpuBridge.BridgeFX.Pipeline.Stages;

/// <summary>
/// Filter stage
/// </summary>
internal sealed class FilterStage<T> : IPipelineStage
    where T : notnull
{
    private readonly Func<T, bool> _predicate;

    public Type InputType => typeof(T);
    public Type OutputType => typeof(T);

    public FilterStage(Func<T, bool> predicate)
    {
        _predicate = predicate;
    }

    public Task<object?> ProcessAsync(object input, CancellationToken ct)
    {
        if (input is not T typedInput)
        {
            throw new ArgumentException($"Expected {typeof(T)}, got {input.GetType()}");
        }

        return Task.FromResult<object?>(_predicate(typedInput) ? typedInput : null);
    }
}
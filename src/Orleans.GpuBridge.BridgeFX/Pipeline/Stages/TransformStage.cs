using System;
using System.Threading;
using System.Threading.Tasks;
using Orleans.GpuBridge.BridgeFX.Pipeline.Core;

namespace Orleans.GpuBridge.BridgeFX.Pipeline.Stages;

/// <summary>
/// Transform stage with nullable type support
/// </summary>
internal sealed class TransformStage<TIn, TOut> : IPipelineStage
{
    private readonly Func<TIn, TOut> _transform;

    public Type InputType => typeof(TIn);
    public Type OutputType => typeof(TOut);

    public TransformStage(Func<TIn, TOut> transform)
    {
        _transform = transform;
    }

    public Task<object?> ProcessAsync(object? input, CancellationToken ct)
    {
        // Handle null inputs for nullable type support
        if (input is not TIn typedInput)
        {
            // Special handling for null values with nullable types
            if (input == null && Nullable.GetUnderlyingType(typeof(TIn)) != null)
            {
                // Allow null for nullable types (e.g., int?, string?)
                typedInput = default(TIn)!;
            }
            else
            {
                throw new ArgumentException(
                    $"Expected {typeof(TIn)}, got {input?.GetType()?.ToString() ?? "null"}");
            }
        }

        var result = _transform(typedInput);
        return Task.FromResult<object?>(result);
    }
}
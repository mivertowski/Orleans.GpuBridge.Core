using System;
using System.Threading;
using System.Threading.Tasks;
using Orleans.GpuBridge.BridgeFX.Pipeline.Core;

namespace Orleans.GpuBridge.BridgeFX.Pipeline.Stages;

/// <summary>
/// Async transform stage with nullable type support
/// </summary>
internal sealed class AsyncTransformStage<TIn, TOut> : IPipelineStage
{
    private readonly Func<TIn, Task<TOut>> _transform;

    public Type InputType => typeof(TIn);
    public Type OutputType => typeof(TOut);

    public AsyncTransformStage(Func<TIn, Task<TOut>> transform)
    {
        _transform = transform;
    }

    public async Task<object?> ProcessAsync(object? input, CancellationToken ct)
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

        // Check for cancellation before processing
        ct.ThrowIfCancellationRequested();

        var result = await _transform(typedInput);

        // Check for cancellation after processing completes
        // This catches cancellations that occur during transform execution
        ct.ThrowIfCancellationRequested();

        return result;
    }
}

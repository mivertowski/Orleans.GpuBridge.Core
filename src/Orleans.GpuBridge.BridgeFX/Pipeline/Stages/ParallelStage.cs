using System;
using System.Threading;
using System.Threading.Tasks;
using Orleans.GpuBridge.BridgeFX.Pipeline.Core;

namespace Orleans.GpuBridge.BridgeFX.Pipeline.Stages;

/// <summary>
/// Parallel processing stage
/// </summary>
internal sealed class ParallelStage<TIn, TOut> : IPipelineStage
    where TIn : notnull
    where TOut : notnull
{
    private readonly Func<TIn, Task<TOut>> _processor;
    private readonly SemaphoreSlim _semaphore;

    public Type InputType => typeof(TIn);
    public Type OutputType => typeof(TOut);

    public ParallelStage(Func<TIn, Task<TOut>> processor, int maxConcurrency)
    {
        _processor = processor;
        _semaphore = new SemaphoreSlim(maxConcurrency, maxConcurrency);
    }

    public async Task<object?> ProcessAsync(object input, CancellationToken ct)
    {
        if (input is not TIn typedInput)
        {
            throw new ArgumentException($"Expected {typeof(TIn)}, got {input.GetType()}");
        }

        await _semaphore.WaitAsync(ct);
        try
        {
            var result = await _processor(typedInput);
            return result;
        }
        finally
        {
            _semaphore.Release();
        }
    }
}
using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Threading;
using System.Threading.Tasks;
using Orleans.GpuBridge.Abstractions;
using Orleans.GpuBridge.Abstractions.Kernels;

namespace Orleans.GpuBridge.Runtime;

/// <summary>
/// Generic CPU-based fallback kernel that provides passthrough execution
/// when a GPU kernel is not available or GPU acceleration is disabled.
/// Attempts type casting and conversion for input-to-output transformation.
/// </summary>
/// <typeparam name="TIn">The input type.</typeparam>
/// <typeparam name="TOut">The output type.</typeparam>
internal sealed class CpuPassthroughKernel<TIn, TOut> : IGpuKernel<TIn, TOut>
    where TIn : notnull
    where TOut : notnull
{
    private readonly Dictionary<string, IReadOnlyList<TIn>> _batches = new();

    /// <summary>
    /// Submits a batch of items for processing.
    /// </summary>
    /// <param name="items">The items to process.</param>
    /// <param name="hints">Optional execution hints (ignored for CPU execution).</param>
    /// <param name="ct">Cancellation token.</param>
    /// <returns>A handle to retrieve results.</returns>
    public ValueTask<KernelHandle> SubmitBatchAsync(
        IReadOnlyList<TIn> items,
        GpuExecutionHints? hints = null,
        CancellationToken ct = default)
    {
        var handle = KernelHandle.Create();
        _batches[handle.Id] = items;
        return new(handle);
    }

    /// <summary>
    /// Reads results asynchronously from a previously submitted batch.
    /// Attempts direct casting if types match, otherwise attempts conversion.
    /// </summary>
    /// <param name="handle">The batch handle.</param>
    /// <param name="ct">Cancellation token.</param>
    /// <returns>An async enumerable of results.</returns>
    public async IAsyncEnumerable<TOut> ReadResultsAsync(
        KernelHandle handle,
        [EnumeratorCancellation] CancellationToken ct = default)
    {
        await Task.Yield(); // Ensure async
        if (!_batches.TryGetValue(handle.Id, out var items))
        {
            yield break;
        }

        // For passthrough, attempt to cast directly if types match
        if (typeof(TIn) == typeof(TOut))
        {
            foreach (var item in items)
            {
                ct.ThrowIfCancellationRequested();

                // Process item asynchronously for large datasets
                var result = await Task.Run(() =>
                {
                    if (item is TOut directResult)
                    {
                        return directResult;
                    }
                    else
                    {
                        return default(TOut)!;
                    }
                }, ct).ConfigureAwait(false);

                yield return result;
            }
        }
        else
        {
            // For different types, use async conversion
            foreach (var item in items)
            {
                ct.ThrowIfCancellationRequested();

                // Try to convert using common patterns asynchronously
                var result = await Task.Run(() =>
                {
                    TOut convertedResult = default(TOut)!;
                    try
                    {
                        if (item is IConvertible convertible)
                        {
                            var converted = Convert.ChangeType(item, typeof(TOut));
                            if (converted is TOut typedResult)
                            {
                                convertedResult = typedResult;
                            }
                        }
                    }
                    catch
                    {
                        // Use default for conversion failures
                    }

                    return convertedResult;
                }, ct).ConfigureAwait(false);

                yield return result;
            }
        }

        // Clean up the batch after processing
        _batches.Remove(handle.Id);
    }

    /// <summary>
    /// Gets information about this kernel.
    /// </summary>
    /// <param name="ct">Cancellation token.</param>
    /// <returns>Kernel information.</returns>
    public ValueTask<KernelInfo> GetInfoAsync(CancellationToken ct = default)
    {
        return new(new KernelInfo(
            new KernelId("cpu-passthrough"),
            "CPU passthrough kernel",
            typeof(TIn),
            typeof(TOut),
            false,
            1024));
    }
}

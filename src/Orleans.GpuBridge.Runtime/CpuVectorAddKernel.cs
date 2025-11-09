using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Threading;
using System.Threading.Tasks;
using Orleans.GpuBridge.Abstractions;
using Orleans.GpuBridge.Abstractions.Kernels;

namespace Orleans.GpuBridge.Runtime;

/// <summary>
/// CPU-based vector addition kernel that processes pairs of float arrays
/// and returns their element-wise sum. Used for testing and CPU fallback scenarios.
/// </summary>
public sealed class CpuVectorAddKernel : IGpuKernel<float[], float>
{
    private readonly Dictionary<string, IReadOnlyList<float[]>> _batches = new();

    /// <summary>
    /// Executes vector addition on two float arrays, returning the sum of all element-wise additions.
    /// </summary>
    /// <param name="a">First vector.</param>
    /// <param name="b">Second vector.</param>
    /// <returns>The sum of element-wise additions.</returns>
    public static float Execute(float[] a, float[] b)
    {
        var n = Math.Min(a.Length, b.Length);
        var sum = 0f;
        for (int i = 0; i < n; i++)
            sum += a[i] + b[i];
        return sum;
    }

    /// <summary>
    /// Submits a batch of float arrays for vector addition processing.
    /// Arrays are processed in pairs.
    /// </summary>
    /// <param name="items">The float arrays to process.</param>
    /// <param name="hints">Optional execution hints (ignored for CPU execution).</param>
    /// <param name="ct">Cancellation token.</param>
    /// <returns>A handle to retrieve results.</returns>
    public ValueTask<KernelHandle> SubmitBatchAsync(
        IReadOnlyList<float[]> items,
        GpuExecutionHints? hints = null,
        CancellationToken ct = default)
    {
        var handle = KernelHandle.Create();
        _batches[handle.Id] = items;
        return new(handle);
    }

    /// <summary>
    /// Reads results asynchronously from a previously submitted batch.
    /// Processes arrays in pairs and returns the sum of each pair.
    /// If an odd number of arrays is provided, the last array's sum is returned.
    /// </summary>
    /// <param name="handle">The batch handle.</param>
    /// <param name="ct">Cancellation token.</param>
    /// <returns>An async enumerable of results.</returns>
    public async IAsyncEnumerable<float> ReadResultsAsync(
        KernelHandle handle,
        [EnumeratorCancellation] CancellationToken ct = default)
    {
        await Task.Yield(); // Ensure async
        if (!_batches.TryGetValue(handle.Id, out var items))
        {
            yield break;
        }

        // Process pairs of vectors asynchronously for large datasets
        for (int i = 0; i < items.Count - 1; i += 2)
        {
            ct.ThrowIfCancellationRequested();

            var a = items[i];
            var b = items[i + 1];

            // Execute computation asynchronously for large vectors
            var result = await Task.Run(() => Execute(a, b), ct).ConfigureAwait(false);
            yield return result;
        }

        // If odd number of items, return sum of last vector
        if (items.Count % 2 == 1)
        {
            var lastVector = items[items.Count - 1];
            var sum = await Task.Run(() => lastVector.Sum(), ct).ConfigureAwait(false);
            yield return sum;
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
            new KernelId("cpu-vector-add"),
            "CPU vector addition kernel",
            typeof(float[]),
            typeof(float),
            false,
            1024));
    }
}

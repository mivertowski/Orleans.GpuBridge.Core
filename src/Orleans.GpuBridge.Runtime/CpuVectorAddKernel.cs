using System;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Orleans.GpuBridge.Abstractions.Kernels;

namespace Orleans.GpuBridge.Runtime;

/// <summary>
/// CPU-based vector addition kernel that processes float arrays
/// and returns their element-wise sum. Used for testing and CPU fallback scenarios.
/// </summary>
public sealed class CpuVectorAddKernel : GpuKernelBase<float[], float>
{
    /// <summary>
    /// Kernel unique identifier
    /// </summary>
    public override string KernelId => "cpu-vector-add";

    /// <summary>
    /// Kernel display name
    /// </summary>
    public override string DisplayName => "CPU Vector Addition Kernel";

    /// <summary>
    /// Backend provider name
    /// </summary>
    public override string BackendProvider => "CPU";

    /// <summary>
    /// Whether kernel uses GPU acceleration (always false for CPU)
    /// </summary>
    public override bool IsGpuAccelerated => false;

    /// <summary>
    /// Executes vector addition on a float array, returning the sum of all elements.
    /// </summary>
    /// <param name="input">Input vector (float array).</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>The sum of all elements.</returns>
    public override async Task<float> ExecuteAsync(float[] input, CancellationToken cancellationToken = default)
    {
        EnsureInitialized();
        EnsureNotDisposed();

        // Execute computation asynchronously for large vectors
        return await Task.Run(() =>
        {
            float sum = 0f;
            for (int i = 0; i < input.Length; i++)
            {
                sum += input[i];
            }
            return sum;
        }, cancellationToken).ConfigureAwait(false);
    }

    /// <summary>
    /// Execute vector addition for batch of inputs (optimized for CPU).
    /// </summary>
    /// <param name="inputs">Array of input vectors.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Array of sums (one per input vector).</returns>
    public override async Task<float[]> ExecuteBatchAsync(float[][] inputs, CancellationToken cancellationToken = default)
    {
        EnsureInitialized();
        EnsureNotDisposed();

        var results = new float[inputs.Length];

        // Process in parallel on CPU for better throughput
        await Task.Run(() =>
        {
            Parallel.For(0, inputs.Length, new ParallelOptions
            {
                CancellationToken = cancellationToken,
                MaxDegreeOfParallelism = Environment.ProcessorCount
            }, i =>
            {
                float sum = 0f;
                for (int j = 0; j < inputs[i].Length; j++)
                {
                    sum += inputs[i][j];
                }
                results[i] = sum;
            });
        }, cancellationToken).ConfigureAwait(false);

        return results;
    }

    /// <summary>
    /// Get estimated execution time for input size.
    /// CPU vector addition is ~1μs per 1000 elements.
    /// </summary>
    public override long GetEstimatedExecutionTimeMicroseconds(int inputSize)
    {
        // ~1μs per 1000 elements for vector addition on CPU
        return Math.Max(1, inputSize / 1000);
    }

    /// <summary>
    /// Get memory requirements for vector addition.
    /// </summary>
    public override KernelMemoryRequirements GetMemoryRequirements()
    {
        return new KernelMemoryRequirements(
            InputMemoryBytes: 4096,  // Assume 1024 floats (4KB)
            OutputMemoryBytes: 4,    // Single float result
            WorkingMemoryBytes: 0,   // No working memory
            TotalMemoryBytes: 4100);
    }

    /// <summary>
    /// Validate input vector.
    /// </summary>
    public override KernelValidationResult ValidateInput(float[] input)
    {
        if (input == null)
            return KernelValidationResult.Invalid("Input vector cannot be null");

        if (input.Length == 0)
            return KernelValidationResult.Invalid("Input vector cannot be empty");

        return KernelValidationResult.Valid();
    }
}

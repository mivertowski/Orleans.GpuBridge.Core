using System;
using System.Threading;
using System.Threading.Tasks;
using Orleans.GpuBridge.Abstractions.Kernels;

namespace Orleans.GpuBridge.Runtime;

/// <summary>
/// Generic CPU-based fallback kernel that provides passthrough execution
/// when a GPU kernel is not available or GPU acceleration is disabled.
/// Attempts type casting and conversion for input-to-output transformation.
/// </summary>
/// <typeparam name="TIn">The input type.</typeparam>
/// <typeparam name="TOut">The output type.</typeparam>
internal sealed class CpuPassthroughKernel<TIn, TOut> : GpuKernelBase<TIn, TOut>
{
    /// <summary>
    /// Kernel unique identifier
    /// </summary>
    public override string KernelId => "cpu-passthrough";

    /// <summary>
    /// Kernel display name
    /// </summary>
    public override string DisplayName => "CPU Passthrough Kernel";

    /// <summary>
    /// Backend provider name
    /// </summary>
    public override string BackendProvider => "CPU";

    /// <summary>
    /// Whether kernel uses GPU acceleration (always false for CPU)
    /// </summary>
    public override bool IsGpuAccelerated => false;

    /// <summary>
    /// Executes passthrough transformation from TIn to TOut.
    /// Attempts direct casting if types match, otherwise attempts conversion.
    /// </summary>
    /// <param name="input">The input value.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Transformed output value.</returns>
    public override async Task<TOut> ExecuteAsync(TIn input, CancellationToken cancellationToken = default)
    {
        EnsureInitialized();
        EnsureNotDisposed();

        // For passthrough, attempt to cast directly if types match
        if (typeof(TIn) == typeof(TOut))
        {
            if (input is TOut directResult)
            {
                return directResult;
            }
        }

        // For different types, use async conversion
        return await Task.Run(() =>
        {
            TOut convertedResult = default(TOut)!;
            try
            {
                if (input is IConvertible convertible)
                {
                    var converted = Convert.ChangeType(input, typeof(TOut));
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
        }, cancellationToken).ConfigureAwait(false);
    }

    /// <summary>
    /// Execute passthrough transformation for batch of inputs.
    /// Optimized for parallel CPU execution.
    /// </summary>
    /// <param name="inputs">Input array.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Transformed output array.</returns>
    public override async Task<TOut[]> ExecuteBatchAsync(TIn[] inputs, CancellationToken cancellationToken = default)
    {
        EnsureInitialized();
        EnsureNotDisposed();

        var results = new TOut[inputs.Length];

        // Process in parallel on CPU for better throughput
        await Task.Run(() =>
        {
            Parallel.For(0, inputs.Length, new ParallelOptions
            {
                CancellationToken = cancellationToken,
                MaxDegreeOfParallelism = Environment.ProcessorCount
            }, i =>
            {
                // Direct cast if types match
                if (typeof(TIn) == typeof(TOut) && inputs[i] is TOut directResult)
                {
                    results[i] = directResult;
                }
                else
                {
                    // Attempt conversion
                    try
                    {
                        if (inputs[i] is IConvertible convertible)
                        {
                            var converted = Convert.ChangeType(inputs[i], typeof(TOut));
                            if (converted is TOut typedResult)
                            {
                                results[i] = typedResult;
                            }
                        }
                    }
                    catch
                    {
                        results[i] = default(TOut)!;
                    }
                }
            });
        }, cancellationToken).ConfigureAwait(false);

        return results;
    }

    /// <summary>
    /// Get estimated execution time for input size.
    /// CPU passthrough is very fast (1μs per element).
    /// </summary>
    public override long GetEstimatedExecutionTimeMicroseconds(int inputSize)
    {
        return inputSize; // 1μs per element for CPU passthrough
    }

    /// <summary>
    /// Get memory requirements for CPU passthrough (minimal).
    /// </summary>
    public override KernelMemoryRequirements GetMemoryRequirements()
    {
        return new KernelMemoryRequirements(
            InputMemoryBytes: 64,  // Small buffer
            OutputMemoryBytes: 64, // Small buffer
            WorkingMemoryBytes: 0, // No working memory
            TotalMemoryBytes: 128);
    }

    /// <summary>
    /// Validate input (CPU passthrough accepts all inputs).
    /// </summary>
    public override KernelValidationResult ValidateInput(TIn input)
    {
        if (input == null)
            return KernelValidationResult.Invalid("Input cannot be null");

        return KernelValidationResult.Valid();
    }
}

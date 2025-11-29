using Microsoft.Extensions.Logging;
using Orleans.GpuBridge.Backends.DotCompute.Enums;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

namespace Orleans.GpuBridge.Backends.DotCompute.Execution;

/// <summary>
/// Parallel kernel executor for DotCompute backend
/// </summary>
public class ParallelKernelExecutor
{
    private readonly ILogger<ParallelKernelExecutor> _logger;

    /// <summary>
    /// Initializes a new instance of the <see cref="ParallelKernelExecutor"/> class
    /// </summary>
    /// <param name="logger">Logger for execution diagnostics</param>
    public ParallelKernelExecutor(ILogger<ParallelKernelExecutor> logger)
    {
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
    }

    /// <summary>
    /// Execute vectorized operation on float array
    /// </summary>
    public async Task<float[]> ExecuteVectorizedAsync(
        float[] input,
        VectorOperation operation,
        float[] parameters,
        CancellationToken cancellationToken = default)
    {
        await Task.Delay(1, cancellationToken); // Simulate async work

        return operation switch
        {
            VectorOperation.Add => input.Select(x => x + parameters[0]).ToArray(),
            VectorOperation.Multiply => input.Select(x => x * parameters[0]).ToArray(),
            VectorOperation.FusedMultiplyAdd => input.Select(x => x * parameters[0] + parameters[1]).ToArray(),
            VectorOperation.Sqrt => input.Select(x => MathF.Sqrt(Math.Max(0f, x))).ToArray(),
            VectorOperation.Abs => input.Select(Math.Abs).ToArray(),
            _ => throw new NotSupportedException($"Operation {operation} not supported")
        };
    }

    /// <summary>
    /// Execute parallel operation with custom function
    /// </summary>
    public async Task<TOut[]> ExecuteAsync<TIn, TOut>(
        TIn[] input,
        Func<TIn, TOut> kernel,
        ParallelExecutionOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        var parallelOptions = new ParallelOptions
        {
            CancellationToken = cancellationToken,
            MaxDegreeOfParallelism = options?.MaxDegreeOfParallelism ?? Environment.ProcessorCount
        };

        var result = new TOut[input.Length];

        await Task.Run(() =>
        {
            Parallel.For(0, input.Length, parallelOptions, i =>
            {
                result[i] = kernel(input[i]);
            });
        }, cancellationToken);

        return result;
    }
}

/// <summary>
/// Options for parallel execution
/// </summary>
public class ParallelExecutionOptions
{
    /// <summary>
    /// Maximum degree of parallelism
    /// </summary>
    public int MaxDegreeOfParallelism { get; set; } = Environment.ProcessorCount;
}
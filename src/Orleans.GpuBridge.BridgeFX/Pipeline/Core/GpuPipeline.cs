using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Orleans;
using Orleans.GpuBridge.Abstractions;
using Orleans.GpuBridge.BridgeFX.Pipeline.Stages;
using Orleans.GpuBridge.BridgeFX.Pipeline.Core;
using Orleans.GpuBridge.Grains.Batch;

namespace Orleans.GpuBridge.BridgeFX;

/// <summary>
/// Fluent pipeline builder for GPU processing
/// </summary>
public sealed class GpuPipeline
{
    private readonly List<IPipelineStage> _stages = new();
    private readonly ILogger<GpuPipeline> _logger;
    private readonly IGpuBridge _bridge;

    /// <summary>
    /// Initializes a new instance of the <see cref="GpuPipeline"/> class.
    /// </summary>
    /// <param name="bridge">The GPU bridge instance.</param>
    /// <param name="logger">The logger instance.</param>
    public GpuPipeline(IGpuBridge bridge, ILogger<GpuPipeline> logger)
    {
        _bridge = bridge;
        _logger = logger;
    }

    /// <summary>
    /// Creates a typed GPU pipeline for a specific kernel
    /// </summary>
    public static GpuPipelineBuilder<TIn, TOut> For<TIn, TOut>(IGrainFactory grainFactory, string kernelId)
        where TIn : notnull
        where TOut : notnull
    {
        return new GpuPipelineBuilder<TIn, TOut>(grainFactory, new KernelId(kernelId));
    }

    /// <summary>
    /// Adds a kernel stage to the pipeline
    /// </summary>
    public GpuPipeline AddKernel<TIn, TOut>(
        KernelId kernelId,
        Func<TIn, bool>? filter = null)
        where TIn : notnull
        where TOut : notnull
    {
        _stages.Add(new KernelStage<TIn, TOut>(kernelId, _bridge, filter));
        return this;
    }

    /// <summary>
    /// Adds a transform stage to the pipeline
    /// </summary>
    public GpuPipeline Transform<TIn, TOut>(
        Func<TIn, TOut> transform)
        where TIn : notnull
        where TOut : notnull
    {
        _stages.Add(new TransformStage<TIn, TOut>(transform));
        return this;
    }

    /// <summary>
    /// Adds an async transform stage to the pipeline
    /// </summary>
    public GpuPipeline Transform<TIn, TOut>(
        Func<TIn, Task<TOut>> asyncTransform)
        where TIn : notnull
        where TOut : notnull
    {
        _stages.Add(new AsyncTransformStage<TIn, TOut>(asyncTransform));
        return this;
    }

    /// <summary>
    /// Adds a batch stage to the pipeline
    /// </summary>
    public GpuPipeline Batch<T>(int batchSize, TimeSpan? timeout = null)
        where T : notnull
    {
        _stages.Add(new BatchStage<T>(batchSize, timeout ?? TimeSpan.FromMilliseconds(100)));
        return this;
    }

    /// <summary>
    /// Adds a parallel stage to the pipeline
    /// </summary>
    public GpuPipeline Parallel<TIn, TOut>(
        Func<TIn, Task<TOut>> processor,
        int maxConcurrency = 0)
        where TIn : notnull
        where TOut : notnull
    {
        var concurrency = maxConcurrency > 0 ? maxConcurrency : Environment.ProcessorCount;
        _stages.Add(new ParallelStage<TIn, TOut>(processor, concurrency));
        return this;
    }

    /// <summary>
    /// Adds a filter stage to the pipeline
    /// </summary>
    public GpuPipeline Filter<T>(Func<T, bool> predicate)
        where T : notnull
    {
        _stages.Add(new FilterStage<T>(predicate));
        return this;
    }

    /// <summary>
    /// Adds a tap stage for side effects
    /// </summary>
    public GpuPipeline Tap<T>(Action<T> action)
        where T : notnull
    {
        _stages.Add(new TapStage<T>(action));
        return this;
    }

    /// <summary>
    /// Builds and returns an executable pipeline
    /// </summary>
    public IPipeline<TInput, TOutput> Build<TInput, TOutput>()
        where TInput : notnull
        where TOutput : notnull
    {
        return new ExecutablePipeline<TInput, TOutput>(_stages, _logger);
    }
}

/// <summary>
/// Typed GPU pipeline builder for strongly-typed kernel execution
/// </summary>
public sealed class GpuPipelineBuilder<TIn, TOut>
    where TIn : notnull
    where TOut : notnull
{
    private readonly IGrainFactory _grainFactory;
    private readonly KernelId _kernelId;
    private int _batchSize = 100;
    private int _maxConcurrency = 1;

    /// <summary>Maximum retry attempts for failed batches</summary>
    private const int MaxRetryAttempts = 3;

    /// <summary>Base delay for exponential backoff (100ms)</summary>
    private static readonly TimeSpan BaseRetryDelay = TimeSpan.FromMilliseconds(100);

    /// <summary>
    /// Initializes a new instance of the <see cref="GpuPipelineBuilder{TIn, TOut}"/> class.
    /// </summary>
    /// <param name="grainFactory">The Orleans grain factory.</param>
    /// <param name="kernelId">The kernel identifier.</param>
    public GpuPipelineBuilder(IGrainFactory grainFactory, KernelId kernelId)
    {
        _grainFactory = grainFactory;
        _kernelId = kernelId;
    }

    /// <summary>
    /// Sets the batch size for processing
    /// </summary>
    public GpuPipelineBuilder<TIn, TOut> WithBatchSize(int batchSize)
    {
        _batchSize = batchSize;
        return this;
    }

    /// <summary>
    /// Sets the maximum concurrency for parallel batch execution
    /// </summary>
    /// <param name="maxConcurrency">Maximum number of concurrent batch operations</param>
    /// <returns>The pipeline builder for fluent chaining</returns>
    /// <remarks>
    /// Controls how many batches can be processed in parallel.
    /// Default is 1 (sequential processing).
    /// TODO: Implement parallel batch execution using SemaphoreSlim or similar
    /// </remarks>
    public GpuPipelineBuilder<TIn, TOut> WithMaxConcurrency(int maxConcurrency)
    {
        if (maxConcurrency < 1)
            throw new ArgumentOutOfRangeException(nameof(maxConcurrency), "Max concurrency must be at least 1");

        _maxConcurrency = maxConcurrency;
        return this;
    }

    /// <summary>
    /// Executes the pipeline with the given input data using parallel batch processing
    /// </summary>
    /// <remarks>
    /// Uses SemaphoreSlim for concurrency control and implements retry logic with
    /// exponential backoff for failed batches (up to 3 attempts).
    /// </remarks>
    public async Task<IReadOnlyList<TOut>> ExecuteAsync(
        IReadOnlyList<TIn> input,
        CancellationToken cancellationToken = default)
    {
        if (input.Count == 0)
        {
            return Array.Empty<TOut>();
        }

        // Prepare batches
        var batches = new List<(int Index, List<TIn> Items)>();
        for (int i = 0; i < input.Count; i += _batchSize)
        {
            var batchEnd = Math.Min(i + _batchSize, input.Count);
            var batch = input.Skip(i).Take(batchEnd - i).ToList();
            batches.Add((i / _batchSize, batch));
        }

        // Results collection (thread-safe, indexed by batch)
        var batchResults = new ConcurrentDictionary<int, IReadOnlyList<TOut>>();

        // Use SemaphoreSlim for concurrency control
        using var semaphore = new SemaphoreSlim(_maxConcurrency, _maxConcurrency);

        // Execute batches in parallel with controlled concurrency
        var batchTasks = batches.Select(batch => ProcessBatchWithRetryAsync(
            batch.Index,
            batch.Items,
            batchResults,
            semaphore,
            cancellationToken)).ToList();

        await Task.WhenAll(batchTasks);

        // Aggregate results in original batch order
        var orderedResults = new List<TOut>();
        for (int i = 0; i < batches.Count; i++)
        {
            if (batchResults.TryGetValue(i, out var results))
            {
                orderedResults.AddRange(results);
            }
        }

        return orderedResults;
    }

    /// <summary>
    /// Processes a single batch with retry logic and exponential backoff
    /// </summary>
    private async Task ProcessBatchWithRetryAsync(
        int batchIndex,
        List<TIn> batch,
        ConcurrentDictionary<int, IReadOnlyList<TOut>> results,
        SemaphoreSlim semaphore,
        CancellationToken cancellationToken)
    {
        await semaphore.WaitAsync(cancellationToken);

        try
        {
            for (int attempt = 1; attempt <= MaxRetryAttempts; attempt++)
            {
                cancellationToken.ThrowIfCancellationRequested();

                try
                {
                    // Get a batch grain with unique ID for parallel execution
                    var grain = _grainFactory.GetGrain<IGpuBatchGrain<TIn, TOut>>(
                        Guid.NewGuid(), _kernelId.Value, null);

                    var batchResult = await grain.ExecuteAsync(batch);

                    if (batchResult.Success && batchResult.Results != null)
                    {
                        results[batchIndex] = batchResult.Results;
                        return; // Success - exit retry loop
                    }

                    // Batch failed but no exception - treat as retriable failure
                    if (attempt < MaxRetryAttempts)
                    {
                        var delay = CalculateExponentialBackoff(attempt);
                        await Task.Delay(delay, cancellationToken);
                    }
                }
                catch (OperationCanceledException)
                {
                    throw; // Don't retry on cancellation
                }
                catch (Exception) when (attempt < MaxRetryAttempts)
                {
                    // Retry with exponential backoff
                    var delay = CalculateExponentialBackoff(attempt);
                    await Task.Delay(delay, cancellationToken);
                }
            }

            // All retries exhausted - store empty result
            results[batchIndex] = Array.Empty<TOut>();
        }
        finally
        {
            semaphore.Release();
        }
    }

    /// <summary>
    /// Calculates exponential backoff delay: BaseDelay * 2^(attempt-1)
    /// </summary>
    private static TimeSpan CalculateExponentialBackoff(int attempt)
    {
        // 100ms, 200ms, 400ms for attempts 1, 2, 3
        var multiplier = Math.Pow(2, attempt - 1);
        return TimeSpan.FromMilliseconds(BaseRetryDelay.TotalMilliseconds * multiplier);
    }
}
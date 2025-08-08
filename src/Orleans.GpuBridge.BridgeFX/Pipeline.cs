using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Channels;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Orleans.GpuBridge.Abstractions;

namespace Orleans.GpuBridge.BridgeFX;

/// <summary>
/// Fluent pipeline builder for GPU processing
/// </summary>
public sealed class GpuPipeline
{
    private readonly List<IPipelineStage> _stages = new();
    private readonly ILogger<GpuPipeline> _logger;
    private readonly IGpuBridge _bridge;
    
    public GpuPipeline(IGpuBridge bridge, ILogger<GpuPipeline> logger)
    {
        _bridge = bridge;
        _logger = logger;
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
/// Interface for pipeline stages
/// </summary>
public interface IPipelineStage
{
    Type InputType { get; }
    Type OutputType { get; }
    Task<object?> ProcessAsync(object input, CancellationToken ct = default);
}

/// <summary>
/// Interface for executable pipeline
/// </summary>
public interface IPipeline<TInput, TOutput>
    where TInput : notnull
    where TOutput : notnull
{
    /// <summary>
    /// Processes a single item through the pipeline
    /// </summary>
    Task<TOutput> ProcessAsync(TInput input, CancellationToken ct = default);
    
    /// <summary>
    /// Processes multiple items through the pipeline
    /// </summary>
    IAsyncEnumerable<TOutput> ProcessManyAsync(
        IAsyncEnumerable<TInput> inputs,
        CancellationToken ct = default);
    
    /// <summary>
    /// Processes items from a channel
    /// </summary>
    Task ProcessChannelAsync(
        ChannelReader<TInput> input,
        ChannelWriter<TOutput> output,
        CancellationToken ct = default);
}

/// <summary>
/// Kernel processing stage
/// </summary>
internal sealed class KernelStage<TIn, TOut> : IPipelineStage
    where TIn : notnull
    where TOut : notnull
{
    private readonly KernelId _kernelId;
    private readonly IGpuBridge _bridge;
    private readonly Func<TIn, bool>? _filter;
    private IGpuKernel<TIn, TOut>? _kernel;
    
    public Type InputType => typeof(TIn);
    public Type OutputType => typeof(TOut);
    
    public KernelStage(KernelId kernelId, IGpuBridge bridge, Func<TIn, bool>? filter)
    {
        _kernelId = kernelId;
        _bridge = bridge;
        _filter = filter;
    }
    
    public async Task<object?> ProcessAsync(object input, CancellationToken ct)
    {
        if (input is not TIn typedInput)
        {
            throw new ArgumentException($"Expected {typeof(TIn)}, got {input.GetType()}");
        }
        
        // Apply filter if specified
        if (_filter != null && !_filter(typedInput))
        {
            return null;
        }
        
        // Get kernel lazily
        _kernel ??= await _bridge.GetKernelAsync<TIn, TOut>(_kernelId, ct);
        
        // Process single item
        var handle = await _kernel.SubmitBatchAsync(new[] { typedInput }, null, ct);
        
        await foreach (var result in _kernel.ReadResultsAsync(handle, ct))
        {
            return result;
        }
        
        return null;
    }
}

/// <summary>
/// Transform stage
/// </summary>
internal sealed class TransformStage<TIn, TOut> : IPipelineStage
    where TIn : notnull
    where TOut : notnull
{
    private readonly Func<TIn, TOut> _transform;
    
    public Type InputType => typeof(TIn);
    public Type OutputType => typeof(TOut);
    
    public TransformStage(Func<TIn, TOut> transform)
    {
        _transform = transform;
    }
    
    public Task<object?> ProcessAsync(object input, CancellationToken ct)
    {
        if (input is not TIn typedInput)
        {
            throw new ArgumentException($"Expected {typeof(TIn)}, got {input.GetType()}");
        }
        
        var result = _transform(typedInput);
        return Task.FromResult<object?>(result);
    }
}

/// <summary>
/// Batch collection stage
/// </summary>
internal sealed class BatchStage<T> : IPipelineStage
    where T : notnull
{
    private readonly int _batchSize;
    private readonly TimeSpan _timeout;
    private readonly List<T> _buffer = new();
    private DateTime _lastFlush = DateTime.UtcNow;
    
    public Type InputType => typeof(T);
    public Type OutputType => typeof(IReadOnlyList<T>);
    
    public BatchStage(int batchSize, TimeSpan timeout)
    {
        _batchSize = batchSize;
        _timeout = timeout;
    }
    
    public Task<object?> ProcessAsync(object input, CancellationToken ct)
    {
        if (input is not T typedInput)
        {
            throw new ArgumentException($"Expected {typeof(T)}, got {input.GetType()}");
        }
        
        lock (_buffer)
        {
            _buffer.Add(typedInput);
            
            var shouldFlush = _buffer.Count >= _batchSize ||
                             DateTime.UtcNow - _lastFlush >= _timeout;
            
            if (shouldFlush)
            {
                var batch = _buffer.ToList();
                _buffer.Clear();
                _lastFlush = DateTime.UtcNow;
                return Task.FromResult<object?>(batch);
            }
        }
        
        return Task.FromResult<object?>(null);
    }
}

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

/// <summary>
/// Tap stage for side effects
/// </summary>
internal sealed class TapStage<T> : IPipelineStage
    where T : notnull
{
    private readonly Action<T> _action;
    
    public Type InputType => typeof(T);
    public Type OutputType => typeof(T);
    
    public TapStage(Action<T> action)
    {
        _action = action;
    }
    
    public Task<object?> ProcessAsync(object input, CancellationToken ct)
    {
        if (input is not T typedInput)
        {
            throw new ArgumentException($"Expected {typeof(T)}, got {input.GetType()}");
        }
        
        _action(typedInput);
        return Task.FromResult<object?>(typedInput);
    }
}

/// <summary>
/// Executable pipeline implementation
/// </summary>
internal sealed class ExecutablePipeline<TInput, TOutput> : IPipeline<TInput, TOutput>
    where TInput : notnull
    where TOutput : notnull
{
    private readonly IReadOnlyList<IPipelineStage> _stages;
    private readonly ILogger _logger;
    
    public ExecutablePipeline(IReadOnlyList<IPipelineStage> stages, ILogger logger)
    {
        _stages = stages;
        _logger = logger;
        ValidatePipeline();
    }
    
    private void ValidatePipeline()
    {
        if (_stages.Count == 0)
        {
            throw new InvalidOperationException("Pipeline must have at least one stage");
        }
        
        // Validate type flow
        var expectedInput = typeof(TInput);
        
        for (int i = 0; i < _stages.Count; i++)
        {
            var stage = _stages[i];
            
            if (i == 0 && !stage.InputType.IsAssignableFrom(expectedInput))
            {
                throw new InvalidOperationException(
                    $"First stage expects {stage.InputType} but pipeline input is {expectedInput}");
            }
            
            if (i > 0)
            {
                var prevOutput = _stages[i - 1].OutputType;
                if (!stage.InputType.IsAssignableFrom(prevOutput))
                {
                    throw new InvalidOperationException(
                        $"Stage {i} expects {stage.InputType} but previous stage outputs {prevOutput}");
                }
            }
        }
        
        var lastOutput = _stages[^1].OutputType;
        if (!typeof(TOutput).IsAssignableFrom(lastOutput))
        {
            throw new InvalidOperationException(
                $"Pipeline output is {typeof(TOutput)} but last stage outputs {lastOutput}");
        }
    }
    
    public async Task<TOutput> ProcessAsync(TInput input, CancellationToken ct = default)
    {
        object? current = input;
        
        foreach (var stage in _stages)
        {
            if (current == null)
            {
                throw new InvalidOperationException("Pipeline stage returned null");
            }
            
            current = await stage.ProcessAsync(current, ct);
        }
        
        if (current is not TOutput output)
        {
            throw new InvalidOperationException(
                $"Pipeline produced {current?.GetType()} but expected {typeof(TOutput)}");
        }
        
        return output;
    }
    
    public async IAsyncEnumerable<TOutput> ProcessManyAsync(
        IAsyncEnumerable<TInput> inputs,
        [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken ct = default)
    {
        await foreach (var input in inputs.WithCancellation(ct))
        {
            TOutput output;
            try
            {
                output = await ProcessAsync(input, ct);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Pipeline processing failed for input");
                continue;
            }
            
            yield return output;
        }
    }
    
    public async Task ProcessChannelAsync(
        ChannelReader<TInput> input,
        ChannelWriter<TOutput> output,
        CancellationToken ct = default)
    {
        try
        {
            await foreach (var item in input.ReadAllAsync(ct))
            {
                try
                {
                    var result = await ProcessAsync(item, ct);
                    await output.WriteAsync(result, ct);
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Failed to process channel item");
                }
            }
        }
        finally
        {
            output.TryComplete();
        }
    }
}
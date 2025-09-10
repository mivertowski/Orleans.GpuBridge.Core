using System;
using System.Collections.Generic;
using System.Linq;
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
    /// Executes the pipeline with the given input data
    /// </summary>
    public async Task<IReadOnlyList<TOut>> ExecuteAsync(IReadOnlyList<TIn> input)
    {
        var results = new List<TOut>();
        
        // Process input in batches using Orleans grains
        for (int i = 0; i < input.Count; i += _batchSize)
        {
            var batchEnd = Math.Min(i + _batchSize, input.Count);
            var batch = input.Skip(i).Take(batchEnd - i).ToList();
            
            // Get a batch grain and execute
            var grain = _grainFactory.GetGrain<IGpuBatchGrain<TIn, TOut>>(
                Guid.NewGuid(), _kernelId.Value, null);
            
            var batchResult = await grain.ExecuteAsync(batch);
            if (batchResult.Success && batchResult.Results != null)
            {
                results.AddRange(batchResult.Results);
            }
        }
        
        return results;
    }
}
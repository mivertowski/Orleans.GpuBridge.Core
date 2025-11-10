using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Channels;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Orleans.GpuBridge.BridgeFX.Pipeline.Core;

namespace Orleans.GpuBridge.BridgeFX;

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
            // Allow null values to flow through the pipeline for nullable type support
            // Stages are responsible for handling null appropriately
            current = await stage.ProcessAsync(current, ct);
        }

        if (current is not TOutput output)
        {
            throw new InvalidOperationException(
                $"Pipeline produced {current?.GetType() ?? typeof(object)} but expected {typeof(TOutput)}");
        }

        return output;
    }
    
    public async IAsyncEnumerable<TOutput> ProcessManyAsync(
        IAsyncEnumerable<TInput> inputs,
        [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken ct = default)
    {
        // Check if pipeline has parallel stages for concurrent processing
        var hasParallelStage = _stages.Any(s => s.GetType().Name.Contains("Parallel"));

        if (hasParallelStage)
        {
            // Use channels for concurrent processing with parallel stages
            var inputChannel = Channel.CreateUnbounded<TInput>();
            var outputChannel = Channel.CreateUnbounded<TOutput>();

            // Background task to feed inputs into channel
            var feedTask = Task.Run(async () =>
            {
                try
                {
                    await foreach (var input in inputs.WithCancellation(ct))
                    {
                        await inputChannel.Writer.WriteAsync(input, ct);
                    }
                }
                finally
                {
                    inputChannel.Writer.Complete();
                }
            }, ct);

            // Background task to process items concurrently
            var processTask = Task.Run(async () =>
            {
                var tasks = new List<Task>();
                try
                {
                    await foreach (var input in inputChannel.Reader.ReadAllAsync(ct))
                    {
                        var task = Task.Run(async () =>
                        {
                            try
                            {
                                var output = await ProcessAsync(input, ct);
                                await outputChannel.Writer.WriteAsync(output, ct);
                            }
                            catch (OperationCanceledException)
                            {
                                throw;
                            }
                            catch (Exception ex)
                            {
                                _logger.LogError(ex, "Pipeline processing failed for input");
                            }
                        }, ct);
                        tasks.Add(task);
                    }

                    await Task.WhenAll(tasks);
                }
                finally
                {
                    outputChannel.Writer.Complete();
                }
            }, ct);

            // Yield results as they become available
            await foreach (var output in outputChannel.Reader.ReadAllAsync(ct))
            {
                yield return output;
            }

            await feedTask;
            await processTask;
        }
        else
        {
            // Sequential processing for pipelines without parallel stages
            await foreach (var input in inputs.WithCancellation(ct))
            {
                TOutput output;
                try
                {
                    output = await ProcessAsync(input, ct);
                }
                catch (OperationCanceledException)
                {
                    // Re-throw cancellation exceptions to properly propagate cancellation
                    throw;
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Pipeline processing failed for input");
                    continue;
                }

                yield return output;
            }
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
                catch (OperationCanceledException)
                {
                    // Re-throw cancellation exceptions to properly propagate cancellation
                    throw;
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Failed to process channel item");
                    // Continue processing remaining items on non-cancellation errors
                }
            }
        }
        finally
        {
            output.TryComplete();
        }
    }
}
using System;
using System.Collections.Generic;
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
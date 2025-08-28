using System.Collections.Generic;
using System.Threading;
using System.Threading.Channels;
using System.Threading.Tasks;

namespace Orleans.GpuBridge.BridgeFX;

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
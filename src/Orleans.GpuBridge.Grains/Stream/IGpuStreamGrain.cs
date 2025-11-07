using System.Threading.Tasks;
using Orleans;
using Orleans.GpuBridge.Abstractions;
using Orleans.GpuBridge.Grains.Batch;
using Orleans.Streams;

namespace Orleans.GpuBridge.Grains.Stream;

/// <summary>
/// Interface for stream processing grain that provides real-time GPU-accelerated stream processing.
/// This grain enables continuous processing of streaming data through GPU kernels with batching and buffering.
/// </summary>
public interface IGpuStreamGrain<TIn, TOut> : IGrainWithStringKey
    where TIn : notnull
    where TOut : notnull
{
    /// <summary>
    /// Starts processing from input stream to output stream using Orleans Streams infrastructure.
    /// This method subscribes to the input stream and publishes processed results to the output stream.
    /// </summary>
    /// <param name="inputStream">The Orleans StreamId to consume input data from.</param>
    /// <param name="outputStream">The Orleans StreamId to publish processed results to.</param>
    /// <param name="hints">Optional execution hints for GPU kernel optimization (batch size, device preferences, etc.).</param>
    /// <returns>A task representing the asynchronous start operation.</returns>
    /// <exception cref="InvalidOperationException">Thrown when stream processing is already running.</exception>
    Task StartProcessingAsync(
        StreamId inputStream,
        StreamId outputStream,
        GpuExecutionHints? hints = null);

    /// <summary>
    /// Starts a custom stream processing pipeline with observer pattern for result delivery.
    /// This method provides more direct control over result handling compared to Orleans Streams.
    /// </summary>
    /// <param name="streamId">The identifier for this stream processing session.</param>
    /// <param name="observer">The observer that will receive processed results via callback.</param>
    /// <param name="hints">Optional execution hints for GPU kernel optimization.</param>
    /// <returns>A task representing the asynchronous start operation.</returns>
    /// <exception cref="InvalidOperationException">Thrown when stream processing is already running.</exception>
    /// <exception cref="ArgumentNullException">Thrown when observer is null.</exception>
    Task StartStreamAsync(
        string streamId,
        IGpuResultObserver<TOut> observer,
        GpuExecutionHints? hints = null);

    /// <summary>
    /// Processes a single item through the GPU kernel asynchronously.
    /// Items are buffered and batched internally for efficient GPU execution.
    /// Results are delivered via the observer registered in StartStreamAsync.
    /// </summary>
    /// <param name="item">The input item to process through the GPU kernel.</param>
    /// <returns>A task representing the asynchronous processing operation.</returns>
    /// <exception cref="InvalidOperationException">Thrown when stream processing is not started.</exception>
    Task ProcessItemAsync(TIn item);

    /// <summary>
    /// Flushes any buffered items and ensures all pending processing is completed.
    /// This method blocks until all items in the internal buffer are processed and results delivered.
    /// </summary>
    /// <returns>A task representing the asynchronous flush operation.</returns>
    /// <exception cref="InvalidOperationException">Thrown when stream processing is not started.</exception>
    Task FlushStreamAsync();

    /// <summary>
    /// Stops stream processing and cleans up resources.
    /// Any buffered items will be flushed before stopping.
    /// </summary>
    /// <returns>A task representing the asynchronous stop operation.</returns>
    Task StopProcessingAsync();

    /// <summary>
    /// Gets the current processing status of the stream grain.
    /// </summary>
    /// <returns>
    /// A task that resolves to the current <see cref="StreamProcessingStatus"/>
    /// (Idle, Starting, Processing, Stopping, Stopped, or Failed).
    /// </returns>
    Task<StreamProcessingStatus> GetStatusAsync();

    /// <summary>
    /// Gets comprehensive processing statistics including throughput, latency, and error counts.
    /// </summary>
    /// <returns>
    /// A task that resolves to <see cref="StreamProcessingStats"/> containing:
    /// - Items processed and failed counts
    /// - Total processing time and average latency
    /// - Start time and last processed timestamp
    /// </returns>
    Task<StreamProcessingStats> GetStatsAsync();
}
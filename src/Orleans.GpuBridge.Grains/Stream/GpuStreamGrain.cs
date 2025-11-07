using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Channels;
using System.Threading.Tasks;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Orleans;
using Orleans.Concurrency;
using Orleans.GpuBridge.Abstractions;
using Orleans.GpuBridge.Grains.Batch;
using Orleans.GpuBridge.Grains.Stream.Internal;
using Orleans.GpuBridge.Runtime;
using Orleans.Streams;

namespace Orleans.GpuBridge.Grains.Stream;

/// <summary>
/// GPU stream processing grain
/// </summary>
[Reentrant]
public sealed class GpuStreamGrain<TIn, TOut> : Grain, IGpuStreamGrain<TIn, TOut>
    where TIn : notnull
    where TOut : notnull
{
    private readonly ILogger<GpuStreamGrain<TIn, TOut>> _logger;
    private readonly Channel<TIn> _buffer;
    private IGpuKernel<TIn, TOut> _kernel = default!;
    private IAsyncStream<TIn> _inputStream = default!;
    private IAsyncStream<TOut> _outputStream = default!;
    private StreamSubscriptionHandle<TIn>? _subscription;
    private CancellationTokenSource? _cts;
    private Task? _processingTask;
    private StreamProcessingStatus _status = StreamProcessingStatus.Idle;
    private readonly StreamProcessingStatsTracker _stats = new();
    
    public GpuStreamGrain(ILogger<GpuStreamGrain<TIn, TOut>> logger)
    {
        _logger = logger;
        _buffer = Channel.CreateUnbounded<TIn>(
            new UnboundedChannelOptions
            {
                SingleReader = true,
                SingleWriter = false
            });
    }
    
    public async Task StartProcessingAsync(
        StreamId inputStream,
        StreamId outputStream,
        GpuExecutionHints? hints = null)
    {
        if (_status == StreamProcessingStatus.Processing)
        {
            throw new InvalidOperationException("Already processing");
        }
        
        _status = StreamProcessingStatus.Starting;
        
        try
        {
            // Get kernel
            var kernelId = KernelId.Parse(this.GetPrimaryKeyString());
            var bridge = ServiceProvider.GetRequiredService<IGpuBridge>();
            _kernel = await bridge.GetKernelAsync<TIn, TOut>(kernelId);
            
            // Get streams
            var streamProvider = this.GetStreamProvider("Default");
            _inputStream = streamProvider.GetStream<TIn>(inputStream);
            _outputStream = streamProvider.GetStream<TOut>(outputStream);
            
            // Subscribe to input stream
            _subscription = await _inputStream.SubscribeAsync(
                async (item, token) =>
                {
                    await _buffer.Writer.WriteAsync(item);
                });
            
            // Start processing loop
            _cts = new CancellationTokenSource();
            _processingTask = ProcessStreamAsync(hints, _cts.Token);
            
            _status = StreamProcessingStatus.Processing;
            _stats.Start();
            
            _logger.LogInformation(
                "Started stream processing from {Input} to {Output}",
                inputStream, outputStream);
        }
        catch (Exception ex)
        {
            _status = StreamProcessingStatus.Failed;
            _logger.LogError(ex, "Failed to start stream processing");
            throw;
        }
    }
    
    public async Task StopProcessingAsync()
    {
        if (_status != StreamProcessingStatus.Processing)
        {
            return;
        }
        
        _status = StreamProcessingStatus.Stopping;
        
        try
        {
            // Unsubscribe from input
            if (_subscription != null)
            {
                await _subscription.UnsubscribeAsync();
                _subscription = null;
            }
            
            // Signal completion
            _buffer.Writer.TryComplete();
            
            // Cancel processing
            _cts?.Cancel();
            
            // Wait for processing to complete
            if (_processingTask != null)
            {
                await _processingTask;
            }
            
            _status = StreamProcessingStatus.Stopped;
            
            _logger.LogInformation(
                "Stopped stream processing. Processed {Count} items",
                _stats.ItemsProcessed);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error stopping stream processing");
            _status = StreamProcessingStatus.Failed;
        }
    }
    
    public Task<StreamProcessingStatus> GetStatusAsync()
    {
        return Task.FromResult(_status);
    }
    
    public Task<StreamProcessingStats> GetStatsAsync()
    {
        return Task.FromResult(_stats.GetStats());
    }

    public Task StartStreamAsync(
        string streamId,
        IGpuResultObserver<TOut> observer,
        GpuExecutionHints? hints = null)
    {
        ArgumentNullException.ThrowIfNull(observer);
        ArgumentException.ThrowIfNullOrEmpty(streamId);

        if (_status == StreamProcessingStatus.Processing)
        {
            throw new InvalidOperationException("Stream processing is already active");
        }

        _logger.LogInformation(
            "Starting custom stream processing with observer for stream {StreamId}",
            streamId);

        _status = StreamProcessingStatus.Starting;

        try
        {
            // Get kernel
            var kernelId = KernelId.Parse(this.GetPrimaryKeyString());
            var bridge = ServiceProvider.GetRequiredService<IGpuBridge>();
            _kernel = bridge.GetKernelAsync<TIn, TOut>(kernelId).GetAwaiter().GetResult();

            // Start processing loop with observer
            _cts = new CancellationTokenSource();
            _processingTask = ProcessStreamWithObserverAsync(observer, hints, _cts.Token);

            _status = StreamProcessingStatus.Processing;
            _stats.Start();

            _logger.LogInformation("Started custom stream processing for {StreamId}", streamId);

            return Task.CompletedTask;
        }
        catch (Exception ex)
        {
            _status = StreamProcessingStatus.Failed;
            _logger.LogError(ex, "Failed to start custom stream processing");
            throw;
        }
    }

    public async Task ProcessItemAsync(TIn item)
    {
        if (_status != StreamProcessingStatus.Processing)
        {
            throw new InvalidOperationException("Stream processing is not active. Call StartStreamAsync first.");
        }

        await _buffer.Writer.WriteAsync(item);

        _logger.LogTrace("Queued item for processing");
    }

    public async Task FlushStreamAsync()
    {
        if (_status != StreamProcessingStatus.Processing)
        {
            throw new InvalidOperationException("Stream processing is not active");
        }

        _logger.LogDebug("Flushing stream buffer");

        // Wait until buffer is empty
        while (_buffer.Reader.Count > 0)
        {
            await Task.Delay(50);
        }

        _logger.LogInformation("Stream buffer flushed");
    }

    private async Task ProcessStreamWithObserverAsync(
        IGpuResultObserver<TOut> observer,
        GpuExecutionHints? hints,
        CancellationToken ct)
    {
        const int defaultBatchSize = 128;
        var batchSize = hints?.MaxMicroBatch ?? defaultBatchSize;
        var batch = new List<TIn>(batchSize);
        var timer = new PeriodicTimer(TimeSpan.FromMilliseconds(100));

        try
        {
            while (!ct.IsCancellationRequested)
            {
                // Collect batch
                while (batch.Count < batchSize &&
                       _buffer.Reader.TryRead(out var item))
                {
                    batch.Add(item);
                }

                // Process if we have items or timeout
                if (batch.Count > 0)
                {
                    var shouldProcess = batch.Count >= batchSize ||
                                       await timer.WaitForNextTickAsync(ct);

                    if (shouldProcess)
                    {
                        await ProcessBatchWithObserverAsync(batch, observer, hints, ct);
                        batch.Clear();
                    }
                }
                else
                {
                    // Wait for data
                    await Task.Delay(10, ct);
                }
            }
        }
        catch (OperationCanceledException)
        {
            _logger.LogInformation("Stream processing with observer cancelled");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Stream processing with observer failed");
            _status = StreamProcessingStatus.Failed;
            await observer.OnErrorAsync(ex);
        }
        finally
        {
            timer.Dispose();
            await observer.OnCompletedAsync();
        }
    }

    private async Task ProcessBatchWithObserverAsync(
        List<TIn> batch,
        IGpuResultObserver<TOut> observer,
        GpuExecutionHints? hints,
        CancellationToken ct)
    {
        var stopwatch = System.Diagnostics.Stopwatch.StartNew();

        try
        {
            var handle = await _kernel.SubmitBatchAsync(batch, hints, ct);

            await foreach (var result in _kernel.ReadResultsAsync(handle, ct))
            {
                await observer.OnNextAsync(result);
            }

            stopwatch.Stop();
            _stats.RecordSuccess(batch.Count, stopwatch.Elapsed);

            _logger.LogDebug(
                "Processed batch of {Count} items in {ElapsedMs}ms via observer",
                batch.Count, stopwatch.ElapsedMilliseconds);
        }
        catch (Exception ex)
        {
            _stats.RecordFailure(batch.Count);
            _logger.LogError(ex,
                "Failed to process batch of {Count} items via observer",
                batch.Count);
            await observer.OnErrorAsync(ex);
        }
    }

    private async Task ProcessStreamAsync(
        GpuExecutionHints? hints,
        CancellationToken ct)
    {
        const int defaultBatchSize = 128;
        var batchSize = hints?.MaxMicroBatch ?? defaultBatchSize;
        var batch = new List<TIn>(batchSize);
        var timer = new PeriodicTimer(TimeSpan.FromMilliseconds(100));
        
        try
        {
            while (!ct.IsCancellationRequested)
            {
                // Collect batch
                while (batch.Count < batchSize &&
                       _buffer.Reader.TryRead(out var item))
                {
                    batch.Add(item);
                }
                
                // Process if we have items or timeout
                if (batch.Count > 0)
                {
                    var shouldProcess = batch.Count >= batchSize ||
                                       await timer.WaitForNextTickAsync(ct);
                    
                    if (shouldProcess)
                    {
                        await ProcessBatchAsync(batch, hints, ct);
                        batch.Clear();
                    }
                }
                else
                {
                    // Wait for data
                    await Task.Delay(10, ct);
                }
            }
        }
        catch (OperationCanceledException)
        {
            _logger.LogInformation("Stream processing cancelled");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Stream processing failed");
            _status = StreamProcessingStatus.Failed;
        }
        finally
        {
            timer.Dispose();
        }
    }
    
    private async Task ProcessBatchAsync(
        List<TIn> batch,
        GpuExecutionHints? hints,
        CancellationToken ct)
    {
        var stopwatch = System.Diagnostics.Stopwatch.StartNew();
        
        try
        {
            var handle = await _kernel.SubmitBatchAsync(batch, hints, ct);
            
            await foreach (var result in _kernel.ReadResultsAsync(handle, ct))
            {
                await _outputStream.OnNextAsync(result);
            }
            
            stopwatch.Stop();
            _stats.RecordSuccess(batch.Count, stopwatch.Elapsed);
            
            _logger.LogDebug(
                "Processed batch of {Count} items in {ElapsedMs}ms",
                batch.Count, stopwatch.ElapsedMilliseconds);
        }
        catch (Exception ex)
        {
            _stats.RecordFailure(batch.Count);
            _logger.LogError(ex,
                "Failed to process batch of {Count} items",
                batch.Count);
        }
    }
}
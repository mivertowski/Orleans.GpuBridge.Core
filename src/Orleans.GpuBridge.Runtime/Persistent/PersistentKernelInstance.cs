// Copyright (c) 2025 Michael Ivertowski. All Rights Reserved.
// Licensed under the Apache License, Version 2.0.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Orleans.GpuBridge.Abstractions;
using Orleans.GpuBridge.Abstractions.Kernels;

namespace Orleans.GpuBridge.Runtime.Persistent;

/// <summary>
/// Represents a running persistent kernel instance
/// </summary>
public sealed class PersistentKernelInstance : IDisposable
{
    private readonly IGpuKernel<byte[], byte[]> _kernel;
    private readonly PersistentKernelOptions _options;
    private readonly ILogger _logger;
    private readonly CancellationTokenSource _cts;
    private Task? _executionTask;
    private KernelState _state;
    private DateTime _startTime;
    private DateTime _lastActivity;
    private long _processedBatches;
    private long _failedBatches;
    private long _totalBytesProcessed;
    private Exception? _lastError;
    private bool _disposed;

    public string InstanceId { get; }
    public KernelId KernelId { get; }

    public PersistentKernelInstance(
        string instanceId,
        KernelId kernelId,
        IGpuKernel<byte[], byte[]> kernel,
        PersistentKernelOptions options,
        ILogger logger)
    {
        InstanceId = instanceId;
        KernelId = kernelId;
        _kernel = kernel;
        _options = options;
        _logger = logger;
        _cts = new CancellationTokenSource();
        _state = KernelState.Idle;
        _lastActivity = DateTime.UtcNow;
    }

    public async Task StartAsync(CancellationToken ct = default)
    {
        if (_state == KernelState.Running)
            return;

        _state = KernelState.Starting;
        _startTime = DateTime.UtcNow;
        _lastActivity = _startTime;

        try
        {
            // Initialize kernel
            await _kernel.InitializeAsync(ct);

            _logger.LogDebug(
                "Starting persistent kernel {KernelId} (Provider: {Provider}) with batch size {BatchSize}",
                _kernel.KernelId, _kernel.BackendProvider, _options.BatchSize);

            // Start execution loop
            _executionTask = ExecutionLoopAsync(_cts.Token);

            _state = KernelState.Running;
        }
        catch (Exception ex)
        {
            _state = KernelState.Failed;
            _lastError = ex;
            throw;
        }
    }

    public async Task StopAsync(CancellationToken ct = default)
    {
        if (_state != KernelState.Running && _state != KernelState.Starting)
            return;

        _logger.LogDebug("Stopping kernel instance {InstanceId}", InstanceId);

        _state = KernelState.Stopping;
        _cts.Cancel();

        if (_executionTask != null)
        {
            try
            {
                await _executionTask.WaitAsync(TimeSpan.FromSeconds(30), ct);
            }
            catch (TimeoutException)
            {
                _logger.LogWarning(
                    "Kernel instance {InstanceId} did not stop gracefully",
                    InstanceId);
            }
        }

        _state = KernelState.Stopped;
        _lastActivity = DateTime.UtcNow;
    }

    public async Task RestartAsync(CancellationToken ct = default)
    {
        await StopAsync(ct);
        await StartAsync(ct);
    }

    private async Task ExecutionLoopAsync(CancellationToken ct)
    {
        var batchBuffer = new List<byte[]>(_options.BatchSize);
        var lastFlush = DateTime.UtcNow;

        try
        {
            while (!ct.IsCancellationRequested)
            {
                try
                {
                    // Simulate getting data from ring buffer
                    // In real implementation, this would read from RingBuffer
                    await Task.Delay(10, ct);

                    // Check if we should flush the batch
                    var shouldFlush = batchBuffer.Count >= _options.BatchSize ||
                                     (batchBuffer.Count > 0 &&
                                      DateTime.UtcNow - lastFlush > _options.MaxBatchWaitTime);

                    if (shouldFlush && batchBuffer.Count > 0)
                    {
                        await ProcessBatchAsync(batchBuffer, ct);
                        batchBuffer.Clear();
                        lastFlush = DateTime.UtcNow;
                    }
                }
                catch (OperationCanceledException)
                {
                    break;
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Error in kernel execution loop");
                    _lastError = ex;
                    Interlocked.Increment(ref _failedBatches);

                    if (_options.RestartOnError)
                    {
                        _state = KernelState.Failed;
                        break;
                    }

                    // Continue after error with delay
                    await Task.Delay(1000, ct);
                }
            }
        }
        finally
        {
            // Process remaining items
            if (batchBuffer.Count > 0)
            {
                try
                {
                    await ProcessBatchAsync(batchBuffer, CancellationToken.None);
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Error processing final batch");
                }
            }
        }
    }

    private async Task ProcessBatchAsync(IReadOnlyList<byte[]> batch, CancellationToken ct)
    {
        var stopwatch = Stopwatch.StartNew();

        try
        {
            // Execute batch using new API
            var batchArray = batch.ToArray();
            var results = await _kernel.ExecuteBatchAsync(batchArray, ct);

            stopwatch.Stop();

            // Update statistics
            Interlocked.Increment(ref _processedBatches);
            var bytesProcessed = batch.Sum(b => b.Length);
            Interlocked.Add(ref _totalBytesProcessed, bytesProcessed);
            _lastActivity = DateTime.UtcNow;

            _logger.LogTrace(
                "Processed batch of {Count} items in {Time}ms ({Throughput:F2} MB/s) - {ResultCount} results",
                batch.Count,
                stopwatch.ElapsedMilliseconds,
                (bytesProcessed / 1024.0 / 1024.0) / stopwatch.Elapsed.TotalSeconds,
                results.Length);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to process batch");
            throw;
        }
    }

    public KernelInstanceStatus GetStatus()
    {
        return new KernelInstanceStatus
        {
            InstanceId = InstanceId,
            KernelId = KernelId,
            State = _state,
            StartTime = _startTime,
            LastActivity = _lastActivity,
            ProcessedBatches = Interlocked.Read(ref _processedBatches),
            FailedBatches = Interlocked.Read(ref _failedBatches),
            TotalBytesProcessed = Interlocked.Read(ref _totalBytesProcessed),
            LastError = _lastError?.Message,
            AutoRestart = _options.RestartOnError,
            Uptime = _state == KernelState.Running
                ? DateTime.UtcNow - _startTime
                : TimeSpan.Zero
        };
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        _cts?.Cancel();
        _cts?.Dispose();
        _executionTask?.Dispose();
    }
}

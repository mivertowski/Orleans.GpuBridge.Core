// Copyright (c) 2025 Michael Ivertowski. All Rights Reserved.
// Licensed under the Apache License, Version 2.0.

using System;
using System.Collections.Concurrent;
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
/// Manages the lifecycle of persistent GPU kernels
/// </summary>
public sealed class KernelLifecycleManager : IDisposable
{
    private readonly ILogger<KernelLifecycleManager> _logger;
    private readonly ConcurrentDictionary<string, PersistentKernelInstance> _instances;
    private readonly Timer _healthCheckTimer;
    private bool _disposed;
    
    public KernelLifecycleManager(ILogger<KernelLifecycleManager> logger)
    {
        _logger = logger;
        _instances = new ConcurrentDictionary<string, PersistentKernelInstance>();
        
        // Start health monitoring
        _healthCheckTimer = new Timer(
            CheckKernelHealth,
            null,
            TimeSpan.FromSeconds(10),
            TimeSpan.FromSeconds(10));
    }
    
    /// <summary>
    /// Starts a new persistent kernel instance
    /// </summary>
    public async Task<PersistentKernelInstance> StartKernelAsync(
        KernelId kernelId,
        IGpuKernel<byte[], byte[]> kernel,
        PersistentKernelOptions options,
        CancellationToken ct = default)
    {
        var instanceId = $"{kernelId.Value}-{Guid.NewGuid():N}";
        
        _logger.LogInformation(
            "Starting persistent kernel {KernelId} with instance {InstanceId}",
            kernelId.Value, instanceId);
        
        var instance = new PersistentKernelInstance(
            instanceId,
            kernelId,
            kernel,
            options,
            _logger);
        
        if (!_instances.TryAdd(instanceId, instance))
        {
            throw new InvalidOperationException($"Instance {instanceId} already exists");
        }
        
        try
        {
            await instance.StartAsync(ct);
            
            _logger.LogInformation(
                "Started persistent kernel {KernelId} successfully",
                kernelId.Value);
            
            return instance;
        }
        catch (Exception ex)
        {
            _instances.TryRemove(instanceId, out _);
            _logger.LogError(ex, "Failed to start kernel {KernelId}", kernelId.Value);
            throw;
        }
    }
    
    /// <summary>
    /// Stops a persistent kernel instance
    /// </summary>
    public async Task StopKernelAsync(string instanceId, CancellationToken ct = default)
    {
        if (!_instances.TryRemove(instanceId, out var instance))
        {
            _logger.LogWarning("Instance {InstanceId} not found", instanceId);
            return;
        }
        
        _logger.LogInformation("Stopping kernel instance {InstanceId}", instanceId);
        
        try
        {
            await instance.StopAsync(ct);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error stopping kernel instance {InstanceId}", instanceId);
        }
        finally
        {
            instance.Dispose();
        }
    }
    
    /// <summary>
    /// Restarts a kernel instance
    /// </summary>
    public async Task RestartKernelAsync(string instanceId, CancellationToken ct = default)
    {
        if (!_instances.TryGetValue(instanceId, out var instance))
        {
            throw new InvalidOperationException($"Instance {instanceId} not found");
        }
        
        _logger.LogInformation("Restarting kernel instance {InstanceId}", instanceId);
        
        await instance.RestartAsync(ct);
    }
    
    /// <summary>
    /// Gets the status of a kernel instance
    /// </summary>
    public KernelInstanceStatus? GetStatus(string instanceId)
    {
        return _instances.TryGetValue(instanceId, out var instance) 
            ? instance.GetStatus() 
            : null;
    }
    
    /// <summary>
    /// Gets status of all kernel instances
    /// </summary>
    public Dictionary<string, KernelInstanceStatus> GetAllStatuses()
    {
        return _instances.ToDictionary(
            kvp => kvp.Key,
            kvp => kvp.Value.GetStatus());
    }
    
    private void CheckKernelHealth(object? state)
    {
        if (_disposed) return;
        
        foreach (var instance in _instances.Values)
        {
            try
            {
                var status = instance.GetStatus();
                
                if (status.State == KernelState.Failed && status.AutoRestart)
                {
                    _logger.LogWarning(
                        "Kernel instance {InstanceId} failed, attempting restart",
                        instance.InstanceId);
                    
                    _ = Task.Run(async () =>
                    {
                        try
                        {
                            await instance.RestartAsync();
                        }
                        catch (Exception ex)
                        {
                            _logger.LogError(ex, 
                                "Failed to restart kernel instance {InstanceId}",
                                instance.InstanceId);
                        }
                    });
                }
                else if (status.State == KernelState.Running)
                {
                    // Check for hangs
                    if (status.LastActivity < DateTime.UtcNow.AddMinutes(-5) &&
                        status.ProcessedBatches > 0)
                    {
                        _logger.LogWarning(
                            "Kernel instance {InstanceId} appears hung (no activity for 5 minutes)",
                            instance.InstanceId);
                    }
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, 
                    "Error checking health of kernel instance {InstanceId}",
                    instance.InstanceId);
            }
        }
    }
    
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        
        _healthCheckTimer?.Dispose();
        
        // Stop all kernel instances
        var stopTasks = _instances.Values
            .Select(i => Task.Run(async () =>
            {
                try
                {
                    await i.StopAsync();
                    i.Dispose();
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Error stopping kernel instance {InstanceId}", i.InstanceId);
                }
            }))
            .ToArray();
        
        Task.WaitAll(stopTasks);
        _instances.Clear();
    }
}

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
            var info = await _kernel.GetInfoAsync(ct);
            
            _logger.LogDebug(
                "Starting persistent kernel {KernelId} with batch size {BatchSize}",
                info.Id, _options.BatchSize);
            
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
            // Submit batch to kernel
            var handle = await _kernel.SubmitBatchAsync(batch, null, ct);
            
            // Read results
            var results = new List<byte[]>();
            await foreach (var result in _kernel.ReadResultsAsync(handle, ct))
            {
                results.Add(result);
            }
            
            stopwatch.Stop();
            
            // Update statistics
            Interlocked.Increment(ref _processedBatches);
            var bytesProcessed = batch.Sum(b => b.Length);
            Interlocked.Add(ref _totalBytesProcessed, bytesProcessed);
            _lastActivity = DateTime.UtcNow;
            
            _logger.LogTrace(
                "Processed batch of {Count} items in {Time}ms ({Throughput:F2} MB/s)",
                batch.Count,
                stopwatch.ElapsedMilliseconds,
                (bytesProcessed / 1024.0 / 1024.0) / stopwatch.Elapsed.TotalSeconds);
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

/// <summary>
/// Options for persistent kernel execution
/// </summary>
public sealed class PersistentKernelOptions
{
    public int BatchSize { get; set; } = 100;
    public TimeSpan MaxBatchWaitTime { get; set; } = TimeSpan.FromMilliseconds(100);
    public bool RestartOnError { get; set; } = true;
    public int MaxRetries { get; set; } = 3;
    public TimeSpan RetryDelay { get; set; } = TimeSpan.FromSeconds(1);
}

/// <summary>
/// Status of a persistent kernel instance
/// </summary>
public sealed class KernelInstanceStatus
{
    public string InstanceId { get; init; } = string.Empty;
    public KernelId KernelId { get; init; } = new("unknown");
    public KernelState State { get; init; }
    public DateTime StartTime { get; init; }
    public DateTime LastActivity { get; init; }
    public long ProcessedBatches { get; init; }
    public long FailedBatches { get; init; }
    public long TotalBytesProcessed { get; init; }
    public string? LastError { get; init; }
    public bool AutoRestart { get; init; }
    public TimeSpan Uptime { get; init; }
    public double SuccessRate => ProcessedBatches > 0 
        ? (ProcessedBatches - FailedBatches) / (double)ProcessedBatches * 100
        : 0;
}

/// <summary>
/// State of a kernel instance
/// </summary>
public enum KernelState
{
    Idle,
    Starting,
    Running,
    Stopping,
    Stopped,
    Failed
}
// Copyright (c) 2025 Michael Ivertowski. All Rights Reserved.
// Licensed under the Apache License, Version 2.0.

using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
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

    /// <summary>
    /// Initializes a new instance of the <see cref="KernelLifecycleManager"/> class.
    /// </summary>
    /// <param name="logger">The logger for diagnostic output.</param>
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

    /// <inheritdoc/>
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

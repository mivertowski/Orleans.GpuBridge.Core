using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Threading;
using Microsoft.Extensions.Logging;
using Orleans.GpuBridge.Abstractions.Memory;

namespace Orleans.GpuBridge.Runtime;

/// <summary>
/// Memory pool manager for different types with size limits
/// </summary>
public sealed class MemoryPoolManager : IDisposable
{
    private readonly ConcurrentDictionary<Type, object> _pools;
    private readonly ILoggerFactory _loggerFactory;
    private readonly ILogger<MemoryPoolManager> _logger;
    private readonly MemoryPoolOptions _options;
    private long _totalMemoryLimit;
    private long _currentTotalUsage;
    private bool _disposed;

    /// <summary>
    /// Initializes a new instance of the <see cref="MemoryPoolManager"/> class.
    /// </summary>
    /// <param name="loggerFactory">The logger factory for creating diagnostic loggers.</param>
    /// <param name="options">Optional memory pool configuration options.</param>
    public MemoryPoolManager(
        ILoggerFactory loggerFactory,
        MemoryPoolOptions? options = null)
    {
        _loggerFactory = loggerFactory;
        _logger = loggerFactory.CreateLogger<MemoryPoolManager>();
        _pools = new ConcurrentDictionary<Type, object>();
        _options = options ?? new MemoryPoolOptions();
        _totalMemoryLimit = _options.MaxTotalMemoryBytes;

        _logger.LogInformation(
            "Memory pool manager initialized with {Limit:N0} bytes total limit",
            _totalMemoryLimit);
    }

    /// <summary>
    /// Gets or creates a memory pool for the specified element type.
    /// </summary>
    /// <typeparam name="T">The unmanaged element type.</typeparam>
    /// <returns>A memory pool for the specified type.</returns>
    public IGpuMemoryPool<T> GetPool<T>() where T : unmanaged
    {
        return (IGpuMemoryPool<T>)_pools.GetOrAdd(
            typeof(T),
            type =>
            {
                var perTypeLimit = _options.PerTypeMemoryLimits.GetValueOrDefault(
                    type,
                    _options.DefaultPerTypeLimit);

                return new AdvancedMemoryPool<T>(
                    _loggerFactory.CreateLogger<AdvancedMemoryPool<T>>(),
                    maxBufferSize: (int)(perTypeLimit / Unsafe.SizeOf<T>()),
                    maxPooledBuffers: _options.MaxPooledBuffersPerType);
            });
    }

    /// <summary>
    /// Attempts to allocate the specified number of bytes from the total memory pool.
    /// </summary>
    /// <param name="requestedBytes">The number of bytes to allocate.</param>
    /// <param name="reason">If allocation fails, contains the reason for failure.</param>
    /// <returns>True if allocation succeeded; otherwise, false.</returns>
    public bool TryAllocate(long requestedBytes, out string? reason)
    {
        reason = null;

        // Check against total limit
        var newTotal = Interlocked.Add(ref _currentTotalUsage, requestedBytes);
        if (newTotal > _totalMemoryLimit)
        {
            Interlocked.Add(ref _currentTotalUsage, -requestedBytes);
            reason = $"Would exceed total memory limit of {_totalMemoryLimit:N0} bytes";
            _logger.LogWarning(
                "Memory allocation denied: {Reason}. Current: {Current:N0}, Requested: {Requested:N0}",
                reason, _currentTotalUsage, requestedBytes);
            return false;
        }

        return true;
    }

    /// <summary>
    /// Releases a previously allocated number of bytes back to the pool.
    /// </summary>
    /// <param name="bytes">The number of bytes to release.</param>
    public void ReleaseAllocation(long bytes)
    {
        Interlocked.Add(ref _currentTotalUsage, -bytes);
    }

    /// <summary>
    /// Gets statistics for all memory pools.
    /// </summary>
    /// <returns>A dictionary mapping element types to their pool statistics.</returns>
    public Dictionary<Type, MemoryPoolStats> GetAllStats()
    {
        var stats = new Dictionary<Type, MemoryPoolStats>();

        foreach (var (type, pool) in _pools)
        {
            if (pool is IGpuMemoryPool<byte> bytePool)
                stats[type] = bytePool.GetStats();
            else if (pool is IGpuMemoryPool<float> floatPool)
                stats[type] = floatPool.GetStats();
            else if (pool is IGpuMemoryPool<double> doublePool)
                stats[type] = doublePool.GetStats();
            else if (pool is IGpuMemoryPool<int> intPool)
                stats[type] = intPool.GetStats();
        }

        return stats;
    }

    /// <summary>
    /// Gets the current health status of the memory pool system.
    /// </summary>
    /// <returns>The health status information.</returns>
    public MemoryPoolHealth GetHealthStatus()
    {
        var usage = Interlocked.Read(ref _currentTotalUsage);
        var limit = _totalMemoryLimit;
        var utilizationPercent = limit > 0 ? (usage / (double)limit) * 100 : 0;

        var status = utilizationPercent switch
        {
            < 50 => HealthStatus.Healthy,
            < 80 => HealthStatus.Warning,
            _ => HealthStatus.Critical
        };

        return new MemoryPoolHealth
        {
            Status = status,
            TotalUsageBytes = usage,
            TotalLimitBytes = limit,
            UtilizationPercent = utilizationPercent,
            PoolCount = _pools.Count,
            Message = status switch
            {
                HealthStatus.Critical => "Memory pool critically low",
                HealthStatus.Warning => "Memory pool usage high",
                _ => "Memory pool healthy"
            }
        };
    }

    /// <summary>
    /// Updates the total memory limit for all pools.
    /// </summary>
    /// <param name="newTotalLimit">The new total memory limit in bytes.</param>
    public void UpdateLimits(long newTotalLimit)
    {
        _totalMemoryLimit = newTotalLimit;
        _logger.LogInformation(
            "Memory pool total limit updated to {Limit:N0} bytes",
            newTotalLimit);
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        foreach (var pool in _pools.Values)
        {
            if (pool is IDisposable disposable)
            {
                disposable.Dispose();
            }
        }

        _pools.Clear();

        _logger.LogInformation(
            "Memory pool manager disposed. Final usage: {Usage:N0} bytes",
            _currentTotalUsage);
    }
}

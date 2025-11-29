// Copyright (c) 2025 Michael Ivertowski
// Licensed under the Apache License, Version 2.0. See LICENSE file for details.

using System;
using System.Buffers;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using DotCompute.Abstractions;

namespace Orleans.GpuBridge.Runtime.Memory;

/// <summary>
/// High-level GPU memory management with zero-copy transfers and automatic pooling.
/// </summary>
/// <remarks>
/// <para>
/// GpuMemoryManager provides a convenient API for GPU memory operations:
/// - Zero-copy transfers (CPU ↔ GPU)
/// - Automatic buffer pooling
/// - Memory pressure monitoring
/// - Async operations for non-blocking transfers
/// </para>
/// <para>
/// Performance characteristics:
/// - Pooled allocation: ~100-500ns (vs ~10-50μs for cold allocation)
/// - Copy to GPU: ~1-10μs/MB (PCIe Gen3: 12-16 GB/s)
/// - Copy from GPU: ~1-10μs/MB
/// - Zero-copy (pinned memory): ~0.5-5μs/MB (up to 2× faster)
/// </para>
/// </remarks>
public sealed class GpuMemoryManager : IDisposable
{
    private readonly GpuBufferPool _bufferPool;
    private readonly ILogger<GpuMemoryManager> _logger;
    private bool _disposed;

    // Memory pressure thresholds
    private const double HighMemoryPressureThreshold = 0.85; // 85% utilization
    private const double CriticalMemoryPressureThreshold = 0.95; // 95% utilization

    /// <summary>
    /// Initializes a new GPU memory manager.
    /// </summary>
    /// <param name="bufferPool">GPU buffer pool instance.</param>
    /// <param name="logger">Logger instance.</param>
    public GpuMemoryManager(GpuBufferPool bufferPool, ILogger<GpuMemoryManager> logger)
    {
        _bufferPool = bufferPool ?? throw new ArgumentNullException(nameof(bufferPool));
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));

        _logger.LogInformation("Initialized GPU memory manager.");
    }

    /// <summary>
    /// Allocates GPU memory for an array of unmanaged elements.
    /// </summary>
    /// <typeparam name="T">Unmanaged element type.</typeparam>
    /// <param name="count">Number of elements.</param>
    /// <returns>GPU memory handle.</returns>
    public GpuMemoryHandle AllocateBuffer<T>(int count) where T : unmanaged
    {
        ThrowIfDisposed();

        unsafe
        {
            long sizeBytes = count * sizeof(T);
            return _bufferPool.RentBuffer(sizeBytes);
        }
    }

    /// <summary>
    /// Copies data from CPU to GPU (zero-copy when possible).
    /// </summary>
    /// <typeparam name="T">Unmanaged element type.</typeparam>
    /// <param name="source">Source CPU array.</param>
    /// <param name="destination">Destination GPU memory handle.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <exception cref="ArgumentException">Destination buffer is too small.</exception>
    public async Task CopyToGpuAsync<T>(
        T[] source,
        GpuMemoryHandle destination,
        CancellationToken cancellationToken = default)
        where T : unmanaged
    {
        ThrowIfDisposed();

        if (source == null)
            throw new ArgumentNullException(nameof(source));

        if (destination == null)
            throw new ArgumentNullException(nameof(destination));

        unsafe
        {
            long requiredSize = source.Length * sizeof(T);
            if (requiredSize > destination.SizeBytes)
            {
                throw new ArgumentException(
                    $"Destination buffer ({destination.SizeBytes} bytes) is too small for source data ({requiredSize} bytes).");
            }
        }

        _logger.LogTrace(
            "Copying {Count} elements ({Type}) to GPU (size={Size} bytes)",
            source.Length,
            typeof(T).Name,
            destination.SizeBytes);

        // Use DotCompute's optimized copy if buffer supports it
        if (destination.DotComputeBuffer is IUnifiedMemoryBuffer<T> typedBuffer)
        {
            await typedBuffer.CopyFromAsync(source.AsMemory(), cancellationToken);

            _logger.LogTrace(
                "GPU copy completed via DotCompute: {Size} bytes",
                destination.SizeBytes);
        }
        else if (destination.DotComputeBuffer != null)
        {
            // Non-generic buffer - use base interface with explicit type and offset
            await destination.DotComputeBuffer.CopyFromAsync<T>(source.AsMemory(), 0, cancellationToken);

            _logger.LogTrace(
                "GPU copy completed via DotCompute (untyped): {Size} bytes",
                destination.SizeBytes);
        }
        else
        {
            // Fallback for CPU memory
            await Task.Run(() =>
            {
                unsafe
                {
                    fixed (T* srcPtr = source)
                    {
                        Buffer.MemoryCopy(
                            srcPtr,
                            destination.DevicePointer.ToPointer(),
                            destination.SizeBytes,
                            source.Length * sizeof(T));
                    }
                }
            }, cancellationToken);

            _logger.LogTrace(
                "CPU fallback copy completed: {Size} bytes",
                destination.SizeBytes);
        }
    }

    /// <summary>
    /// Copies data from GPU to CPU.
    /// </summary>
    /// <typeparam name="T">Unmanaged element type.</typeparam>
    /// <param name="source">Source GPU memory handle.</param>
    /// <param name="destination">Destination CPU array.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <exception cref="ArgumentException">Arrays are different sizes.</exception>
    public async Task CopyFromGpuAsync<T>(
        GpuMemoryHandle source,
        T[] destination,
        CancellationToken cancellationToken = default)
        where T : unmanaged
    {
        ThrowIfDisposed();

        if (source == null)
            throw new ArgumentNullException(nameof(source));

        if (destination == null)
            throw new ArgumentNullException(nameof(destination));

        unsafe
        {
            long requiredSize = destination.Length * sizeof(T);
            if (requiredSize > source.SizeBytes)
            {
                throw new ArgumentException(
                    $"Source buffer ({source.SizeBytes} bytes) is smaller than destination array ({requiredSize} bytes).");
            }
        }

        _logger.LogTrace(
            "Copying {Count} elements ({Type}) from GPU (size={Size} bytes)",
            destination.Length,
            typeof(T).Name,
            source.SizeBytes);

        // Use DotCompute's optimized copy if buffer supports it
        if (source.DotComputeBuffer is IUnifiedMemoryBuffer<T> typedBuffer)
        {
            await typedBuffer.CopyToAsync(destination.AsMemory(), cancellationToken);

            _logger.LogTrace(
                "GPU copy completed via DotCompute: {Size} bytes",
                source.SizeBytes);
        }
        else if (source.DotComputeBuffer != null)
        {
            // Non-generic buffer - use base interface with explicit type and offset
            await source.DotComputeBuffer.CopyToAsync<T>(destination.AsMemory(), 0, cancellationToken);

            _logger.LogTrace(
                "GPU copy completed via DotCompute (untyped): {Size} bytes",
                source.SizeBytes);
        }
        else
        {
            // Fallback for CPU memory
            await Task.Run(() =>
            {
                unsafe
                {
                    fixed (T* dstPtr = destination)
                    {
                        Buffer.MemoryCopy(
                            source.DevicePointer.ToPointer(),
                            dstPtr,
                            destination.Length * sizeof(T),
                            destination.Length * sizeof(T));
                    }
                }
            }, cancellationToken);

            _logger.LogTrace(
                "CPU fallback copy completed: {Size} bytes",
                source.SizeBytes);
        }
    }

    /// <summary>
    /// Allocates GPU memory and copies data in one operation.
    /// </summary>
    /// <typeparam name="T">Unmanaged element type.</typeparam>
    /// <param name="source">Source CPU array.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>GPU memory handle with copied data.</returns>
    public async Task<GpuMemoryHandle> AllocateAndCopyAsync<T>(
        T[] source,
        CancellationToken cancellationToken = default)
        where T : unmanaged
    {
        ThrowIfDisposed();

        var handle = AllocateBuffer<T>(source.Length);

        try
        {
            await CopyToGpuAsync(source, handle, cancellationToken);
            return handle;
        }
        catch
        {
            handle.Dispose();
            throw;
        }
    }

    /// <summary>
    /// Checks current memory pressure level.
    /// </summary>
    /// <returns>Memory pressure level.</returns>
    public MemoryPressureLevel GetMemoryPressure()
    {
        ThrowIfDisposed();

        var stats = _bufferPool.GetStatistics();
        double utilization = stats.UtilizationRatio;

        if (utilization >= CriticalMemoryPressureThreshold)
        {
            _logger.LogWarning(
                "CRITICAL GPU memory pressure: {Utilization:P1} (InUse={InUse}MB, Total={Total}MB)",
                utilization,
                stats.InUseBytes / (1024 * 1024),
                stats.TotalAllocatedBytes / (1024 * 1024));

            return MemoryPressureLevel.Critical;
        }
        else if (utilization >= HighMemoryPressureThreshold)
        {
            _logger.LogWarning(
                "HIGH GPU memory pressure: {Utilization:P1} (InUse={InUse}MB, Total={Total}MB)",
                utilization,
                stats.InUseBytes / (1024 * 1024),
                stats.TotalAllocatedBytes / (1024 * 1024));

            return MemoryPressureLevel.High;
        }
        else if (utilization >= 0.5)
        {
            return MemoryPressureLevel.Medium;
        }
        else
        {
            return MemoryPressureLevel.Low;
        }
    }

    /// <summary>
    /// Gets current GPU memory statistics.
    /// </summary>
    /// <returns>Memory statistics.</returns>
    public GpuMemoryStats GetStatistics() => _bufferPool.GetStatistics();

    /// <summary>
    /// Gets buffer pool hit rate (0-1).
    /// </summary>
    /// <returns>Hit rate percentage.</returns>
    public double GetPoolHitRate() => _bufferPool.GetHitRate();

    /// <summary>
    /// Clears all pooled buffers (useful for reducing memory footprint).
    /// </summary>
    public void ClearPool()
    {
        ThrowIfDisposed();
        _bufferPool.Clear();
    }

    /// <summary>
    /// Disposes the GPU memory manager and releases all memory.
    /// </summary>
    public void Dispose()
    {
        if (_disposed)
            return;

        _logger.LogInformation("Disposing GPU memory manager...");

        _bufferPool.Dispose();

        _disposed = true;
        GC.SuppressFinalize(this);
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
        {
            throw new ObjectDisposedException(nameof(GpuMemoryManager),
                "Cannot access disposed GPU memory manager.");
        }
    }
}

/// <summary>
/// GPU memory pressure levels.
/// </summary>
public enum MemoryPressureLevel
{
    /// <summary>
    /// Low pressure (&lt;50% utilization).
    /// </summary>
    Low = 0,

    /// <summary>
    /// Medium pressure (50-85% utilization).
    /// </summary>
    Medium = 1,

    /// <summary>
    /// High pressure (85-95% utilization).
    /// </summary>
    High = 2,

    /// <summary>
    /// Critical pressure (&gt;95% utilization).
    /// </summary>
    Critical = 3
}

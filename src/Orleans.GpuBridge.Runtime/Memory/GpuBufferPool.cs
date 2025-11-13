// Copyright (c) 2025 Michael Ivertowski
// Licensed under the MIT License.

using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using Microsoft.Extensions.Logging;
using DotCompute.Abstractions;
using DotCompute.Abstractions.Memory;
using DotCompute.Backends.CUDA;
using DotCompute.Backends.CUDA.Memory;

namespace Orleans.GpuBridge.Runtime.Memory;

/// <summary>
/// Pool of reusable GPU memory buffers organized by size buckets.
/// </summary>
/// <remarks>
/// <para>
/// GpuBufferPool dramatically reduces GPU memory allocation overhead by reusing
/// previously allocated buffers. Buffers are organized into size buckets (powers of 2)
/// for efficient lookup and allocation.
/// </para>
/// <para>
/// Performance characteristics:
/// - Cold allocation: ~10-50μs (GPU cudaMalloc)
/// - Pool allocation: ~100-500ns (lock-free queue)
/// - **50-500× speedup** for repeated allocations
/// </para>
/// <para>
/// Thread-safe: Uses ConcurrentQueue for lock-free buffer management.
/// </para>
/// </remarks>
public sealed class GpuBufferPool : IDisposable
{
    private readonly ILogger<GpuBufferPool> _logger;
    private readonly ConcurrentDictionary<int, ConcurrentQueue<GpuMemoryHandle>> _buckets = new();
    private readonly ConcurrentDictionary<ulong, GpuMemoryHandle> _activeAllocations = new();
    private readonly object _statsLock = new();
    private readonly CudaContext? _cudaContext;
    private readonly CudaMemoryManager? _memoryManager;
    private bool _disposed;

    // Statistics
    private long _totalAllocations;
    private long _poolHits;
    private long _poolMisses;
    private long _totalBytesAllocated;
    private long _currentBytesInUse;

    /// <summary>
    /// Minimum buffer size (1 KB).
    /// </summary>
    public const long MinBufferSize = 1024;

    /// <summary>
    /// Maximum buffer size (1 GB).
    /// </summary>
    public const long MaxBufferSize = 1024L * 1024 * 1024;

    /// <summary>
    /// Maximum number of buffers to keep in each size bucket.
    /// </summary>
    public int MaxBuffersPerBucket { get; set; } = 16;

    /// <summary>
    /// Initializes a new GPU buffer pool.
    /// </summary>
    /// <param name="logger">Logger instance.</param>
    /// <param name="cudaContext">Optional CUDA context for GPU memory allocation.</param>
    /// <param name="memoryManager">Optional CUDA memory manager for GPU memory allocation.</param>
    /// <remarks>
    /// If cudaContext and memoryManager are provided, the pool will use actual GPU unified memory.
    /// If not provided, the pool will use CPU memory with CPU-GPU transfer support via DotCompute.
    /// </remarks>
    public GpuBufferPool(
        ILogger<GpuBufferPool> logger,
        CudaContext? cudaContext = null,
        CudaMemoryManager? memoryManager = null)
    {
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        _cudaContext = cudaContext;
        _memoryManager = memoryManager;

        var mode = _memoryManager != null ? "GPU unified memory" : "CPU memory with GPU transfer";
        _logger.LogInformation(
            "Initialized GPU buffer pool ({Mode}, min={MinSize}KB, max={MaxSize}MB, bucketsPerSize={MaxPerBucket})",
            mode,
            MinBufferSize / 1024,
            MaxBufferSize / (1024 * 1024),
            MaxBuffersPerBucket);
    }

    /// <summary>
    /// Rents a GPU buffer of at least the specified size.
    /// </summary>
    /// <param name="sizeBytes">Required size in bytes.</param>
    /// <returns>GPU memory handle.</returns>
    /// <exception cref="ArgumentOutOfRangeException">Size is out of valid range.</exception>
    /// <exception cref="ObjectDisposedException">Pool has been disposed.</exception>
    public GpuMemoryHandle RentBuffer(long sizeBytes)
    {
        ThrowIfDisposed();

        if (sizeBytes <= 0)
            throw new ArgumentOutOfRangeException(nameof(sizeBytes), "Size must be positive.");

        if (sizeBytes > MaxBufferSize)
            throw new ArgumentOutOfRangeException(nameof(sizeBytes),
                $"Size {sizeBytes} exceeds maximum buffer size {MaxBufferSize}.");

        // Round up to next power of 2 for efficient bucketing
        int bucketKey = GetBucketKey(sizeBytes);
        long bucketSize = 1L << bucketKey;

        // Try to get from pool first
        if (_buckets.TryGetValue(bucketKey, out var bucket) &&
            bucket.TryDequeue(out var pooledHandle))
        {
            Interlocked.Increment(ref _poolHits);
            Interlocked.Increment(ref _totalAllocations);
            Interlocked.Add(ref _currentBytesInUse, bucketSize);

            _logger.LogTrace(
                "Rented GPU buffer from pool: size={RequestedSize}→{BucketSize} bytes (bucket={BucketKey}, hit)",
                sizeBytes,
                bucketSize,
                bucketKey);

            // Track as active allocation
            _activeAllocations[pooledHandle.HandleId] = pooledHandle;

            return pooledHandle.AddReference();
        }

        // Pool miss - allocate new buffer
        Interlocked.Increment(ref _poolMisses);
        Interlocked.Increment(ref _totalAllocations);

        var newHandle = AllocateGpuBuffer(bucketSize);

        Interlocked.Add(ref _totalBytesAllocated, bucketSize);
        Interlocked.Add(ref _currentBytesInUse, bucketSize);

        _logger.LogDebug(
            "Allocated new GPU buffer: size={RequestedSize}→{BucketSize} bytes (bucket={BucketKey}, miss)",
            sizeBytes,
            bucketSize,
            bucketKey);

        // Track as active allocation
        _activeAllocations[newHandle.HandleId] = newHandle;

        return newHandle;
    }

    /// <summary>
    /// Returns a buffer to the pool for reuse.
    /// </summary>
    /// <param name="handle">GPU memory handle to return.</param>
    internal void ReturnBuffer(GpuMemoryHandle handle)
    {
        if (handle == null)
            return;

        ThrowIfDisposed();

        int bucketKey = GetBucketKey(handle.SizeBytes);
        long bucketSize = 1L << bucketKey;

        // Remove from active allocations
        _activeAllocations.TryRemove(handle.HandleId, out _);
        Interlocked.Add(ref _currentBytesInUse, -bucketSize);

        // Get or create bucket
        var bucket = _buckets.GetOrAdd(bucketKey, _ => new ConcurrentQueue<GpuMemoryHandle>());

        // Only return to pool if under capacity
        if (bucket.Count < MaxBuffersPerBucket)
        {
            bucket.Enqueue(handle);

            _logger.LogTrace(
                "Returned GPU buffer to pool: size={Size} bytes (bucket={BucketKey}, count={Count})",
                bucketSize,
                bucketKey,
                bucket.Count);
        }
        else
        {
            // Pool full, free the buffer directly
            FreeGpuBuffer(handle);

            Interlocked.Add(ref _totalBytesAllocated, -bucketSize);

            _logger.LogTrace(
                "Freed GPU buffer (pool full): size={Size} bytes (bucket={BucketKey})",
                bucketSize,
                bucketKey);
        }
    }

    /// <summary>
    /// Gets current memory statistics.
    /// </summary>
    public GpuMemoryStats GetStatistics()
    {
        ThrowIfDisposed();

        int totalPooledBuffers = _buckets.Values.Sum(q => q.Count);

        return new GpuMemoryStats
        {
            TotalAllocatedBytes = Interlocked.Read(ref _totalBytesAllocated),
            InUseBytes = Interlocked.Read(ref _currentBytesInUse),
            AvailableBytes = Interlocked.Read(ref _totalBytesAllocated) - Interlocked.Read(ref _currentBytesInUse),
            ActiveAllocations = _activeAllocations.Count,
            PooledBuffers = totalPooledBuffers
        };
    }

    /// <summary>
    /// Gets pool hit rate (0-1).
    /// </summary>
    public double GetHitRate()
    {
        long total = Interlocked.Read(ref _totalAllocations);
        if (total == 0)
            return 0.0;

        long hits = Interlocked.Read(ref _poolHits);
        return (double)hits / total;
    }

    /// <summary>
    /// Clears all buffers from the pool and releases GPU memory.
    /// </summary>
    public void Clear()
    {
        ThrowIfDisposed();

        _logger.LogInformation("Clearing GPU buffer pool...");

        foreach (var bucket in _buckets.Values)
        {
            while (bucket.TryDequeue(out var handle))
            {
                FreeGpuBuffer(handle);
                Interlocked.Add(ref _totalBytesAllocated, -handle.SizeBytes);
            }
        }

        _logger.LogInformation("GPU buffer pool cleared.");
    }

    /// <summary>
    /// Disposes the buffer pool and releases all GPU memory.
    /// </summary>
    public void Dispose()
    {
        if (_disposed)
            return;

        _logger.LogInformation(
            "Disposing GPU buffer pool (allocations={TotalAllocs}, hits={Hits}, misses={Misses}, hit rate={HitRate:P1})",
            Interlocked.Read(ref _totalAllocations),
            Interlocked.Read(ref _poolHits),
            Interlocked.Read(ref _poolMisses),
            GetHitRate());

        Clear();

        // Free any remaining active allocations
        foreach (var handle in _activeAllocations.Values)
        {
            FreeGpuBuffer(handle);
        }

        _activeAllocations.Clear();

        _disposed = true;
        GC.SuppressFinalize(this);
    }

    /// <summary>
    /// Calculates bucket key (power of 2) for a given size.
    /// </summary>
    /// <param name="sizeBytes">Size in bytes.</param>
    /// <returns>Bucket key (log2 of bucket size).</returns>
    private static int GetBucketKey(long sizeBytes)
    {
        // Round up to next power of 2
        long size = Math.Max(sizeBytes, MinBufferSize);

        // Find log2(size) using bit manipulation
        int key = 0;
        long s = size - 1;
        while (s > 0)
        {
            s >>= 1;
            key++;
        }

        return key;
    }

    /// <summary>
    /// Allocates GPU memory via DotCompute CUDA backend.
    /// </summary>
    /// <param name="sizeBytes">Size in bytes.</param>
    /// <returns>GPU memory handle.</returns>
    /// <remarks>
    /// <para>
    /// Uses CudaMemoryManager.AllocateAsync with MemoryOptions.Unified when available.
    /// Unified memory provides zero-copy access from both CPU and GPU.
    /// </para>
    /// <para>
    /// Falls back to CPU memory (Marshal.AllocHGlobal) when CUDA is not available.
    /// CPU memory still works with DotCompute's copy operations for explicit transfers.
    /// </para>
    /// </remarks>
    private GpuMemoryHandle AllocateGpuBuffer(long sizeBytes)
    {
        // Try GPU unified memory allocation if CudaMemoryManager is available
        if (_memoryManager != null)
        {
            try
            {
                // Allocate unified memory (accessible from both CPU and GPU)
                var buffer = _memoryManager.AllocateAsync<byte>(
                    sizeBytes,
                    MemoryOptions.Unified,
                    CancellationToken.None).GetAwaiter().GetResult();

                // Get device memory pointer via DotCompute's API
                var deviceMemory = buffer.GetDeviceMemory();
                IntPtr devicePtr = deviceMemory.Handle;

                _logger.LogTrace(
                    "Allocated GPU unified memory via DotCompute: {Size} bytes at 0x{Pointer:X}",
                    sizeBytes,
                    devicePtr.ToInt64());

                return new GpuMemoryHandle(devicePtr, sizeBytes, this, buffer);
            }
            catch (Exception ex)
            {
                _logger.LogWarning(
                    ex,
                    "Failed to allocate GPU unified memory, falling back to CPU memory: {Size} bytes",
                    sizeBytes);

                // Fall through to CPU memory allocation
            }
        }

        // CPU memory fallback
        try
        {
            IntPtr cpuPtr = Marshal.AllocHGlobal((int)sizeBytes);

            _logger.LogTrace(
                "Allocated CPU memory (CUDA unavailable): {Size} bytes at 0x{Pointer:X}",
                sizeBytes,
                cpuPtr.ToInt64());

            return new GpuMemoryHandle(cpuPtr, sizeBytes, this);
        }
        catch (Exception ex)
        {
            _logger.LogError(
                ex,
                "Failed to allocate memory: {Size} bytes",
                sizeBytes);

            throw;
        }
    }

    /// <summary>
    /// Frees GPU memory via DotCompute or fallback.
    /// </summary>
    /// <param name="handle">GPU memory handle.</param>
    private void FreeGpuBuffer(GpuMemoryHandle handle)
    {
        try
        {
            if (handle.DotComputeBuffer != null)
            {
                // DotCompute buffer handles its own disposal (calls cudaFree internally)
                handle.DotComputeBuffer.Dispose();

                _logger.LogTrace(
                    "Freed GPU memory via DotCompute: {Size} bytes",
                    handle.SizeBytes);
            }
            else
            {
                // Fallback for CPU-allocated memory
                if (handle.DevicePointer != IntPtr.Zero)
                {
                    Marshal.FreeHGlobal(handle.DevicePointer);

                    _logger.LogTrace(
                        "Freed CPU fallback memory: {Size} bytes",
                        handle.SizeBytes);
                }
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(
                ex,
                "Error freeing GPU memory (ptr=0x{Pointer:X}, size={Size} bytes)",
                handle.DevicePointer.ToInt64(),
                handle.SizeBytes);
        }
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
        {
            throw new ObjectDisposedException(nameof(GpuBufferPool),
                "Cannot access disposed GPU buffer pool.");
        }
    }
}

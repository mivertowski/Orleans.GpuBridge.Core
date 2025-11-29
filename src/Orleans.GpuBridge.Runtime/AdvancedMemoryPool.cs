using System;
using System.Buffers;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Orleans.GpuBridge.Abstractions.Memory;

namespace Orleans.GpuBridge.Runtime;

/// <summary>
/// Advanced memory pool with allocation tracking and statistics
/// </summary>
public sealed class AdvancedMemoryPool<T> : IGpuMemoryPool<T>, IDisposable
    where T : unmanaged
{
    private readonly ILogger<AdvancedMemoryPool<T>> _logger;
    private readonly ArrayPool<T> _arrayPool;
    private readonly ConcurrentDictionary<int, PooledSegment> _segments;
    private readonly ConcurrentDictionary<IntPtr, AllocationInfo> _allocations;
    private readonly Timer _gcTimer;
    private readonly int _maxBufferSize;
    private readonly int _maxPooledBuffers;
    private long _totalAllocated;
    private long _totalReturned;
    private long _currentInUse;
    private long _peakUsage;
    private bool _disposed;

    /// <summary>
    /// Initializes a new instance of the <see cref="AdvancedMemoryPool{T}"/> class
    /// </summary>
    /// <param name="logger">Logger instance for diagnostics</param>
    /// <param name="maxBufferSize">Maximum size of buffers in the pool (default: 16MB)</param>
    /// <param name="maxPooledBuffers">Maximum number of buffers to maintain in the pool (default: 100)</param>
    public AdvancedMemoryPool(
        ILogger<AdvancedMemoryPool<T>> logger,
        int maxBufferSize = 1024 * 1024 * 16, // 16MB for T
        int maxPooledBuffers = 100)
    {
        _logger = logger;
        _maxBufferSize = maxBufferSize;
        _maxPooledBuffers = maxPooledBuffers;
        _arrayPool = ArrayPool<T>.Create(maxBufferSize, maxPooledBuffers);
        _segments = new ConcurrentDictionary<int, PooledSegment>();
        _allocations = new ConcurrentDictionary<IntPtr, AllocationInfo>();

        // Start GC timer to clean up unused segments
        _gcTimer = new Timer(
            CollectUnusedSegments,
            null,
            TimeSpan.FromMinutes(1),
            TimeSpan.FromMinutes(1));
    }

    /// <summary>
    /// Rents a memory segment from the pool with at least the specified size
    /// </summary>
    /// <param name="minSize">Minimum size of the memory segment in elements</param>
    /// <returns>A pooled GPU memory instance</returns>
    public IGpuMemory<T> Rent(int minSize)
    {
        if (_disposed)
            throw new ObjectDisposedException(nameof(AdvancedMemoryPool<T>));

        if (minSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(minSize));

        // Try to find a reusable segment
        var segment = FindReusableSegment(minSize);
        if (segment != null)
        {
            _logger.LogTrace("Reusing pooled segment of size {Size}", segment.Size);
            // Convert segment to PooledGpuMemory
            return new PooledGpuMemory(this, segment.Array, segment.Size);
        }

        // Allocate new segment
        var array = _arrayPool.Rent(minSize);
        var memory = new PooledGpuMemory(this, array, minSize);

        // Track allocation
        var allocationInfo = new AllocationInfo
        {
            Size = array.Length * Unsafe.SizeOf<T>(),
            AllocatedAt = DateTime.UtcNow,
            StackTrace = Environment.StackTrace
        };

        unsafe
        {
            fixed (T* ptr = array)
            {
                _allocations.TryAdd((IntPtr)ptr, allocationInfo);
            }
        }

        // Update statistics
        Interlocked.Add(ref _totalAllocated, array.Length * Unsafe.SizeOf<T>());
        Interlocked.Add(ref _currentInUse, array.Length * Unsafe.SizeOf<T>());

        // Update peak usage
        var current = Interlocked.Read(ref _currentInUse);
        var peak = Interlocked.Read(ref _peakUsage);
        while (current > peak)
        {
            Interlocked.CompareExchange(ref _peakUsage, current, peak);
            peak = Interlocked.Read(ref _peakUsage);
        }

        _logger.LogDebug(
            "Allocated {Size:N0} bytes, current usage: {Current:N0} bytes",
            allocationInfo.Size, current);

        return memory;
    }

    /// <summary>
    /// Returns a memory segment to the pool for reuse
    /// </summary>
    /// <param name="memory">The memory instance to return</param>
    public void Return(IGpuMemory<T> memory)
    {
        if (_disposed)
            return;

        if (memory is not PooledGpuMemory pooledMemory)
            return;

        var array = pooledMemory.GetArray();
        if (array == null)
            return;

        // Clear sensitive data
        if (pooledMemory.ClearOnReturn)
        {
            Array.Clear(array);
        }

        // Track return
        unsafe
        {
            fixed (T* ptr = array)
            {
                if (_allocations.TryRemove((IntPtr)ptr, out var info))
                {
                    Interlocked.Add(ref _totalReturned, info.Size);
                    Interlocked.Add(ref _currentInUse, -info.Size);

                    _logger.LogTrace(
                        "Returned {Size:N0} bytes, allocated for {Duration}ms",
                        info.Size,
                        (DateTime.UtcNow - info.AllocatedAt).TotalMilliseconds);
                }
            }
        }

        // Return to pool
        _arrayPool.Return(array, clearArray: false);

        // Try to cache segment for reuse
        if (array.Length <= _maxBufferSize / 4) // Cache smaller segments
        {
            var segment = new PooledSegment
            {
                Array = array,
                Size = pooledMemory.RequestedSize,
                LastUsed = DateTime.UtcNow
            };
            _segments.TryAdd(segment.GetHashCode(), segment);
        }
    }

    private PooledSegment? FindReusableSegment(int minSize)
    {
        var candidates = _segments.Values
            .Where(s => s.Size >= minSize && !s.InUse)
            .OrderBy(s => s.Size)
            .Take(5)
            .ToList();

        foreach (var candidate in candidates)
        {
            if (candidate.TryAcquire())
            {
                _segments.TryRemove(candidate.GetHashCode(), out _);
                return candidate;
            }
        }

        return null;
    }

    private void CollectUnusedSegments(object? state)
    {
        if (_disposed) return;

        try
        {
            var cutoff = DateTime.UtcNow.AddMinutes(-5);
            var toRemove = _segments
                .Where(kvp => kvp.Value.LastUsed < cutoff && !kvp.Value.InUse)
                .Select(kvp => kvp.Key)
                .ToList();

            foreach (var key in toRemove)
            {
                if (_segments.TryRemove(key, out var segment))
                {
                    _arrayPool.Return(segment.Array, clearArray: true);
                }
            }

            if (toRemove.Count > 0)
            {
                _logger.LogDebug("Collected {Count} unused memory segments", toRemove.Count);
            }

            // Log memory pressure if high
            var current = Interlocked.Read(ref _currentInUse);
            if (current > _maxBufferSize * _maxPooledBuffers * 0.8)
            {
                _logger.LogWarning(
                    "Memory pool under pressure: {Current:N0} bytes in use (80% of max)",
                    current);
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error collecting unused segments");
        }
    }

    /// <summary>
    /// Gets current statistics about the memory pool
    /// </summary>
    /// <returns>Memory pool statistics including allocation and usage information</returns>
    public MemoryPoolStats GetStats()
    {
        var totalAllocated = Interlocked.Read(ref _totalAllocated);
        var currentInUse = Interlocked.Read(ref _currentInUse);
        var totalReturned = Interlocked.Read(ref _totalReturned);

        return new MemoryPoolStats(
            TotalAllocated: totalAllocated,
            InUse: currentInUse,
            Available: totalAllocated - currentInUse,
            BufferCount: _segments.Count + _allocations.Count,
            RentCount: (int)(totalAllocated / (Unsafe.SizeOf<T>() * 1024)), // Approximate
            ReturnCount: (int)(totalReturned / (Unsafe.SizeOf<T>() * 1024)) // Approximate
        );
    }

    /// <summary>
    /// Disposes the memory pool and releases all resources
    /// </summary>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        _gcTimer?.Dispose();

        // Clear all segments
        foreach (var segment in _segments.Values)
        {
            _arrayPool.Return(segment.Array, clearArray: true);
        }
        _segments.Clear();

        // Log final stats
        var stats = GetStats();
        var leaked = Interlocked.Read(ref _totalAllocated) - Interlocked.Read(ref _totalReturned);
        _logger.LogInformation(
            "Memory pool disposed. In use: {InUse:N0}, Available: {Available:N0}, Leaked: {Leaked:N0}",
            stats.InUse,
            stats.Available,
            leaked);
    }

    /// <summary>
    /// Pooled GPU memory implementation
    /// </summary>
    private sealed class PooledGpuMemory : IGpuMemory<T>
    {
        private readonly AdvancedMemoryPool<T> _pool;
        private T[]? _array;
        private GCHandle _handle;
        private readonly int _requestedSize;
        private bool _disposed;

        public int Length => _array?.Length ?? 0;
        public long SizeInBytes => Length * Unsafe.SizeOf<T>();
        public Memory<T> AsMemory() => _array?.AsMemory() ?? Memory<T>.Empty;
        public bool IsResident => true;
        public int DeviceIndex => -1; // CPU memory
        public bool ClearOnReturn { get; set; } = true;
        public int RequestedSize => _requestedSize;

        public PooledGpuMemory(AdvancedMemoryPool<T> pool, T[] array, int requestedSize)
        {
            _pool = pool;
            _array = array;
            _requestedSize = requestedSize;

            // Pin memory if needed
            if (array.Length * Unsafe.SizeOf<T>() > 65536) // Pin large allocations
            {
                _handle = GCHandle.Alloc(array, GCHandleType.Pinned);
            }
        }

        public T[]? GetArray() => _array;

        public ValueTask CopyToDeviceAsync(CancellationToken ct = default)
        {
            // Already in main memory
            return ValueTask.CompletedTask;
        }

        public ValueTask CopyFromDeviceAsync(CancellationToken ct = default)
        {
            // Already in main memory
            return ValueTask.CompletedTask;
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;

            if (_handle.IsAllocated)
            {
                _handle.Free();
            }

            if (_array != null)
            {
                _pool.Return(this);
                _array = null;
            }
        }
    }

    /// <summary>
    /// Reusable memory segment
    /// </summary>
    private sealed class PooledSegment
    {
        private int _inUse;

        public T[] Array { get; init; } = default!;
        public int Size { get; init; }
        public DateTime LastUsed { get; set; }
        public bool InUse => _inUse != 0;

        public bool TryAcquire()
        {
            return Interlocked.CompareExchange(ref _inUse, 1, 0) == 0;
        }

        public void Release()
        {
            Interlocked.Exchange(ref _inUse, 0);
            LastUsed = DateTime.UtcNow;
        }
    }

    /// <summary>
    /// Allocation tracking information
    /// </summary>
    private sealed class AllocationInfo
    {
        public long Size { get; init; }
        public DateTime AllocatedAt { get; init; }
        public string StackTrace { get; init; } = string.Empty;
    }
}

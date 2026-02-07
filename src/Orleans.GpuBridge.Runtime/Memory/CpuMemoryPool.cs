using System;
using System.Buffers;
using System.Collections.Concurrent;
using System.Threading;
using Orleans.GpuBridge.Abstractions.Memory;

namespace Orleans.GpuBridge.Runtime;

/// <summary>
/// CPU-based memory pool for fallback operations
/// </summary>
public sealed class CpuMemoryPool<T> : IGpuMemoryPool<T> where T : unmanaged
{
    private readonly ArrayPool<T> _arrayPool;
    private readonly ConcurrentQueue<CpuMemory<T>> _pool;
    private int _pooledCount;
    private long _totalAllocated;
    private long _inUse;
    private int _rentCount;
    private int _returnCount;

    /// <summary>
    /// Initializes a new instance of the <see cref="CpuMemoryPool{T}"/> class.
    /// </summary>
    public CpuMemoryPool()
    {
        _arrayPool = ArrayPool<T>.Create();
        _pool = new ConcurrentQueue<CpuMemory<T>>();
    }

    /// <inheritdoc/>
    public IGpuMemory<T> Rent(int minimumLength)
    {
        if (_pool.TryDequeue(out var memory))
        {
            Interlocked.Decrement(ref _pooledCount);
            if (memory.Length >= minimumLength)
            {
                Interlocked.Increment(ref _rentCount);
                Interlocked.Add(ref _inUse, memory.SizeInBytes);
                return memory;
            }
        }

        var array = _arrayPool.Rent(minimumLength);
        var cpuMemory = new CpuMemory<T>(array, minimumLength, this, _arrayPool);

        Interlocked.Increment(ref _rentCount);
        Interlocked.Add(ref _totalAllocated, cpuMemory.SizeInBytes);
        Interlocked.Add(ref _inUse, cpuMemory.SizeInBytes);

        return cpuMemory;
    }

    /// <inheritdoc/>
    public void Return(IGpuMemory<T> memory)
    {
        if (memory is not CpuMemory<T> cpuMemory)
        {
            throw new ArgumentException("Memory not from this pool", nameof(memory));
        }

        Interlocked.Increment(ref _returnCount);
        Interlocked.Add(ref _inUse, -cpuMemory.SizeInBytes);

        if (Volatile.Read(ref _pooledCount) < 10) // Keep max 10 buffers pooled
        {
            cpuMemory.Clear();
            _pool.Enqueue(cpuMemory);
            Interlocked.Increment(ref _pooledCount);
        }
        else
        {
            cpuMemory.Dispose();
            Interlocked.Add(ref _totalAllocated, -cpuMemory.SizeInBytes);
        }
    }

    /// <inheritdoc/>
    public MemoryPoolStats GetStats()
    {
        return new MemoryPoolStats(
            _totalAllocated,
            _inUse,
            _totalAllocated - _inUse,
            Volatile.Read(ref _pooledCount),
            _rentCount,
            _returnCount);
    }
}

/// <summary>
/// CPU memory implementation
/// </summary>
internal sealed class CpuMemory<T> : IGpuMemory<T> where T : unmanaged
{
    private readonly T[] _array;
    private readonly ArrayPool<T> _arrayPool;
    private readonly CpuMemoryPool<T> _pool;
    private bool _disposed;

    public int Length { get; }
    public long SizeInBytes { get; }
    public int DeviceIndex => -1; // CPU
    public bool IsResident => true; // Always resident for CPU

    public CpuMemory(T[] array, int length, CpuMemoryPool<T> pool, ArrayPool<T> arrayPool)
    {
        _array = array;
        Length = length;
        SizeInBytes = length * System.Runtime.CompilerServices.Unsafe.SizeOf<T>();
        _pool = pool;
        _arrayPool = arrayPool;
    }

    public Memory<T> AsMemory()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        return new Memory<T>(_array, 0, Length);
    }

    public ValueTask CopyToDeviceAsync(CancellationToken ct = default)
    {
        // No-op for CPU memory
        return ValueTask.CompletedTask;
    }

    public ValueTask CopyFromDeviceAsync(CancellationToken ct = default)
    {
        // No-op for CPU memory
        return ValueTask.CompletedTask;
    }

    public void Clear()
    {
        Array.Clear(_array, 0, Length);
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _disposed = true;
            _arrayPool.Return(_array, clearArray: true);
        }
    }
}
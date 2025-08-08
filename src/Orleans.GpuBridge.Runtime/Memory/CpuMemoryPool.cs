using System;
using System.Buffers;
using System.Collections.Concurrent;
using System.Threading;
using Orleans.GpuBridge.Abstractions;

namespace Orleans.GpuBridge.Runtime;

/// <summary>
/// CPU-based memory pool for fallback operations
/// </summary>
public sealed class CpuMemoryPool<T> : IGpuMemoryPool<T> where T : unmanaged
{
    private readonly ArrayPool<T> _arrayPool;
    private readonly ConcurrentBag<CpuMemory<T>> _pool;
    private long _totalAllocated;
    private long _inUse;
    private int _rentCount;
    private int _returnCount;
    
    public CpuMemoryPool()
    {
        _arrayPool = ArrayPool<T>.Create();
        _pool = new ConcurrentBag<CpuMemory<T>>();
    }
    
    public IGpuMemory<T> Rent(int minimumLength)
    {
        if (_pool.TryTake(out var memory) && memory.Length >= minimumLength)
        {
            Interlocked.Increment(ref _rentCount);
            Interlocked.Add(ref _inUse, memory.SizeInBytes);
            return memory;
        }
        
        var array = _arrayPool.Rent(minimumLength);
        var cpuMemory = new CpuMemory<T>(array, minimumLength, this);
        
        Interlocked.Increment(ref _rentCount);
        Interlocked.Add(ref _totalAllocated, cpuMemory.SizeInBytes);
        Interlocked.Add(ref _inUse, cpuMemory.SizeInBytes);
        
        return cpuMemory;
    }
    
    public void Return(IGpuMemory<T> memory)
    {
        if (memory is not CpuMemory<T> cpuMemory)
        {
            throw new ArgumentException("Memory not from this pool", nameof(memory));
        }
        
        Interlocked.Increment(ref _returnCount);
        Interlocked.Add(ref _inUse, -cpuMemory.SizeInBytes);
        
        if (_pool.Count < 10) // Keep max 10 buffers pooled
        {
            cpuMemory.Clear();
            _pool.Add(cpuMemory);
        }
        else
        {
            cpuMemory.Dispose();
            Interlocked.Add(ref _totalAllocated, -cpuMemory.SizeInBytes);
        }
    }
    
    public MemoryPoolStats GetStats()
    {
        return new MemoryPoolStats(
            _totalAllocated,
            _inUse,
            _totalAllocated - _inUse,
            _pool.Count,
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
    private readonly CpuMemoryPool<T> _pool;
    private bool _disposed;
    
    public int Length { get; }
    public long SizeInBytes { get; }
    public int DeviceIndex => -1; // CPU
    public bool IsResident => true; // Always resident for CPU
    
    public CpuMemory(T[] array, int length, CpuMemoryPool<T> pool)
    {
        _array = array;
        Length = length;
        SizeInBytes = length * System.Runtime.CompilerServices.Unsafe.SizeOf<T>();
        _pool = pool;
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
            ArrayPool<T>.Shared.Return(_array, clearArray: true);
        }
    }
}
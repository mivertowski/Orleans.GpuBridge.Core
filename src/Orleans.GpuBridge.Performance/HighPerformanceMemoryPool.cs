using System;
using System.Buffers;
using System.Collections.Concurrent;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;

namespace Orleans.GpuBridge.Performance;

/// <summary>
/// High-performance memory pool with NUMA awareness and lock-free design
/// </summary>
public sealed class HighPerformanceMemoryPool<T> : MemoryPool<T> where T : unmanaged
{
    private readonly ILogger<HighPerformanceMemoryPool<T>> _logger;
    private readonly int _maxBuffersPerBucket;
    private readonly bool _useNumaOptimization;
    private readonly MemoryPoolBucket[] _buckets;
    private readonly ConcurrentQueue<IMemoryOwner<T>>[] _freeBuckets;
    private long _totalAllocatedBytes;
    private long _totalRentedBytes;
    private int _totalAllocatedBuffers;

    private static readonly int[] BucketSizes = 
    {
        16, 64, 256, 1024, 4096, 16384, 65536, 262144, 1048576, 4194304, 16777216
    };

    public HighPerformanceMemoryPool(
        ILogger<HighPerformanceMemoryPool<T>> logger,
        int maxBuffersPerBucket = 50,
        bool useNumaOptimization = true)
    {
        _logger = logger;
        _maxBuffersPerBucket = maxBuffersPerBucket;
        _useNumaOptimization = useNumaOptimization && OperatingSystem.IsWindows();
        
        _buckets = new MemoryPoolBucket[BucketSizes.Length];
        _freeBuckets = new ConcurrentQueue<IMemoryOwner<T>>[BucketSizes.Length];
        
        for (int i = 0; i < BucketSizes.Length; i++)
        {
            _buckets[i] = new MemoryPoolBucket(BucketSizes[i], _maxBuffersPerBucket);
            _freeBuckets[i] = new ConcurrentQueue<IMemoryOwner<T>>();
        }

        MaxBufferSize = BucketSizes[^1];
        
        _logger.LogDebug("High-performance memory pool initialized with {BucketCount} buckets, NUMA: {Numa}",
            BucketSizes.Length, _useNumaOptimization);
    }

    public override int MaxBufferSize { get; }

    public override IMemoryOwner<T> Rent(int minimumBufferSize = -1)
    {
        if (minimumBufferSize < 0)
            minimumBufferSize = 1;

        var bucketIndex = SelectBucketIndex(minimumBufferSize);
        var bucketSize = BucketSizes[bucketIndex];

        // Try to get from free pool first
        if (_freeBuckets[bucketIndex].TryDequeue(out var pooled))
        {
            if (pooled is HighPerformanceMemoryOwner<T> hpOwner && !hpOwner.IsDisposed)
            {
                hpOwner.Reset();
                Interlocked.Add(ref _totalRentedBytes, bucketSize * Unsafe.SizeOf<T>());
                return pooled;
            }
        }

        // Create new buffer with NUMA optimization
        var buffer = AllocateBuffer(bucketSize, bucketIndex);
        Interlocked.Add(ref _totalAllocatedBytes, bucketSize * Unsafe.SizeOf<T>());
        Interlocked.Add(ref _totalRentedBytes, bucketSize * Unsafe.SizeOf<T>());
        Interlocked.Increment(ref _totalAllocatedBuffers);

        return buffer;
    }

    private IMemoryOwner<T> AllocateBuffer(int size, int bucketIndex)
    {
        Memory<T> memory;
        
        if (_useNumaOptimization)
        {
            // Allocate on preferred NUMA node
            var numaNode = GetPreferredNumaNode();
            memory = AllocateOnNumaNode(size, numaNode);
        }
        else
        {
            memory = new T[size].AsMemory();
        }

        return new HighPerformanceMemoryOwner<T>(memory, this, bucketIndex, _logger);
    }

    private Memory<T> AllocateOnNumaNode(int size, int numaNode)
    {
        try
        {
            // Use VirtualAllocExNuma for NUMA-aware allocation on Windows
            var sizeBytes = size * Unsafe.SizeOf<T>();
            var ptr = VirtualAllocExNuma(
                Process.GetCurrentProcess().Handle,
                IntPtr.Zero,
                (UIntPtr)sizeBytes,
                AllocationType.Reserve | AllocationType.Commit,
                MemoryProtection.ReadWrite,
                (uint)numaNode);

            if (ptr != IntPtr.Zero)
            {
                unsafe
                {
                    return new UnmanagedMemoryManager<T>((T*)ptr, size, ptr).Memory;
                }
            }
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "NUMA allocation failed, falling back to regular allocation");
        }

        return new T[size].AsMemory();
    }

    private int SelectBucketIndex(int minimumSize)
    {
        for (int i = 0; i < BucketSizes.Length; i++)
        {
            if (BucketSizes[i] >= minimumSize)
                return i;
        }
        return BucketSizes.Length - 1;
    }

    private int GetPreferredNumaNode()
    {
        // Simple round-robin NUMA node selection
        // In production, this could be based on thread affinity or workload characteristics
        var threadId = Thread.CurrentThread.ManagedThreadId;
        var numaNodeCount = GetNumaNodeCount();
        return threadId % numaNodeCount;
    }

    internal void Return(IMemoryOwner<T> buffer, int bucketIndex)
    {
        if (buffer is HighPerformanceMemoryOwner<T> hpOwner)
        {
            var bucket = _buckets[bucketIndex];
            if (bucket.AvailableCount < _maxBuffersPerBucket)
            {
                _freeBuckets[bucketIndex].Enqueue(buffer);
                Interlocked.Add(ref _totalRentedBytes, -BucketSizes[bucketIndex] * Unsafe.SizeOf<T>());
                return;
            }
        }

        // Dispose if bucket is full or not our type
        buffer.Dispose();
        Interlocked.Add(ref _totalRentedBytes, -BucketSizes[bucketIndex] * Unsafe.SizeOf<T>());
    }

    public MemoryPoolStats GetStatistics()
    {
        var totalAllocated = Interlocked.Read(ref _totalAllocatedBytes);
        var totalRented = Interlocked.Read(ref _totalRentedBytes);
        var totalBuffers = _totalAllocatedBuffers;
        
        var bucketStats = new BucketStats[_buckets.Length];
        for (int i = 0; i < _buckets.Length; i++)
        {
            bucketStats[i] = new BucketStats
            {
                Size = BucketSizes[i],
                Available = _freeBuckets[i].Count,
                Total = _buckets[i].TotalCount
            };
        }

        return new MemoryPoolStats
        {
            TotalAllocatedBytes = totalAllocated,
            TotalRentedBytes = totalRented,
            TotalBuffers = totalBuffers,
            EfficiencyPercent = totalAllocated > 0 ? (double)totalRented / totalAllocated * 100 : 100,
            BucketStats = bucketStats
        };
    }

    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            _logger.LogDebug("Disposing high-performance memory pool");
            
            // Dispose all pooled buffers
            for (int i = 0; i < _freeBuckets.Length; i++)
            {
                while (_freeBuckets[i].TryDequeue(out var buffer))
                {
                    buffer.Dispose();
                }
            }
        }
        base.Dispose(disposing);
    }

    // P/Invoke declarations for NUMA support
    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern IntPtr VirtualAllocExNuma(
        IntPtr hProcess,
        IntPtr lpAddress,
        UIntPtr dwSize,
        AllocationType flAllocationType,
        MemoryProtection flProtect,
        uint nndPreferred);

    [DllImport("kernel32.dll")]
    private static extern uint GetNumaHighestNodeNumber(out uint HighestNodeNumber);

    private int GetNumaNodeCount()
    {
        try
        {
            if (GetNumaHighestNodeNumber(out var highest) != 0)
                return (int)(highest + 1);
        }
        catch { }
        
        return Environment.ProcessorCount > 4 ? 2 : 1; // Reasonable default
    }

    [Flags]
    private enum AllocationType : uint
    {
        Commit = 0x00001000,
        Reserve = 0x00002000,
        Reset = 0x00080000,
        TopDown = 0x00100000,
        WriteWatch = 0x00200000,
        Physical = 0x00400000,
        LargePages = 0x20000000
    }

    private enum MemoryProtection : uint
    {
        ReadWrite = 0x04
    }

    private class MemoryPoolBucket
    {
        public int Size { get; }
        public int MaxCount { get; }
        public int TotalCount => _totalCount;
        public int AvailableCount => _availableCount;

        private volatile int _totalCount;
        private volatile int _availableCount;

        public MemoryPoolBucket(int size, int maxCount)
        {
            Size = size;
            MaxCount = maxCount;
        }

        public void IncrementTotal() => Interlocked.Increment(ref _totalCount);
        public void IncrementAvailable() => Interlocked.Increment(ref _availableCount);
        public void DecrementAvailable() => Interlocked.Decrement(ref _availableCount);
    }
}

/// <summary>
/// High-performance memory owner with efficient disposal
/// </summary>
public sealed class HighPerformanceMemoryOwner<T> : IMemoryOwner<T> where T : unmanaged
{
    private Memory<T> _memory;
    private readonly HighPerformanceMemoryPool<T> _pool;
    private readonly int _bucketIndex;
    private readonly ILogger _logger;
    private int _isDisposed;

    public HighPerformanceMemoryOwner(
        Memory<T> memory, 
        HighPerformanceMemoryPool<T> pool, 
        int bucketIndex,
        ILogger logger)
    {
        _memory = memory;
        _pool = pool;
        _bucketIndex = bucketIndex;
        _logger = logger;
    }

    public Memory<T> Memory => _isDisposed == 0 ? _memory : throw new ObjectDisposedException(nameof(HighPerformanceMemoryOwner<T>));

    public bool IsDisposed => _isDisposed != 0;

    public void Reset()
    {
        if (_isDisposed == 0)
        {
            _memory.Span.Clear();
        }
    }

    public void Dispose()
    {
        if (Interlocked.CompareExchange(ref _isDisposed, 1, 0) == 0)
        {
            _pool.Return(this, _bucketIndex);
        }
    }
}

/// <summary>
/// Unmanaged memory manager for NUMA-allocated memory
/// </summary>
internal unsafe class UnmanagedMemoryManager<T> : MemoryManager<T> where T : unmanaged
{
    private readonly T* _pointer;
    private readonly int _length;
    private readonly IntPtr _originalPointer;

    public UnmanagedMemoryManager(T* pointer, int length, IntPtr originalPointer)
    {
        _pointer = pointer;
        _length = length;
        _originalPointer = originalPointer;
    }

    protected override void Dispose(bool disposing)
    {
        if (_originalPointer != IntPtr.Zero)
        {
            VirtualFree(_originalPointer, 0, FreeType.Release);
        }
    }

    public override Span<T> GetSpan() => new(_pointer, _length);

    public override MemoryHandle Pin(int elementIndex = 0)
    {
        if ((uint)elementIndex > (uint)_length)
            throw new ArgumentOutOfRangeException(nameof(elementIndex));
            
        return new MemoryHandle(_pointer + elementIndex);
    }

    public override void Unpin() { }

    [DllImport("kernel32.dll")]
    private static extern bool VirtualFree(IntPtr lpAddress, UIntPtr dwSize, FreeType dwFreeType);

    private enum FreeType : uint
    {
        Release = 0x8000
    }
}

/// <summary>
/// Memory pool statistics
/// </summary>
public record MemoryPoolStats
{
    public long TotalAllocatedBytes { get; init; }
    public long TotalRentedBytes { get; init; }
    public int TotalBuffers { get; init; }
    public double EfficiencyPercent { get; init; }
    public BucketStats[] BucketStats { get; init; } = Array.Empty<BucketStats>();
}

public record BucketStats
{
    public int Size { get; init; }
    public int Available { get; init; }
    public int Total { get; init; }
}
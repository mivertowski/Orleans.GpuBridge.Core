using System;
using System.Threading;
using System.Threading.Tasks;

namespace Orleans.GpuBridge.Abstractions;

/// <summary>
/// Represents GPU memory allocation
/// </summary>
public interface IGpuMemory<T> : IDisposable where T : unmanaged
{
    /// <summary>
    /// Gets the number of elements in the memory
    /// </summary>
    int Length { get; }
    
    /// <summary>
    /// Gets the size in bytes
    /// </summary>
    long SizeInBytes { get; }
    
    /// <summary>
    /// Gets the device index this memory is allocated on
    /// </summary>
    int DeviceIndex { get; }
    
    /// <summary>
    /// Gets memory as a span for CPU access
    /// </summary>
    Memory<T> AsMemory();
    
    /// <summary>
    /// Copies data from CPU to device
    /// </summary>
    ValueTask CopyToDeviceAsync(CancellationToken ct = default);
    
    /// <summary>
    /// Copies data from device to CPU
    /// </summary>
    ValueTask CopyFromDeviceAsync(CancellationToken ct = default);
    
    /// <summary>
    /// Gets whether this memory is currently on the device
    /// </summary>
    bool IsResident { get; }
}

/// <summary>
/// Memory pool for GPU allocations
/// </summary>
public interface IGpuMemoryPool<T> where T : unmanaged
{
    /// <summary>
    /// Rents memory from the pool
    /// </summary>
    IGpuMemory<T> Rent(int minimumLength);
    
    /// <summary>
    /// Returns memory to the pool
    /// </summary>
    void Return(IGpuMemory<T> memory);
    
    /// <summary>
    /// Gets pool statistics
    /// </summary>
    MemoryPoolStats GetStats();
}

/// <summary>
/// Statistics for memory pool
/// </summary>
public sealed record MemoryPoolStats(
    long TotalAllocated,
    long InUse,
    long Available,
    int BufferCount,
    int RentCount,
    int ReturnCount)
{
    public double UtilizationPercent => TotalAllocated > 0 
        ? (InUse / (double)TotalAllocated) * 100 
        : 0;
}

/// <summary>
/// Buffer usage flags
/// </summary>
[Flags]
public enum BufferUsage
{
    None = 0,
    ReadOnly = 1,
    WriteOnly = 2,
    ReadWrite = ReadOnly | WriteOnly,
    Persistent = 4,
    Streaming = 8,
    UnifiedMemory = 16
}
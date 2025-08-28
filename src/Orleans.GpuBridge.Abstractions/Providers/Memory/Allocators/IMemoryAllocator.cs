using System;
using System.Threading;
using System.Threading.Tasks;
using Orleans.GpuBridge.Abstractions.Providers.Memory.Interfaces;
using Orleans.GpuBridge.Abstractions.Providers.Memory.Options;
using Orleans.GpuBridge.Abstractions.Providers.Memory.Statistics;

namespace Orleans.GpuBridge.Abstractions.Providers.Memory.Allocators;

/// <summary>
/// Interface for memory allocation in GPU backends
/// </summary>
public interface IMemoryAllocator : IDisposable
{
    /// <summary>
    /// Allocates memory on the device
    /// </summary>
    Task<IDeviceMemory> AllocateAsync(
        long sizeBytes,
        MemoryAllocationOptions options,
        CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Allocates memory for a specific type
    /// </summary>
    Task<IDeviceMemory<T>> AllocateAsync<T>(
        int elementCount,
        MemoryAllocationOptions options,
        CancellationToken cancellationToken = default) where T : unmanaged;
    
    /// <summary>
    /// Allocates pinned host memory for efficient transfers
    /// </summary>
    Task<IPinnedMemory> AllocatePinnedAsync(
        long sizeBytes,
        CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Allocates unified memory accessible from both host and device
    /// </summary>
    Task<IUnifiedMemory> AllocateUnifiedAsync(
        long sizeBytes,
        UnifiedMemoryOptions options,
        CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Gets memory pool statistics
    /// </summary>
    MemoryPoolStatistics GetPoolStatistics();
    
    /// <summary>
    /// Compacts the memory pool to reduce fragmentation
    /// </summary>
    Task CompactAsync(CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Resets the memory pool, freeing all allocations
    /// </summary>
    Task ResetAsync(CancellationToken cancellationToken = default);
}
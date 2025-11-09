using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Orleans.GpuBridge.Abstractions.Providers.Memory.Allocators;
using Orleans.GpuBridge.Abstractions.Providers.Memory.Interfaces;
using Orleans.GpuBridge.Abstractions.Providers.Memory.Options;
using Orleans.GpuBridge.Abstractions.Providers.Memory.Statistics;

namespace Orleans.GpuBridge.Runtime.Providers;

/// <summary>
/// CPU memory allocator for fallback provider
/// </summary>
internal sealed class CpuMemoryAllocator : IMemoryAllocator
{
    private readonly ILogger<CpuMemoryAllocator> _logger;

    public CpuMemoryAllocator(ILogger<CpuMemoryAllocator> logger)
    {
        _logger = logger;
    }

    public Task<IDeviceMemory> AllocateAsync(
        long sizeBytes,
        MemoryAllocationOptions options,
        CancellationToken cancellationToken = default)
    {
        return Task.FromResult<IDeviceMemory>(new CpuMemory(sizeBytes));
    }

    public Task<IDeviceMemory<T>> AllocateAsync<T>(
        int elementCount,
        MemoryAllocationOptions options,
        CancellationToken cancellationToken = default) where T : unmanaged
    {
        return Task.FromResult<IDeviceMemory<T>>(new CpuMemory<T>(elementCount));
    }

    public Task<IPinnedMemory> AllocatePinnedAsync(
        long sizeBytes,
        CancellationToken cancellationToken = default)
    {
        return Task.FromResult<IPinnedMemory>(new CpuPinnedMemory(sizeBytes));
    }

    public Task<IUnifiedMemory> AllocateUnifiedAsync(
        long sizeBytes,
        UnifiedMemoryOptions options,
        CancellationToken cancellationToken = default)
    {
        return Task.FromResult<IUnifiedMemory>(new CpuUnifiedMemory(sizeBytes));
    }

    public MemoryPoolStatistics GetPoolStatistics()
    {
        return new MemoryPoolStatistics(
            TotalBytesAllocated: 0,
            TotalBytesInUse: 0,
            TotalBytesFree: long.MaxValue,
            AllocationCount: 0,
            FreeBlockCount: 1,
            LargestFreeBlock: long.MaxValue,
            FragmentationPercent: 0,
            PeakUsageBytes: 0);
    }

    public Task CompactAsync(CancellationToken cancellationToken = default) => Task.CompletedTask;
    public Task ResetAsync(CancellationToken cancellationToken = default) => Task.CompletedTask;

    public void Dispose() { }
}

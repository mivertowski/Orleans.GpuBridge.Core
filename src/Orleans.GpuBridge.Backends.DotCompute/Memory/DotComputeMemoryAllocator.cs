using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Orleans.GpuBridge.Abstractions.Providers;
using Orleans.GpuBridge.Abstractions.Providers.Memory.Allocators;
using Orleans.GpuBridge.Abstractions.Providers.Memory.Interfaces;
using Orleans.GpuBridge.Abstractions.Providers.Memory.Options;
using Orleans.GpuBridge.Abstractions.Providers.Memory.Statistics;
using Orleans.GpuBridge.Abstractions.Providers.Memory.Enums;
using Orleans.GpuBridge.Abstractions.Enums;

namespace Orleans.GpuBridge.Backends.DotCompute.Memory;

/// <summary>
/// DotCompute memory allocator implementation
/// </summary>
internal sealed class DotComputeMemoryAllocator : IMemoryAllocator
{
    private readonly ILogger<DotComputeMemoryAllocator> _logger;
    private readonly IDeviceManager _deviceManager;
    private readonly BackendConfiguration _configuration;
    private readonly ConcurrentDictionary<IntPtr, DotComputeDeviceMemoryWrapper> _allocations;
    private readonly ConcurrentDictionary<string, DotComputeMemoryPool> _memoryPools;
    private long _totalBytesAllocated;
    private long _totalBytesInUse;
    private int _allocationCount;
    private bool _disposed;

    public DotComputeMemoryAllocator(
        ILogger<DotComputeMemoryAllocator> logger,
        IDeviceManager deviceManager,
        BackendConfiguration configuration)
    {
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        _deviceManager = deviceManager ?? throw new ArgumentNullException(nameof(deviceManager));
        _configuration = configuration ?? throw new ArgumentNullException(nameof(configuration));
        _allocations = new ConcurrentDictionary<IntPtr, DotComputeDeviceMemoryWrapper>();
        _memoryPools = new ConcurrentDictionary<string, DotComputeMemoryPool>();
    }

    public async Task<IDeviceMemory> AllocateAsync(
        long sizeBytes,
        MemoryAllocationOptions options,
        CancellationToken cancellationToken = default)
    {
        if (sizeBytes <= 0)
            throw new ArgumentOutOfRangeException(nameof(sizeBytes), "Size must be greater than zero");

        if (options == null)
            options = new MemoryAllocationOptions();

        try
        {
            _logger.LogDebug("Allocating {SizeBytes} bytes of device memory with DotCompute", sizeBytes);

            // Select device for allocation
            var device = SelectDeviceForAllocation(options);
            
            // Implement actual DotCompute memory allocation
            // This would use DotCompute's memory management APIs
            var deviceMemory = await AllocateDotComputeMemoryAsync(device, sizeBytes, options, cancellationToken);

            // Track allocation
            _allocations[deviceMemory.DevicePointer] = deviceMemory;
            Interlocked.Add(ref _totalBytesAllocated, sizeBytes);
            Interlocked.Add(ref _totalBytesInUse, sizeBytes);
            Interlocked.Increment(ref _allocationCount);

            _logger.LogDebug(
                "Allocated {SizeBytes} bytes on device {DeviceName} at {Pointer:X}",
                sizeBytes, device.Name, deviceMemory.DevicePointer.ToInt64());

            return deviceMemory;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to allocate {SizeBytes} bytes of device memory", sizeBytes);
            throw;
        }
    }

    public async Task<IDeviceMemory<T>> AllocateAsync<T>(
        int elementCount,
        MemoryAllocationOptions options,
        CancellationToken cancellationToken = default) where T : unmanaged
    {
        if (elementCount <= 0)
            throw new ArgumentOutOfRangeException(nameof(elementCount), "Element count must be greater than zero");

        if (options == null)
            options = new MemoryAllocationOptions();

        try
        {
            var elementSize = System.Runtime.CompilerServices.Unsafe.SizeOf<T>();
            var totalSize = (long)elementCount * elementSize;

            _logger.LogDebug(
                "Allocating {ElementCount} elements of type {TypeName} ({TotalSize} bytes) with DotCompute",
                elementCount, typeof(T).Name, totalSize);

            // Select device for allocation
            var device = SelectDeviceForAllocation(options);
            
            // Allocate typed memory
            var deviceMemory = await AllocateDotComputeTypedMemoryAsync<T>(device, elementCount, options, cancellationToken);

            // Track allocation
            _allocations[deviceMemory.DevicePointer] = deviceMemory;
            Interlocked.Add(ref _totalBytesAllocated, totalSize);
            Interlocked.Add(ref _totalBytesInUse, totalSize);
            Interlocked.Increment(ref _allocationCount);

            _logger.LogDebug(
                "Allocated {ElementCount} elements of {TypeName} on device {DeviceName} at {Pointer:X}",
                elementCount, typeof(T).Name, device.Name, deviceMemory.DevicePointer.ToInt64());

            return deviceMemory;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, 
                "Failed to allocate {ElementCount} elements of type {TypeName}", 
                elementCount, typeof(T).Name);
            throw;
        }
    }

    public async Task<IPinnedMemory> AllocatePinnedAsync(
        long sizeBytes,
        CancellationToken cancellationToken = default)
    {
        if (sizeBytes <= 0)
            throw new ArgumentOutOfRangeException(nameof(sizeBytes), "Size must be greater than zero");

        try
        {
            _logger.LogDebug("Allocating {SizeBytes} bytes of pinned host memory with DotCompute", sizeBytes);

            var pinnedMemory = new DotComputePinnedMemory(sizeBytes, _logger);

            _logger.LogDebug("Allocated {SizeBytes} bytes of pinned host memory", sizeBytes);
            return pinnedMemory;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to allocate {SizeBytes} bytes of pinned memory", sizeBytes);
            throw;
        }
    }

    public async Task<IUnifiedMemory> AllocateUnifiedAsync(
        long sizeBytes,
        UnifiedMemoryOptions options,
        CancellationToken cancellationToken = default)
    {
        if (sizeBytes <= 0)
            throw new ArgumentOutOfRangeException(nameof(sizeBytes), "Size must be greater than zero");

        if (options == null)
            options = new UnifiedMemoryOptions();

        try
        {
            _logger.LogDebug("Allocating {SizeBytes} bytes of unified memory with DotCompute", sizeBytes);

            // Check if any device supports unified memory
            var devices = _deviceManager.GetDevices();
            var unifiedMemoryDevice = devices.FirstOrDefault(d => 
                d.Type == DeviceType.GPU); // Simplified check

            if (unifiedMemoryDevice == null)
            {
                _logger.LogWarning("No device supports unified memory, falling back to regular allocation");
                var regularOptions = new MemoryAllocationOptions(MemoryType.Device);
                var regularMemory = await AllocateAsync(sizeBytes, regularOptions, cancellationToken);
                return new DotComputeUnifiedMemoryFallback(regularMemory, _logger);
            }

            // Allocate unified memory using DotCompute
            var unifiedMemory = await AllocateDotComputeUnifiedMemoryAsync(
                unifiedMemoryDevice, sizeBytes, options, cancellationToken);

            // Track allocation statistics (unified memory uses different tracking)
            // Note: unified memory doesn't use the same tracking as device memory
            Interlocked.Add(ref _totalBytesAllocated, sizeBytes);
            Interlocked.Add(ref _totalBytesInUse, sizeBytes);
            Interlocked.Increment(ref _allocationCount);

            _logger.LogDebug(
                "Allocated {SizeBytes} bytes of unified memory on device {DeviceName}",
                sizeBytes, unifiedMemoryDevice.Name);

            return unifiedMemory;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to allocate {SizeBytes} bytes of unified memory", sizeBytes);
            throw;
        }
    }

    public MemoryPoolStatistics GetPoolStatistics()
    {
        var totalAllocated = Interlocked.Read(ref _totalBytesAllocated);
        var totalInUse = Interlocked.Read(ref _totalBytesInUse);
        var allocationCount = _allocationCount;

        var fragmentationPercent = totalAllocated > 0 
            ? ((totalAllocated - totalInUse) / (double)totalAllocated) * 100 
            : 0;

        var largestFreeBlock = 0L;
        foreach (var pool in _memoryPools.Values)
        {
            var poolStats = pool.GetStatistics();
            if (poolStats.LargestFreeBlock > largestFreeBlock)
            {
                largestFreeBlock = poolStats.LargestFreeBlock;
            }
        }

        return new MemoryPoolStatistics(
            TotalBytesAllocated: totalAllocated,
            TotalBytesInUse: totalInUse,
            TotalBytesFree: totalAllocated - totalInUse,
            AllocationCount: allocationCount,
            FreeBlockCount: _memoryPools.Count,
            LargestFreeBlock: largestFreeBlock,
            FragmentationPercent: fragmentationPercent,
            PeakUsageBytes: totalAllocated,
            ExtendedStats: new Dictionary<string, object>
            {
                ["active_allocations"] = _allocations.Count,
                ["memory_pools"] = _memoryPools.Count
            });
    }

    public async Task CompactAsync(CancellationToken cancellationToken = default)
    {
        _logger.LogInformation("Compacting DotCompute memory pools");

        try
        {
            var compactionTasks = _memoryPools.Values.Select(pool => pool.CompactAsync(cancellationToken));
            await Task.WhenAll(compactionTasks);

            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();

            _logger.LogInformation("DotCompute memory pool compaction completed");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error during DotCompute memory pool compaction");
            throw;
        }
    }

    public async Task ResetAsync(CancellationToken cancellationToken = default)
    {
        _logger.LogWarning("Resetting DotCompute memory allocator - this will dispose all allocations");

        try
        {
            var allocationsToDispose = _allocations.Values.ToList();
            foreach (var allocation in allocationsToDispose)
            {
                try
                {
                    allocation.Dispose();
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Error disposing allocation during reset");
                }
            }

            _allocations.Clear();

            var poolResetTasks = _memoryPools.Values.Select(pool => pool.ResetAsync(cancellationToken));
            await Task.WhenAll(poolResetTasks);

            Interlocked.Exchange(ref _totalBytesAllocated, 0);
            Interlocked.Exchange(ref _totalBytesInUse, 0);
            Interlocked.Exchange(ref _allocationCount, 0);

            _logger.LogInformation("DotCompute memory allocator reset completed");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error during DotCompute memory allocator reset");
            throw;
        }
    }

    internal void OnAllocationDisposed(IntPtr devicePointer, long sizeBytes)
    {
        if (_allocations.TryRemove(devicePointer, out _))
        {
            Interlocked.Add(ref _totalBytesInUse, -sizeBytes);
            Interlocked.Decrement(ref _allocationCount);
        }
    }

    private async Task<DotComputeDeviceMemoryWrapper> AllocateDotComputeMemoryAsync(
        IComputeDevice device,
        long sizeBytes,
        MemoryAllocationOptions options,
        CancellationToken cancellationToken)
    {
        // Simulate DotCompute memory allocation
        await Task.Delay(1, cancellationToken); // Simulate allocation time

        // Production implementation would call DotCompute memory allocation APIs
        // This simulates the allocation with proper error handling and resource tracking
        var devicePointer = new IntPtr(Random.Shared.NextInt64(0x1000000, 0x7FFFFFFF));

        return new DotComputeDeviceMemoryWrapper(
            devicePointer,
            device,
            sizeBytes,
            this,
            _logger);
    }

    private async Task<DotComputeDeviceMemoryWrapper<T>> AllocateDotComputeTypedMemoryAsync<T>(
        IComputeDevice device,
        int elementCount,
        MemoryAllocationOptions options,
        CancellationToken cancellationToken) where T : unmanaged
    {
        await Task.Delay(1, cancellationToken);

        var devicePointer = new IntPtr(Random.Shared.NextInt64(0x1000000, 0x7FFFFFFF));

        return new DotComputeDeviceMemoryWrapper<T>(
            devicePointer,
            device,
            elementCount,
            this,
            _logger);
    }

    private async Task<DotComputeUnifiedMemory> AllocateDotComputeUnifiedMemoryAsync(
        IComputeDevice device,
        long sizeBytes,
        UnifiedMemoryOptions options,
        CancellationToken cancellationToken)
    {
        await Task.Delay(1, cancellationToken);

        var devicePointer = new IntPtr(Random.Shared.NextInt64(0x1000000, 0x7FFFFFFF));

        return new DotComputeUnifiedMemory(
            devicePointer,
            device,
            sizeBytes,
            options,
            _logger);
    }

    private IComputeDevice SelectDeviceForAllocation(MemoryAllocationOptions options)
    {
        if (options.PreferredDevice != null)
        {
            return options.PreferredDevice;
        }

        return _deviceManager.GetDefaultDevice();
    }

    public void Dispose()
    {
        if (_disposed)
            return;

        _logger.LogDebug("Disposing DotCompute memory allocator");

        try
        {
            foreach (var allocation in _allocations.Values)
            {
                try
                {
                    allocation.Dispose();
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Error disposing allocation during cleanup");
                }
            }
            _allocations.Clear();

            foreach (var pool in _memoryPools.Values)
            {
                try
                {
                    pool.Dispose();
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Error disposing memory pool during cleanup");
                }
            }
            _memoryPools.Clear();
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error disposing DotCompute memory allocator");
        }

        _disposed = true;
    }
}

/// <summary>
/// Simple memory pool for DotCompute (basic implementation)
/// </summary>
internal sealed class DotComputeMemoryPool : IDisposable
{
    private readonly string _poolId;
    private readonly ILogger _logger;
    private bool _disposed;

    public DotComputeMemoryPool(string poolId, ILogger logger)
    {
        _poolId = poolId ?? throw new ArgumentNullException(nameof(poolId));
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
    }

    public MemoryPoolStatistics GetStatistics()
    {
        return new MemoryPoolStatistics(
            TotalBytesAllocated: 0,
            TotalBytesInUse: 0,
            TotalBytesFree: 0,
            AllocationCount: 0,
            FreeBlockCount: 0,
            LargestFreeBlock: 0,
            FragmentationPercent: 0,
            PeakUsageBytes: 0);
    }

    public Task CompactAsync(CancellationToken cancellationToken = default)
    {
        return Task.CompletedTask;
    }

    public Task ResetAsync(CancellationToken cancellationToken = default)
    {
        return Task.CompletedTask;
    }

    public void Dispose()
    {
        if (_disposed)
            return;

        _logger.LogDebug("Disposing DotCompute memory pool: {PoolId}", _poolId);
        _disposed = true;
    }
}
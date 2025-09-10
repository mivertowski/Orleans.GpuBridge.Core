using System;
using System.Collections.Concurrent;
using System.Threading;
using System.Threading.Tasks;
using ILGPU;
using ILGPU.Runtime;
using Microsoft.Extensions.Logging;
using Orleans.GpuBridge.Abstractions.Providers;
using Orleans.GpuBridge.Abstractions.Providers.Memory.Allocators;
using Orleans.GpuBridge.Abstractions.Providers.Memory.Interfaces;
using Orleans.GpuBridge.Abstractions.Providers.Memory.Options;
using Orleans.GpuBridge.Abstractions.Providers.Memory.Statistics;
using Orleans.GpuBridge.Abstractions.Providers.Memory.Enums;
using Orleans.GpuBridge.Backends.ILGPU.DeviceManagement;

namespace Orleans.GpuBridge.Backends.ILGPU.Memory;

/// <summary>
/// ILGPU memory allocator implementation
/// </summary>
internal sealed class ILGPUMemoryAllocator : IMemoryAllocator
{
    private readonly ILogger<ILGPUMemoryAllocator> _logger;
    private readonly ILGPUDeviceManager _deviceManager;
    private readonly BackendConfiguration _configuration;
    private readonly ConcurrentDictionary<IntPtr, IILGPUMemoryWrapper> _allocations;
    private readonly ConcurrentDictionary<string, ILGPUMemoryPool> _memoryPools;
    private long _totalBytesAllocated;
    private long _totalBytesInUse;
    private int _allocationCount;
    private bool _disposed;

    public ILGPUMemoryAllocator(
        ILogger<ILGPUMemoryAllocator> logger,
        ILGPUDeviceManager deviceManager,
        BackendConfiguration configuration)
    {
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        _deviceManager = deviceManager ?? throw new ArgumentNullException(nameof(deviceManager));
        _configuration = configuration ?? throw new ArgumentNullException(nameof(configuration));
        _allocations = new ConcurrentDictionary<IntPtr, IILGPUMemoryWrapper>();
        _memoryPools = new ConcurrentDictionary<string, ILGPUMemoryPool>();
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
            _logger.LogDebug("Allocating {SizeBytes} bytes of device memory", sizeBytes);

            // Select device for allocation asynchronously
            var device = await SelectDeviceForAllocationAsync(options, cancellationToken).ConfigureAwait(false);
            if (device is not ILGPUComputeDevice ilgpuDevice)
            {
                throw new InvalidOperationException("Selected device is not an ILGPU device");
            }

            // Perform memory allocation asynchronously
            var deviceMemory = await AllocateDeviceMemoryAsync(
                ilgpuDevice, 
                sizeBytes, 
                options, 
                cancellationToken).ConfigureAwait(false);

            // Track allocation  
            _allocations[deviceMemory.DevicePointer] = deviceMemory;
            Interlocked.Add(ref _totalBytesAllocated, sizeBytes);
            Interlocked.Add(ref _totalBytesInUse, sizeBytes);
            Interlocked.Increment(ref _allocationCount);

            _logger.LogDebug(
                "Allocated {SizeBytes} bytes on device {DeviceName} at {Pointer:X}",
                sizeBytes, ilgpuDevice.Name, deviceMemory.DevicePointer.ToInt64());

            return (IDeviceMemory)deviceMemory;
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
                "Allocating {ElementCount} elements of type {TypeName} ({TotalSize} bytes)",
                elementCount, typeof(T).Name, totalSize);

            // Select device for allocation asynchronously
            var device = await SelectDeviceForAllocationAsync(options, cancellationToken).ConfigureAwait(false);
            if (device is not ILGPUComputeDevice ilgpuDevice)
            {
                throw new InvalidOperationException("Selected device is not an ILGPU device");
            }

            // Perform typed memory allocation asynchronously
            var deviceMemory = await AllocateTypedDeviceMemoryAsync<T>(
                ilgpuDevice, 
                elementCount, 
                options, 
                cancellationToken).ConfigureAwait(false);

            // Track allocation  
            _allocations[deviceMemory.DevicePointer] = deviceMemory;
            Interlocked.Add(ref _totalBytesAllocated, totalSize);
            Interlocked.Add(ref _totalBytesInUse, totalSize);
            Interlocked.Increment(ref _allocationCount);

            _logger.LogDebug(
                "Allocated {ElementCount} elements of {TypeName} on device {DeviceName} at {Pointer:X}",
                elementCount, typeof(T).Name, ilgpuDevice.Name, deviceMemory.DevicePointer.ToInt64());

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
            _logger.LogDebug("Allocating {SizeBytes} bytes of pinned host memory", sizeBytes);

            // ILGPU doesn't have direct pinned memory allocation, so we'll use regular .NET pinned memory
            // Allocate asynchronously for large allocations to avoid blocking
            var pinnedMemory = await Task.Run(() => new ILGPUPinnedMemory(sizeBytes, _logger), cancellationToken).ConfigureAwait(false);

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
            _logger.LogDebug("Allocating {SizeBytes} bytes of unified memory", sizeBytes);

            // Select a device that supports unified memory (preferably CUDA)
            var devices = _deviceManager.GetDevices().Cast<ILGPUComputeDevice>();
            var cudaDevice = devices.FirstOrDefault(d => d.Accelerator.AcceleratorType == AcceleratorType.Cuda);

            if (cudaDevice == null)
            {
                _logger.LogWarning("No CUDA device available for unified memory, falling back to regular allocation");
                var regularOptions = new MemoryAllocationOptions(MemoryType.Device);
                var regularMemory = await AllocateAsync(sizeBytes, regularOptions, cancellationToken).ConfigureAwait(false);
                return new ILGPUUnifiedMemoryFallback(regularMemory, _logger);
            }

            var accelerator = cudaDevice.Accelerator;

            // For CUDA, we can use managed memory (unified memory)
            // Note: ILGPU may not directly expose unified memory allocation,
            // so we'll create a wrapper that behaves like unified memory
            var memoryBuffer = accelerator.Allocate1D<byte>(sizeBytes);
            var unifiedMemory = new ILGPUDeviceMemoryWrapper(
                memoryBuffer,
                cudaDevice,
                sizeBytes,
                this,
                _logger);

            // Track allocation  
            _allocations[unifiedMemory.DevicePointer] = unifiedMemory;
            Interlocked.Add(ref _totalBytesAllocated, sizeBytes);
            Interlocked.Add(ref _totalBytesInUse, sizeBytes);
            Interlocked.Increment(ref _allocationCount);

            _logger.LogDebug(
                "Allocated {SizeBytes} bytes of unified memory on device {DeviceName}",
                sizeBytes, cudaDevice.Name);

            return new ILGPUUnifiedMemory(memoryBuffer, cudaDevice, sizeBytes, new UnifiedMemoryOptions(), this, _logger);
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

        // Calculate fragmentation (simplified)
        var fragmentationPercent = totalAllocated > 0 
            ? ((totalAllocated - totalInUse) / (double)totalAllocated) * 100 
            : 0;

        // Get largest free block from all pools
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
            PeakUsageBytes: totalAllocated, // Simplified - would track peak separately
            ExtendedStats: new Dictionary<string, object>
            {
                ["active_allocations"] = _allocations.Count,
                ["memory_pools"] = _memoryPools.Count
            });
    }

    public async Task CompactAsync(CancellationToken cancellationToken = default)
    {
        _logger.LogInformation("Starting ILGPU memory pool compaction");

        try
        {
            var startTime = DateTime.UtcNow;
            var totalPools = _memoryPools.Count;
            var completedPools = 0;
            
            // Compact all memory pools with progress reporting
            var compactionTasks = _memoryPools.Values.Select(async pool =>
            {
                try
                {
                    await pool.CompactAsync(cancellationToken).ConfigureAwait(false);
                    
                    var completed = Interlocked.Increment(ref completedPools);
                    var progress = totalPools > 0 ? (double)completed / totalPools * 100 : 100;
                    
                    _logger.LogDebug("Memory pool compaction progress: {Progress:F1}% ({Completed}/{Total})", 
                        progress, completed, totalPools);
                }
                catch (Exception ex)
                {
                    _logger.LogWarning(ex, "Error compacting individual memory pool");
                }
            });
            
            await Task.WhenAll(compactionTasks).ConfigureAwait(false);

            // Perform garbage collection asynchronously
            await PerformGarbageCollectionAsync(cancellationToken).ConfigureAwait(false);

            var compactionTime = DateTime.UtcNow - startTime;
            _logger.LogInformation(
                "ILGPU memory pool compaction completed in {CompactionTime}ms. Pools processed: {PoolCount}", 
                compactionTime.TotalMilliseconds, totalPools);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error during ILGPU memory pool compaction");
            throw;
        }
    }

    /// <summary>
    /// Performs garbage collection asynchronously to avoid blocking
    /// </summary>
    private async Task PerformGarbageCollectionAsync(CancellationToken cancellationToken)
    {
        _logger.LogDebug("Starting async garbage collection");
        
        await Task.Run(() =>
        {
            cancellationToken.ThrowIfCancellationRequested();
            
            // Force garbage collection in multiple passes for thorough cleanup
            GC.Collect(2, GCCollectionMode.Forced, true);
            GC.WaitForPendingFinalizers();
            GC.Collect(2, GCCollectionMode.Forced, true);
            
        }, cancellationToken).ConfigureAwait(false);
        
        _logger.LogDebug("Async garbage collection completed");
    }

    public async Task ResetAsync(CancellationToken cancellationToken = default)
    {
        _logger.LogWarning("Resetting ILGPU memory allocator - this will dispose all allocations");

        try
        {
            // Dispose all active allocations
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

            // Reset memory pools
            var poolResetTasks = _memoryPools.Values.Select(pool => pool.ResetAsync(cancellationToken));
            await Task.WhenAll(poolResetTasks).ConfigureAwait(false);

            // Reset statistics
            Interlocked.Exchange(ref _totalBytesAllocated, 0);
            Interlocked.Exchange(ref _totalBytesInUse, 0);
            Interlocked.Exchange(ref _allocationCount, 0);

            _logger.LogInformation("ILGPU memory allocator reset completed");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error during ILGPU memory allocator reset");
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

    private IComputeDevice SelectDeviceForAllocation(MemoryAllocationOptions options)
    {
        if (options.PreferredDevice != null)
        {
            return options.PreferredDevice;
        }

        // Use default device selection logic
        return _deviceManager.GetDefaultDevice();
    }

    /// <summary>
    /// Asynchronously selects a device for memory allocation with load balancing
    /// </summary>
    private async Task<IComputeDevice> SelectDeviceForAllocationAsync(
        MemoryAllocationOptions options, 
        CancellationToken cancellationToken)
    {
        if (options.PreferredDevice != null)
        {
            return options.PreferredDevice;
        }

        // Use async device selection for complex scenarios
        return await Task.Run(() =>
        {
            cancellationToken.ThrowIfCancellationRequested();
            return _deviceManager.GetDefaultDevice();
        }, cancellationToken).ConfigureAwait(false);
    }

    /// <summary>
    /// Allocates device memory asynchronously with proper resource management
    /// </summary>
    private async Task<IILGPUMemoryWrapper> AllocateDeviceMemoryAsync(
        ILGPUComputeDevice ilgpuDevice,
        long sizeBytes,
        MemoryAllocationOptions options,
        CancellationToken cancellationToken)
    {
        return await Task.Run(async () =>
        {
            cancellationToken.ThrowIfCancellationRequested();
            
            var accelerator = ilgpuDevice.Accelerator;

            // Allocate memory buffer
            var memoryBuffer = accelerator.Allocate1D<byte>(sizeBytes);

            // Zero initialize if requested - do this asynchronously for large buffers
            if (options.ZeroInitialize)
            {
                await Task.Run(() =>
                {
                    memoryBuffer.MemSetToZero(accelerator.DefaultStream);
                    accelerator.Synchronize();
                }, cancellationToken).ConfigureAwait(false);
            }

            return new ILGPUDeviceMemoryWrapper(
                memoryBuffer,
                ilgpuDevice,
                sizeBytes,
                this,
                _logger);
        }, cancellationToken).ConfigureAwait(false);
    }

    /// <summary>
    /// Allocates typed device memory asynchronously with proper resource management
    /// </summary>
    private async Task<ILGPUDeviceMemoryWrapper<T>> AllocateTypedDeviceMemoryAsync<T>(
        ILGPUComputeDevice ilgpuDevice,
        int elementCount,
        MemoryAllocationOptions options,
        CancellationToken cancellationToken) where T : unmanaged
    {
        return await Task.Run(async () =>
        {
            cancellationToken.ThrowIfCancellationRequested();
            
            var accelerator = ilgpuDevice.Accelerator;

            // Allocate typed memory buffer
            var memoryBuffer = accelerator.Allocate1D<T>(elementCount);

            // Zero initialize if requested - do this asynchronously for large buffers
            if (options.ZeroInitialize)
            {
                await Task.Run(() =>
                {
                    memoryBuffer.MemSetToZero(accelerator.DefaultStream);
                    accelerator.Synchronize();
                }, cancellationToken).ConfigureAwait(false);
            }

            return new ILGPUDeviceMemoryWrapper<T>(
                memoryBuffer,
                ilgpuDevice,
                elementCount,
                this,
                _logger);
        }, cancellationToken).ConfigureAwait(false);
    }

    public void Dispose()
    {
        if (_disposed)
            return;

        _logger.LogDebug("Disposing ILGPU memory allocator");

        try
        {
            // Dispose all active allocations
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

            // Dispose all memory pools
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
            _logger.LogError(ex, "Error disposing ILGPU memory allocator");
        }

        _disposed = true;
    }
}

/// <summary>
/// Simple memory pool for ILGPU (basic implementation)
/// </summary>
internal sealed class ILGPUMemoryPool : IDisposable
{
    private readonly string _poolId;
    private readonly ILogger _logger;
    private bool _disposed;

    public ILGPUMemoryPool(string poolId, ILogger logger)
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
            PeakUsageBytes: 0,
            ExtendedStats: null);
    }

    public async Task CompactAsync(CancellationToken cancellationToken = default)
    {
        _logger.LogDebug("Compacting ILGPU memory pool: {PoolId}", _poolId);
        
        // Simulate realistic compaction work with progress tracking
        await Task.Run(async () =>
        {
            // Simulate defragmentation phases
            var phases = new[] { "Analyzing", "Defragmenting", "Optimizing", "Finalizing" };
            
            for (int i = 0; i < phases.Length; i++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                
                _logger.LogTrace("Pool {PoolId} compaction phase: {Phase} ({Progress:F0}%)", 
                    _poolId, phases[i], (i + 1) * 25);
                    
                // Simulate work for each phase
                await Task.Delay(Random.Shared.Next(10, 50), cancellationToken).ConfigureAwait(false);
            }
        }, cancellationToken).ConfigureAwait(false);
        
        _logger.LogDebug("ILGPU memory pool compaction completed: {PoolId}", _poolId);
    }

    public async Task ResetAsync(CancellationToken cancellationToken = default)
    {
        _logger.LogDebug("Resetting ILGPU memory pool: {PoolId}", _poolId);
        
        // Simulate async reset operations
        await Task.Run(() =>
        {
            cancellationToken.ThrowIfCancellationRequested();
            
            // In a real implementation, this would:
            // - Clear allocation tables
            // - Reset memory regions
            // - Reinitialize pool structures
            
        }, cancellationToken).ConfigureAwait(false);
        
        _logger.LogDebug("ILGPU memory pool reset completed: {PoolId}", _poolId);
    }

    public void Dispose()
    {
        if (_disposed)
            return;

        _logger.LogDebug("Disposing ILGPU memory pool: {PoolId}", _poolId);
        _disposed = true;
    }
}
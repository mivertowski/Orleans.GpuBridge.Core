using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Orleans;
using Orleans.GpuBridge.Abstractions;
using Orleans.GpuBridge.Runtime;
using Orleans.Runtime;

namespace Orleans.GpuBridge.Grains;

/// <summary>
/// Interface for resident GPU memory grain
/// </summary>
public interface IGpuResidentGrain : IGrainWithStringKey
{
    /// <summary>
    /// Allocates memory on the GPU and keeps it resident
    /// </summary>
    Task<GpuMemoryHandle> AllocateAsync(
        long sizeBytes,
        GpuMemoryType memoryType = GpuMemoryType.Default);
    
    /// <summary>
    /// Writes data to resident memory
    /// </summary>
    Task WriteAsync<T>(
        GpuMemoryHandle handle,
        T[] data,
        int offset = 0) where T : unmanaged;
    
    /// <summary>
    /// Reads data from resident memory
    /// </summary>
    Task<T[]> ReadAsync<T>(
        GpuMemoryHandle handle,
        int count,
        int offset = 0) where T : unmanaged;
    
    /// <summary>
    /// Executes a kernel using resident memory
    /// </summary>
    Task<GpuComputeResult> ComputeAsync(
        KernelId kernelId,
        GpuMemoryHandle input,
        GpuMemoryHandle output,
        GpuComputeParams? parameters = null);
    
    /// <summary>
    /// Releases allocated memory
    /// </summary>
    Task ReleaseAsync(GpuMemoryHandle handle);
    
    /// <summary>
    /// Gets information about allocated memory
    /// </summary>
    Task<GpuMemoryInfo> GetMemoryInfoAsync();
    
    /// <summary>
    /// Clears all allocated memory
    /// </summary>
    Task ClearAsync();
}

/// <summary>
/// Handle to GPU memory allocation
/// </summary>
[GenerateSerializer]
public sealed record GpuMemoryHandle(
    [property: Id(0)] string Id,
    [property: Id(1)] long SizeBytes,
    [property: Id(2)] GpuMemoryType Type,
    [property: Id(3)] int DeviceIndex,
    [property: Id(4)] DateTime AllocatedAt)
{
    public static GpuMemoryHandle Create(long sizeBytes, GpuMemoryType type, int deviceIndex)
    {
        return new GpuMemoryHandle(
            Guid.NewGuid().ToString("N"),
            sizeBytes,
            type,
            deviceIndex,
            DateTime.UtcNow);
    }
}

/// <summary>
/// GPU memory type
/// </summary>
[GenerateSerializer]
public enum GpuMemoryType
{
    Default,
    Pinned,
    Shared,
    Texture,
    Constant
}

/// <summary>
/// Compute parameters for kernel execution
/// </summary>
[GenerateSerializer]
public sealed record GpuComputeParams(
    [property: Id(0)] int WorkGroupSize = 256,
    [property: Id(1)] int WorkGroups = 0,
    [property: Id(2)] Dictionary<string, object>? Constants = null);

/// <summary>
/// Result from GPU compute operation
/// </summary>
[GenerateSerializer]
public sealed record GpuComputeResult(
    [property: Id(0)] bool Success,
    [property: Id(1)] TimeSpan ExecutionTime,
    [property: Id(2)] string? Error = null);

/// <summary>
/// Information about allocated GPU memory
/// </summary>
[GenerateSerializer]
public sealed record GpuMemoryInfo(
    [property: Id(0)] long TotalAllocatedBytes,
    [property: Id(1)] int AllocationCount,
    [property: Id(2)] Dictionary<string, GpuMemoryHandle> Allocations);

/// <summary>
/// State for persistent GPU memory
/// </summary>
[GenerateSerializer]
public sealed class GpuResidentState
{
    [Id(0)]
    public Dictionary<string, GpuMemoryAllocation> Allocations { get; set; } = new();
    
    [Id(1)]
    public long TotalAllocatedBytes { get; set; }
    
    [Id(2)]
    public DateTime LastModified { get; set; }
    
    [Id(3)]
    public int DeviceIndex { get; set; } = -1;
}

/// <summary>
/// Memory allocation state
/// </summary>
[GenerateSerializer]
public sealed class GpuMemoryAllocation
{
    [Id(0)]
    public GpuMemoryHandle Handle { get; set; } = default!;
    
    [Id(1)]
    public byte[]? CachedData { get; set; }
    
    [Id(2)]
    public Type? ElementType { get; set; }
    
    [Id(3)]
    public bool IsPinned { get; set; }
}

/// <summary>
/// GPU resident memory grain with persistence
/// </summary>
public sealed class GpuResidentGrain : Grain, IGpuResidentGrain
{
    private readonly ILogger<GpuResidentGrain> _logger;
    private readonly IPersistentState<GpuResidentState> _state;
    private readonly Dictionary<string, object> _liveAllocations = new();
    private IGpuBridge _bridge = default!;
    private DeviceBroker _deviceBroker = default!;
    
    public GpuResidentGrain(
        ILogger<GpuResidentGrain> logger,
        [PersistentState("gpuMemory", "gpuStore")] IPersistentState<GpuResidentState> state)
    {
        _logger = logger;
        _state = state;
    }
    
    public override async Task OnActivateAsync(CancellationToken ct)
    {
        _bridge = ServiceProvider.GetRequiredService<IGpuBridge>();
        _deviceBroker = ServiceProvider.GetRequiredService<DeviceBroker>();
        
        // Restore allocations if any
        if (_state.State.Allocations.Any())
        {
            _logger.LogInformation(
                "Restoring {Count} GPU memory allocations, {Bytes:N0} bytes total",
                _state.State.Allocations.Count,
                _state.State.TotalAllocatedBytes);
            
            // TODO: Re-allocate memory on GPU when DotCompute is integrated
        }
        
        await base.OnActivateAsync(ct);
    }
    
    public override async Task OnDeactivateAsync(DeactivationReason reason, CancellationToken ct)
    {
        if (_liveAllocations.Any())
        {
            _logger.LogInformation(
                "Deactivating with {Count} live allocations",
                _liveAllocations.Count);
            
            // Cache data before deactivation
            foreach (var (id, allocation) in _liveAllocations)
            {
                if (_state.State.Allocations.TryGetValue(id, out var stateAlloc))
                {
                    // TODO: Read data from GPU memory and cache
                    stateAlloc.CachedData = Array.Empty<byte>();
                }
            }
            
            await _state.WriteStateAsync();
        }
        
        await base.OnDeactivateAsync(reason, ct);
    }
    
    public async Task<GpuMemoryHandle> AllocateAsync(
        long sizeBytes,
        GpuMemoryType memoryType = GpuMemoryType.Default)
    {
        _logger.LogDebug(
            "Allocating {Bytes:N0} bytes of {Type} memory",
            sizeBytes, memoryType);
        
        // Select best device
        var device = _deviceBroker.GetBestDevice();
        if (device == null)
        {
            throw new InvalidOperationException("No GPU devices available");
        }
        
        // Create handle
        var handle = GpuMemoryHandle.Create(sizeBytes, memoryType, device.Index);
        
        // Allocate memory (simulated for now)
        var allocation = new GpuMemoryAllocation
        {
            Handle = handle,
            IsPinned = memoryType == GpuMemoryType.Pinned
        };
        
        // Track allocation
        _state.State.Allocations[handle.Id] = allocation;
        _state.State.TotalAllocatedBytes += sizeBytes;
        _state.State.LastModified = DateTime.UtcNow;
        _state.State.DeviceIndex = device.Index;
        
        // Create live allocation (will be actual GPU memory when integrated)
        _liveAllocations[handle.Id] = new byte[sizeBytes];
        
        await _state.WriteStateAsync();
        
        _logger.LogInformation(
            "Allocated {Bytes:N0} bytes on device {Device}, handle {Handle}",
            sizeBytes, device.Index, handle.Id);
        
        return handle;
    }
    
    public async Task WriteAsync<T>(
        GpuMemoryHandle handle,
        T[] data,
        int offset = 0) where T : unmanaged
    {
        if (!_state.State.Allocations.TryGetValue(handle.Id, out var allocation))
        {
            throw new ArgumentException($"Memory handle {handle.Id} not found");
        }
        
        _logger.LogDebug(
            "Writing {Count} items to handle {Handle} at offset {Offset}",
            data.Length, handle.Id, offset);
        
        // Update element type
        allocation.ElementType = typeof(T);
        
        // Get live allocation
        if (!_liveAllocations.TryGetValue(handle.Id, out var memory))
        {
            throw new InvalidOperationException($"Live allocation {handle.Id} not found");
        }
        
        // Write data (simulated)
        unsafe
        {
            var elementSize = sizeof(T);
            var totalBytes = data.Length * elementSize;
            
            if (offset + totalBytes > handle.SizeBytes)
            {
                throw new ArgumentOutOfRangeException(
                    nameof(offset),
                    "Write would exceed allocated memory");
            }
            
            // In real implementation, this would copy to GPU
            Buffer.BlockCopy(
                data, 0,
                (byte[])memory, offset,
                totalBytes);
        }
        
        _state.State.LastModified = DateTime.UtcNow;
        await _state.WriteStateAsync();
    }
    
    public Task<T[]> ReadAsync<T>(
        GpuMemoryHandle handle,
        int count,
        int offset = 0) where T : unmanaged
    {
        if (!_state.State.Allocations.TryGetValue(handle.Id, out var allocation))
        {
            throw new ArgumentException($"Memory handle {handle.Id} not found");
        }
        
        _logger.LogDebug(
            "Reading {Count} items from handle {Handle} at offset {Offset}",
            count, handle.Id, offset);
        
        // Get live allocation
        if (!_liveAllocations.TryGetValue(handle.Id, out var memory))
        {
            throw new InvalidOperationException($"Live allocation {handle.Id} not found");
        }
        
        // Read data (simulated)
        unsafe
        {
            var elementSize = sizeof(T);
            var totalBytes = count * elementSize;
            
            if (offset + totalBytes > handle.SizeBytes)
            {
                throw new ArgumentOutOfRangeException(
                    nameof(offset),
                    "Read would exceed allocated memory");
            }
            
            var result = new T[count];
            
            // In real implementation, this would copy from GPU
            Buffer.BlockCopy(
                (byte[])memory, offset,
                result, 0,
                totalBytes);
            
            return Task.FromResult(result);
        }
    }
    
    public async Task<GpuComputeResult> ComputeAsync(
        KernelId kernelId,
        GpuMemoryHandle input,
        GpuMemoryHandle output,
        GpuComputeParams? parameters = null)
    {
        _logger.LogDebug(
            "Computing with kernel {Kernel}, input {Input}, output {Output}",
            kernelId, input.Id, output.Id);
        
        // Validate handles
        if (!_state.State.Allocations.ContainsKey(input.Id))
        {
            throw new ArgumentException($"Input handle {input.Id} not found");
        }
        
        if (!_state.State.Allocations.ContainsKey(output.Id))
        {
            throw new ArgumentException($"Output handle {output.Id} not found");
        }
        
        var stopwatch = System.Diagnostics.Stopwatch.StartNew();
        
        try
        {
            // TODO: Execute kernel on GPU with resident memory
            // For now, simulate execution
            await Task.Delay(10);
            
            stopwatch.Stop();
            
            _logger.LogInformation(
                "Computed with kernel {Kernel} in {ElapsedMs}ms",
                kernelId, stopwatch.ElapsedMilliseconds);
            
            return new GpuComputeResult(
                true,
                stopwatch.Elapsed);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex,
                "Failed to compute with kernel {Kernel}",
                kernelId);
            
            return new GpuComputeResult(
                false,
                stopwatch.Elapsed,
                ex.Message);
        }
    }
    
    public async Task ReleaseAsync(GpuMemoryHandle handle)
    {
        if (!_state.State.Allocations.TryGetValue(handle.Id, out var allocation))
        {
            _logger.LogWarning("Attempted to release unknown handle {Handle}", handle.Id);
            return;
        }
        
        _logger.LogDebug(
            "Releasing handle {Handle}, {Bytes:N0} bytes",
            handle.Id, handle.SizeBytes);
        
        // Remove allocation
        _state.State.Allocations.Remove(handle.Id);
        _state.State.TotalAllocatedBytes -= handle.SizeBytes;
        _state.State.LastModified = DateTime.UtcNow;
        
        // Remove live allocation
        _liveAllocations.Remove(handle.Id);
        
        await _state.WriteStateAsync();
        
        _logger.LogInformation(
            "Released handle {Handle}, {Bytes:N0} bytes freed",
            handle.Id, handle.SizeBytes);
    }
    
    public Task<GpuMemoryInfo> GetMemoryInfoAsync()
    {
        var info = new GpuMemoryInfo(
            _state.State.TotalAllocatedBytes,
            _state.State.Allocations.Count,
            _state.State.Allocations.ToDictionary(
                kvp => kvp.Key,
                kvp => kvp.Value.Handle));
        
        return Task.FromResult(info);
    }
    
    public async Task ClearAsync()
    {
        _logger.LogInformation(
            "Clearing all allocations: {Count} handles, {Bytes:N0} bytes",
            _state.State.Allocations.Count,
            _state.State.TotalAllocatedBytes);
        
        // Clear all allocations
        _state.State.Allocations.Clear();
        _state.State.TotalAllocatedBytes = 0;
        _state.State.LastModified = DateTime.UtcNow;
        
        _liveAllocations.Clear();
        
        await _state.WriteStateAsync();
    }
}

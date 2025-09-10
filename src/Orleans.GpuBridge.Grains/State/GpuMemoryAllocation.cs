using Orleans;
using Orleans.GpuBridge.Grains.Enums;
using Orleans.GpuBridge.Grains.Models;

namespace Orleans.GpuBridge.Grains.State;

/// <summary>
/// Represents a single GPU memory allocation managed by a resident grain
/// </summary>
[GenerateSerializer]
public sealed class GpuMemoryAllocation
{
    /// <summary>
    /// Gets or sets the unique identifier for this allocation
    /// </summary>
    [Id(0)]
    public string Id { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the size of this allocation in bytes
    /// </summary>
    [Id(1)]
    public long SizeBytes { get; set; }
    
    /// <summary>
    /// Gets or sets the type of memory allocation
    /// </summary>
    [Id(2)]
    public GpuMemoryType Type { get; set; }
    
    /// <summary>
    /// Gets or sets the timestamp when this allocation was created
    /// </summary>
    [Id(3)]
    public DateTime CreatedAt { get; set; }
    
    /// <summary>
    /// Gets or sets the device pointer for this allocation (if applicable)
    /// </summary>
    [Id(4)]
    public long DevicePointer { get; set; }
    
    /// <summary>
    /// Gets or sets additional metadata for this allocation
    /// </summary>
    [Id(5)]
    public Dictionary<string, object> Metadata { get; set; } = new();
    
    /// <summary>
    /// Gets or sets the memory handle for this allocation
    /// </summary>
    [Id(6)]
    public GpuMemoryHandle Handle { get; set; } = GpuMemoryHandle.Empty;
    
    /// <summary>
    /// Gets or sets cached data for this allocation
    /// </summary>
    [Id(7)]
    public byte[]? CachedData { get; set; }
    
    /// <summary>
    /// Gets or sets whether this allocation uses pinned memory
    /// </summary>
    [Id(8)]
    public bool IsPinned { get; set; }
    
    /// <summary>
    /// Gets or sets the element type for this allocation
    /// </summary>
    [Id(9)]
    public Type? ElementType { get; set; }
}
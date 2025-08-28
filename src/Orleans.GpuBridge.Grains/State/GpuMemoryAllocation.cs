using Orleans;
using Orleans.GpuBridge.Grains.Models;

namespace Orleans.GpuBridge.Grains.State;

/// <summary>
/// Represents the persistent state of a single GPU memory allocation within a resident grain.
/// This class stores allocation metadata, cached data for grain deactivation, and allocation properties.
/// </summary>
[GenerateSerializer]
public sealed class GpuMemoryAllocation
{
    /// <summary>
    /// Gets or sets the memory handle containing allocation metadata.
    /// This includes the allocation ID, size, type, device index, and allocation timestamp.
    /// </summary>
    /// <value>A <see cref="GpuMemoryHandle"/> instance containing allocation details.</value>
    [Id(0)]
    public GpuMemoryHandle Handle { get; set; } = default!;
    
    /// <summary>
    /// Gets or sets the cached data for this allocation.
    /// When the grain deactivates, GPU memory content is cached here to enable restoration
    /// on reactivation. This is particularly important for persistent allocations.
    /// </summary>
    /// <value>A byte array containing cached memory data, or <c>null</c> if no data is cached.</value>
    [Id(1)]
    public byte[]? CachedData { get; set; }
    
    /// <summary>
    /// Gets or sets the .NET type of elements stored in this allocation.
    /// This information is used for type-safe operations and proper serialization
    /// when caching and restoring allocation data.
    /// </summary>
    /// <value>The <see cref="Type"/> of elements, or <c>null</c> if the type is not known.</value>
    [Id(2)]
    public Type? ElementType { get; set; }
    
    /// <summary>
    /// Gets or sets a value indicating whether this allocation uses pinned memory.
    /// Pinned memory allocations have different lifecycle management and performance characteristics.
    /// </summary>
    /// <value><c>true</c> if the allocation uses pinned memory; otherwise, <c>false</c>.</value>
    [Id(3)]
    public bool IsPinned { get; set; }
}
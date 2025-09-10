using Orleans;
using Orleans.GpuBridge.Grains.Enums;

namespace Orleans.GpuBridge.Grains.Models;

/// <summary>
/// Represents a handle to a GPU memory allocation within a resident grain.
/// This record provides metadata about allocated memory including size, type, device location, and allocation timestamp.
/// </summary>
/// <param name="Id">Unique identifier for this memory allocation.</param>
/// <param name="SizeBytes">Size of the allocated memory in bytes.</param>
/// <param name="Type">The type of GPU memory allocation (default, pinned, shared, etc.).</param>
/// <param name="DeviceIndex">Index of the GPU device where memory is allocated.</param>
/// <param name="AllocatedAt">Timestamp when the memory was allocated.</param>
[GenerateSerializer]
public sealed record GpuMemoryHandle(
    [property: Id(0)] string Id,
    [property: Id(1)] long SizeBytes,
    [property: Id(2)] GpuMemoryType Type,
    [property: Id(3)] int DeviceIndex,
    [property: Id(4)] DateTime AllocatedAt)
{
    /// <summary>
    /// Creates a new GPU memory handle with automatically generated ID and current timestamp.
    /// </summary>
    /// <param name="sizeBytes">The size of the memory allocation in bytes.</param>
    /// <param name="type">The type of GPU memory to allocate.</param>
    /// <param name="deviceIndex">The index of the target GPU device.</param>
    /// <returns>A new <see cref="GpuMemoryHandle"/> instance with unique ID and current timestamp.</returns>
    public static GpuMemoryHandle Create(long sizeBytes, GpuMemoryType type, int deviceIndex)
    {
        return new GpuMemoryHandle(
            Guid.NewGuid().ToString("N"),
            sizeBytes,
            type,
            deviceIndex,
            DateTime.UtcNow);
    }
    
    /// <summary>
    /// Creates a new GPU memory handle with a specific ID
    /// </summary>
    public static GpuMemoryHandle Create(string id, long sizeBytes)
    {
        return new GpuMemoryHandle(
            id,
            sizeBytes,
            GpuMemoryType.Default,
            0,
            DateTime.UtcNow);
    }
    
    /// <summary>
    /// Empty GPU memory handle
    /// </summary>
    public static GpuMemoryHandle Empty { get; } = new GpuMemoryHandle(
        string.Empty,
        0,
        GpuMemoryType.Default,
        -1,
        DateTime.MinValue);
}
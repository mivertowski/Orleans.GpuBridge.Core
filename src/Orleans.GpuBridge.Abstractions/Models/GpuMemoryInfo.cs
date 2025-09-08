using System;
using Orleans;

namespace Orleans.GpuBridge.Abstractions.Models;

/// <summary>
/// Represents GPU memory information and statistics
/// </summary>
[GenerateSerializer]
[Immutable]
public sealed record GpuMemoryInfo
{
    /// <summary>
    /// Gets the total memory available on the GPU in bytes
    /// </summary>
    [Id(0)]
    public long TotalMemoryBytes { get; init; }
    
    /// <summary>
    /// Gets the currently allocated memory on the GPU in bytes
    /// </summary>
    [Id(1)]
    public long AllocatedMemoryBytes { get; init; }
    
    /// <summary>
    /// Gets the free memory available on the GPU in bytes
    /// </summary>
    [Id(2)]
    public long FreeMemoryBytes { get; init; }
    
    /// <summary>
    /// Gets the reserved memory on the GPU in bytes
    /// </summary>
    [Id(3)]
    public long ReservedMemoryBytes { get; init; }
    
    /// <summary>
    /// Gets the memory used by persistent kernels in bytes
    /// </summary>
    [Id(4)]
    public long PersistentKernelMemoryBytes { get; init; }
    
    /// <summary>
    /// Gets the memory used by buffers in bytes
    /// </summary>
    [Id(5)]
    public long BufferMemoryBytes { get; init; }
    
    /// <summary>
    /// Gets the memory used by textures in bytes
    /// </summary>
    [Id(6)]
    public long TextureMemoryBytes { get; init; }
    
    /// <summary>
    /// Gets the memory fragmentation percentage (0-100)
    /// </summary>
    [Id(7)]
    public double FragmentationPercentage { get; init; }
    
    /// <summary>
    /// Gets the memory utilization percentage (0-100)
    /// </summary>
    [Id(8)]
    public double UtilizationPercentage { get; init; }
    
    /// <summary>
    /// Gets the timestamp when this information was captured
    /// </summary>
    [Id(9)]
    public DateTime Timestamp { get; init; }
    
    /// <summary>
    /// Gets the device index this memory info belongs to
    /// </summary>
    [Id(10)]
    public int DeviceIndex { get; init; }
    
    /// <summary>
    /// Gets the device name
    /// </summary>
    [Id(11)]
    public string DeviceName { get; init; } = string.Empty;
    
    /// <summary>
    /// Creates a new instance of GpuMemoryInfo with calculated derived values
    /// </summary>
    public static GpuMemoryInfo Create(
        long totalBytes,
        long allocatedBytes,
        int deviceIndex = 0,
        string deviceName = "")
    {
        var freeBytes = totalBytes - allocatedBytes;
        var utilizationPercentage = totalBytes > 0 
            ? (allocatedBytes * 100.0) / totalBytes 
            : 0;
        
        return new GpuMemoryInfo
        {
            TotalMemoryBytes = totalBytes,
            AllocatedMemoryBytes = allocatedBytes,
            FreeMemoryBytes = freeBytes,
            ReservedMemoryBytes = 0,
            PersistentKernelMemoryBytes = 0,
            BufferMemoryBytes = 0,
            TextureMemoryBytes = 0,
            FragmentationPercentage = 0,
            UtilizationPercentage = utilizationPercentage,
            Timestamp = DateTime.UtcNow,
            DeviceIndex = deviceIndex,
            DeviceName = deviceName
        };
    }
    
    /// <summary>
    /// Creates an empty GpuMemoryInfo instance
    /// </summary>
    public static GpuMemoryInfo Empty => new()
    {
        Timestamp = DateTime.UtcNow
    };
}
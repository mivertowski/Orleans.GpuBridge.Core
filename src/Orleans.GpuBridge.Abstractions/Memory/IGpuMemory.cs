using System;
using System.Threading;
using System.Threading.Tasks;

namespace Orleans.GpuBridge.Abstractions.Memory;

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
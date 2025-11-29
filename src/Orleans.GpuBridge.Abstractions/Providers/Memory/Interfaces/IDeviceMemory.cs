using System;
using System.Threading;
using System.Threading.Tasks;
using Orleans.GpuBridge.Abstractions.Providers;

namespace Orleans.GpuBridge.Abstractions.Providers.Memory.Interfaces;

/// <summary>
/// Represents device memory allocation
/// </summary>
public interface IDeviceMemory : IDisposable
{
    /// <summary>
    /// Size of the allocation in bytes
    /// </summary>
    long SizeBytes { get; }

    /// <summary>
    /// Device pointer to the memory
    /// </summary>
    IntPtr DevicePointer { get; }

    /// <summary>
    /// Associated device
    /// </summary>
    IComputeDevice Device { get; }

    /// <summary>
    /// Copies data from host to device
    /// </summary>
    Task CopyFromHostAsync(
        IntPtr hostPointer,
        long offsetBytes,
        long sizeBytes,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Copies data from device to host
    /// </summary>
    Task CopyToHostAsync(
        IntPtr hostPointer,
        long offsetBytes,
        long sizeBytes,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Copies data between device allocations
    /// </summary>
    Task CopyFromAsync(
        IDeviceMemory source,
        long sourceOffset,
        long destinationOffset,
        long sizeBytes,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Fills the memory with a byte value
    /// </summary>
    Task FillAsync(
        byte value,
        long offsetBytes,
        long sizeBytes,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Creates a view into a portion of this memory
    /// </summary>
    IDeviceMemory CreateView(long offsetBytes, long sizeBytes);
}
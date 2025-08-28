using System;
using System.Threading;
using System.Threading.Tasks;
using Orleans.GpuBridge.Abstractions.Providers;

namespace Orleans.GpuBridge.Abstractions.Providers.Memory.Interfaces;

/// <summary>
/// Pinned host memory for efficient GPU transfers
/// </summary>
public interface IPinnedMemory : IDisposable
{
    /// <summary>
    /// Size in bytes
    /// </summary>
    long SizeBytes { get; }
    
    /// <summary>
    /// Host pointer to the pinned memory
    /// </summary>
    IntPtr HostPointer { get; }
    
    /// <summary>
    /// Gets a span view of the memory
    /// </summary>
    Span<byte> AsSpan();
    
    /// <summary>
    /// Registers this memory for use with a specific device
    /// </summary>
    Task RegisterWithDeviceAsync(
        IComputeDevice device,
        CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Unregisters this memory from a device
    /// </summary>
    Task UnregisterFromDeviceAsync(
        IComputeDevice device,
        CancellationToken cancellationToken = default);
}
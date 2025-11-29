using System;
using System.Threading;
using System.Threading.Tasks;

namespace Orleans.GpuBridge.Abstractions.Providers.Memory.Interfaces;

/// <summary>
/// Typed device memory allocation
/// </summary>
public interface IDeviceMemory<T> : IDeviceMemory where T : unmanaged
{
    /// <summary>
    /// Number of elements
    /// </summary>
    int Length { get; }

    /// <summary>
    /// Gets a span view of the memory (if accessible from host)
    /// </summary>
    Span<T> AsSpan();

    /// <summary>
    /// Copies data from host array to device
    /// </summary>
    Task CopyFromHostAsync(
        T[] source,
        int sourceOffset,
        int destinationOffset,
        int count,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Copies data from device to host array
    /// </summary>
    Task CopyToHostAsync(
        T[] destination,
        int sourceOffset,
        int destinationOffset,
        int count,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Fills the memory with a value
    /// </summary>
    Task FillAsync(
        T value,
        int offset,
        int count,
        CancellationToken cancellationToken = default);
}
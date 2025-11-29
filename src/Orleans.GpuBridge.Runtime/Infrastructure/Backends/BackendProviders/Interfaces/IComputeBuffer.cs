using Orleans.GpuBridge.Abstractions.Memory;

namespace Orleans.GpuBridge.Runtime.BackendProviders.Interfaces;

/// <summary>
/// Compute buffer interface for GPU memory management.
/// </summary>
/// <typeparam name="T">The unmanaged element type stored in the buffer.</typeparam>
public interface IComputeBuffer<T> : IDisposable where T : unmanaged
{
    /// <summary>
    /// Gets the number of elements in the buffer.
    /// </summary>
    int Size { get; }

    /// <summary>
    /// Gets the intended usage pattern for this buffer.
    /// </summary>
    BufferUsage Usage { get; }

    /// <summary>
    /// Writes data from the host to the compute buffer.
    /// </summary>
    /// <param name="data">The source data to write.</param>
    void Write(ReadOnlySpan<T> data);

    /// <summary>
    /// Reads data from the compute buffer to the host.
    /// </summary>
    /// <param name="data">The destination span to receive the data.</param>
    void Read(Span<T> data);

    /// <summary>
    /// Copies the contents of this buffer to another compute buffer.
    /// </summary>
    /// <param name="destination">The target buffer to copy data to.</param>
    void CopyTo(IComputeBuffer<T> destination);
}

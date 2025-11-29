using Orleans.GpuBridge.Abstractions.Memory;

namespace Orleans.GpuBridge.Runtime.BackendProviders.Interfaces;

/// <summary>
/// Compute buffer interface
/// </summary>
public interface IComputeBuffer<T> : IDisposable where T : unmanaged
{
    int Size { get; }
    BufferUsage Usage { get; }

    void Write(ReadOnlySpan<T> data);
    void Read(Span<T> data);
    void CopyTo(IComputeBuffer<T> destination);
}
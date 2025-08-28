namespace Orleans.GpuBridge.DotCompute.Memory;

/// <summary>
/// Unified buffer for efficient memory management
/// </summary>
public interface IUnifiedBuffer<T> : IDisposable where T : unmanaged
{
    int Length { get; }
    Memory<T> Memory { get; }
    bool IsResident { get; }
    
    Task CopyToDeviceAsync(CancellationToken ct = default);
    Task CopyFromDeviceAsync(CancellationToken ct = default);
    Task<IUnifiedBuffer<T>> CloneAsync(CancellationToken ct = default);
}
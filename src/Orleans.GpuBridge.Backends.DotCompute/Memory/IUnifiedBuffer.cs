namespace Orleans.GpuBridge.Backends.DotCompute.Memory;

/// <summary>
/// Unified buffer for efficient memory management
/// </summary>
public interface IUnifiedBuffer<T> : IDisposable where T : unmanaged
{
    /// <summary>
    /// Gets the number of elements in the buffer
    /// </summary>
    int Length { get; }

    /// <summary>
    /// Gets the managed memory view of the buffer
    /// </summary>
    Memory<T> Memory { get; }

    /// <summary>
    /// Gets a value indicating whether the buffer is currently resident in device memory
    /// </summary>
    bool IsResident { get; }

    /// <summary>
    /// Asynchronously copies the buffer data to device memory
    /// </summary>
    /// <param name="ct">Cancellation token for the async operation</param>
    /// <returns>A task representing the asynchronous copy operation</returns>
    Task CopyToDeviceAsync(CancellationToken ct = default);

    /// <summary>
    /// Asynchronously copies the buffer data from device memory to host memory
    /// </summary>
    /// <param name="ct">Cancellation token for the async operation</param>
    /// <returns>A task representing the asynchronous copy operation</returns>
    Task CopyFromDeviceAsync(CancellationToken ct = default);

    /// <summary>
    /// Creates a deep copy of this buffer asynchronously
    /// </summary>
    /// <param name="ct">Cancellation token for the async operation</param>
    /// <returns>A task containing the cloned buffer</returns>
    Task<IUnifiedBuffer<T>> CloneAsync(CancellationToken ct = default);
}
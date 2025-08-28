using Orleans;
using Orleans.GpuBridge.Abstractions;
using Orleans.GpuBridge.Grains.Models;
using Orleans.GpuBridge.Grains.Enums;

namespace Orleans.GpuBridge.Grains.Interfaces;

/// <summary>
/// Interface for a grain that manages resident GPU memory allocations.
/// This grain provides persistent storage for GPU memory buffers and enables
/// efficient kernel execution on pre-allocated memory without repeated transfers.
/// </summary>
public interface IGpuResidentGrain : IGrainWithStringKey
{
    /// <summary>
    /// Allocates memory on the GPU device and keeps it resident for subsequent operations.
    /// The allocated memory persists across grain activations and deactivations.
    /// </summary>
    /// <param name="sizeBytes">The size of memory to allocate in bytes.</param>
    /// <param name="memoryType">The type of memory allocation (default, pinned, shared, etc.).</param>
    /// <returns>
    /// A task that resolves to a <see cref="GpuMemoryHandle"/> representing the allocated memory.
    /// This handle can be used for subsequent read/write operations and kernel executions.
    /// </returns>
    /// <exception cref="InvalidOperationException">Thrown when no GPU devices are available for allocation.</exception>
    /// <exception cref="OutOfMemoryException">Thrown when insufficient GPU memory is available.</exception>
    Task<GpuMemoryHandle> AllocateAsync(
        long sizeBytes,
        GpuMemoryType memoryType = GpuMemoryType.Default);
    
    /// <summary>
    /// Writes data to a previously allocated resident memory buffer.
    /// The data is transferred to GPU memory and becomes available for kernel operations.
    /// </summary>
    /// <typeparam name="T">The unmanaged type of data elements to write.</typeparam>
    /// <param name="handle">The memory handle identifying the target allocation.</param>
    /// <param name="data">The array of data to write to GPU memory.</param>
    /// <param name="offset">The byte offset within the allocation to start writing at.</param>
    /// <returns>A task representing the asynchronous write operation.</returns>
    /// <exception cref="ArgumentException">Thrown when the memory handle is not found.</exception>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when the write would exceed allocated memory bounds.</exception>
    Task WriteAsync<T>(
        GpuMemoryHandle handle,
        T[] data,
        int offset = 0) where T : unmanaged;
    
    /// <summary>
    /// Reads data from a resident memory buffer back to host memory.
    /// This operation transfers data from GPU memory to a host-accessible array.
    /// </summary>
    /// <typeparam name="T">The unmanaged type of data elements to read.</typeparam>
    /// <param name="handle">The memory handle identifying the source allocation.</param>
    /// <param name="count">The number of elements to read.</param>
    /// <param name="offset">The byte offset within the allocation to start reading from.</param>
    /// <returns>A task that resolves to an array containing the read data.</returns>
    /// <exception cref="ArgumentException">Thrown when the memory handle is not found.</exception>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when the read would exceed allocated memory bounds.</exception>
    Task<T[]> ReadAsync<T>(
        GpuMemoryHandle handle,
        int count,
        int offset = 0) where T : unmanaged;
    
    /// <summary>
    /// Executes a GPU kernel using resident memory buffers as input and output.
    /// This enables high-performance computation without memory transfer overhead.
    /// </summary>
    /// <param name="kernelId">The identifier of the kernel to execute.</param>
    /// <param name="input">The memory handle containing input data for the kernel.</param>
    /// <param name="output">The memory handle where kernel results will be written.</param>
    /// <param name="parameters">Optional kernel execution parameters (work group size, constants, etc.).</param>
    /// <returns>A task that resolves to a <see cref="GpuComputeResult"/> containing execution status and timing information.</returns>
    /// <exception cref="ArgumentException">Thrown when input or output handles are not found.</exception>
    Task<GpuComputeResult> ComputeAsync(
        KernelId kernelId,
        GpuMemoryHandle input,
        GpuMemoryHandle output,
        GpuComputeParams? parameters = null);
    
    /// <summary>
    /// Releases a previously allocated memory buffer, freeing GPU memory resources.
    /// After this operation, the memory handle becomes invalid and cannot be used.
    /// </summary>
    /// <param name="handle">The memory handle to release.</param>
    /// <returns>A task representing the asynchronous release operation.</returns>
    Task ReleaseAsync(GpuMemoryHandle handle);
    
    /// <summary>
    /// Retrieves information about all currently allocated memory buffers.
    /// This includes total memory usage, allocation count, and individual allocation details.
    /// </summary>
    /// <returns>
    /// A task that resolves to a <see cref="GpuMemoryInfo"/> containing comprehensive memory usage information.
    /// </returns>
    Task<GpuMemoryInfo> GetMemoryInfoAsync();
    
    /// <summary>
    /// Releases all allocated memory buffers and clears the resident state.
    /// This operation frees all GPU memory associated with this grain instance.
    /// </summary>
    /// <returns>A task representing the asynchronous clear operation.</returns>
    Task ClearAsync();
}
using Orleans.GpuBridge.Abstractions.Enums;
using Orleans.GpuBridge.Abstractions.Memory;

namespace Orleans.GpuBridge.Runtime.BackendProviders.Interfaces;

/// <summary>
/// Compute context interface for GPU operations.
/// </summary>
public interface IComputeContext : IDisposable
{
    /// <summary>
    /// Gets the GPU backend type for this context.
    /// </summary>
    GpuBackend Backend { get; }

    /// <summary>
    /// Gets the device index this context is bound to.
    /// </summary>
    int DeviceIndex { get; }

    /// <summary>
    /// Creates a compute buffer for the specified element type.
    /// </summary>
    /// <typeparam name="T">The unmanaged element type.</typeparam>
    /// <param name="size">The number of elements in the buffer.</param>
    /// <param name="usage">The intended usage pattern for the buffer.</param>
    /// <returns>A new compute buffer instance.</returns>
    IComputeBuffer<T> CreateBuffer<T>(int size, BufferUsage usage) where T : unmanaged;

    /// <summary>
    /// Compiles a kernel from source code.
    /// </summary>
    /// <param name="source">The kernel source code.</param>
    /// <param name="entryPoint">The name of the entry point function.</param>
    /// <returns>A compiled compute kernel.</returns>
    IComputeKernel CompileKernel(string source, string entryPoint);

    /// <summary>
    /// Executes a compute kernel with the specified work size.
    /// </summary>
    /// <param name="kernel">The kernel to execute.</param>
    /// <param name="workSize">The number of work items to process.</param>
    void Execute(IComputeKernel kernel, int workSize);

    /// <summary>
    /// Synchronizes execution, waiting for all pending operations to complete.
    /// </summary>
    void Synchronize();
}

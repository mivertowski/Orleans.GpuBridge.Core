using System;
using Orleans.GpuBridge.Abstractions.Models.Compilation;

namespace Orleans.GpuBridge.Abstractions.Models;

/// <summary>
/// Represents a compiled GPU kernel that can be executed on compute devices.
/// </summary>
/// <remarks>
/// This class encapsulates the compiled binary code, metadata, and native handles
/// required for kernel execution. It implements IDisposable to ensure proper cleanup
/// of native GPU resources.
/// </remarks>
public sealed class CompiledKernel : IDisposable
{
    /// <summary>
    /// Gets the unique identifier for this compiled kernel.
    /// </summary>
    /// <value>
    /// A unique string identifier used to reference the kernel in the system.
    /// This ID is typically generated during compilation and remains constant
    /// throughout the kernel's lifetime.
    /// </value>
    public string KernelId { get; init; } = string.Empty;

    /// <summary>
    /// Gets the human-readable name of the kernel.
    /// </summary>
    /// <value>
    /// A descriptive name for the kernel, often derived from the original
    /// method or function name. Used for debugging and logging purposes.
    /// </value>
    public string Name { get; init; } = string.Empty;

    /// <summary>
    /// Gets the compiled binary code for the kernel.
    /// </summary>
    /// <value>
    /// The platform-specific compiled binary code (e.g., PTX for CUDA,
    /// SPIR-V for OpenCL/Vulkan) that can be loaded and executed by the GPU driver.
    /// </value>
    public byte[] CompiledCode { get; init; } = Array.Empty<byte>();

    /// <summary>
    /// Gets the metadata associated with the compiled kernel.
    /// </summary>
    /// <value>
    /// Detailed metadata including resource requirements, execution constraints,
    /// and optimization information used for efficient kernel scheduling and execution.
    /// </value>
    public KernelMetadata Metadata { get; init; } = new();

    /// <summary>
    /// Gets the native handle to the kernel object in the GPU driver.
    /// </summary>
    /// <value>
    /// A platform-specific native pointer or handle that references the kernel
    /// object in the GPU runtime. This handle is used for direct API calls to
    /// execute the kernel. A value of IntPtr.Zero indicates no native handle.
    /// </value>
    public IntPtr NativeHandle { get; init; }

    /// <summary>
    /// Gets or sets backend-specific data associated with this compiled kernel.
    /// </summary>
    /// <value>
    /// An object containing backend-specific execution context, such as a DotCompute
    /// kernel adapter, CUDA module reference, or other runtime-specific data needed
    /// for kernel execution. The exact type depends on the backend provider.
    /// </value>
    /// <remarks>
    /// This property allows backend providers to attach their specific execution
    /// context to the compiled kernel without requiring changes to the core abstraction.
    /// Backend implementations should cast this to their expected type before use.
    /// </remarks>
    public object? BackendData { get; init; }

    /// <summary>
    /// Gets a value indicating whether this kernel instance has been disposed.
    /// </summary>
    /// <value>
    /// <c>true</c> if the kernel has been disposed and its resources released;
    /// otherwise, <c>false</c>. Once disposed, the kernel cannot be used for execution.
    /// </value>
    public bool IsDisposed { get; private set; }

    /// <summary>
    /// Releases all resources used by the <see cref="CompiledKernel"/>.
    /// </summary>
    /// <remarks>
    /// This method releases native GPU resources associated with the kernel,
    /// including any driver-level handles or memory allocations. After disposal,
    /// the kernel cannot be used for execution and any attempts to do so should
    /// result in an <see cref="ObjectDisposedException"/>.
    /// 
    /// This method is safe to call multiple times; subsequent calls have no effect.
    /// </remarks>
    public void Dispose()
    {
        if (!IsDisposed)
        {
            // Implement proper cleanup of native GPU resources
            // This includes releasing driver handles, freeing GPU memory,
            // and any other platform-specific cleanup operations

            try
            {
                // Clean up compiled code resources
                if (CompiledCode != null)
                {
                    // For managed byte arrays, no explicit cleanup needed
                    // Native resources would be handled by backend-specific disposal
                }

                // Clean up backend-specific data
                if (BackendData is IDisposable disposableBackend)
                {
                    disposableBackend.Dispose();
                }
                else if (BackendData is IAsyncDisposable asyncDisposableBackend)
                {
                    asyncDisposableBackend.DisposeAsync().AsTask().GetAwaiter().GetResult();
                }

                // Clean up metadata resources
                if (Metadata?.ExtendedMetadata != null)
                {
                    foreach (var item in Metadata.ExtendedMetadata)
                    {
                        if (item.Value is IDisposable disposableResource)
                        {
                            disposableResource.Dispose();
                        }
                    }
                }

                // Log disposal for debugging
                System.Diagnostics.Debug.WriteLine($"Disposed CompiledKernel: {Name} (ID: {KernelId})");
            }
            catch (Exception ex)
            {
                // Log disposal errors but don't throw from Dispose
                System.Diagnostics.Debug.WriteLine($"Error disposing CompiledKernel {KernelId}: {ex.Message}");
            }

            IsDisposed = true;
            GC.SuppressFinalize(this);
        }
    }
}
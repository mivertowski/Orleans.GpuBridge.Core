using System;
using System.Threading;
using System.Threading.Tasks;
using Orleans.GpuBridge.Abstractions.Enums;

namespace Orleans.GpuBridge.DotCompute.Interfaces;

/// <summary>
/// Represents a compute accelerator device for kernel execution
/// </summary>
/// <remarks>
/// This interface provides abstraction over different compute backends (CPU, CUDA, etc.)
/// and manages kernel execution, memory allocation, and device state.
/// </remarks>
public interface IAccelerator : IDisposable
{
    /// <summary>
    /// Gets the unique identifier for this accelerator
    /// </summary>
    string AcceleratorId { get; }
    
    /// <summary>
    /// Gets the display name of this accelerator
    /// </summary>
    string Name { get; }
    
    /// <summary>
    /// Gets the type of this accelerator
    /// </summary>
    DeviceType DeviceType { get; }
    
    /// <summary>
    /// Gets the index of this accelerator
    /// </summary>
    int DeviceIndex { get; }
    
    /// <summary>
    /// Gets the total memory available on this accelerator in bytes
    /// </summary>
    long TotalMemoryBytes { get; }
    
    /// <summary>
    /// Gets the available memory on this accelerator in bytes
    /// </summary>
    long AvailableMemoryBytes { get; }
    
    /// <summary>
    /// Gets the number of compute units (e.g., CUDA cores, CPU threads)
    /// </summary>
    int ComputeUnits { get; }
    
    /// <summary>
    /// Gets the maximum number of threads per block/workgroup
    /// </summary>
    int MaxThreadsPerBlock { get; }
    
    /// <summary>
    /// Gets the warp/wavefront size for this accelerator
    /// </summary>
    int WarpSize { get; }
    
    /// <summary>
    /// Gets whether this accelerator is currently available for use
    /// </summary>
    bool IsAvailable { get; }
    
    /// <summary>
    /// Gets whether this accelerator supports unified memory
    /// </summary>
    bool SupportsUnifiedMemory { get; }
    
    /// <summary>
    /// Gets whether this accelerator supports persistent kernels
    /// </summary>
    bool SupportsPersistentKernels { get; }
    
    /// <summary>
    /// Initializes the accelerator for use
    /// </summary>
    /// <param name="cancellationToken">Cancellation token</param>
    /// <returns>Task representing the initialization operation</returns>
    Task InitializeAsync(CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Creates a new stream for kernel execution
    /// </summary>
    /// <returns>A new accelerator stream</returns>
    IAcceleratorStream CreateStream();
    
    /// <summary>
    /// Synchronizes all operations on this accelerator
    /// </summary>
    /// <param name="cancellationToken">Cancellation token</param>
    /// <returns>Task representing the synchronization operation</returns>
    Task SynchronizeAsync(CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Resets the accelerator state
    /// </summary>
    /// <param name="cancellationToken">Cancellation token</param>
    /// <returns>Task representing the reset operation</returns>
    Task ResetAsync(CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Gets performance metrics for this accelerator
    /// </summary>
    /// <returns>Performance metrics</returns>
    IAcceleratorMetrics GetMetrics();
}
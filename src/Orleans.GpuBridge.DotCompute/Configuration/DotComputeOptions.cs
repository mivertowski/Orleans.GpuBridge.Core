using Orleans.GpuBridge.Abstractions.Enums;

namespace Orleans.GpuBridge.DotCompute.Configuration;

/// <summary>
/// Configuration options for the DotCompute backend, controlling which compute backends are enabled
/// and various performance and caching settings.
/// </summary>
/// <remarks>
/// These options control the initialization and behavior of the DotCompute device manager,
/// including which compute backends (CUDA, OpenCL, DirectCompute, Metal) are available for use,
/// kernel compilation caching, and memory pooling settings.
/// </remarks>
public sealed class DotComputeOptions
{
    /// <summary>
    /// Gets or sets whether CUDA backend support is enabled.
    /// </summary>
    /// <value>
    /// <c>true</c> to enable CUDA backend (default); <c>false</c> to disable.
    /// </value>
    /// <remarks>
    /// CUDA support is only available on Windows and Linux platforms with compatible NVIDIA GPUs
    /// and CUDA runtime libraries installed.
    /// </remarks>
    public bool EnableCuda { get; set; } = true;

    /// <summary>
    /// Gets or sets whether OpenCL backend support is enabled.
    /// </summary>
    /// <value>
    /// <c>true</c> to enable OpenCL backend (default); <c>false</c> to disable.
    /// </value>
    /// <remarks>
    /// OpenCL provides cross-platform compute support for various GPU and CPU devices
    /// from different vendors.
    /// </remarks>
    public bool EnableOpenCL { get; set; } = true;

    /// <summary>
    /// Gets or sets whether DirectCompute backend support is enabled.
    /// </summary>
    /// <value>
    /// <c>true</c> to enable DirectCompute backend (default); <c>false</c> to disable.
    /// </value>
    /// <remarks>
    /// DirectCompute is only available on Windows platforms and provides GPU compute
    /// capabilities through DirectX 11 and later.
    /// </remarks>
    public bool EnableDirectCompute { get; set; } = true;

    /// <summary>
    /// Gets or sets whether Metal backend support is enabled.
    /// </summary>
    /// <value>
    /// <c>true</c> to enable Metal backend (default); <c>false</c> to disable.
    /// </value>
    /// <remarks>
    /// Metal is only available on macOS and provides GPU compute capabilities
    /// for Apple's graphics hardware.
    /// </remarks>
    public bool EnableMetal { get; set; } = true;

    /// <summary>
    /// Gets or sets the preferred compute backend for device selection.
    /// </summary>
    /// <value>
    /// The preferred <see cref="GpuBackend"/> to use when multiple backends are available.
    /// Defaults to <see cref="GpuBackend.Auto"/>.
    /// </value>
    /// <remarks>
    /// When set to <see cref="GpuBackend.Auto"/>, the device manager will automatically
    /// select the best available backend based on platform and device capabilities.
    /// </remarks>
    public GpuBackend PreferredBackend { get; set; } = GpuBackend.Auto;

    /// <summary>
    /// Gets or sets whether compiled kernel caching is enabled.
    /// </summary>
    /// <value>
    /// <c>true</c> to enable kernel caching (default); <c>false</c> to disable.
    /// </value>
    /// <remarks>
    /// Kernel caching can significantly improve performance by avoiding recompilation
    /// of frequently used kernels. Cached kernels are stored in memory.
    /// </remarks>
    public bool EnableKernelCaching { get; set; } = true;

    /// <summary>
    /// Gets or sets the maximum number of kernels to cache in memory.
    /// </summary>
    /// <value>
    /// The maximum number of cached kernels. Defaults to 100.
    /// </value>
    /// <remarks>
    /// When the cache is full, the least recently used kernels will be evicted
    /// to make room for new ones.
    /// </remarks>
    public int MaxCachedKernels { get; set; } = 100;

    /// <summary>
    /// Gets or sets whether memory pooling is enabled for buffer allocation.
    /// </summary>
    /// <value>
    /// <c>true</c> to enable memory pooling (default); <c>false</c> to disable.
    /// </value>
    /// <remarks>
    /// Memory pooling can improve performance by reusing previously allocated
    /// GPU memory buffers, reducing allocation overhead.
    /// </remarks>
    public bool EnableMemoryPooling { get; set; } = true;

    /// <summary>
    /// Gets or sets the maximum amount of memory to pool in bytes.
    /// </summary>
    /// <value>
    /// The maximum pooled memory size in bytes. Defaults to 1GB (1,073,741,824 bytes).
    /// </value>
    /// <remarks>
    /// This limit prevents excessive memory usage from pooled buffers. When the limit
    /// is reached, older buffers will be freed to make room for new allocations.
    /// </remarks>
    public long MaxPooledMemoryBytes { get; set; } = 1024L * 1024 * 1024; // 1GB
}
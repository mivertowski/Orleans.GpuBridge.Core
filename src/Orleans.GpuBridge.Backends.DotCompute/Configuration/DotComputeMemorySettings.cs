namespace Orleans.GpuBridge.Backends.DotCompute.Configuration;

/// <summary>
/// Memory management configuration settings for the DotCompute backend.
/// </summary>
/// <remarks>
/// This class provides comprehensive configuration options for GPU memory management
/// in the DotCompute backend, including memory pooling, allocation strategies,
/// defragmentation, and platform-specific optimizations.
/// 
/// <para>
/// Proper memory configuration is critical for GPU performance. Poor memory management
/// can lead to:
/// - Frequent allocations causing performance bottlenecks
/// - Memory fragmentation reducing available memory
/// - Excessive memory transfers between CPU and GPU
/// - Out-of-memory errors in compute-intensive workloads
/// </para>
/// 
/// <para>
/// The default settings are optimized for typical GPU compute workloads with moderate
/// memory usage patterns. Adjust settings based on your specific workload characteristics:
/// - Large dataset processing: Increase pool sizes and alignment
/// - Frequent small allocations: Enable pooling and defragmentation
/// - Memory-constrained systems: Reduce pool sizes and enable aggressive cleanup
/// - High-performance computing: Enable unified memory and pinned memory features
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Configuration for large dataset processing
/// var memorySettings = new DotComputeMemorySettings
/// {
///     EnableMemoryPooling = true,
///     InitialPoolSize = 2L * 1024 * 1024 * 1024, // 2 GB
///     MaxPoolSize = 16L * 1024 * 1024 * 1024,    // 16 GB
///     AllocationAlignment = 4096,                 // 4KB alignment
///     EnableDefragmentation = true,
///     DefragmentationThreshold = 0.15             // Defrag at 15% fragmentation
/// };
/// 
/// // Configuration for memory-constrained environments
/// var constrainedSettings = new DotComputeMemorySettings
/// {
///     EnableMemoryPooling = true,
///     InitialPoolSize = 128 * 1024 * 1024,       // 128 MB
///     MaxPoolSize = 1024 * 1024 * 1024,          // 1 GB
///     EnableDefragmentation = true,
///     DefragmentationThreshold = 0.10,           // Aggressive defrag at 10%
///     PreferUnifiedMemory = false                // Avoid unified memory overhead
/// };
/// </code>
/// </example>
public class DotComputeMemorySettings
{
    /// <summary>
    /// Gets or sets whether to enable memory pooling for GPU allocations.
    /// </summary>
    /// <value>
    /// <c>true</c> to enable memory pooling; otherwise, <c>false</c>. Default is <c>true</c>.
    /// </value>
    /// <remarks>
    /// Memory pooling pre-allocates blocks of GPU memory and reuses them for subsequent
    /// allocations, significantly reducing allocation overhead and memory fragmentation.
    /// This is especially beneficial for workloads with frequent memory allocations.
    /// 
    /// <para>
    /// Benefits of memory pooling:
    /// - Faster allocation and deallocation (10-100x speedup)
    /// - Reduced memory fragmentation
    /// - More predictable memory usage patterns
    /// - Better utilization of large memory blocks
    /// - Reduced GPU driver overhead
    /// </para>
    /// 
    /// <para>
    /// Considerations:
    /// - Initial memory overhead from pre-allocated pools
    /// - May hold memory longer than necessary
    /// - Requires tuning of pool sizes for optimal efficiency
    /// - Most effective for repeated allocation patterns
    /// </para>
    /// 
    /// <para>
    /// Recommended to enable for most production workloads unless memory is severely
    /// constrained or allocation patterns are highly irregular.
    /// </para>
    /// </remarks>
    public bool EnableMemoryPooling { get; set; } = true;

    /// <summary>
    /// Gets or sets the initial memory pool size per GPU device in bytes.
    /// </summary>
    /// <value>
    /// The initial pool size in bytes. Default is 512 MB (536,870,912 bytes).
    /// Must be a positive value.
    /// </value>
    /// <remarks>
    /// This setting determines the amount of GPU memory pre-allocated for the memory
    /// pool when the backend is initialized. The pool will grow up to <see cref="MaxPoolSize"/>
    /// as needed, but starts with this initial allocation.
    /// 
    /// <para>
    /// Sizing considerations:
    /// - Should accommodate typical working set size
    /// - Larger initial size reduces early allocations and fragmentation
    /// - Too large may waste memory or cause initialization failures
    /// - Should be significantly smaller than total GPU memory
    /// - Consider multiple GPU scenarios (allocation per device)
    /// </para>
    /// 
    /// <para>
    /// Recommended sizing guidelines:
    /// - Development: 128-256 MB
    /// - Small workloads: 256-512 MB
    /// - Medium workloads: 512 MB - 2 GB
    /// - Large workloads: 2-8 GB (depending on GPU memory capacity)
    /// - Memory-constrained: 64-128 MB
    /// </para>
    /// 
    /// <para>
    /// This setting is only effective when <see cref="EnableMemoryPooling"/> is <c>true</c>.
    /// </para>
    /// </remarks>
    /// <example>
    /// <code>
    /// // For 8GB GPU, allocate 1GB initially
    /// memorySettings.InitialPoolSize = 1024 * 1024 * 1024; // 1 GB
    /// 
    /// // For memory-constrained 4GB GPU
    /// memorySettings.InitialPoolSize = 256 * 1024 * 1024;  // 256 MB
    /// </code>
    /// </example>
    /// <exception cref="ArgumentOutOfRangeException">
    /// Thrown when the value is less than or equal to zero, or exceeds <see cref="MaxPoolSize"/>.
    /// </exception>
    public long InitialPoolSize { get; set; } = 512 * 1024 * 1024; // 512 MB

    /// <summary>
    /// Gets or sets the maximum memory pool size per GPU device in bytes.
    /// </summary>
    /// <value>
    /// The maximum pool size in bytes. Default is 4 GB (4,294,967,296 bytes).
    /// Must be greater than or equal to <see cref="InitialPoolSize"/>.
    /// </value>
    /// <remarks>
    /// This setting defines the upper limit for memory pool growth. When the pool
    /// reaches this size, new allocations will either reuse existing pool memory
    /// or fall back to direct GPU memory allocation if pooling is insufficient.
    /// 
    /// <para>
    /// The maximum pool size serves as a safety mechanism to:
    /// - Prevent unbounded memory growth
    /// - Reserve GPU memory for other applications
    /// - Avoid out-of-memory conditions
    /// - Maintain system stability under load
    /// </para>
    /// 
    /// <para>
    /// Sizing guidelines:
    /// - Should be 60-80% of total GPU memory for dedicated workloads
    /// - Should be 40-60% of total GPU memory for shared environments
    /// - Consider OS and driver memory overhead (typically 200-500 MB)
    /// - Account for concurrent applications using GPU memory
    /// - Leave buffer for temporary allocations and driver operations
    /// </para>
    /// 
    /// <para>
    /// Common configurations by GPU memory:
    /// - 4 GB GPU: 2-3 GB max pool size
    /// - 8 GB GPU: 5-7 GB max pool size  
    /// - 16 GB GPU: 10-14 GB max pool size
    /// - 32+ GB GPU: 20-28 GB max pool size
    /// </para>
    /// 
    /// <para>
    /// This setting is only effective when <see cref="EnableMemoryPooling"/> is <c>true</c>.
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentOutOfRangeException">
    /// Thrown when the value is less than <see cref="InitialPoolSize"/>.
    /// </exception>
    public long MaxPoolSize { get; set; } = 4L * 1024 * 1024 * 1024; // 4 GB

    /// <summary>
    /// Gets or sets the memory allocation alignment in bytes.
    /// </summary>
    /// <value>
    /// The alignment requirement in bytes. Default is 256 bytes.
    /// Must be a power of 2 and at least 1.
    /// </value>
    /// <remarks>
    /// Memory alignment ensures that allocated memory addresses meet hardware requirements
    /// for optimal performance. Different GPU architectures and data types have varying
    /// alignment requirements for maximum memory bandwidth utilization.
    /// 
    /// <para>
    /// Alignment benefits:
    /// - Optimal memory access patterns and bandwidth
    /// - Efficient vectorized operations and coalesced access
    /// - Compatibility with hardware-specific optimizations
    /// - Reduced memory access penalties and cache misses
    /// </para>
    /// 
    /// <para>
    /// Platform-specific recommendations:
    /// - CUDA: 256-512 bytes (matches warp/coalescing requirements)
    /// - OpenCL: 128-256 bytes (platform dependent)
    /// - DirectCompute: 256 bytes (typical D3D11/12 alignment)
    /// - Metal: 256 bytes (typical Metal buffer alignment)
    /// - Vulkan: 256 bytes (typical Vulkan buffer alignment)
    /// </para>
    /// 
    /// <para>
    /// Data type considerations:
    /// - Single precision (float): 128-256 byte alignment
    /// - Double precision (double): 256-512 byte alignment
    /// - Mixed data types: Use largest required alignment
    /// - Structured data: Align to largest member + padding
    /// </para>
    /// 
    /// <para>
    /// Higher alignment values may waste some memory due to padding but can significantly
    /// improve performance for memory-intensive kernels.
    /// </para>
    /// </remarks>
    /// <example>
    /// <code>
    /// // High-performance configuration for float4 data
    /// memorySettings.AllocationAlignment = 512;  // 512-byte alignment
    /// 
    /// // Memory-efficient configuration
    /// memorySettings.AllocationAlignment = 128;  // 128-byte alignment
    /// 
    /// // Conservative configuration for mixed workloads
    /// memorySettings.AllocationAlignment = 256;  // 256-byte alignment (default)
    /// </code>
    /// </example>
    /// <exception cref="ArgumentOutOfRangeException">
    /// Thrown when the value is not a power of 2 or is less than 1.
    /// </exception>
    public int AllocationAlignment { get; set; } = 256;

    /// <summary>
    /// Gets or sets whether to enable automatic memory defragmentation.
    /// </summary>
    /// <value>
    /// <c>true</c> to enable automatic defragmentation; otherwise, <c>false</c>. Default is <c>true</c>.
    /// </value>
    /// <remarks>
    /// Automatic defragmentation periodically reorganizes memory pool allocations to
    /// reduce fragmentation and maximize available contiguous memory blocks. This helps
    /// maintain allocation performance over time, especially for long-running applications.
    /// 
    /// <para>
    /// Defragmentation benefits:
    /// - Maintains allocation performance over time
    /// - Reduces memory waste from fragmentation
    /// - Enables larger contiguous allocations
    /// - Improves memory utilization efficiency
    /// - Prevents gradual performance degradation
    /// </para>
    /// 
    /// <para>
    /// Defragmentation costs:
    /// - Temporary performance impact during defrag operations
    /// - Memory copying overhead for active allocations
    /// - Potential kernel execution delays during defrag
    /// - CPU processing time for fragmentation analysis
    /// </para>
    /// 
    /// <para>
    /// Defragmentation is triggered when fragmentation exceeds the threshold specified
    /// by <see cref="DefragmentationThreshold"/>. The operation is performed during
    /// idle periods when possible to minimize performance impact.
    /// </para>
    /// 
    /// <para>
    /// Recommended for most workloads, especially:
    /// - Long-running applications
    /// - Workloads with variable allocation sizes
    /// - Applications with frequent allocation/deallocation cycles
    /// - Memory-constrained environments
    /// </para>
    /// </remarks>
    public bool EnableDefragmentation { get; set; } = true;

    /// <summary>
    /// Gets or sets the fragmentation threshold that triggers automatic defragmentation.
    /// </summary>
    /// <value>
    /// The fragmentation threshold as a percentage (0.0 to 1.0). Default is 0.25 (25%).
    /// </value>
    /// <remarks>
    /// This threshold determines when automatic defragmentation is triggered. The value
    /// represents the percentage of memory pool space that is fragmented (unusable for
    /// contiguous allocations due to fragmentation).
    /// 
    /// <para>
    /// The fragmentation metric considers:
    /// - Free memory blocks too small for typical allocations
    /// - Scattered free regions preventing large allocations
    /// - Overall efficiency of memory space utilization
    /// - Impact on allocation success rates
    /// </para>
    /// 
    /// <para>
    /// Threshold selection guidelines:
    /// - Lower values (0.10-0.20): More frequent defrag, better memory efficiency
    /// - Medium values (0.20-0.30): Balanced performance and efficiency (recommended)
    /// - Higher values (0.30-0.50): Less frequent defrag, may impact large allocations
    /// - Very high values (&gt;0.50): Minimal defrag, risk of allocation failures
    /// </para>
    /// 
    /// <para>
    /// Workload-specific recommendations:
    /// - Uniform allocation sizes: 0.30-0.40 (fragmentation less problematic)
    /// - Mixed allocation sizes: 0.20-0.30 (fragmentation more problematic)
    /// - Memory-constrained: 0.10-0.20 (aggressive defragmentation)
    /// - Performance-critical: 0.25-0.35 (balance efficiency and performance)
    /// </para>
    /// 
    /// <para>
    /// This setting is only effective when <see cref="EnableDefragmentation"/> is <c>true</c>.
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentOutOfRangeException">
    /// Thrown when the value is less than 0.0 or greater than 1.0.
    /// </exception>
    public double DefragmentationThreshold { get; set; } = 0.25; // 25%

    /// <summary>
    /// Gets or sets whether to prefer unified memory when available on the platform.
    /// </summary>
    /// <value>
    /// <c>true</c> to prefer unified memory; otherwise, <c>false</c>. Default is <c>true</c>.
    /// </value>
    /// <remarks>
    /// Unified memory (also known as managed memory) provides a single address space
    /// shared between CPU and GPU, allowing automatic data migration and simplified
    /// memory management. This feature is available on select platforms and hardware.
    /// 
    /// <para>
    /// Unified memory benefits:
    /// - Simplified programming model (no explicit transfers)
    /// - Automatic data migration between CPU and GPU
    /// - Reduced memory duplication for shared data
    /// - Better memory utilization on memory-constrained systems
    /// - Easier debugging and memory profiling
    /// </para>
    /// 
    /// <para>
    /// Unified memory limitations:
    /// - Platform availability (CUDA 6.0+, some OpenCL implementations)
    /// - Performance overhead from automatic migration
    /// - Potential page faults during GPU execution
    /// - Limited control over data placement and timing
    /// - May not be optimal for all memory access patterns
    /// </para>
    /// 
    /// <para>
    /// Platform support:
    /// - CUDA: Unified Memory available on Pascal+ architectures (GTX 10xx+)
    /// - OpenCL: Limited support in OpenCL 2.0+ (implementation dependent)
    /// - DirectCompute: Not typically supported
    /// - Metal: Shared memory on Apple Silicon (M1/M2 series)
    /// - Vulkan: Limited through extensions (implementation dependent)
    /// </para>
    /// 
    /// <para>
    /// When unified memory is not available or not beneficial for the workload,
    /// the backend will fall back to discrete memory management with explicit transfers.
    /// </para>
    /// </remarks>
    public bool PreferUnifiedMemory { get; set; } = true;

    /// <summary>
    /// Gets or sets whether to use pinned (page-locked) memory for CPU-GPU transfers.
    /// </summary>
    /// <value>
    /// <c>true</c> to use pinned memory; otherwise, <c>false</c>. Default is <c>true</c>.
    /// </value>
    /// <remarks>
    /// Pinned memory (also called page-locked or non-pageable memory) is CPU memory
    /// that is locked in physical RAM and cannot be swapped to disk. This enables
    /// faster and more efficient transfers between CPU and GPU memory.
    /// 
    /// <para>
    /// Pinned memory benefits:
    /// - Significantly faster CPU-GPU transfer speeds (2-3x improvement)
    /// - Enables asynchronous memory transfers
    /// - Reduced CPU usage during memory operations
    /// - More predictable transfer performance
    /// - Better overlap of computation and communication
    /// </para>
    /// 
    /// <para>
    /// Pinned memory costs:
    /// - Consumes system RAM that cannot be swapped
    /// - Limited resource (typically 50-75% of system RAM)
    /// - Slower allocation/deallocation compared to pageable memory
    /// - May impact system performance if overused
    /// - Not suitable for very large or temporary buffers
    /// </para>
    /// 
    /// <para>
    /// Usage recommendations:
    /// - Enable for frequently transferred data
    /// - Use for streaming or pipeline workloads
    /// - Ideal for intermediate-sized buffers (MB to GB range)
    /// - Avoid for very large datasets that exceed system RAM
    /// - Consider system RAM capacity and other applications
    /// </para>
    /// 
    /// <para>
    /// Platform considerations:
    /// - CUDA: cudaMallocHost() for pinned allocations
    /// - OpenCL: CL_MEM_ALLOC_HOST_PTR flag
    /// - DirectCompute: D3D11_USAGE_STAGING with appropriate flags
    /// - Metal: Shared memory pools on macOS/iOS
    /// - Vulkan: Host-visible memory with coherent flag
    /// </para>
    /// </remarks>
    public bool UsePinnedMemory { get; set; } = true;
}
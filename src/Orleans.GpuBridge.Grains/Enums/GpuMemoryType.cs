using Orleans;

namespace Orleans.GpuBridge.Grains.Enums;

/// <summary>
/// Specifies the type of GPU memory allocation to request.
/// Different memory types have different performance characteristics and access patterns.
/// </summary>
[GenerateSerializer]
public enum GpuMemoryType
{
    /// <summary>
    /// Default device memory allocation.
    /// This provides standard GPU memory with optimal access from GPU kernels
    /// but requires explicit transfers between host and device.
    /// </summary>
    Default,

    /// <summary>
    /// Pinned (page-locked) host memory allocation.
    /// This memory remains in physical RAM and provides faster transfer rates
    /// to/from GPU memory compared to pageable memory.
    /// </summary>
    Pinned,

    /// <summary>
    /// Shared memory allocation accessible by both CPU and GPU.
    /// This enables unified memory access patterns but may have
    /// performance implications due to coherency requirements.
    /// </summary>
    Shared,

    /// <summary>
    /// Texture memory allocation optimized for spatial locality.
    /// This memory type provides hardware-accelerated filtering and
    /// caching optimized for 2D/3D spatial access patterns.
    /// </summary>
    Texture,

    /// <summary>
    /// Constant memory allocation for read-only data.
    /// This memory is cached and optimized for broadcast access patterns
    /// where all threads read the same data simultaneously.
    /// </summary>
    Constant
}
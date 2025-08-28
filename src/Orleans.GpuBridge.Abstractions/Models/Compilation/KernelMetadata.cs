using System.Collections.Generic;

namespace Orleans.GpuBridge.Abstractions.Models.Compilation;

/// <summary>
/// Contains metadata information about a compiled GPU kernel's resource requirements and capabilities.
/// </summary>
/// <param name="RequiredSharedMemory">
/// The amount of shared memory in bytes that the kernel requires per thread block.
/// Shared memory is fast, on-chip memory shared among all threads in a block.
/// A value of 0 indicates the kernel does not use shared memory. This value is
/// used for occupancy calculations and resource allocation. Default is 0.
/// </param>
/// <param name="RequiredRegisters">
/// The number of registers required per thread for kernel execution.
/// Registers are the fastest form of memory on the GPU, and their usage
/// directly affects occupancy. A value of 0 indicates the compiler determined
/// register usage automatically. Used for performance analysis and scheduling
/// decisions. Default is 0.
/// </param>
/// <param name="MaxThreadsPerBlock">
/// The maximum number of threads that can be launched in a single thread block
/// for this kernel. This limit is determined by the kernel's resource usage
/// and GPU architecture constraints. Must be a power of 2 and not exceed
/// the device's maximum block size. Default is 256.
/// </param>
/// <param name="PreferredBlockSize">
/// The preferred or optimal number of threads per block for this kernel.
/// This value represents the block size that typically yields the best
/// performance for the kernel. A value of 0 indicates no specific preference,
/// allowing the runtime to choose automatically. Default is 0.
/// </param>
/// <param name="UsesAtomics">
/// Indicates whether the kernel uses atomic operations.
/// Atomic operations provide thread-safe access to memory locations but may
/// impact performance due to serialization. This information is used for
/// scheduling and optimization decisions. Default is <c>false</c>.
/// </param>
/// <param name="UsesSharedMemory">
/// Indicates whether the kernel uses shared memory.
/// Shared memory enables fast data sharing between threads in the same block.
/// This flag helps with resource planning and occupancy optimization.
/// Should be consistent with <paramref name="RequiredSharedMemory"/> being greater than 0.
/// Default is <c>false</c>.
/// </param>
/// <param name="UsesDynamicParallelism">
/// Indicates whether the kernel launches child kernels (dynamic parallelism).
/// Dynamic parallelism allows kernels to launch other kernels without CPU
/// involvement but requires additional GPU resources and may limit occupancy.
/// This capability is not available on all GPU architectures. Default is <c>false</c>.
/// </param>
/// <param name="ExtendedMetadata">
/// Additional metadata specific to particular GPU backends or use cases.
/// This dictionary allows storage of backend-specific information that doesn't
/// fit into the standard metadata fields. Keys should be descriptive names,
/// values can be any serializable object. Default is <c>null</c>.
/// </param>
/// <remarks>
/// This metadata is primarily gathered during kernel compilation and analysis.
/// It provides crucial information for:
/// - Occupancy calculation and optimization
/// - Resource allocation and scheduling
/// - Performance analysis and tuning
/// - Compatibility checking with target devices
/// 
/// <para>
/// The metadata values should accurately reflect the kernel's actual resource
/// usage to enable proper scheduling and optimization. Inaccurate metadata
/// can lead to suboptimal performance or runtime errors.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Metadata for a simple kernel
/// var basicMetadata = new KernelMetadata(
///     MaxThreadsPerBlock: 1024,
///     PreferredBlockSize: 256);
/// 
/// // Metadata for a shared memory kernel
/// var sharedMemoryMetadata = new KernelMetadata(
///     RequiredSharedMemory: 4096,
///     RequiredRegisters: 32,
///     MaxThreadsPerBlock: 512,
///     PreferredBlockSize: 256,
///     UsesSharedMemory: true);
/// 
/// // Metadata for a complex kernel with atomics
/// var complexMetadata = new KernelMetadata(
///     RequiredSharedMemory: 2048,
///     RequiredRegisters: 48,
///     MaxThreadsPerBlock: 256,
///     PreferredBlockSize: 128,
///     UsesAtomics: true,
///     UsesSharedMemory: true,
///     ExtendedMetadata: new Dictionary&lt;string, object&gt;
///     {
///         ["WarpSize"] = 32,
///         ["MinComputeCapability"] = "6.0"
///     });
/// </code>
/// </example>
public sealed record KernelMetadata(
    int RequiredSharedMemory = 0,
    int RequiredRegisters = 0,
    int MaxThreadsPerBlock = 256,
    int PreferredBlockSize = 0,
    bool UsesAtomics = false,
    bool UsesSharedMemory = false,
    bool UsesDynamicParallelism = false,
    IReadOnlyDictionary<string, object>? ExtendedMetadata = null);
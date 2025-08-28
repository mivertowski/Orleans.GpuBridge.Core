using System.Collections.Generic;

namespace Orleans.GpuBridge.Abstractions.Models;

/// <summary>
/// Defines the parameters required for launching a kernel on a compute device.
/// These parameters specify the execution configuration, work distribution,
/// memory requirements, and kernel arguments needed for proper kernel execution.
/// </summary>
/// <param name="GlobalWorkSize">
/// The global work size for kernel execution, specified as an array of dimensions.
/// This defines the total amount of work to be performed across all compute units.
/// For example, [1024, 512] might specify a 2D grid with 1024 items in the X dimension
/// and 512 items in the Y dimension. The number of dimensions must match the kernel's
/// expected dimensionality (typically 1D, 2D, or 3D).
/// </param>
/// <param name="LocalWorkSize">
/// The local work size (workgroup size) for kernel execution, specified as an array of dimensions.
/// This defines how the global work is divided into local workgroups that execute together
/// on the same compute unit. If null, the backend will automatically select an appropriate
/// local work size based on device characteristics and kernel requirements.
/// When specified, the dimensions must match those of GlobalWorkSize.
/// </param>
/// <param name="DynamicSharedMemoryBytes">
/// The amount of dynamic shared memory to allocate per workgroup, in bytes.
/// Shared memory is fast on-chip memory that can be accessed by all threads
/// within a workgroup. This is in addition to any statically declared shared memory
/// in the kernel code. Default is 0, meaning no additional shared memory is allocated.
/// </param>
/// <param name="Arguments">
/// Kernel arguments as key-value pairs where keys are parameter names and values are argument data.
/// This dictionary contains all the parameters that need to be passed to the kernel function.
/// The keys should match the parameter names in the kernel signature, and values must be
/// compatible with the expected parameter types. If null, the kernel is assumed to have no parameters.
/// </param>
public sealed record KernelLaunchParameters(
    int[] GlobalWorkSize,
    int[]? LocalWorkSize = null,
    int DynamicSharedMemoryBytes = 0,
    IReadOnlyDictionary<string, object>? Arguments = null);
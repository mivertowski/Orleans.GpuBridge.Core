using Orleans;

namespace Orleans.GpuBridge.Grains.Models;

/// <summary>
/// Parameters for configuring GPU kernel execution within resident grains.
/// These parameters control work group sizing, thread organization, and kernel constants.
/// </summary>
/// <param name="WorkGroupSize">
/// The number of threads per work group (also known as block size in CUDA terminology).
/// This affects resource utilization and should be a multiple of the warp/wavefront size.
/// Default is 256 threads per work group.
/// </param>
/// <param name="WorkGroups">
/// The total number of work groups to launch (also known as grid size in CUDA terminology).
/// When set to 0, the system will automatically calculate the optimal number based on data size.
/// Default is 0 (automatic calculation).
/// </param>
/// <param name="Constants">
/// A dictionary of named constants to pass to the kernel.
/// These values are accessible within the kernel code and can include scalars, arrays, or other serializable objects.
/// Default is null (no constants).
/// </param>
[GenerateSerializer]
public sealed record GpuComputeParams(
    [property: Id(0)] int WorkGroupSize = 256,
    [property: Id(1)] int WorkGroups = 0,
    [property: Id(2)] Dictionary<string, object>? Constants = null);
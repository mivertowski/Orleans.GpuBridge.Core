using Orleans;

namespace Orleans.GpuBridge.Grains.Models;

/// <summary>
/// Represents the result of a GPU kernel execution within a resident grain.
/// This record contains execution status, timing information, and optional error details.
/// </summary>
/// <param name="Success">
/// Indicates whether the kernel execution completed successfully.
/// <c>true</c> if the kernel executed without errors; <c>false</c> if an error occurred.
/// </param>
/// <param name="ExecutionTime">
/// The total time taken to execute the kernel, including launch overhead.
/// This measurement covers the entire execution from kernel launch to completion.
/// </param>
/// <param name="Error">
/// Optional error message providing details when <paramref name="Success"/> is <c>false</c>.
/// Contains information about what went wrong during kernel execution.
/// Default is <c>null</c> (no error).
/// </param>
[GenerateSerializer]
public sealed record GpuComputeResult(
    [property: Id(0)] bool Success,
    [property: Id(1)] TimeSpan ExecutionTime,
    [property: Id(2)] string? Error = null);
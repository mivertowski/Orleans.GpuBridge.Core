using Orleans.GpuBridge.Abstractions.Enums;

namespace Orleans.GpuBridge.Abstractions.Kernels;

/// <summary>
/// Handle for a submitted kernel execution
/// </summary>
public sealed record KernelHandle(
    string Id,
    DateTimeOffset SubmittedAt,
    KernelStatus Status = KernelStatus.Queued)
{
    /// <summary>
    /// Creates a new kernel handle with a unique ID and current timestamp.
    /// </summary>
    /// <returns>A new <see cref="KernelHandle"/> with status set to Queued.</returns>
    public static KernelHandle Create() => new(
        Guid.NewGuid().ToString("N"),
        DateTimeOffset.UtcNow);
}
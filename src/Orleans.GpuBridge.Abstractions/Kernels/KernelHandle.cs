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
    public static KernelHandle Create() => new(
        Guid.NewGuid().ToString("N"),
        DateTimeOffset.UtcNow);
}
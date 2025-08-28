namespace Orleans.GpuBridge.Abstractions.Enums;

/// <summary>
/// Status of kernel execution
/// </summary>
public enum KernelStatus
{
    Queued,
    Running,
    Completed,
    Failed,
    Cancelled
}
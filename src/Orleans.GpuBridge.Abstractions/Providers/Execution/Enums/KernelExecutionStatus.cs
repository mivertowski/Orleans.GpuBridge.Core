namespace Orleans.GpuBridge.Abstractions.Providers.Execution.Enums;

/// <summary>
/// Kernel execution status
/// </summary>
public enum KernelExecutionStatus
{
    Queued,
    Running,
    Completed,
    Failed,
    Cancelled,
    Timeout
}
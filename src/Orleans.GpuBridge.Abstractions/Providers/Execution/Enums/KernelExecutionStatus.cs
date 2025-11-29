namespace Orleans.GpuBridge.Abstractions.Providers.Execution.Enums;

/// <summary>
/// Kernel execution status
/// </summary>
public enum KernelExecutionStatus
{
    /// <summary>
    /// Kernel has been queued for execution but not yet started.
    /// </summary>
    Queued,

    /// <summary>
    /// Kernel is currently running on the GPU.
    /// </summary>
    Running,

    /// <summary>
    /// Kernel execution completed successfully.
    /// </summary>
    Completed,

    /// <summary>
    /// Kernel execution failed with an error.
    /// </summary>
    Failed,

    /// <summary>
    /// Kernel execution was cancelled before completion.
    /// </summary>
    Cancelled,

    /// <summary>
    /// Kernel execution exceeded the timeout limit.
    /// </summary>
    Timeout
}
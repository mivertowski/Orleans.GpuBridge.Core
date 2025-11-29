namespace Orleans.GpuBridge.Abstractions.Enums;

/// <summary>
/// Status of kernel execution
/// </summary>
public enum KernelStatus
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
    Cancelled
}
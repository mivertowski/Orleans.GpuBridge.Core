namespace Orleans.GpuBridge.HealthChecks.Exceptions;

/// <summary>
/// Exception thrown when GPU kernel execution fails.
/// </summary>
public class GpuKernelException : GpuOperationException
{
    /// <summary>
    /// Gets the name of the kernel that caused the exception.
    /// </summary>
    public string KernelName { get; }

    /// <summary>
    /// Initializes a new instance of the <see cref="GpuKernelException"/> class.
    /// </summary>
    /// <param name="kernelName">The name of the kernel that failed.</param>
    /// <param name="message">The error message.</param>
    public GpuKernelException(string kernelName, string message)
        : base($"Kernel '{kernelName}' failed: {message}")
    {
        KernelName = kernelName;
    }
}

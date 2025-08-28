namespace Orleans.GpuBridge.HealthChecks.Exceptions;

/// <summary>
/// Exception thrown when GPU kernel execution fails.
/// </summary>
public class GpuKernelException : GpuOperationException
{
    public string KernelName { get; }
    
    public GpuKernelException(string kernelName, string message) 
        : base($"Kernel '{kernelName}' failed: {message}")
    {
        KernelName = kernelName;
    }
}
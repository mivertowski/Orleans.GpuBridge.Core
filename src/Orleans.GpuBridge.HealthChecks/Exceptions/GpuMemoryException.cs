namespace Orleans.GpuBridge.HealthChecks.Exceptions;

/// <summary>
/// Exception thrown when GPU memory operations fail.
/// </summary>
public class GpuMemoryException : GpuOperationException
{
    public bool IsTransient { get; }
    
    public GpuMemoryException(string message, bool isTransient = true) 
        : base(message)
    {
        IsTransient = isTransient;
    }
}
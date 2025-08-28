namespace Orleans.GpuBridge.HealthChecks.Exceptions;

/// <summary>
/// Base exception for GPU operations.
/// </summary>
public class GpuOperationException : Exception
{
    public GpuOperationException(string message) : base(message) { }
    public GpuOperationException(string message, Exception innerException) 
        : base(message, innerException) { }
}
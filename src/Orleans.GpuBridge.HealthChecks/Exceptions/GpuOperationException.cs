namespace Orleans.GpuBridge.HealthChecks.Exceptions;

/// <summary>
/// Base exception for GPU operations.
/// </summary>
public class GpuOperationException : Exception
{
    /// <summary>
    /// Initializes a new instance of the <see cref="GpuOperationException"/> class.
    /// </summary>
    /// <param name="message">The error message.</param>
    public GpuOperationException(string message) : base(message) { }

    /// <summary>
    /// Initializes a new instance of the <see cref="GpuOperationException"/> class.
    /// </summary>
    /// <param name="message">The error message.</param>
    /// <param name="innerException">The inner exception.</param>
    public GpuOperationException(string message, Exception innerException)
        : base(message, innerException) { }
}

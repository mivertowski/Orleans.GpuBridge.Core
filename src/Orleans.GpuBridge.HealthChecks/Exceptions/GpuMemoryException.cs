namespace Orleans.GpuBridge.HealthChecks.Exceptions;

/// <summary>
/// Exception thrown when GPU memory operations fail.
/// </summary>
public class GpuMemoryException : GpuOperationException
{
    /// <summary>
    /// Gets a value indicating whether the error is transient and may succeed on retry.
    /// </summary>
    public bool IsTransient { get; }

    /// <summary>
    /// Initializes a new instance of the <see cref="GpuMemoryException"/> class.
    /// </summary>
    /// <param name="message">The error message.</param>
    /// <param name="isTransient">Whether the error is transient.</param>
    public GpuMemoryException(string message, bool isTransient = true)
        : base(message)
    {
        IsTransient = isTransient;
    }
}

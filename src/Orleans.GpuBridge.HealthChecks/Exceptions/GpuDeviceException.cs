namespace Orleans.GpuBridge.HealthChecks.Exceptions;

/// <summary>
/// Exception thrown when GPU device operations fail.
/// </summary>
public class GpuDeviceException : GpuOperationException
{
    /// <summary>
    /// Gets the index of the GPU device that caused the exception.
    /// </summary>
    public int DeviceIndex { get; }

    /// <summary>
    /// Initializes a new instance of the <see cref="GpuDeviceException"/> class.
    /// </summary>
    /// <param name="deviceIndex">The index of the GPU device.</param>
    /// <param name="message">The error message.</param>
    public GpuDeviceException(int deviceIndex, string message)
        : base($"GPU device {deviceIndex} error: {message}")
    {
        DeviceIndex = deviceIndex;
    }
}

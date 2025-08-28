namespace Orleans.GpuBridge.HealthChecks.Exceptions;

/// <summary>
/// Exception thrown when GPU device operations fail.
/// </summary>
public class GpuDeviceException : GpuOperationException
{
    public int DeviceIndex { get; }
    
    public GpuDeviceException(int deviceIndex, string message) 
        : base($"GPU device {deviceIndex} error: {message}")
    {
        DeviceIndex = deviceIndex;
    }
}
namespace Orleans.GpuBridge.Abstractions.Enums;

/// <summary>
/// Represents the current operational status of a compute device.
/// This enumeration provides insight into device availability and health,
/// enabling applications to make informed decisions about device selection
/// and workload distribution.
/// </summary>
public enum DeviceStatus
{
    /// <summary>
    /// The device is available and ready to accept new work.
    /// This is the normal operational state where the device can be used
    /// for kernel execution and memory operations.
    /// </summary>
    Available,

    /// <summary>
    /// The device is currently busy executing work.
    /// While busy, the device may still accept additional work that will be queued,
    /// but performance may be impacted due to resource contention.
    /// </summary>
    Busy,

    /// <summary>
    /// The device is offline or not responding.
    /// This may indicate hardware issues, driver problems, or that the device
    /// has been disabled. The device should not be used until it returns to
    /// an Available status.
    /// </summary>
    Offline,

    /// <summary>
    /// The device has encountered an error condition.
    /// This indicates that the device experienced a failure during operation
    /// and may require reset or recovery procedures before it can be used again.
    /// </summary>
    Error,

    /// <summary>
    /// The device is currently being reset.
    /// During reset operations, the device is temporarily unavailable and
    /// all existing contexts and queues may be invalidated. Applications
    /// should wait for the device to return to Available status.
    /// </summary>
    Resetting,

    /// <summary>
    /// The device status is unknown or could not be determined.
    /// This may occur during system initialization or when communication
    /// with the device is temporarily unavailable. Applications should
    /// use caution when working with devices in this state.
    /// </summary>
    Unknown
}
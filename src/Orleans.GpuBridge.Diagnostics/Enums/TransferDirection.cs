namespace Orleans.GpuBridge.Diagnostics.Enums;

/// <summary>
/// Specifies the direction of GPU memory transfer operations for telemetry tracking.
/// This enum helps categorize and monitor different types of data movement patterns
/// to analyze bandwidth utilization and identify performance optimization opportunities.
/// </summary>
public enum TransferDirection
{
    /// <summary>
    /// Data transfer from host (CPU) memory to device (GPU) memory.
    /// This represents the typical pattern of uploading input data for GPU processing.
    /// Common scenarios include uploading textures, vertex data, or computation input arrays.
    /// </summary>
    HostToDevice,
    
    /// <summary>
    /// Data transfer from device (GPU) memory to host (CPU) memory.
    /// This represents downloading processed results from GPU back to CPU memory.
    /// Common scenarios include reading computation results, rendered images, or processed data arrays.
    /// </summary>
    DeviceToHost,
    
    /// <summary>
    /// Data transfer between different memory regions on the same GPU device.
    /// This includes transfers between different memory types (e.g., global to shared memory)
    /// or memory-to-memory copy operations within the same device.
    /// </summary>
    DeviceToDevice,
    
    /// <summary>
    /// Data transfer between different GPU devices (peer-to-peer transfer).
    /// This represents direct GPU-to-GPU transfers in multi-GPU systems,
    /// which can provide higher bandwidth than going through host memory.
    /// Requires compatible hardware and driver support.
    /// </summary>
    PeerToPeer
}
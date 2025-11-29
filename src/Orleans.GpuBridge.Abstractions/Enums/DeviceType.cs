namespace Orleans.GpuBridge.Abstractions.Enums;

/// <summary>
/// Type of compute device
/// </summary>
public enum DeviceType
{
    /// <summary>
    /// CPU-based compute device
    /// </summary>
    CPU,

    /// <summary>
    /// Generic GPU device
    /// </summary>
    GPU,

    /// <summary>
    /// Generic hardware accelerator
    /// </summary>
    Accelerator,

    /// <summary>
    /// Custom or proprietary device type
    /// </summary>
    Custom,

    /// <summary>
    /// NVIDIA CUDA device
    /// </summary>
    CUDA,

    /// <summary>
    /// OpenCL compatible device
    /// </summary>
    OpenCL,

    /// <summary>
    /// Microsoft DirectCompute device
    /// </summary>
    DirectCompute,

    /// <summary>
    /// Apple Metal device
    /// </summary>
    Metal,

    /// <summary>
    /// Field-Programmable Gate Array
    /// </summary>
    FPGA,

    /// <summary>
    /// Application-Specific Integrated Circuit
    /// </summary>
    ASIC
}
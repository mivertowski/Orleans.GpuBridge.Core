namespace Orleans.GpuBridge.Abstractions.Enums;

/// <summary>
/// Type of compute device
/// </summary>
public enum DeviceType
{
    Cpu,
    Gpu,
    Accelerator,
    Custom,
    Cuda,
    OpenCl,
    DirectCompute,
    Metal,
    Fpga,
    Asic
}
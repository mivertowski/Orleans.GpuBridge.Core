namespace Orleans.GpuBridge.Abstractions.Enums;

/// <summary>
/// Supported GPU backends
/// </summary>
public enum GpuBackend
{
    Cpu,
    Cuda,
    OpenCL,
    DirectCompute,
    Metal,
    Vulkan
}
namespace Orleans.GpuBridge.Runtime.BackendProviders.Enums;

/// <summary>
/// Backend type enumeration
/// </summary>
public enum BackendType
{
    Cpu,
    Cuda,
    OpenCL,
    DirectCompute,
    Metal,
    Vulkan
}
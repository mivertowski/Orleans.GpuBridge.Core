namespace Orleans.GpuBridge.Backends.ILGPU.Kernels;

/// <summary>
/// Kernel categories
/// </summary>
public enum KernelCategory
{
    Vector,
    Reduction,
    Matrix,
    Image,
    Custom
}

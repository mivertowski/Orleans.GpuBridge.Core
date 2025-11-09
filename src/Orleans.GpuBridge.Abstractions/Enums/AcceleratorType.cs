namespace Orleans.GpuBridge.Abstractions.Enums;

/// <summary>
/// Accelerator type enumeration for device selection across all backend providers.
/// </summary>
/// <remarks>
/// This enum represents the supported compute accelerator types that can be used
/// with Orleans.GpuBridge. Backend providers may support a subset of these types
/// based on platform and hardware availability.
/// </remarks>
public enum AcceleratorType
{
    /// <summary>
    /// Automatically select the best available device based on capabilities and performance.
    /// The backend provider will choose the most appropriate accelerator type.
    /// </summary>
    Auto,

    /// <summary>
    /// Use CPU for computation. This provides maximum compatibility but may have
    /// lower performance compared to GPU accelerators.
    /// </summary>
    CPU,

    /// <summary>
    /// Use NVIDIA CUDA-capable GPU. Requires NVIDIA GPU hardware and CUDA runtime.
    /// Provides high performance for parallel compute workloads.
    /// </summary>
    CUDA,

    /// <summary>
    /// Use OpenCL-capable device. Supports multiple vendors (NVIDIA, AMD, Intel)
    /// and provides cross-platform GPU acceleration.
    /// </summary>
    OpenCL,

    /// <summary>
    /// Use Apple Metal on macOS/iOS. Provides optimized GPU acceleration on
    /// Apple platforms with Metal-capable hardware.
    /// </summary>
    Metal,

    /// <summary>
    /// Use DirectCompute (DirectX Compute Shader) on Windows. Provides GPU
    /// acceleration on Windows platforms with DirectX 11+ capable hardware.
    /// </summary>
    DirectCompute,

    /// <summary>
    /// Use Vulkan compute. Provides cross-platform GPU acceleration on devices
    /// with Vulkan support, offering low-level control and high performance.
    /// </summary>
    Vulkan
}

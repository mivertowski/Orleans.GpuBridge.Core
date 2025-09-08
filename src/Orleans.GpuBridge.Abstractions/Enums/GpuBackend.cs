namespace Orleans.GpuBridge.Abstractions.Enums;

/// <summary>
/// Defines the available compute backend types for GPU and CPU processing.
/// </summary>
/// <remarks>
/// This enumeration specifies which compute backend should be used for executing
/// GPU kernels. Each backend corresponds to a different compute API or platform.
/// The device manager uses this information to select the appropriate backend
/// based on platform capabilities and user preferences.
/// </remarks>
public enum GpuBackend
{
    /// <summary>
    /// Automatically select the best available backend based on platform and device capabilities.
    /// </summary>
    /// <remarks>
    /// When Auto is selected, the system will evaluate available backends
    /// and choose the most appropriate one based on performance characteristics
    /// and platform support.
    /// </remarks>
    Auto,

    /// <summary>
    /// CPU backend for fallback processing when GPU is not available.
    /// </summary>
    /// <remarks>
    /// The CPU backend provides a fallback option when no GPU backends are available
    /// or when CPU processing is preferred. It uses multi-threaded SIMD operations
    /// to maximize performance on CPU architectures.
    /// </remarks>
    CPU,

    /// <summary>
    /// NVIDIA CUDA backend for high-performance GPU computing.
    /// </summary>
    /// <remarks>
    /// CUDA provides optimal performance on NVIDIA GPUs and is available on
    /// Windows and Linux platforms. Requires CUDA runtime libraries to be installed.
    /// </remarks>
    CUDA,

    /// <summary>
    /// OpenCL backend for cross-platform GPU and CPU computing.
    /// </summary>
    /// <remarks>
    /// OpenCL provides broad compatibility across different GPU vendors (NVIDIA, AMD, Intel)
    /// and platforms (Windows, Linux, macOS). Performance may vary depending on
    /// the specific OpenCL implementation.
    /// </remarks>
    OpenCL,

    /// <summary>
    /// Microsoft DirectCompute backend for Windows GPU computing.
    /// </summary>
    /// <remarks>
    /// DirectCompute is available on Windows platforms through DirectX 11 and later.
    /// It provides good integration with Windows graphics APIs and supports
    /// most modern GPUs on Windows.
    /// </remarks>
    DirectCompute,

    /// <summary>
    /// Apple Metal backend for macOS GPU computing.
    /// </summary>
    /// <remarks>
    /// Metal provides optimized performance on macOS with Apple's graphics hardware.
    /// It offers low-level access to GPU resources and is the preferred backend
    /// for Apple platforms.
    /// </remarks>
    Metal,

    /// <summary>
    /// Khronos Vulkan backend for modern cross-platform GPU computing.
    /// </summary>
    /// <remarks>
    /// Vulkan provides low-level GPU access with minimal driver overhead.
    /// It supports Windows, Linux, and Android platforms with modern GPUs.
    /// </remarks>
    Vulkan
}
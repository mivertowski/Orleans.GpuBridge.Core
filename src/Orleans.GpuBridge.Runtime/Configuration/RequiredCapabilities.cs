namespace Orleans.GpuBridge.Runtime.Configuration;

/// <summary>
/// Specifies the capabilities that must be supported by a backend provider for it to be considered during selection.
/// These capabilities filter available backends based on specific feature requirements.
/// </summary>
public class RequiredCapabilities
{
    /// <summary>
    /// Gets or sets a value indicating whether Just-In-Time (JIT) compilation support is required.
    /// JIT compilation allows kernels to be compiled at runtime for optimal performance on the target device.
    /// </summary>
    /// <value><c>true</c> if JIT compilation is required; otherwise, <c>false</c>. Default is <c>false</c>.</value>
    public bool RequireJitCompilation { get; set; } = false;

    /// <summary>
    /// Gets or sets a value indicating whether unified memory support is required.
    /// Unified memory allows seamless data sharing between CPU and GPU without explicit memory transfers.
    /// </summary>
    /// <value><c>true</c> if unified memory is required; otherwise, <c>false</c>. Default is <c>false</c>.</value>
    public bool RequireUnifiedMemory { get; set; } = false;

    /// <summary>
    /// Gets or sets a value indicating whether atomic operations support is required.
    /// Atomic operations enable thread-safe read-modify-write operations in parallel kernels.
    /// </summary>
    /// <value><c>true</c> if atomic operations are required; otherwise, <c>false</c>. Default is <c>false</c>.</value>
    public bool RequireAtomicOperations { get; set; } = false;

    /// <summary>
    /// Gets or sets a value indicating whether tensor operations support is required.
    /// Tensor operations provide optimized implementations for deep learning and machine learning workloads.
    /// </summary>
    /// <value><c>true</c> if tensor operations are required; otherwise, <c>false</c>. Default is <c>false</c>.</value>
    public bool RequireTensorOperations { get; set; } = false;

    /// <summary>
    /// Gets or sets a value indicating whether profiling and performance monitoring support is required.
    /// Profiling enables detailed analysis of kernel execution times and resource utilization.
    /// </summary>
    /// <value><c>true</c> if profiling support is required; otherwise, <c>false</c>. Default is <c>false</c>.</value>
    public bool RequireProfiling { get; set; } = false;

    /// <summary>
    /// Gets or sets the minimum compute capability version required for the backend.
    /// This is typically used for CUDA backends to specify the minimum SM (Streaming Multiprocessor) version.
    /// </summary>
    /// <value>A <see cref="Version"/> representing the minimum compute capability, or <c>null</c> if no minimum is required.</value>
    public Version? MinimumComputeCapability { get; set; }

    /// <summary>
    /// Gets or sets the list of kernel programming languages that must be supported.
    /// Examples include "CUDA", "OpenCL", "HLSL", "GLSL", etc.
    /// </summary>
    /// <value>A list of required kernel language names. Default is an empty list (no language restrictions).</value>
    public List<string> RequiredKernelLanguages { get; set; } = new();
}
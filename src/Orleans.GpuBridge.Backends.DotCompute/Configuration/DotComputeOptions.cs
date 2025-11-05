namespace Orleans.GpuBridge.Backends.DotCompute.Configuration;

/// <summary>
/// Configuration options for the DotCompute backend provider.
/// </summary>
public sealed class DotComputeOptions
{
    /// <summary>
    /// Gets or sets the default accelerator type to use.
    /// If null, DotCompute will automatically select the best available device.
    /// </summary>
    public AcceleratorType? DefaultAccelerator { get; set; }

    /// <summary>
    /// Gets or sets a value indicating whether to enable automatic performance optimization.
    /// Default is true.
    /// </summary>
    public bool EnableAutoOptimization { get; set; } = true;

    /// <summary>
    /// Gets or sets a value indicating whether to enable telemetry collection.
    /// Default is false.
    /// </summary>
    public bool EnableTelemetry { get; set; } = false;

    /// <summary>
    /// Gets or sets a value indicating whether to enable kernel compilation caching.
    /// Default is true for significant performance improvement on repeated kernel execution.
    /// </summary>
    public bool EnableKernelCaching { get; set; } = true;

    /// <summary>
    /// Gets or sets a value indicating whether to enable production debugging features.
    /// Default is false. Set to true to enable cross-backend validation and detailed diagnostics.
    /// </summary>
    public bool EnableProductionDebugging { get; set; } = false;

    /// <summary>
    /// Gets or sets the maximum number of concurrent kernel executions.
    /// If null, uses DotCompute default based on device capabilities.
    /// </summary>
    public int? MaxConcurrentExecutions { get; set; }

    /// <summary>
    /// Gets or sets the kernel compilation timeout in milliseconds.
    /// Default is 30000 (30 seconds).
    /// </summary>
    public int CompilationTimeoutMs { get; set; } = 30000;

    /// <summary>
    /// Gets or sets the kernel execution timeout in milliseconds.
    /// Default is 60000 (60 seconds).
    /// </summary>
    public int ExecutionTimeoutMs { get; set; } = 60000;
}

/// <summary>
/// Accelerator type enumeration for device selection.
/// </summary>
public enum AcceleratorType
{
    /// <summary>
    /// Automatically select the best available device.
    /// </summary>
    Auto,

    /// <summary>
    /// Use CPU for computation.
    /// </summary>
    CPU,

    /// <summary>
    /// Use NVIDIA CUDA-capable GPU.
    /// </summary>
    CUDA,

    /// <summary>
    /// Use OpenCL-capable device.
    /// </summary>
    OpenCL,

    /// <summary>
    /// Use Apple Metal on macOS/iOS.
    /// </summary>
    Metal,

    /// <summary>
    /// Use DirectCompute on Windows.
    /// </summary>
    DirectCompute,

    /// <summary>
    /// Use Vulkan compute.
    /// </summary>
    Vulkan
}

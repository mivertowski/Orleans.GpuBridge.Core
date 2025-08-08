namespace Orleans.GpuBridge.Abstractions;

/// <summary>
/// Configuration options for GPU Bridge
/// </summary>
public sealed class GpuBridgeOptions 
{ 
    /// <summary>
    /// Whether to prefer GPU execution over CPU
    /// </summary>
    public bool PreferGpu { get; set; } = true;
    
    /// <summary>
    /// Default micro-batch size for kernel execution
    /// </summary>
    public int DefaultMicroBatch { get; set; } = 8192;
    
    /// <summary>
    /// Maximum concurrent kernels per device
    /// </summary>
    public int MaxConcurrentKernels { get; set; } = 100;
    
    /// <summary>
    /// Memory pool size in MB
    /// </summary>
    public int MemoryPoolSizeMB { get; set; } = 1024;
    
    /// <summary>
    /// Enable GPU Direct Storage if available
    /// </summary>
    public bool EnableGpuDirectStorage { get; set; } = false;
    
    /// <summary>
    /// Maximum number of devices to use
    /// </summary>
    public int MaxDevices { get; set; } = 4;
    
    /// <summary>
    /// Enable kernel profiling
    /// </summary>
    public bool EnableProfiling { get; set; } = false;
    
    /// <summary>
    /// Telemetry options
    /// </summary>
    public TelemetryOptions Telemetry { get; set; } = new();
}

/// <summary>
/// Telemetry configuration options
/// </summary>
public sealed class TelemetryOptions
{
    /// <summary>
    /// Enable metrics collection
    /// </summary>
    public bool EnableMetrics { get; set; } = true;
    
    /// <summary>
    /// Enable distributed tracing
    /// </summary>
    public bool EnableTracing { get; set; } = true;
    
    /// <summary>
    /// Sampling rate for traces (0.0 to 1.0)
    /// </summary>
    public double SamplingRate { get; set; } = 0.1;
}

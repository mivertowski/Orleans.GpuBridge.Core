namespace Orleans.GpuBridge.Diagnostics.Configuration;

/// <summary>
/// Configuration options for GPU metrics collection and monitoring.
/// </summary>
/// <remarks>
/// These options control the behavior of the GPU metrics collector, including
/// collection intervals, feature enablement, and device limits. The configuration
/// can be customized based on monitoring requirements and system resources.
/// </remarks>
public class GpuMetricsOptions
{
    /// <summary>
    /// Gets or sets the interval between automatic metrics collection cycles.
    /// </summary>
    /// <value>
    /// The time interval between collections. Default is 10 seconds.
    /// </value>
    /// <remarks>
    /// <para>
    /// Shorter intervals provide more frequent metrics updates but increase system overhead.
    /// Longer intervals reduce overhead but may miss transient performance issues.
    /// </para>
    /// <para>
    /// Recommended values:
    /// - High-frequency monitoring: 1-5 seconds
    /// - Standard monitoring: 10-30 seconds  
    /// - Low-overhead monitoring: 60+ seconds
    /// </para>
    /// </remarks>
    public TimeSpan CollectionInterval { get; set; } = TimeSpan.FromSeconds(10);

    /// <summary>
    /// Gets or sets a value indicating whether system-level metrics collection is enabled.
    /// </summary>
    /// <value>
    /// <c>true</c> to collect system metrics (CPU, memory, threads, handles); otherwise, <c>false</c>.
    /// Default is <c>true</c>.
    /// </value>
    /// <remarks>
    /// System metrics include process-level resource consumption data that can be useful
    /// for capacity planning and troubleshooting. Disabling this option can reduce overhead
    /// when only GPU metrics are needed.
    /// </remarks>
    public bool EnableSystemMetrics { get; set; } = true;

    /// <summary>
    /// Gets or sets a value indicating whether GPU device metrics collection is enabled.
    /// </summary>
    /// <value>
    /// <c>true</c> to collect GPU metrics (utilization, memory, temperature, power); otherwise, <c>false</c>.
    /// Default is <c>true</c>.
    /// </value>
    /// <remarks>
    /// GPU metrics require platform-specific tools (nvidia-smi, rocm-smi) to be available.
    /// If these tools are not installed, GPU metrics collection will fail gracefully.
    /// Disabling this option prevents GPU monitoring attempts entirely.
    /// </remarks>
    public bool EnableGpuMetrics { get; set; } = true;

    /// <summary>
    /// Gets or sets a value indicating whether detailed diagnostic logging is enabled.
    /// </summary>
    /// <value>
    /// <c>true</c> to enable verbose logging of metrics collection activities; otherwise, <c>false</c>.
    /// Default is <c>false</c>.
    /// </value>
    /// <remarks>
    /// <para>
    /// When enabled, the metrics collector will log detailed information about:
    /// - Individual device query attempts and failures
    /// - System metrics values
    /// - Collection timing and performance
    /// </para>
    /// <para>
    /// This option should typically be disabled in production to avoid log volume issues,
    /// but can be useful for troubleshooting metrics collection problems.
    /// </para>
    /// </remarks>
    public bool EnableDetailedLogging { get; set; } = false;

    /// <summary>
    /// Gets or sets the maximum number of GPU devices to monitor.
    /// </summary>
    /// <value>
    /// The maximum device index to query (0-based). Default is 8 devices (indices 0-7).
    /// </value>
    /// <remarks>
    /// <para>
    /// This limit prevents excessive overhead when systems have many GPU devices.
    /// The collector will attempt to query devices from index 0 up to (MaxDevices - 1).
    /// </para>
    /// <para>
    /// Recommended values:
    /// - Single-GPU systems: 1-2
    /// - Multi-GPU workstations: 4-8
    /// - GPU clusters: 16+ (adjust based on monitoring requirements)
    /// </para>
    /// <para>
    /// Note: Devices that don't exist or can't be queried will be skipped without error.
    /// </para>
    /// </remarks>
    public int MaxDevices { get; set; } = 8;
}
namespace Orleans.GpuBridge.HealthChecks.Configuration;

/// <summary>
/// Configuration options for GPU health check monitoring and alerting thresholds.
/// These options control when the health check reports healthy, degraded, or unhealthy status
/// based on GPU hardware metrics and operational parameters.
/// </summary>
/// <remarks>
/// The health check evaluates GPU systems across multiple dimensions:
/// - Hardware health (temperature, power consumption)
/// - Memory utilization and availability  
/// - Functional testing (kernel execution verification)
/// - Resource utilization patterns
/// 
/// Status determination follows a hierarchical approach:
/// - Unhealthy: Critical thresholds exceeded (temperature, memory exhaustion)
/// - Degraded: Warning thresholds exceeded or performance issues detected
/// - Healthy: All metrics within acceptable ranges
/// </remarks>
public class GpuHealthCheckOptions
{
    /// <summary>
    /// Gets or sets a value indicating whether the presence of at least one functional GPU device is required.
    /// When <c>true</c>, the health check will report unhealthy if no GPU devices are available.
    /// When <c>false</c>, the absence of GPU devices is treated as degraded (with CPU fallback active).
    /// </summary>
    /// <value><c>true</c> to require GPU presence for healthy status; otherwise, <c>false</c>. Default is <c>false</c>.</value>
    /// <remarks>
    /// Set this to <c>true</c> for GPU-dependent applications where CPU fallback is not acceptable.
    /// Set to <c>false</c> for applications that can gracefully degrade to CPU execution.
    /// </remarks>
    public bool RequireGpu { get; set; } = false;

    /// <summary>
    /// Gets or sets a value indicating whether to perform functional kernel execution testing.
    /// When enabled, the health check will execute a simple test kernel to verify GPU compute functionality.
    /// This provides functional verification beyond just device presence and hardware metrics.
    /// </summary>
    /// <value><c>true</c> to enable kernel execution testing; otherwise, <c>false</c>. Default is <c>true</c>.</value>
    /// <remarks>
    /// Kernel testing adds execution overhead but provides confidence that GPU compute operations work correctly.
    /// Disable this in environments where kernel testing may interfere with critical workloads.
    /// </remarks>
    public bool TestKernelExecution { get; set; } = true;

    /// <summary>
    /// Gets or sets the maximum acceptable GPU temperature in Celsius before reporting unhealthy status.
    /// Temperatures at or above this threshold indicate thermal issues that could damage hardware
    /// or cause thermal throttling affecting performance.
    /// </summary>
    /// <value>The maximum temperature threshold in Celsius. Default is 85.0°C.</value>
    /// <remarks>
    /// This threshold should be set based on:
    /// - GPU hardware specifications and thermal design power (TDP)
    /// - Cooling system capabilities and ambient temperature
    /// - Performance requirements and throttling behavior
    /// 
    /// Typical values:
    /// - Consumer GPUs: 80-85°C
    /// - Professional/Server GPUs: 85-90°C  
    /// - Specialized cooling: 75-80°C
    /// </remarks>
    public double MaxTemperatureCelsius { get; set; } = 85.0;

    /// <summary>
    /// Gets or sets the temperature threshold in Celsius for reporting degraded status.
    /// Temperatures at or above this threshold indicate elevated thermal conditions
    /// that warrant monitoring but don't require immediate intervention.
    /// </summary>
    /// <value>The warning temperature threshold in Celsius. Default is 75.0°C.</value>
    /// <remarks>
    /// This threshold should provide sufficient headroom before reaching the maximum temperature
    /// to allow for temperature monitoring, alerting, and potential workload adjustment.
    /// Typically set 10-15°C below the maximum temperature threshold.
    /// </remarks>
    public double WarnTemperatureCelsius { get; set; } = 75.0;

    /// <summary>
    /// Gets or sets the maximum acceptable GPU memory utilization percentage before reporting unhealthy status.
    /// Memory usage at or above this threshold indicates memory exhaustion conditions
    /// that will likely cause allocation failures and application errors.
    /// </summary>
    /// <value>The maximum memory utilization percentage (0.0-100.0). Default is 95.0%.</value>
    /// <remarks>
    /// GPU memory exhaustion can cause:
    /// - Allocation failures and out-of-memory errors
    /// - Application crashes or degraded functionality
    /// - Performance degradation due to memory pressure
    /// 
    /// Set based on:
    /// - Application memory patterns and peak usage
    /// - Memory fragmentation considerations
    /// - Need for memory headroom during processing spikes
    /// </remarks>
    public double MaxMemoryUsagePercent { get; set; } = 95.0;

    /// <summary>
    /// Gets or sets the memory utilization percentage threshold for reporting degraded status.
    /// Memory usage at or above this threshold indicates memory pressure conditions
    /// that warrant monitoring and potential optimization.
    /// </summary>
    /// <value>The warning memory utilization percentage (0.0-100.0). Default is 80.0%.</value>
    /// <remarks>
    /// This threshold provides early warning of approaching memory limits,
    /// allowing for proactive memory management, garbage collection,
    /// or workload redistribution before reaching critical levels.
    /// </remarks>
    public double WarnMemoryUsagePercent { get; set; } = 80.0;

    /// <summary>
    /// Gets or sets the minimum expected GPU utilization percentage for optimal operation.
    /// Utilization below this threshold indicates potential underutilization or performance issues
    /// that may warrant investigation or workload optimization.
    /// </summary>
    /// <value>The minimum utilization percentage (0.0-100.0). Default is 5.0%.</value>
    /// <remarks>
    /// Low GPU utilization may indicate:
    /// - CPU-bound workloads not effectively using GPU resources
    /// - Inefficient data transfer patterns causing GPU starvation
    /// - Suboptimal workload sizing or batching
    /// - System bottlenecks preventing effective GPU utilization
    /// 
    /// Set based on expected workload characteristics and performance requirements.
    /// For batch processing: higher thresholds (10-20%)
    /// For interactive applications: lower thresholds (1-5%)
    /// </remarks>
    public double MinUtilizationPercent { get; set; } = 5.0;
}
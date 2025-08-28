namespace Orleans.GpuBridge.Diagnostics.Models;

/// <summary>
/// Represents performance and resource utilization metrics for a GPU device.
/// </summary>
/// <remarks>
/// This class captures comprehensive metrics from GPU devices including utilization,
/// memory consumption, thermal status, and power usage. The metrics are vendor-agnostic
/// and support NVIDIA, AMD, and Intel GPU monitoring tools.
/// </remarks>
public class GpuDeviceMetrics
{
    /// <summary>
    /// Gets or sets the zero-based index of the GPU device.
    /// </summary>
    /// <value>
    /// The device index as reported by the GPU driver. Typically starts from 0.
    /// </value>
    /// <remarks>
    /// Device indices are assigned by the GPU driver and may not be consecutive
    /// if some devices are unavailable or disabled.
    /// </remarks>
    public int DeviceIndex { get; set; }

    /// <summary>
    /// Gets or sets the human-readable name of the GPU device.
    /// </summary>
    /// <value>
    /// The device name as reported by the GPU driver (e.g., "NVIDIA GeForce RTX 4090", "AMD Radeon RX 7900 XTX").
    /// Default is an empty string.
    /// </value>
    /// <remarks>
    /// Device names are provided by the GPU vendor and may include model numbers,
    /// memory sizes, or other identifying information.
    /// </remarks>
    public string DeviceName { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the type/vendor of the GPU device.
    /// </summary>
    /// <value>
    /// The GPU vendor identifier (e.g., "NVIDIA", "AMD", "Intel").
    /// Default is an empty string.
    /// </value>
    /// <remarks>
    /// This property indicates which monitoring tool was used to collect the metrics
    /// and can be used for vendor-specific processing or display logic.
    /// </remarks>
    public string DeviceType { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the GPU compute utilization percentage.
    /// </summary>
    /// <value>
    /// GPU utilization as a percentage (0.0 to 100.0). Represents the fraction of time
    /// the GPU was actively processing during the measurement period.
    /// </value>
    /// <remarks>
    /// High utilization (>80%) may indicate compute bottlenecks, while low utilization
    /// may suggest underutilized GPU resources or CPU-bound workloads.
    /// </remarks>
    public double GpuUtilization { get; set; }

    /// <summary>
    /// Gets or sets the GPU memory controller utilization percentage.
    /// </summary>
    /// <value>
    /// Memory subsystem utilization as a percentage (0.0 to 100.0). Represents how
    /// actively the memory controller was used during the measurement period.
    /// </value>
    /// <remarks>
    /// Memory utilization can be high even when compute utilization is low if the
    /// workload is memory-bandwidth intensive or involves large data transfers.
    /// </remarks>
    public double MemoryUtilization { get; set; }

    /// <summary>
    /// Gets or sets the amount of GPU memory currently in use.
    /// </summary>
    /// <value>
    /// Used GPU memory in megabytes (MB). Includes memory allocated for compute kernels,
    /// textures, buffers, and driver overhead.
    /// </value>
    /// <remarks>
    /// This represents allocated memory that may include both active data and cached
    /// allocations. Memory usage approaching the total capacity may cause allocation
    /// failures or performance degradation.
    /// </remarks>
    public long MemoryUsedMB { get; set; }

    /// <summary>
    /// Gets or sets the total amount of GPU memory available on the device.
    /// </summary>
    /// <value>
    /// Total GPU memory capacity in megabytes (MB). This is the maximum memory
    /// that can be allocated by applications and the GPU driver.
    /// </value>
    /// <remarks>
    /// Total memory is typically slightly less than the advertised memory capacity
    /// due to driver and hardware reservations.
    /// </remarks>
    public long MemoryTotalMB { get; set; }

    /// <summary>
    /// Gets or sets the current GPU core temperature.
    /// </summary>
    /// <value>
    /// GPU temperature in degrees Celsius. Normal operating temperatures vary by
    /// GPU model but typically range from 30-85°C under load.
    /// </value>
    /// <remarks>
    /// <para>
    /// Temperature monitoring is important for:
    /// - Thermal throttling detection (typically occurs around 80-95°C)
    /// - Cooling system effectiveness evaluation
    /// - Hardware health monitoring
    /// </para>
    /// <para>
    /// Sustained high temperatures may indicate inadequate cooling or high workloads.
    /// </para>
    /// </remarks>
    public double TemperatureCelsius { get; set; }

    /// <summary>
    /// Gets or sets the current GPU power consumption.
    /// </summary>
    /// <value>
    /// Power usage in watts (W). Represents the electrical power being consumed
    /// by the GPU during the measurement period.
    /// </value>
    /// <remarks>
    /// <para>
    /// Power consumption varies based on:
    /// - GPU utilization and clock speeds
    /// - Memory bandwidth usage
    /// - Operating temperature
    /// - Power management settings
    /// </para>
    /// <para>
    /// High power usage may indicate intensive workloads or inefficient algorithms.
    /// Power limits are typically enforced by the GPU to prevent overheating.
    /// </para>
    /// </remarks>
    public double PowerUsageWatts { get; set; }

    /// <summary>
    /// Gets or sets the timestamp when these metrics were collected.
    /// </summary>
    /// <value>
    /// UTC timestamp indicating when the metrics were captured from the GPU.
    /// </value>
    /// <remarks>
    /// This timestamp can be used to determine metric age, calculate rates of change,
    /// or synchronize metrics across multiple devices.
    /// </remarks>
    public DateTimeOffset Timestamp { get; set; }

    /// <summary>
    /// Gets the amount of GPU memory that is currently available for allocation.
    /// </summary>
    /// <value>
    /// Available GPU memory in megabytes (MB), calculated as the difference between
    /// total memory and used memory.
    /// </value>
    /// <remarks>
    /// Available memory indicates how much additional memory can potentially be allocated.
    /// However, memory fragmentation may prevent allocation of the full available amount.
    /// </remarks>
    public long MemoryAvailableMB => MemoryTotalMB - MemoryUsedMB;

    /// <summary>
    /// Gets the GPU memory usage as a percentage of total memory capacity.
    /// </summary>
    /// <value>
    /// Memory usage percentage (0.0 to 100.0), calculated as the ratio of used memory
    /// to total memory capacity.
    /// </value>
    /// <remarks>
    /// <para>
    /// Memory usage percentages help identify potential memory pressure:
    /// - 0-70%: Normal usage with adequate free memory
    /// - 70-90%: High usage, may impact performance
    /// - 90-100%: Critical usage, likely to cause allocation failures
    /// </para>
    /// <para>
    /// Returns 0 if total memory is not available or invalid.
    /// </para>
    /// </remarks>
    public double MemoryUsagePercent => MemoryTotalMB > 0 ? (MemoryUsedMB * 100.0 / MemoryTotalMB) : 0;
}
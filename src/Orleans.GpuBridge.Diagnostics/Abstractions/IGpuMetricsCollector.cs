using Orleans.GpuBridge.Diagnostics.Models;

namespace Orleans.GpuBridge.Diagnostics.Abstractions;

/// <summary>
/// Defines the contract for collecting GPU and system performance metrics.
/// </summary>
/// <remarks>
/// This interface provides methods for retrieving performance metrics from GPU devices
/// and the host system. Implementations should support multiple GPU vendors and provide
/// fallback mechanisms when hardware monitoring tools are unavailable.
/// </remarks>
public interface IGpuMetricsCollector
{
    /// <summary>
    /// Retrieves performance metrics for a specific GPU device.
    /// </summary>
    /// <param name="deviceIndex">The zero-based index of the GPU device to query.</param>
    /// <returns>
    /// A task that represents the asynchronous operation. The task result contains
    /// the GPU device metrics including utilization, memory usage, temperature, and power consumption.
    /// </returns>
    /// <exception cref="InvalidOperationException">
    /// Thrown when no metrics are available for the specified device index.
    /// </exception>
    /// <remarks>
    /// This method will attempt to collect fresh metrics if cached data is not available.
    /// The method supports multiple GPU vendors (NVIDIA, AMD, Intel) and will automatically
    /// detect the appropriate monitoring tools.
    /// </remarks>
    Task<GpuDeviceMetrics> GetDeviceMetricsAsync(int deviceIndex);

    /// <summary>
    /// Retrieves current system performance metrics for the host process.
    /// </summary>
    /// <returns>
    /// A task that represents the asynchronous operation. The task result contains
    /// system metrics including CPU usage, memory consumption, thread count, and handle count.
    /// </returns>
    /// <remarks>
    /// System metrics provide insight into the resource consumption of the Orleans.GpuBridge
    /// host process and can be used for performance monitoring and capacity planning.
    /// </remarks>
    Task<SystemMetrics> GetSystemMetricsAsync();

    /// <summary>
    /// Retrieves performance metrics for all available GPU devices.
    /// </summary>
    /// <returns>
    /// A task that represents the asynchronous operation. The task result contains
    /// a read-only list of GPU device metrics for all detected devices.
    /// </returns>
    /// <remarks>
    /// This method will attempt to collect metrics from all configured devices up to
    /// the MaxDevices limit specified in GpuMetricsOptions. Devices that cannot be
    /// queried will be omitted from the results rather than causing the operation to fail.
    /// </remarks>
    Task<IReadOnlyList<GpuDeviceMetrics>> GetAllDeviceMetricsAsync();
}
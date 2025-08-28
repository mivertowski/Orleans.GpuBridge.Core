using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using Orleans.GpuBridge.Abstractions.Models;

namespace Orleans.GpuBridge.Abstractions.Providers;

/// <summary>
/// Interface for managing compute devices in GPU backends.
/// The device manager is responsible for discovering, enumerating, and managing
/// compute devices available to the application, as well as creating execution contexts
/// and providing device metrics for monitoring and optimization purposes.
/// </summary>
public interface IDeviceManager : IDisposable
{
    /// <summary>
    /// Initializes the device manager and discovers available devices.
    /// This method performs the initial device enumeration and setup required
    /// before other operations can be performed. It should be called once
    /// during application startup.
    /// </summary>
    /// <param name="cancellationToken">Token to cancel the initialization operation.</param>
    /// <returns>A task that completes when initialization is finished.</returns>
    Task InitializeAsync(CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Gets all available compute devices discovered by the device manager.
    /// This list represents all devices that are currently available for use,
    /// regardless of their current status or utilization.
    /// </summary>
    /// <returns>A read-only list of all discovered compute devices.</returns>
    IReadOnlyList<IComputeDevice> GetDevices();
    
    /// <summary>
    /// Gets a specific device by its index in the device enumeration.
    /// </summary>
    /// <param name="deviceIndex">The zero-based index of the device to retrieve.</param>
    /// <returns>The device at the specified index, or null if the index is invalid.</returns>
    IComputeDevice? GetDevice(int deviceIndex);
    
    /// <summary>
    /// Gets the default device for computation.
    /// This is typically the most capable device available, or the device
    /// designated as the primary compute device by the system or backend provider.
    /// </summary>
    /// <returns>The default compute device for the system.</returns>
    IComputeDevice GetDefaultDevice();
    
    /// <summary>
    /// Selects the best device based on the specified requirements and preferences.
    /// This method evaluates all available devices against the provided criteria
    /// and returns the device that best matches the requirements.
    /// </summary>
    /// <param name="criteria">The selection criteria and preferences for device selection.</param>
    /// <returns>The device that best matches the specified criteria.</returns>
    IComputeDevice SelectDevice(DeviceSelectionCriteria criteria);
    
    /// <summary>
    /// Creates a compute context on the specified device.
    /// A compute context represents an execution environment where kernels can be
    /// compiled, loaded, and executed. Multiple contexts can exist on the same device.
    /// </summary>
    /// <param name="device">The device on which to create the context.</param>
    /// <param name="options">Configuration options for the context creation.</param>
    /// <param name="cancellationToken">Token to cancel the context creation operation.</param>
    /// <returns>A task that completes with the created compute context.</returns>
    Task<IComputeContext> CreateContextAsync(
        IComputeDevice device,
        ContextOptions options,
        CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Gets current utilization and performance metrics for the specified device.
    /// This information can be used for monitoring, load balancing, and performance
    /// optimization decisions.
    /// </summary>
    /// <param name="device">The device for which to retrieve metrics.</param>
    /// <param name="cancellationToken">Token to cancel the metrics retrieval operation.</param>
    /// <returns>A task that completes with the current device metrics.</returns>
    Task<DeviceMetrics> GetDeviceMetricsAsync(
        IComputeDevice device,
        CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Resets the specified device, clearing its memory and resetting its state.
    /// This operation will invalidate all existing contexts and resources associated
    /// with the device. Use this method to recover from error conditions or to
    /// ensure a clean state for new operations.
    /// </summary>
    /// <param name="device">The device to reset.</param>
    /// <param name="cancellationToken">Token to cancel the reset operation.</param>
    /// <returns>A task that completes when the device has been reset.</returns>
    Task ResetDeviceAsync(
        IComputeDevice device,
        CancellationToken cancellationToken = default);
}
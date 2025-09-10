using Orleans.GpuBridge.Abstractions.Enums;

namespace Orleans.GpuBridge.Backends.DotCompute.Models;

/// <summary>
/// Specifies requirements and preferences for compute device selection during GPU operations.
/// </summary>
/// <remarks>
/// This class defines the criteria used by the device manager to select the most appropriate
/// compute device for a given task. The requirements include memory constraints, compute unit
/// preferences, and device type restrictions. The device manager uses these requirements
/// to score available devices and select the best match.
/// </remarks>
public sealed class ComputeRequirements
{
    /// <summary>
    /// Gets or sets whether GPU devices should be preferred over CPU devices.
    /// </summary>
    /// <value>
    /// <c>true</c> to prefer GPU devices (default); <c>false</c> to prefer CPU devices.
    /// </value>
    /// <remarks>
    /// When <c>true</c>, the device scoring algorithm will give higher scores to GPU devices
    /// (CUDA, OpenCL, DirectCompute, Metal) compared to CPU devices. When <c>false</c>,
    /// CPU devices will receive higher scores, making them more likely to be selected.
    /// </remarks>
    public bool PreferGpu { get; set; } = true;

    /// <summary>
    /// Gets or sets the minimum amount of memory required in bytes.
    /// </summary>
    /// <value>
    /// The minimum memory requirement in bytes. Defaults to 256MB (268,435,456 bytes).
    /// </value>
    /// <remarks>
    /// Devices with available memory below this threshold will receive lower scores
    /// during device selection. This ensures that selected devices have sufficient
    /// memory for the intended compute operations. The value should account for
    /// both input data and any intermediate buffers required during computation.
    /// </remarks>
    public long MinMemoryBytes { get; set; } = 256 * 1024 * 1024; // 256MB

    /// <summary>
    /// Gets or sets the minimum number of compute units required.
    /// </summary>
    /// <value>
    /// The minimum number of compute units. Defaults to 1.
    /// </value>
    /// <remarks>
    /// Compute units represent parallel processing capabilities of the device.
    /// For GPUs, this typically corresponds to streaming multiprocessors (CUDA),
    /// compute units (OpenCL), or similar parallel processing blocks.
    /// For CPUs, this represents the number of logical processors.
    /// </remarks>
    public int MinComputeUnits { get; set; } = 1;

    /// <summary>
    /// Gets or sets the required device type, if any specific type is mandated.
    /// </summary>
    /// <value>
    /// The required <see cref="DeviceType"/>, or <c>null</c> if no specific type is required.
    /// </value>
    /// <remarks>
    /// When set to a specific device type, only devices of that type will be considered
    /// during device selection. This is useful when a particular compute backend
    /// is required for compatibility or performance reasons. When <c>null</c>,
    /// all available device types are considered based on other criteria.
    /// </remarks>
    public DeviceType? RequiredType { get; set; }
}
using System.Collections.Generic;
using Orleans.GpuBridge.Abstractions.Enums;

namespace Orleans.GpuBridge.Abstractions.Models;

/// <summary>
/// Defines criteria for selecting a compute device from available devices.
/// This record encapsulates various requirements and preferences that can be used
/// by device managers to automatically select the most suitable device for specific workloads.
/// </summary>
/// <param name="PreferredType">
/// The preferred type of compute device (GPU, CPU, etc.).
/// If null, no specific device type preference is applied.
/// </param>
/// <param name="MinMemoryBytes">
/// The minimum amount of device memory required in bytes.
/// Devices with less memory than this threshold will be excluded from selection.
/// Default is 0, meaning no memory requirement.
/// </param>
/// <param name="MinComputeUnits">
/// The minimum number of compute units required.
/// This represents streaming multiprocessors, compute units, or cores depending on device type.
/// Default is 0, meaning no compute unit requirement.
/// </param>
/// <param name="RequireUnifiedMemory">
/// Indicates whether the device must support unified memory architecture.
/// When true, only devices with unified memory (shared between host and device) are considered.
/// Default is false.
/// </param>
/// <param name="PreferHighestPerformance">
/// When true, selects the device with the highest performance characteristics.
/// This typically considers factors like compute units, memory bandwidth, and clock speeds.
/// Default is true.
/// </param>
/// <param name="RequiredFeature">
/// A specific feature that must be supported by the device.
/// If specified, only devices that report support for this feature will be considered.
/// Common features might include "fp64", "atomics", "images", etc.
/// </param>
/// <param name="ExcludeDevices">
/// A list of device IDs that should be excluded from selection.
/// This allows for explicit blacklisting of problematic or reserved devices.
/// If null or empty, no devices are excluded based on ID.
/// </param>
public sealed record DeviceSelectionCriteria(
    DeviceType? PreferredType = null,
    long MinMemoryBytes = 0,
    int MinComputeUnits = 0,
    bool RequireUnifiedMemory = false,
    bool PreferHighestPerformance = true,
    string? RequiredFeature = null,
    IReadOnlyList<string>? ExcludeDevices = null);
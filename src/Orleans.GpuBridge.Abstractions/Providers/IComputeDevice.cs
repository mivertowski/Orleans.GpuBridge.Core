using System;
using System.Collections.Generic;
using Orleans.GpuBridge.Abstractions.Enums;

namespace Orleans.GpuBridge.Abstractions.Providers;

/// <summary>
/// Represents a compute device (GPU, CPU, etc.) that can execute computational tasks.
/// This interface provides access to device properties, capabilities, and status information
/// needed for device selection and management in the GPU bridge framework.
/// </summary>
public interface IComputeDevice
{
    /// <summary>
    /// Gets the unique identifier for the device.
    /// This ID is persistent across application runs and uniquely identifies the device in the system.
    /// </summary>
    string DeviceId { get; }
    
    /// <summary>
    /// Gets the device index in the system.
    /// This is typically the enumeration order of devices as discovered by the backend provider.
    /// </summary>
    int Index { get; }
    
    /// <summary>
    /// Gets the display name of the device.
    /// This is a human-readable name that can be shown in user interfaces for device selection.
    /// </summary>
    string Name { get; }
    
    /// <summary>
    /// Gets the type of compute device.
    /// Indicates whether this is a GPU, CPU, or other type of compute device.
    /// </summary>
    DeviceType Type { get; }
    
    /// <summary>
    /// Gets the vendor of the device.
    /// Common values include "NVIDIA", "AMD", "Intel", etc.
    /// </summary>
    string Vendor { get; }
    
    /// <summary>
    /// Gets the device architecture identifier.
    /// Examples include "Ampere", "RDNA2", "x86-64", which indicate the underlying hardware architecture.
    /// </summary>
    string Architecture { get; }
    
    /// <summary>
    /// Gets the compute capability version of the device.
    /// This represents the feature set and capabilities supported by the device,
    /// such as CUDA compute capability for NVIDIA GPUs.
    /// </summary>
    Version ComputeCapability { get; }
    
    /// <summary>
    /// Gets the total memory available on the device in bytes.
    /// This is the physical memory capacity of the device.
    /// </summary>
    long TotalMemoryBytes { get; }
    
    /// <summary>
    /// Gets the currently available memory on the device in bytes.
    /// This value may change as memory is allocated and freed during device usage.
    /// </summary>
    long AvailableMemoryBytes { get; }
    
    /// <summary>
    /// Gets the number of compute units available on the device.
    /// This represents streaming multiprocessors (SMs) for NVIDIA, compute units (CUs) for AMD,
    /// or cores for CPU devices.
    /// </summary>
    int ComputeUnits { get; }
    
    /// <summary>
    /// Gets the maximum clock frequency of the device in MHz.
    /// This represents the peak operating frequency of the compute units.
    /// </summary>
    int MaxClockFrequencyMHz { get; }
    
    /// <summary>
    /// Gets the maximum number of threads per block or workgroup.
    /// This is a fundamental limit for kernel launch configurations.
    /// </summary>
    int MaxThreadsPerBlock { get; }
    
    /// <summary>
    /// Gets the maximum workgroup dimensions supported by the device.
    /// This array specifies the maximum size in each dimension (x, y, z) for workgroup configurations.
    /// </summary>
    int[] MaxWorkGroupDimensions { get; }
    
    /// <summary>
    /// Gets the warp size (NVIDIA) or wave size (AMD) for the device.
    /// This represents the number of threads that execute in lockstep within a compute unit.
    /// </summary>
    int WarpSize { get; }
    
    /// <summary>
    /// Gets device-specific properties as a read-only dictionary.
    /// This allows backend providers to expose additional device characteristics
    /// that may not be covered by the standard properties.
    /// </summary>
    IReadOnlyDictionary<string, object> Properties { get; }
    
    /// <summary>
    /// Checks if the device supports a specific feature.
    /// </summary>
    /// <param name="feature">The feature identifier to check for support.</param>
    /// <returns>True if the device supports the specified feature; otherwise, false.</returns>
    bool SupportsFeature(string feature);
    
    /// <summary>
    /// Gets the current operational status of the device.
    /// </summary>
    /// <returns>The current device status indicating availability, utilization, or error states.</returns>
    DeviceStatus GetStatus();
}
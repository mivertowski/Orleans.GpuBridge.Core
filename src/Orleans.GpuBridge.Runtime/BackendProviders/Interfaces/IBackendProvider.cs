using System;
using System.Collections.Generic;
using Orleans.GpuBridge.Abstractions.Enums;
using Orleans.GpuBridge.Runtime.BackendProviders.Models;

namespace Orleans.GpuBridge.Runtime.BackendProviders.Interfaces;

/// <summary>
/// Simplified backend provider interface for GPU compute backends
/// </summary>
/// <remarks>
/// This interface defines the contract for backend providers that manage GPU compute resources.
/// Each provider represents a specific compute backend (CUDA, OpenCL, DirectCompute, etc.) and
/// handles device enumeration, context creation, and resource management for that backend.
/// </remarks>
public interface IBackendProvider : IDisposable
{
    /// <summary>
    /// Gets the display name of this backend provider
    /// </summary>
    string Name { get; }
    
    /// <summary>
    /// Gets the backend type identifier
    /// </summary>
    GpuBackend Type { get; }
    
    /// <summary>
    /// Gets a value indicating whether this backend is currently available
    /// </summary>
    bool IsAvailable { get; }
    
    /// <summary>
    /// Gets the number of available devices for this backend
    /// </summary>
    int DeviceCount { get; }
    
    /// <summary>
    /// Initializes the backend provider
    /// </summary>
    /// <returns>true if initialization was successful; otherwise, false</returns>
    bool Initialize();
    
    /// <summary>
    /// Shuts down the backend provider and releases resources
    /// </summary>
    void Shutdown();
    
    /// <summary>
    /// Creates a compute context for the specified device
    /// </summary>
    /// <param name="deviceIndex">The index of the device to create a context for</param>
    /// <returns>A compute context for the specified device</returns>
    IComputeContext CreateContext(int deviceIndex = 0);
    
    /// <summary>
    /// Gets information about all available devices for this backend
    /// </summary>
    /// <returns>A read-only list of device information</returns>
    IReadOnlyList<DeviceInfo> GetDevices();
}
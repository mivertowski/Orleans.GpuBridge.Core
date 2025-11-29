// Copyright (c) 2025 Michael Ivertowski
// Licensed under the Apache License, Version 2.0. See LICENSE file for details.

using DotCompute.Abstractions;
using Microsoft.Extensions.Logging;
using Orleans.GpuBridge.Abstractions.Enums;
using Orleans.GpuBridge.Abstractions.Providers;
using Orleans.GpuBridge.Abstractions.Models;

namespace Orleans.GpuBridge.Backends.DotCompute.DeviceManagement;

/// <summary>
/// Adapts DotCompute's IAccelerator to Orleans.GpuBridge's IComputeDevice interface
/// </summary>
/// <remarks>
/// This adapter wraps a DotCompute IAccelerator and presents it through the
/// Orleans.GpuBridge.Abstractions.IComputeDevice interface for seamless integration.
///
/// The adapter handles:
/// - Device identity mapping (DotCompute ID → Orleans.GpuBridge ID)
/// - Property translation (AcceleratorInfo → IComputeDevice properties)
/// - Memory management coordination
/// - Health monitoring and metrics
/// - Lifecycle management (initialization, reset, disposal)
/// </remarks>
internal sealed class DotComputeAcceleratorAdapter : IComputeDevice
{
    private readonly IAccelerator _accelerator;
    private readonly ILogger _logger;
    private readonly int _index;
    private bool _disposed;

    /// <summary>
    /// Initializes a new adapter wrapping a DotCompute accelerator
    /// </summary>
    /// <param name="accelerator">The DotCompute accelerator to wrap</param>
    /// <param name="index">Device index for ID generation</param>
    /// <param name="logger">Logger for diagnostics</param>
    public DotComputeAcceleratorAdapter(
        IAccelerator accelerator,
        int index,
        ILogger logger)
    {
        _accelerator = accelerator ?? throw new ArgumentNullException(nameof(accelerator));
        _index = index;
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));

        _logger.LogDebug(
            "Created DotComputeAcceleratorAdapter for device: {DeviceName} (Index: {Index})",
            accelerator.Info.Name,
            index);
    }

    #region IComputeDevice Core Properties

    /// <inheritdoc />
    public string Id => $"dotcompute-{Type.ToString().ToLowerInvariant()}-{_index}";

    /// <inheritdoc />
    public string DeviceId => Id;

    /// <inheritdoc />
    public int Index => _index;

    /// <inheritdoc />
    public string Name => _accelerator.Info.Name;

    /// <inheritdoc />
    public string Vendor => _accelerator.Info.Vendor ?? "DotCompute";

    /// <inheritdoc />
    public DeviceType Type => MapAcceleratorType(_accelerator.Info.Type);

    /// <inheritdoc />
    public int ComputeUnits => _accelerator.Info.ComputeUnits;

    /// <inheritdoc />
    public int MaxWorkGroupSize => _accelerator.Info.MaxWorkGroupSize;

    /// <inheritdoc />
    public long MaxMemoryBytes => _accelerator.Info.TotalMemory;

    /// <inheritdoc />
    public long TotalMemoryBytes => _accelerator.Info.TotalMemory;

    /// <inheritdoc />
    public long AvailableMemoryBytes => (long)(_accelerator.Memory.TotalAvailableMemory * 0.8);

    #endregion

    #region IComputeDevice Extended Properties

    /// <inheritdoc />
    public bool IsHealthy { get; private set; } = true;

    /// <inheritdoc />
    public string? LastError { get; private set; }

    /// <inheritdoc />
    public string Architecture => _accelerator.Info.Architecture ?? "Unknown";

    /// <inheritdoc />
    public int WarpSize => _accelerator.Info.WarpSize;

    /// <inheritdoc />
    public Version ComputeCapability => new(
        _accelerator.Info.MajorVersion,
        _accelerator.Info.MinorVersion);

    /// <inheritdoc />
    public int MaxClockFrequencyMHz => Type == DeviceType.GPU ? 1500 : 3000;

    /// <inheritdoc />
    public int MaxThreadsPerBlock => MaxWorkGroupSize;

    /// <inheritdoc />
    public int[] MaxWorkGroupDimensions => Type == DeviceType.GPU
        ? new[] { 1024, 1024, 64 }
        : new[] { 256, 1, 1 };

    /// <inheritdoc />
    public IReadOnlyDictionary<string, object> Properties => new Dictionary<string, object>
    {
        ["device_type"] = Type.ToString(),
        ["compute_units"] = ComputeUnits,
        ["max_work_group_size"] = MaxWorkGroupSize,
        ["architecture"] = Architecture,
        ["warp_size"] = WarpSize,
        ["extensions"] = _accelerator.Info.Extensions ?? Array.Empty<string>()
    };

    #endregion

    #region IComputeDevice Status Methods

    /// <inheritdoc />
    public bool SupportsFeature(string feature)
    {
        return feature switch
        {
            "double_precision" => Type == DeviceType.GPU,
            "shared_memory" => Type == DeviceType.GPU,
            "async_execution" => true,
            "memory_coalescing" => Type == DeviceType.GPU,
            _ => false
        };
    }

    /// <inheritdoc />
    public DeviceStatus GetStatus()
    {
        if (_disposed)
            return DeviceStatus.Error;

        if (!IsHealthy)
            return DeviceStatus.Error;

        // DotCompute v0.3.0-rc1: AcceleratorInfo doesn't have IsAvailable property
        // Assume device is available if we can access its info
        return DeviceStatus.Available;
    }

    /// <summary>
    /// Sets the health status of the device
    /// </summary>
    public void SetHealthStatus(bool isHealthy, string? error = null)
    {
        IsHealthy = isHealthy;
        LastError = error;

        if (!isHealthy && !string.IsNullOrEmpty(error))
        {
            _logger.LogWarning("Device {DeviceId} health status changed: {Error}", Id, error);
        }
    }

    #endregion

    #region Internal Helper Methods

    /// <summary>
    /// Gets memory usage information for this device
    /// </summary>
    /// <remarks>
    /// Internal method used by DotComputeDeviceManager for metrics gathering
    /// </remarks>
    internal GpuMemoryInfo GetMemoryInfo()
    {
        try
        {
            var memoryManager = _accelerator.Memory;
            var stats = memoryManager.Statistics;

            return GpuMemoryInfo.Create(
                totalBytes: memoryManager.TotalAvailableMemory,
                allocatedBytes: memoryManager.CurrentAllocatedMemory,
                deviceIndex: Index,
                deviceName: Name);
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Failed to get memory info for device {DeviceId}", Id);
            return GpuMemoryInfo.Empty;
        }
    }

    #endregion

    #region Lifetime Management

    /// <inheritdoc />
    public void Dispose()
    {
        if (_disposed)
            return;

        _logger.LogDebug("Disposing DotComputeAcceleratorAdapter for device {DeviceId}", Id);

        // DotCompute IAccelerator doesn't implement IDisposable in v0.3.0-rc1
        // Cleanup is handled by IAcceleratorManager.DisposeAsync()
        // We just mark ourselves as disposed

        _disposed = true;
    }

    #endregion

    #region Helper Methods

    /// <summary>
    /// Maps DotCompute accelerator type string to Orleans.GpuBridge DeviceType
    /// </summary>
    /// <remarks>
    /// DotCompute v0.3.0-rc1: AcceleratorInfo.Type returns a string, not an enum
    /// </remarks>
    private static DeviceType MapAcceleratorType(string acceleratorTypeString)
    {
        return acceleratorTypeString?.ToUpperInvariant() switch
        {
            "GPU" => DeviceType.GPU,
            "CPU" => DeviceType.CPU,
            _ => DeviceType.GPU // Default to GPU for unknown types
        };
    }

    #endregion

    #region Native Accelerator Access

    /// <summary>
    /// Gets the underlying DotCompute accelerator for advanced scenarios
    /// </summary>
    /// <remarks>
    /// Internal access for kernel compilation and execution that needs
    /// direct DotCompute API access
    /// </remarks>
    internal IAccelerator Accelerator => _accelerator;

    #endregion
}

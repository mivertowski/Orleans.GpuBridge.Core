using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Orleans.GpuBridge.Abstractions.Enums;
using Orleans.GpuBridge.Abstractions.Providers;

namespace Orleans.GpuBridge.Backends.DotCompute.DeviceManagement;

/// <summary>
/// DotCompute device manager implementation
/// </summary>
internal sealed class DotComputeDeviceManager : IDeviceManager
{
    private readonly ILogger<DotComputeDeviceManager> _logger;
    private readonly ConcurrentDictionary<string, DotComputeComputeDevice> _devices;
    private bool _initialized;
    private bool _disposed;

    public DotComputeDeviceManager(ILogger<DotComputeDeviceManager> logger)
    {
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        _devices = new ConcurrentDictionary<string, DotComputeComputeDevice>();
    }

    public async Task InitializeAsync(CancellationToken cancellationToken = default)
    {
        if (_initialized)
            return;

        try
        {
            _logger.LogInformation("Initializing DotCompute device manager");

            await DiscoverDevicesAsync(cancellationToken);

            _initialized = true;
            _logger.LogInformation("DotCompute device manager initialized with {DeviceCount} devices", _devices.Count);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to initialize DotCompute device manager");
            throw;
        }
    }

    public IEnumerable<IComputeDevice> GetDevices()
    {
        EnsureInitialized();
        return _devices.Values;
    }

    public IComputeDevice GetDefaultDevice()
    {
        EnsureInitialized();
        
        // Prefer GPU devices over CPU
        var gpuDevice = _devices.Values.FirstOrDefault(d => d.Type != DeviceType.Cpu);
        if (gpuDevice != null)
        {
            return gpuDevice;
        }

        // Fallback to any available device
        var firstDevice = _devices.Values.FirstOrDefault();
        if (firstDevice == null)
        {
            throw new InvalidOperationException("No compute devices available");
        }

        return firstDevice;
    }

    public IComputeDevice? GetDevice(string deviceId)
    {
        EnsureInitialized();
        _devices.TryGetValue(deviceId, out var device);
        return device;
    }

    public IEnumerable<IComputeDevice> GetDevicesByType(DeviceType deviceType)
    {
        EnsureInitialized();
        return _devices.Values.Where(d => d.Type == deviceType);
    }

    public async Task<DeviceHealthInfo> GetDeviceHealthAsync(string deviceId, CancellationToken cancellationToken = default)
    {
        EnsureInitialized();
        
        if (!_devices.TryGetValue(deviceId, out var device))
        {
            throw new ArgumentException($"Device not found: {deviceId}", nameof(deviceId));
        }

        try
        {
            // For DotCompute, we would query actual device health
            // This is a simplified implementation
            var isHealthy = device.IsHealthy;
            var memoryUsage = await GetDeviceMemoryUsageAsync(device, cancellationToken);
            var temperature = await GetDeviceTemperatureAsync(device, cancellationToken);

            return new DeviceHealthInfo(
                IsHealthy: isHealthy,
                MemoryUsagePercent: memoryUsage,
                TemperatureCelsius: temperature,
                LastError: device.LastError,
                ExtendedInfo: new Dictionary<string, object>
                {
                    ["device_type"] = device.Type.ToString(),
                    ["compute_units"] = device.ComputeUnits,
                    ["max_work_group_size"] = device.MaxWorkGroupSize
                });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to get health info for device {DeviceId}", deviceId);
            throw;
        }
    }

    private async Task DiscoverDevicesAsync(CancellationToken cancellationToken)
    {
        // Simulate device discovery - in real implementation this would use DotCompute APIs
        
        // Add a simulated GPU device
        var gpuDevice = new DotComputeComputeDevice(
            id: "dotcompute-gpu-0",
            name: "DotCompute GPU Device 0",
            type: DeviceType.Gpu,
            computeUnits: 32,
            maxWorkGroupSize: 1024,
            maxMemoryBytes: 8L * 1024 * 1024 * 1024, // 8GB
            logger: _logger);

        _devices["dotcompute-gpu-0"] = gpuDevice;

        // Add a CPU fallback device
        var cpuDevice = new DotComputeComputeDevice(
            id: "dotcompute-cpu-0",
            name: "DotCompute CPU Device",
            type: DeviceType.Cpu,
            computeUnits: Environment.ProcessorCount,
            maxWorkGroupSize: 256,
            maxMemoryBytes: GC.GetTotalMemory(false),
            logger: _logger);

        _devices["dotcompute-cpu-0"] = cpuDevice;

        _logger.LogInformation("Discovered {DeviceCount} DotCompute devices", _devices.Count);
    }

    private async Task<double> GetDeviceMemoryUsageAsync(DotComputeComputeDevice device, CancellationToken cancellationToken)
    {
        // Simulate memory usage check
        return await Task.FromResult(45.0); // 45% usage
    }

    private async Task<double?> GetDeviceTemperatureAsync(DotComputeComputeDevice device, CancellationToken cancellationToken)
    {
        // GPU devices might have temperature sensors
        if (device.Type == DeviceType.Gpu)
        {
            return await Task.FromResult(65.0); // 65°C
        }

        return null; // CPU devices typically don't report temperature through DotCompute
    }

    private void EnsureInitialized()
    {
        if (!_initialized)
            throw new InvalidOperationException("DotCompute device manager not initialized");
    }

    public void Dispose()
    {
        if (_disposed)
            return;

        _logger.LogDebug("Disposing DotCompute device manager");

        try
        {
            foreach (var device in _devices.Values)
            {
                try
                {
                    device.Dispose();
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Error disposing device {DeviceId}", device.Id);
                }
            }

            _devices.Clear();
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error disposing DotCompute device manager");
        }

        _disposed = true;
    }
}

/// <summary>
/// DotCompute compute device implementation
/// </summary>
internal sealed class DotComputeComputeDevice : IComputeDevice
{
    private readonly ILogger _logger;
    private bool _disposed;

    public string Id { get; }
    public string Name { get; }
    public DeviceType Type { get; }
    public int ComputeUnits { get; }
    public int MaxWorkGroupSize { get; }
    public long MaxMemoryBytes { get; }
    public bool IsHealthy { get; private set; } = true;
    public string? LastError { get; private set; }

    public DotComputeComputeDevice(
        string id,
        string name,
        DeviceType type,
        int computeUnits,
        int maxWorkGroupSize,
        long maxMemoryBytes,
        ILogger logger)
    {
        Id = id ?? throw new ArgumentNullException(nameof(id));
        Name = name ?? throw new ArgumentNullException(nameof(name));
        Type = type;
        ComputeUnits = computeUnits;
        MaxWorkGroupSize = maxWorkGroupSize;
        MaxMemoryBytes = maxMemoryBytes;
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
    }

    public void SetHealthStatus(bool isHealthy, string? error = null)
    {
        IsHealthy = isHealthy;
        LastError = error;

        if (!isHealthy && !string.IsNullOrEmpty(error))
        {
            _logger.LogWarning("Device {DeviceId} marked as unhealthy: {Error}", Id, error);
        }
    }

    public void Dispose()
    {
        if (_disposed)
            return;

        _logger.LogDebug("Disposing DotCompute device {DeviceId}", Id);
        
        // In a real implementation, we would release DotCompute device resources
        
        _disposed = true;
    }
}

/// <summary>
/// DotCompute command queue implementation
/// </summary>
internal sealed class DotComputeCommandQueue : ICommandQueue
{
    private readonly IComputeDevice _device;
    private readonly ILogger _logger;
    private bool _disposed;

    public IComputeContext Context { get; }

    public DotComputeCommandQueue(IComputeDevice device, ILogger logger)
    {
        _device = device ?? throw new ArgumentNullException(nameof(device));
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        Context = new DotComputeComputeContext(device, logger);
    }

    public Task FlushAsync(CancellationToken cancellationToken = default)
    {
        // In DotCompute, this would flush pending commands
        return Task.CompletedTask;
    }

    public Task SynchronizeAsync(CancellationToken cancellationToken = default)
    {
        // In DotCompute, this would wait for all commands to complete
        return Task.CompletedTask;
    }

    public void Dispose()
    {
        if (_disposed)
            return;

        _logger.LogDebug("Disposing DotCompute command queue");
        Context?.Dispose();
        _disposed = true;
    }
}

/// <summary>
/// DotCompute compute context implementation
/// </summary>
internal sealed class DotComputeComputeContext : IComputeContext
{
    private readonly ILogger _logger;
    private bool _disposed;

    public IComputeDevice Device { get; }

    public DotComputeComputeContext(IComputeDevice device, ILogger logger)
    {
        Device = device ?? throw new ArgumentNullException(nameof(device));
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
    }

    public void Dispose()
    {
        if (_disposed)
            return;

        _logger.LogDebug("Disposing DotCompute context for device {DeviceId}", Device.Id);
        _disposed = true;
    }
}
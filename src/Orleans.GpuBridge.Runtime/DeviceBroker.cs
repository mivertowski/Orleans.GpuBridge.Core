using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using Orleans.GpuBridge.Abstractions;

namespace Orleans.GpuBridge.Runtime;

/// <summary>
/// Manages GPU devices and work distribution
/// </summary>
public sealed class DeviceBroker
{
    private readonly ILogger<DeviceBroker> _logger;
    private readonly GpuBridgeOptions _options;
    private readonly List<GpuDevice> _devices;
    private bool _initialized;
    
    public int DeviceCount => _devices.Count;
    public long TotalMemoryBytes => _devices.Sum(d => d.TotalMemoryBytes);
    public int CurrentQueueDepth => 0; // TODO: Implement queue tracking
    
    public DeviceBroker(
        ILogger<DeviceBroker> logger,
        IOptions<GpuBridgeOptions> options)
    {
        _logger = logger;
        _options = options.Value;
        _devices = new List<GpuDevice>();
    }
    
    public Task InitializeAsync(CancellationToken ct)
    {
        if (_initialized) return Task.CompletedTask;
        
        _logger.LogInformation("Initializing device broker");
        
        // For now, simulate CPU-only device
        var cpuDevice = new GpuDevice(
            Index: 0,
            Name: "CPU Fallback",
            Type: DeviceType.Cpu,
            TotalMemoryBytes: Environment.WorkingSet,
            AvailableMemoryBytes: Environment.WorkingSet / 2,
            ComputeUnits: Environment.ProcessorCount,
            Capabilities: new[] { "CPU", "Fallback" });
        
        _devices.Add(cpuDevice);
        
        // TODO: Detect real GPU devices when DotCompute is integrated
        if (_options.PreferGpu)
        {
            _logger.LogWarning("GPU requested but not available, using CPU fallback");
        }
        
        _initialized = true;
        _logger.LogInformation(
            "Device broker initialized with {Count} devices, {Memory:N0} bytes total memory",
            _devices.Count, TotalMemoryBytes);
        
        return Task.CompletedTask;
    }
    
    public Task ShutdownAsync(CancellationToken ct)
    {
        _logger.LogInformation("Shutting down device broker");
        _devices.Clear();
        _initialized = false;
        return Task.CompletedTask;
    }
    
    public IReadOnlyList<GpuDevice> GetDevices()
    {
        return _devices.AsReadOnly();
    }
    
    public GpuDevice? GetDevice(int index)
    {
        return _devices.FirstOrDefault(d => d.Index == index);
    }
    
    public GpuDevice? GetBestDevice()
    {
        return _devices
            .OrderByDescending(d => d.AvailableMemoryBytes)
            .ThenBy(d => d.Index)
            .FirstOrDefault();
    }
}

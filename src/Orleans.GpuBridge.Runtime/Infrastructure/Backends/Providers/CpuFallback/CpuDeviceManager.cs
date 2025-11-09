using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Orleans.GpuBridge.Abstractions;
using Orleans.GpuBridge.Abstractions.Enums;
using Orleans.GpuBridge.Abstractions.Models;
using Orleans.GpuBridge.Abstractions.Providers;
using Orleans.GpuBridge.Abstractions.Providers.Memory.Options;

namespace Orleans.GpuBridge.Runtime.Providers;

/// <summary>
/// CPU device manager for fallback provider
/// </summary>
internal sealed class CpuDeviceManager : IDeviceManager
{
    private readonly ILogger<CpuDeviceManager> _logger;
    private readonly List<IComputeDevice> _devices;

    public CpuDeviceManager(ILogger<CpuDeviceManager> logger)
    {
        _logger = logger;
        _devices = new List<IComputeDevice>
        {
            new CpuDevice()
        };
    }

    public Task InitializeAsync(CancellationToken cancellationToken = default) => Task.CompletedTask;

    public IReadOnlyList<IComputeDevice> GetDevices() => _devices;

    public IComputeDevice? GetDevice(int deviceIndex) =>
        deviceIndex == 0 ? _devices[0] : null;

    public IComputeDevice GetDefaultDevice() => _devices[0];

    public IComputeDevice SelectDevice(DeviceSelectionCriteria criteria) => _devices[0];

    public Task<IComputeContext> CreateContextAsync(
        IComputeDevice device,
        ContextOptions options,
        CancellationToken cancellationToken = default)
    {
        return Task.FromResult<IComputeContext>(new CpuContext(device));
    }

    public Task<Orleans.GpuBridge.Abstractions.Models.DeviceMetrics> GetDeviceMetricsAsync(
        IComputeDevice device,
        CancellationToken cancellationToken = default)
    {
        return Task.FromResult(new Orleans.GpuBridge.Abstractions.Models.DeviceMetrics
        {
            GpuUtilizationPercent = 0,
            MemoryUtilizationPercent = 50,
            UsedMemoryBytes = GC.GetTotalMemory(false),
            TemperatureCelsius = 45,
            PowerWatts = 0,
            FanSpeedPercent = 0,
            KernelsExecuted = 0,
            BytesTransferred = 0,
            Uptime = TimeSpan.FromMilliseconds(Environment.TickCount64)
        });
    }

    public Task ResetDeviceAsync(
        IComputeDevice device,
        CancellationToken cancellationToken = default) => Task.CompletedTask;

    public void Dispose() { }
}

/// <summary>
/// CPU device abstraction for fallback provider
/// </summary>
internal sealed class CpuDevice : IComputeDevice
{
    public string DeviceId => "cpu-0";
    public int Index => 0;
    public string Name => "CPU";
    public DeviceType Type => DeviceType.CPU;
    public string Vendor => "Generic";
    public string Architecture => "x86-64";
    public Version ComputeCapability => new(1, 0);
    public long TotalMemoryBytes => 8L * 1024 * 1024 * 1024; // 8GB
    public long AvailableMemoryBytes => 4L * 1024 * 1024 * 1024; // 4GB
    public int ComputeUnits => Environment.ProcessorCount;
    public int MaxClockFrequencyMHz => 3000;
    public int MaxThreadsPerBlock => 1024;
    public int[] MaxWorkGroupDimensions => new[] { 1024, 1024, 1024 };
    public int WarpSize => 1;
    public IReadOnlyDictionary<string, object> Properties => new Dictionary<string, object>();

    public bool SupportsFeature(string feature) => false;
    public DeviceStatus GetStatus() => DeviceStatus.Available;
}

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Orleans.GpuBridge.Abstractions;
using Orleans.GpuBridge.Abstractions.Enums;
using Orleans.GpuBridge.Abstractions.Models;

namespace Orleans.GpuBridge.Runtime;

/// <summary>
/// System detection helper methods for DeviceBroker
/// </summary>
public sealed partial class DeviceBroker
{
    /// <summary>
    /// Gets CUDA device information from system APIs
    /// </summary>
    private async Task<List<CudaDeviceInfo>> GetCudaDevicesFromSystem(CancellationToken ct)
    {
        var devices = new List<CudaDeviceInfo>();

        try
        {
            // Try nvidia-smi for device enumeration
            var processInfo = new ProcessStartInfo
            {
                FileName = "nvidia-smi",
                Arguments = "--query-gpu=index,name,pci.bus_id,memory.total,compute_cap --format=csv,noheader,nounits",
                UseShellExecute = false,
                RedirectStandardOutput = true,
                CreateNoWindow = true
            };

            using var process = Process.Start(processInfo);
            if (process != null)
            {
                var output = await process.StandardOutput.ReadToEndAsync();
                await process.WaitForExitAsync(ct);

                if (process.ExitCode == 0)
                {
                    devices.AddRange(ParseNvidiaSmiOutput(output));
                }
            }
        }
        catch (Exception ex)
        {
            _logger.LogDebug(ex, "nvidia-smi not available, falling back to system detection");
        }

        // Fallback to system detection if nvidia-smi fails
        if (devices.Count == 0)
        {
            devices.AddRange(await DetectCudaDevicesManually(ct));
        }

        return devices;
    }

    /// <summary>
    /// Gets OpenCL device information from system
    /// </summary>
    private async Task<List<OpenClDeviceInfo>> GetOpenClDevicesFromSystem(CancellationToken ct)
    {
        var devices = new List<OpenClDeviceInfo>();

        try
        {
            // Use clinfo if available
            var processInfo = new ProcessStartInfo
            {
                FileName = "clinfo",
                Arguments = "-l",
                UseShellExecute = false,
                RedirectStandardOutput = true,
                CreateNoWindow = true
            };

            using var process = Process.Start(processInfo);
            if (process != null)
            {
                var output = await process.StandardOutput.ReadToEndAsync();
                await process.WaitForExitAsync(ct);

                if (process.ExitCode == 0)
                {
                    devices.AddRange(ParseCliInfoOutput(output));
                }
            }
        }
        catch (Exception ex)
        {
            _logger.LogDebug(ex, "clinfo not available, using manual detection");
        }

        // Fallback to manual detection
        if (devices.Count == 0)
        {
            devices.AddRange(await DetectOpenClDevicesManually(ct));
        }

        return devices;
    }

    /// <summary>
    /// Gets Intel Level Zero devices
    /// </summary>
    private async Task<List<GpuDevice>> GetLevelZeroDevices(CancellationToken ct)
    {
        var devices = new List<GpuDevice>();

        // This would require Intel Level Zero API integration
        // For now, return empty list and fall back to OpenCL
        await Task.CompletedTask;

        return devices;
    }

    /// <summary>
    /// Gets Metal device information (macOS only)
    /// </summary>
    private async Task<List<MetalDeviceInfo>> GetMetalDevicesFromSystem(CancellationToken ct)
    {
        var devices = new List<MetalDeviceInfo>();

        if (!OperatingSystem.IsMacOS())
        {
            return devices;
        }

        try
        {
            // Use system_profiler on macOS
            var processInfo = new ProcessStartInfo
            {
                FileName = "system_profiler",
                Arguments = "SPDisplaysDataType -json",
                UseShellExecute = false,
                RedirectStandardOutput = true,
                CreateNoWindow = true
            };

            using var process = Process.Start(processInfo);
            if (process != null)
            {
                var output = await process.StandardOutput.ReadToEndAsync();
                await process.WaitForExitAsync(ct);

                if (process.ExitCode == 0)
                {
                    devices.AddRange(ParseMetalDevicesFromJson(output));
                }
            }
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Failed to detect Metal devices through system_profiler");
        }

        return devices;
    }

    /// <summary>
    /// Comprehensive device health check
    /// </summary>
    private async Task<DeviceHealthInfo> CheckDeviceHealth(GpuDevice device, CancellationToken ct)
    {
        var health = new DeviceHealthInfo
        {
            DeviceId = device.Id,
            LastCheckTime = DateTime.UtcNow
        };

        try
        {
            // Check device availability
            if (!await IsDeviceAccessible(device, ct))
            {
                return health;
            }

            // Get temperature and power info
            var thermalInfo = await GetDeviceThermalInfo(device, ct);
            health = health with
            {
                TemperatureCelsius = thermalInfo.Temperature,
                PowerUsageWatts = thermalInfo.PowerUsage
            };

            // Get memory utilization
            health = health with { MemoryUtilizationPercent = await GetDeviceMemoryUtilization(device, ct) };

            // Health thresholds
            if (health.TemperatureCelsius > 95)
            {
                return health;
            }
            else if (health.MemoryUtilizationPercent > 98.0)
            {
                return health;
            }
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Health check failed for device {DeviceId}", device.Index);
            return health;
        }

        return health;
    }

    /// <summary>
    /// Calculate current device load metrics
    /// </summary>
    private async Task<DeviceLoadInfo> CalculateDeviceLoad(GpuDevice device, CancellationToken ct)
    {
        var loadInfo = new DeviceLoadInfo
        {
            DeviceId = device.Id,
            LastUpdateTime = DateTime.UtcNow
        };

        try
        {
            // Get current utilization
            loadInfo = loadInfo with { CurrentUtilization = await GetDeviceUtilization(device, ct) };

            // Get queue depth from active work items
            loadInfo = loadInfo with
            {
                QueueDepth = _workQueues.TryGetValue(device.Index, out var queue)
                    ? queue.QueuedItems : 0
            };

            // Calculate selection weight based on load
            loadInfo = loadInfo with
            {
                SelectionWeight = CalculateSelectionWeight(loadInfo.CurrentUtilization, loadInfo.QueueDepth)
            };

            // Update performance history
            UpdatePerformanceHistory(loadInfo, loadInfo.CurrentUtilization);
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Load calculation failed for device {DeviceId}", device.Index);
            loadInfo = loadInfo with
            {
                CurrentUtilization = 100.0, // Assume busy if we can't measure
                SelectionWeight = 0.1
            };
        }

        return loadInfo;
    }

    /// <summary>
    /// Determine OpenCL device type from vendor and type string
    /// </summary>
    private DeviceType DetermineOpenClDeviceType(string vendor, string type)
    {
        vendor = vendor.ToLowerInvariant();
        type = type.ToLowerInvariant();

        if (vendor.Contains("nvidia") || vendor.Contains("cuda"))
            return DeviceType.CUDA;

        if (vendor.Contains("amd") || vendor.Contains("advanced micro"))
            return DeviceType.OpenCL;

        if (vendor.Contains("intel"))
            return DeviceType.OpenCL;

        if (type.Contains("cpu"))
            return DeviceType.CPU;

        return DeviceType.OpenCL;
    }

    /// <summary>
    /// Parse nvidia-smi output
    /// </summary>
    private List<CudaDeviceInfo> ParseNvidiaSmiOutput(string output)
    {
        var devices = new List<CudaDeviceInfo>();
        var lines = output.Split('\n', StringSplitOptions.RemoveEmptyEntries);

        foreach (var line in lines)
        {
            var parts = line.Split(',').Select(p => p.Trim()).ToArray();
            if (parts.Length >= 5 && int.TryParse(parts[0], out var index))
            {
                devices.Add(new CudaDeviceInfo
                {
                    Index = index,
                    Name = parts[1],
                    PciBusId = parts[2],
                    TotalMemory = long.TryParse(parts[3], out var mem) ? mem * 1024 * 1024 : 0,
                    ComputeCapabilityMajor = int.TryParse(parts[4].Split('.')[0], out var major) ? major : 1,
                    ComputeCapabilityMinor = int.TryParse(parts[4].Split('.')[1], out var minor) ? minor : 0
                });
            }
        }

        return devices;
    }

    // Helper method stubs for completion
    private Task<List<CudaDeviceInfo>> DetectCudaDevicesManually(CancellationToken ct) =>
        Task.FromResult(new List<CudaDeviceInfo>());

    private Task<List<OpenClDeviceInfo>> DetectOpenClDevicesManually(CancellationToken ct) =>
        Task.FromResult(new List<OpenClDeviceInfo>());

    private List<OpenClDeviceInfo> ParseCliInfoOutput(string output) => new();

    private List<MetalDeviceInfo> ParseMetalDevicesFromJson(string json) => new();

    private Task<bool> IsDeviceAccessible(GpuDevice device, CancellationToken ct) =>
        Task.FromResult(true);

    private async Task<(int Temperature, double PowerUsage)> GetDeviceThermalInfo(GpuDevice device, CancellationToken ct)
    {
        // Placeholder implementation - would use device-specific APIs
        await Task.CompletedTask;
        return (45, 50.0);
    }

    private Task<double> GetDeviceMemoryUtilization(GpuDevice device, CancellationToken ct) =>
        Task.FromResult(25.0);

    private Task<double> GetDeviceUtilization(GpuDevice device, CancellationToken ct) =>
        Task.FromResult(30.0);
}

/// <summary>
/// CUDA device information structure
/// </summary>
public sealed class CudaDeviceInfo
{
    /// <summary>Gets or sets the device index</summary>
    public int Index { get; set; }

    /// <summary>Gets or sets the device name</summary>
    public string Name { get; set; } = "";

    /// <summary>Gets or sets the PCI bus ID</summary>
    public string PciBusId { get; set; } = "";

    /// <summary>Gets or sets the total memory in bytes</summary>
    public long TotalMemory { get; set; }

    /// <summary>Gets or sets the compute capability major version</summary>
    public int ComputeCapabilityMajor { get; set; }

    /// <summary>Gets or sets the compute capability minor version</summary>
    public int ComputeCapabilityMinor { get; set; }

    /// <summary>Gets or sets the maximum threads per block</summary>
    public int MaxThreadsPerBlock { get; set; } = 1024;

    /// <summary>Gets or sets the shared memory per block in bytes</summary>
    public int SharedMemoryPerBlock { get; set; } = 49152;

    /// <summary>Gets or sets the warp size</summary>
    public int WarpSize { get; set; } = 32;

    /// <summary>Gets or sets the maximum grid size</summary>
    public int MaxGridSize { get; set; } = 65536;

    /// <summary>Gets or sets the clock rate in kHz</summary>
    public int ClockRate { get; set; } = 1000000;

    /// <summary>Gets or sets the memory clock rate in kHz</summary>
    public int MemoryClockRate { get; set; } = 2000000;
}

/// <summary>
/// OpenCL device information structure
/// </summary>
public sealed class OpenClDeviceInfo
{
    /// <summary>Gets or sets the platform index</summary>
    public int PlatformIndex { get; set; }

    /// <summary>Gets or sets the device index</summary>
    public int DeviceIndex { get; set; }

    /// <summary>Gets or sets the device name</summary>
    public string Name { get; set; } = "";

    /// <summary>Gets or sets the device vendor</summary>
    public string Vendor { get; set; } = "";

    /// <summary>Gets or sets the device type</summary>
    public string Type { get; set; } = "";

    /// <summary>Gets or sets the PCI bus ID</summary>
    public string PciBusId { get; set; } = "";

    /// <summary>Gets or sets the global memory size in bytes</summary>
    public long GlobalMemorySize { get; set; }

    /// <summary>Gets or sets the local memory size in bytes</summary>
    public long LocalMemorySize { get; set; }

    /// <summary>Gets or sets the maximum work group size</summary>
    public long MaxWorkGroupSize { get; set; }

    /// <summary>Gets or sets the maximum clock frequency in MHz</summary>
    public long MaxClockFrequency { get; set; }

    /// <summary>Gets or sets the supported extensions</summary>
    public string Extensions { get; set; } = "";
}

/// <summary>
/// Metal device information structure (macOS only)
/// </summary>
public sealed class MetalDeviceInfo
{
    /// <summary>Gets or sets the device index</summary>
    public int Index { get; set; }

    /// <summary>Gets or sets the device name</summary>
    public string Name { get; set; } = "";

    /// <summary>Gets or sets the recommended maximum working set size in bytes</summary>
    public long RecommendedMaxWorkingSetSize { get; set; }

    /// <summary>Gets or sets whether this is Apple Silicon</summary>
    public bool IsAppleSilicon { get; set; }
}
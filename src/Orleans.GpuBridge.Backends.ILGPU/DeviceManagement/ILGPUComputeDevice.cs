using System;
using System.Collections.Generic;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using ILGPU.Runtime.OpenCL;
using ILGPU.Runtime.CPU;
using Microsoft.Extensions.Logging;
using Orleans.GpuBridge.Abstractions.Providers;

namespace Orleans.GpuBridge.Backends.ILGPU.DeviceManagement;

/// <summary>
/// ILGPU compute device wrapper
/// </summary>
internal sealed class ILGPUComputeDevice : IComputeDevice, IDisposable
{
    private readonly ILogger _logger;
    private readonly Dictionary<string, object> _properties;
    private bool _disposed;

    public Accelerator Accelerator { get; }

    public string DeviceId { get; }
    public int Index { get; }
    public string Name => Accelerator.Name;
    public DeviceType Type { get; }
    public string Vendor { get; }
    public string Architecture { get; }
    public Version ComputeCapability { get; }
    public long TotalMemoryBytes => Accelerator.MemoryInfo.TotalMemory;
    public long AvailableMemoryBytes => Accelerator.MemoryInfo.AvailableMemory;
    public int ComputeUnits { get; }
    public int MaxClockFrequencyMHz { get; }
    public int MaxThreadsPerBlock { get; }
    public int[] MaxWorkGroupDimensions { get; }
    public int WarpSize { get; }
    public IReadOnlyDictionary<string, object> Properties => _properties;

    public ILGPUComputeDevice(int index, Accelerator accelerator, ILogger logger)
    {
        Index = index;
        Accelerator = accelerator ?? throw new ArgumentNullException(nameof(accelerator));
        _logger = logger;
        
        DeviceId = $"ilgpu-{index}-{accelerator.AcceleratorType}";
        Type = MapAcceleratorType(accelerator.AcceleratorType);
        Vendor = DetermineVendor(accelerator);
        Architecture = DetermineArchitecture(accelerator);
        ComputeCapability = DetermineComputeCapability(accelerator);
        ComputeUnits = DetermineComputeUnits(accelerator);
        MaxClockFrequencyMHz = DetermineMaxClockFrequency(accelerator);
        MaxThreadsPerBlock = DetermineMaxThreadsPerBlock(accelerator);
        MaxWorkGroupDimensions = DetermineMaxWorkGroupDimensions(accelerator);
        WarpSize = DetermineWarpSize(accelerator);
        
        _properties = BuildProperties(accelerator);
    }

    public bool SupportsFeature(string feature)
    {
        return feature.ToLowerInvariant() switch
        {
            "shared-memory" => Accelerator.AcceleratorType != AcceleratorType.CPU,
            "atomics" => true, // ILGPU supports atomics on all accelerators
            "warp-shuffle" => Accelerator.AcceleratorType == AcceleratorType.Cuda,
            "local-memory" => Accelerator.AcceleratorType != AcceleratorType.CPU,
            "64bit-global-memory" => true,
            "unified-memory" => Accelerator is CudaAccelerator,
            "profiling" => true,
            "debug" => true,
            "barriers" => true,
            "group-operations" => Accelerator.AcceleratorType != AcceleratorType.CPU,
            _ => _properties.ContainsKey($"feature_{feature}")
        };
    }

    public DeviceStatus GetStatus()
    {
        try
        {
            if (_disposed || Accelerator.IsDisposed)
                return DeviceStatus.Offline;

            // Try to query memory info to check if device is responsive
            _ = Accelerator.MemoryInfo;
            return DeviceStatus.Available;
        }
        catch (Exception ex)
        {
            _logger.LogDebug(ex, "Device {DeviceName} appears to be offline", Name);
            return DeviceStatus.Error;
        }
    }

    private static DeviceType MapAcceleratorType(AcceleratorType acceleratorType)
    {
        return acceleratorType switch
        {
            AcceleratorType.CPU => DeviceType.Cpu,
            AcceleratorType.Cuda => DeviceType.Cuda,
            AcceleratorType.OpenCL => DeviceType.OpenCl,
            _ => DeviceType.Custom
        };
    }

    private static string DetermineVendor(Accelerator accelerator)
    {
        return accelerator switch
        {
            CudaAccelerator => "NVIDIA",
            CLAccelerator clAcc => clAcc.Vendor ?? "Unknown",
            CPUAccelerator => Environment.Is64BitProcess ? "x64" : "x86",
            _ => "Unknown"
        };
    }

    private static string DetermineArchitecture(Accelerator accelerator)
    {
        return accelerator switch
        {
            CudaAccelerator cuda => $"CUDA SM {cuda.CudaArchitecture.Major}.{cuda.CudaArchitecture.Minor}",
            CLAccelerator cl => cl.Name?.Contains("NVIDIA") == true ? "CUDA via OpenCL" : 
                              cl.Name?.Contains("AMD") == true ? "GCN/RDNA" : 
                              cl.Name?.Contains("Intel") == true ? "Intel GPU" : "Unknown OpenCL",
            CPUAccelerator => Environment.Is64BitProcess ? "x64" : "x86",
            _ => "Unknown"
        };
    }

    private static Version DetermineComputeCapability(Accelerator accelerator)
    {
        return accelerator switch
        {
            CudaAccelerator cuda => new Version(cuda.CudaArchitecture.Major, cuda.CudaArchitecture.Minor),
            CLAccelerator => new Version(2, 0), // Assume OpenCL 2.0
            CPUAccelerator => new Version(1, 0),
            _ => new Version(1, 0)
        };
    }

    private static int DetermineComputeUnits(Accelerator accelerator)
    {
        return accelerator switch
        {
            CudaAccelerator cuda => cuda.NumMultiprocessors,
            CLAccelerator cl when cl.MaxNumComputeUnits.HasValue => cl.MaxNumComputeUnits.Value,
            CPUAccelerator => Environment.ProcessorCount,
            _ => 1
        };
    }

    private static int DetermineMaxClockFrequency(Accelerator accelerator)
    {
        return accelerator switch
        {
            CudaAccelerator cuda => cuda.ClockRate / 1000, // Convert from kHz to MHz
            CLAccelerator cl when cl.MaxClockFrequency.HasValue => cl.MaxClockFrequency.Value,
            CPUAccelerator => 3000, // Assume 3 GHz for CPU
            _ => 1000
        };
    }

    private static int DetermineMaxThreadsPerBlock(Accelerator accelerator)
    {
        return accelerator switch
        {
            CudaAccelerator cuda => cuda.MaxNumThreadsPerGroup,
            CLAccelerator cl when cl.MaxGroupSize.HasValue => (int)cl.MaxGroupSize.Value.Size,
            CPUAccelerator => 1024, // Virtual limit for CPU
            _ => 256
        };
    }

    private static int[] DetermineMaxWorkGroupDimensions(Accelerator accelerator)
    {
        return accelerator switch
        {
            CudaAccelerator cuda => new[] 
            { 
                cuda.MaxGridSize.X, 
                cuda.MaxGridSize.Y, 
                cuda.MaxGridSize.Z 
            },
            CLAccelerator cl when cl.MaxGroupSize.HasValue => new[]
            {
                (int)cl.MaxGroupSize.Value.X,
                (int)cl.MaxGroupSize.Value.Y,
                (int)cl.MaxGroupSize.Value.Z
            },
            CPUAccelerator => new[] { 1024, 1024, 1024 },
            _ => new[] { 256, 256, 256 }
        };
    }

    private static int DetermineWarpSize(Accelerator accelerator)
    {
        return accelerator switch
        {
            CudaAccelerator cuda => cuda.WarpSize,
            CLAccelerator => 32, // Common warp size, varies by hardware
            CPUAccelerator => 1, // CPU doesn't have warps
            _ => 32
        };
    }

    private Dictionary<string, object> BuildProperties(Accelerator accelerator)
    {
        var props = new Dictionary<string, object>
        {
            ["accelerator_type"] = accelerator.AcceleratorType.ToString(),
            ["backend"] = accelerator.AcceleratorType.ToString(),
            ["memory_bandwidth_gb_per_sec"] = 0.0, // Would need device-specific info
            ["supports_double_precision"] = true, // ILGPU supports double precision
            ["supports_int64_atomics"] = accelerator.AcceleratorType != AcceleratorType.CPU
        };

        // Add accelerator-specific properties
        switch (accelerator)
        {
            case CudaAccelerator cuda:
                props["cuda_major"] = cuda.CudaArchitecture.Major;
                props["cuda_minor"] = cuda.CudaArchitecture.Minor;
                props["cuda_driver_version"] = cuda.DriverVersion;
                props["cuda_runtime_version"] = cuda.RuntimeVersion;
                props["multiprocessor_count"] = cuda.NumMultiprocessors;
                props["clock_rate_khz"] = cuda.ClockRate;
                props["memory_clock_rate_khz"] = cuda.MemoryClockRate;
                props["memory_bus_width"] = cuda.MemoryBusWidth;
                props["l2_cache_size"] = cuda.L2CacheSize;
                props["max_shared_memory_per_group"] = cuda.MaxSharedMemoryPerGroup;
                props["feature_shared_memory"] = true;
                props["feature_warp_shuffle"] = true;
                props["feature_unified_memory"] = true;
                break;

            case CLAccelerator cl:
                props["opencl_version"] = cl.CLVersion?.ToString() ?? "Unknown";
                props["opencl_c_version"] = cl.CLCVersion?.ToString() ?? "Unknown";
                if (cl.MaxNumComputeUnits.HasValue)
                    props["compute_units"] = cl.MaxNumComputeUnits.Value;
                if (cl.MaxClockFrequency.HasValue)
                    props["max_clock_frequency_mhz"] = cl.MaxClockFrequency.Value;
                if (cl.MaxGroupSize.HasValue)
                {
                    var groupSize = cl.MaxGroupSize.Value;
                    props["max_work_group_size"] = groupSize.Size;
                    props["max_work_item_dimensions"] = 3;
                }
                props["feature_shared_memory"] = true;
                props["feature_local_memory"] = true;
                break;

            case CPUAccelerator cpu:
                props["cpu_cores"] = Environment.ProcessorCount;
                props["is_64bit"] = Environment.Is64BitProcess;
                props["feature_simd"] = true;
                props["feature_parallel_execution"] = true;
                break;
        }

        return props;
    }

    public void Dispose()
    {
        if (_disposed)
            return;

        try
        {
            if (!Accelerator.IsDisposed)
            {
                Accelerator.Synchronize();
                Accelerator.Dispose();
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error disposing ILGPU compute device: {DeviceName}", Name);
        }

        _disposed = true;
    }
}
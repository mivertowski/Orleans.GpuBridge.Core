using Orleans.GpuBridge.Abstractions.Domain.ValueObjects;
using Orleans.GpuBridge.Abstractions.Enums;
using Orleans.GpuBridge.Abstractions.Models;
using DeviceFeatures = Orleans.GpuBridge.Abstractions.Models.DeviceFeatures;

namespace Orleans.GpuBridge.Abstractions;

/// <summary>
/// Represents a comprehensive GPU device with production-grade capabilities
/// </summary>
public sealed record GpuDevice(
    int Index,
    string Name,
    DeviceType Type,
    long TotalMemoryBytes,
    long AvailableMemoryBytes,
    int ComputeUnits,
    IReadOnlyList<string> Capabilities)
{
    /// <summary>
    /// Unique identifier for the device (derived from Index for compatibility)
    /// </summary>
    public string Id => $"device_{Index}";
    
    /// <summary>
    /// Current memory utilization as a percentage (0.0 to 1.0)
    /// </summary>
    public double MemoryUtilization => TotalMemoryBytes > 0 
        ? (TotalMemoryBytes - AvailableMemoryBytes) / (double)TotalMemoryBytes 
        : 0;

    /// <summary>
    /// Available memory in bytes (alias for consistency)
    /// </summary>
    public long MemoryBytes => TotalMemoryBytes;

    /// <summary>
    /// Maximum threads per block (estimated from device type and compute units)
    /// </summary>
    public int MaxThreadsPerBlock => Type switch
    {
        DeviceType.CUDA => 1024, // Modern CUDA devices
        DeviceType.OpenCL => 256, // Conservative OpenCL estimate
        DeviceType.Metal => 1024, // Metal threadgroup size
        DeviceType.DirectCompute => 1024, // DirectCompute thread group size
        DeviceType.CPU => Environment.ProcessorCount,
        _ => 256 // Conservative default
    };

    /// <summary>
    /// Maximum shared memory per block in bytes
    /// </summary>
    public long MaxSharedMemoryPerBlock => Type switch
    {
        DeviceType.CUDA => 48 * 1024, // 48KB for modern CUDA
        DeviceType.OpenCL => 32 * 1024, // 32KB typical OpenCL
        DeviceType.Metal => 32 * 1024, // Metal threadgroup memory
        DeviceType.DirectCompute => 32 * 1024,
        DeviceType.CPU => 1024 * 1024, // 1MB for CPU cache simulation
        _ => 16 * 1024 // Conservative default
    };

    /// <summary>
    /// Estimated clock rate in MHz
    /// </summary>
    public int ClockRateMHz => ExtractClockRateFromCapabilities() ?? EstimateClockRate();

    /// <summary>
    /// Warp/wavefront size for SIMD execution
    /// </summary>
    public int WarpSize => Type switch
    {
        DeviceType.CUDA => 32, // CUDA warp size
        DeviceType.OpenCL => 64, // AMD wavefront or NVIDIA warp
        DeviceType.Metal => 32, // Metal SIMD group size
        DeviceType.DirectCompute => 32, // DirectCompute wave size
        DeviceType.CPU => 8, // SIMD register width equivalent
        _ => 32 // Default SIMD width
    };

    /// <summary>
    /// Supported features flags
    /// </summary>
    public DeviceFeatures SupportedFeatures => ParseSupportedFeatures();

    /// <summary>
    /// Compute capability for CUDA devices (null for non-CUDA)
    /// </summary>
    public ComputeCapability? ComputeCapability => ExtractComputeCapability();

    /// <summary>
    /// Current device status
    /// </summary>
    public DeviceStatus Status { get; set; } = DeviceStatus.Available;

    /// <summary>
    /// Thermal information if available
    /// </summary>
    public ThermalInfo? ThermalInfo { get; init; }

    /// <summary>
    /// Performance metrics
    /// </summary>
    public PerformanceMetrics? PerformanceMetrics { get; init; }

    private int? ExtractClockRateFromCapabilities()
    {
        foreach (var capability in Capabilities ?? [])
        {
            if (capability.StartsWith("Clock:") && 
                capability.Contains("MHz") &&
                int.TryParse(capability.Split(':')[1].Replace("MHz", "").Trim(), out int clockRate))
            {
                return clockRate;
            }
        }
        return null;
    }

    private int EstimateClockRate()
    {
        return Type switch
        {
            DeviceType.CUDA => 1500, // Modern CUDA base clock
            DeviceType.OpenCL => 1200, // Typical GPU clock
            DeviceType.Metal => 1400, // Apple GPU estimates
            DeviceType.DirectCompute => 1300,
            DeviceType.CPU => 3000, // Modern CPU base clock
            _ => 1000 // Conservative default
        };
    }

    private DeviceFeatures ParseSupportedFeatures()
    {
        var features = DeviceFeatures.None;
        
        foreach (var capability in Capabilities ?? [])
        {
            var cap = capability.ToUpperInvariant();
            
            if (cap.Contains("TENSOR")) features |= DeviceFeatures.TensorCores;
            if (cap.Contains("RT") || cap.Contains("RAYTRACING")) features |= DeviceFeatures.RayTracing;
            if (cap.Contains("UNIFIED") || cap.Contains("UMA")) features |= DeviceFeatures.UnifiedMemory;
            if (cap.Contains("ATOMIC")) features |= DeviceFeatures.Atomics;
            if (cap.Contains("DOUBLE") || cap.Contains("FP64")) features |= DeviceFeatures.DoublePrecision;
            if (cap.Contains("SHARED")) features |= DeviceFeatures.SharedMemory;
            if (cap.Contains("DYNAMIC") || cap.Contains("PARALLEL")) features |= DeviceFeatures.DynamicParallelism;
            if (cap.Contains("MULTI") || cap.Contains("SLI") || cap.Contains("CROSSFIRE")) features |= DeviceFeatures.MultiGpu;
        }

        // Set baseline features based on device type
        features |= DeviceFeatures.Atomics | DeviceFeatures.SharedMemory;
        
        if (Type == DeviceType.CUDA || Type == DeviceType.OpenCL || Type == DeviceType.Metal)
        {
            features |= DeviceFeatures.DoublePrecision;
        }

        return features;
    }

    private ComputeCapability? ExtractComputeCapability()
    {
        if (Type != DeviceType.CUDA) return null;

        foreach (var capability in Capabilities ?? [])
        {
            if (capability.StartsWith("Compute ") && capability.Contains('.'))
            {
                var parts = capability.Replace("Compute ", "").Split('.');
                if (parts.Length == 2 && 
                    int.TryParse(parts[0], out int major) && 
                    int.TryParse(parts[1], out int minor))
                {
                    return new ComputeCapability(major, minor);
                }
            }
        }

        // Fallback estimation based on compute units for CUDA
        return ComputeUnits switch
        {
            > 10000 => new ComputeCapability(8, 6), // Ada Lovelace
            > 5000 => new ComputeCapability(8, 0),  // Ampere
            > 2000 => new ComputeCapability(7, 5),  // Turing
            > 1000 => new ComputeCapability(6, 1),  // Pascal
            _ => new ComputeCapability(5, 2)        // Maxwell fallback
        };
    }
}
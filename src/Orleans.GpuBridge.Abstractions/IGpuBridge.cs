using System.Threading;
using System.Threading.Tasks;

namespace Orleans.GpuBridge.Abstractions;

/// <summary>
/// Main interface for GPU bridge operations
/// </summary>
public interface IGpuBridge
{
    /// <summary>
    /// Gets information about the GPU bridge and available resources
    /// </summary>
    ValueTask<GpuBridgeInfo> GetInfoAsync(CancellationToken ct = default);
    
    /// <summary>
    /// Gets a kernel executor for the specified kernel ID
    /// </summary>
    ValueTask<IGpuKernel<TIn, TOut>> GetKernelAsync<TIn, TOut>(
        KernelId kernelId, 
        CancellationToken ct = default) 
        where TIn : notnull 
        where TOut : notnull;
    
    /// <summary>
    /// Gets the list of available GPU devices
    /// </summary>
    ValueTask<IReadOnlyList<GpuDevice>> GetDevicesAsync(CancellationToken ct = default);
}

/// <summary>
/// Information about the GPU bridge
/// </summary>
public sealed record GpuBridgeInfo(
    string Version,
    int DeviceCount,
    long TotalMemoryBytes,
    GpuBackend Backend,
    bool IsGpuAvailable,
    IReadOnlyDictionary<string, object>? Metadata = null);

/// <summary>
/// Supported GPU backends
/// </summary>
public enum GpuBackend
{
    Cpu,
    Cuda,
    OpenCL,
    DirectCompute,
    Metal,
    Vulkan
}

/// <summary>
/// Represents a GPU device
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
    public double MemoryUtilization => TotalMemoryBytes > 0 
        ? (TotalMemoryBytes - AvailableMemoryBytes) / (double)TotalMemoryBytes 
        : 0;
}

/// <summary>
/// Type of compute device
/// </summary>
public enum DeviceType
{
    Cpu,
    Gpu,
    Accelerator,
    Custom,
    Cuda,
    OpenCl,
    DirectCompute,
    Metal,
    Fpga,
    Asic
}
using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using Orleans.GpuBridge.Abstractions.Enums;

namespace Orleans.GpuBridge.Abstractions.Providers;

/// <summary>
/// Interface for managing compute devices in GPU backends
/// </summary>
public interface IDeviceManager : IDisposable
{
    /// <summary>
    /// Initializes the device manager and discovers available devices
    /// </summary>
    Task InitializeAsync(CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Gets all available compute devices
    /// </summary>
    IReadOnlyList<IComputeDevice> GetDevices();
    
    /// <summary>
    /// Gets a specific device by index
    /// </summary>
    IComputeDevice? GetDevice(int deviceIndex);
    
    /// <summary>
    /// Gets the default device for computation
    /// </summary>
    IComputeDevice GetDefaultDevice();
    
    /// <summary>
    /// Selects the best device based on requirements
    /// </summary>
    IComputeDevice SelectDevice(DeviceSelectionCriteria criteria);
    
    /// <summary>
    /// Creates a compute context on a specific device
    /// </summary>
    Task<IComputeContext> CreateContextAsync(
        IComputeDevice device,
        ContextOptions options,
        CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Gets device utilization metrics
    /// </summary>
    Task<DeviceMetrics> GetDeviceMetricsAsync(
        IComputeDevice device,
        CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Resets a device (clears memory, resets state)
    /// </summary>
    Task ResetDeviceAsync(
        IComputeDevice device,
        CancellationToken cancellationToken = default);
}

/// <summary>
/// Represents a compute device (GPU, CPU, etc.)
/// </summary>
public interface IComputeDevice
{
    /// <summary>
    /// Unique identifier for the device
    /// </summary>
    string DeviceId { get; }
    
    /// <summary>
    /// Device index in the system
    /// </summary>
    int Index { get; }
    
    /// <summary>
    /// Display name of the device
    /// </summary>
    string Name { get; }
    
    /// <summary>
    /// Type of compute device
    /// </summary>
    DeviceType Type { get; }
    
    /// <summary>
    /// Vendor of the device
    /// </summary>
    string Vendor { get; }
    
    /// <summary>
    /// Device architecture (e.g., "Ampere", "RDNA2", "x86-64")
    /// </summary>
    string Architecture { get; }
    
    /// <summary>
    /// Compute capability version
    /// </summary>
    Version ComputeCapability { get; }
    
    /// <summary>
    /// Total memory in bytes
    /// </summary>
    long TotalMemoryBytes { get; }
    
    /// <summary>
    /// Available memory in bytes
    /// </summary>
    long AvailableMemoryBytes { get; }
    
    /// <summary>
    /// Number of compute units (SMs for NVIDIA, CUs for AMD, cores for CPU)
    /// </summary>
    int ComputeUnits { get; }
    
    /// <summary>
    /// Maximum clock frequency in MHz
    /// </summary>
    int MaxClockFrequencyMHz { get; }
    
    /// <summary>
    /// Maximum number of threads per block/workgroup
    /// </summary>
    int MaxThreadsPerBlock { get; }
    
    /// <summary>
    /// Maximum workgroup dimensions
    /// </summary>
    int[] MaxWorkGroupDimensions { get; }
    
    /// <summary>
    /// Warp/wave size
    /// </summary>
    int WarpSize { get; }
    
    /// <summary>
    /// Device-specific properties
    /// </summary>
    IReadOnlyDictionary<string, object> Properties { get; }
    
    /// <summary>
    /// Checks if the device supports a specific feature
    /// </summary>
    bool SupportsFeature(string feature);
    
    /// <summary>
    /// Gets the current device status
    /// </summary>
    DeviceStatus GetStatus();
}

/// <summary>
/// Compute context for kernel execution
/// </summary>
public interface IComputeContext : IDisposable
{
    /// <summary>
    /// Associated device
    /// </summary>
    IComputeDevice Device { get; }
    
    /// <summary>
    /// Context ID
    /// </summary>
    string ContextId { get; }
    
    /// <summary>
    /// Makes this context current for the calling thread
    /// </summary>
    void MakeCurrent();
    
    /// <summary>
    /// Synchronizes all operations in this context
    /// </summary>
    Task SynchronizeAsync(CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Creates a command queue/stream for this context
    /// </summary>
    ICommandQueue CreateCommandQueue(CommandQueueOptions options);
}

/// <summary>
/// Command queue for submitting work to a device
/// </summary>
public interface ICommandQueue : IDisposable
{
    /// <summary>
    /// Queue ID
    /// </summary>
    string QueueId { get; }
    
    /// <summary>
    /// Associated context
    /// </summary>
    IComputeContext Context { get; }
    
    /// <summary>
    /// Enqueues a kernel for execution
    /// </summary>
    Task EnqueueKernelAsync(
        CompiledKernel kernel,
        KernelLaunchParameters parameters,
        CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Enqueues a memory copy operation
    /// </summary>
    Task EnqueueCopyAsync(
        IntPtr source,
        IntPtr destination,
        long sizeBytes,
        CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Waits for all enqueued operations to complete
    /// </summary>
    Task FlushAsync(CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Inserts a barrier that ensures all previous operations complete before continuing
    /// </summary>
    void EnqueueBarrier();
}

/// <summary>
/// Criteria for selecting a compute device
/// </summary>
public sealed record DeviceSelectionCriteria(
    DeviceType? PreferredType = null,
    long MinMemoryBytes = 0,
    int MinComputeUnits = 0,
    bool RequireUnifiedMemory = false,
    bool PreferHighestPerformance = true,
    string? RequiredFeature = null,
    IReadOnlyList<string>? ExcludeDevices = null);

/// <summary>
/// Options for creating a compute context
/// </summary>
public sealed record ContextOptions(
    bool EnableProfiling = false,
    bool EnableDebugMode = false,
    int CommandQueueCount = 1,
    bool EnableOutOfOrderExecution = false,
    IReadOnlyDictionary<string, object>? CustomOptions = null);

/// <summary>
/// Options for creating a command queue
/// </summary>
public sealed record CommandQueueOptions(
    bool EnableProfiling = false,
    bool EnableOutOfOrderExecution = false,
    int Priority = 0);

/// <summary>
/// Device metrics and utilization
/// </summary>
public sealed record DeviceMetrics(
    double GpuUtilizationPercent,
    double MemoryUtilizationPercent,
    long UsedMemoryBytes,
    double TemperatureCelsius,
    double PowerWatts,
    int FanSpeedPercent,
    long KernelsExecuted,
    long BytesTransferred,
    TimeSpan Uptime,
    IReadOnlyDictionary<string, object>? ExtendedMetrics = null);

/// <summary>
/// Status of a compute device
/// </summary>
public enum DeviceStatus
{
    /// <summary>Device is available and ready</summary>
    Available,
    /// <summary>Device is currently busy</summary>
    Busy,
    /// <summary>Device is offline or not responding</summary>
    Offline,
    /// <summary>Device has encountered an error</summary>
    Error,
    /// <summary>Device is being reset</summary>
    Resetting,
    /// <summary>Device status is unknown</summary>
    Unknown
}

/// <summary>
/// Parameters for launching a kernel
/// </summary>
public sealed record KernelLaunchParameters(
    int[] GlobalWorkSize,
    int[]? LocalWorkSize = null,
    int DynamicSharedMemoryBytes = 0,
    IReadOnlyDictionary<string, object>? Arguments = null);
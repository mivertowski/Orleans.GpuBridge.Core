using Orleans.GpuBridge.Abstractions.Enums;

namespace Orleans.GpuBridge.Abstractions.Models;

/// <summary>
/// Criteria for selecting a compute device
/// </summary>
[GenerateSerializer]
public sealed record DeviceSelectionCriteria
{
    /// <summary>
    /// Preferred device type (null for any)
    /// </summary>
    [Id(0)]
    public DeviceType? PreferredType { get; init; }

    /// <summary>
    /// Minimum required memory in bytes
    /// </summary>
    [Id(1)]
    public long MinimumMemoryBytes { get; init; }

    /// <summary>
    /// Minimum compute capability (for CUDA devices)
    /// </summary>
    [Id(2)]
    public Version? MinimumComputeCapability { get; init; }

    /// <summary>
    /// Maximum allowed queue depth
    /// </summary>
    [Id(3)]
    public int MaxQueueDepth { get; init; } = int.MaxValue;

    /// <summary>
    /// Prefer low-latency devices
    /// </summary>
    [Id(4)]
    public bool PreferLowLatency { get; init; }

    /// <summary>
    /// Prefer high-throughput devices
    /// </summary>
    [Id(5)]
    public bool PreferHighThroughput { get; init; }

    /// <summary>
    /// Required device features
    /// </summary>
    [Id(6)]
    public DeviceFeatures RequiredFeatures { get; init; } = DeviceFeatures.None;

    /// <summary>
    /// Device index hint (-1 for any)
    /// </summary>
    [Id(7)]
    public int DeviceIndexHint { get; init; } = -1;

    /// <summary>
    /// Allow CPU fallback if no GPU available
    /// </summary>
    [Id(8)]
    public bool AllowCpuFallback { get; init; } = true;

    /// <summary>
    /// Minimum memory size in bytes (legacy property)
    /// </summary>
    [Id(9)]
    public long MinMemoryBytes { get; init; }

    /// <summary>
    /// Minimum compute units required
    /// </summary>
    [Id(10)]
    public int MinComputeUnits { get; init; }

    /// <summary>
    /// Device IDs to exclude
    /// </summary>
    [Id(11)]
    public List<int> ExcludeDevices { get; init; } = new();

    /// <summary>
    /// Prefer highest performance devices
    /// </summary>
    [Id(12)]
    public bool PreferHighestPerformance { get; init; }

    /// <summary>
    /// Require unified memory support
    /// </summary>
    [Id(13)]
    public bool RequireUnifiedMemory { get; init; }

    /// <summary>
    /// Required feature for device selection
    /// </summary>
    [Id(14)]
    public DeviceFeatures RequiredFeature { get; init; } = DeviceFeatures.None;
}

/// <summary>
/// Device features flags
/// </summary>
[Flags]
[GenerateSerializer]
public enum DeviceFeatures
{
    /// <summary>
    /// No specific features required
    /// </summary>
    None = 0,

    /// <summary>
    /// Double precision floating point support
    /// </summary>
    DoublePrecision = 1 << 0,

    /// <summary>
    /// Atomic operations support
    /// </summary>
    Atomics = 1 << 1,

    /// <summary>
    /// Shared memory support
    /// </summary>
    SharedMemory = 1 << 2,

    /// <summary>
    /// Dynamic parallelism support
    /// </summary>
    DynamicParallelism = 1 << 3,

    /// <summary>
    /// Unified memory support
    /// </summary>
    UnifiedMemory = 1 << 4,

    /// <summary>
    /// Tensor core support
    /// </summary>
    TensorCores = 1 << 5,

    /// <summary>
    /// Ray tracing support
    /// </summary>
    RayTracing = 1 << 6,

    /// <summary>
    /// Multi-GPU support
    /// </summary>
    MultiGpu = 1 << 7
}
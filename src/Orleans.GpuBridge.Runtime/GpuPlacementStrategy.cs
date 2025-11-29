using Orleans.Runtime;
using Orleans.Runtime.Placement;

namespace Orleans.GpuBridge.Runtime;

/// <summary>
/// GPU-aware placement strategy for Orleans grains
/// </summary>
[Serializable]
[GenerateSerializer]
public sealed class GpuPlacementStrategy : PlacementStrategy
{
    /// <summary>
    /// Gets the singleton instance of the GPU placement strategy
    /// </summary>
    public static GpuPlacementStrategy Instance { get; } = new();

    /// <summary>
    /// Gets or sets whether to prefer placing grains on the local silo
    /// </summary>
    [Id(0)]
    public bool PreferLocalPlacement { get; init; }

    /// <summary>
    /// Gets or sets the minimum GPU memory required in MB
    /// </summary>
    [Id(1)]
    public int MinimumGpuMemoryMB { get; init; }
}

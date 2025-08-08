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
    public static GpuPlacementStrategy Instance { get; } = new();
    
    [Id(0)]
    public bool PreferLocalPlacement { get; init; }
    
    [Id(1)]
    public int MinimumGpuMemoryMB { get; init; }
}

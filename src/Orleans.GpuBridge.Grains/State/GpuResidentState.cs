using Orleans;

namespace Orleans.GpuBridge.Grains.State;

/// <summary>
/// Persistent state for GPU resident memory grain that survives grain activation/deactivation cycles.
/// This state is stored in the configured Orleans persistence provider and maintains memory allocation tracking.
/// </summary>
[GenerateSerializer]
public sealed class GpuResidentState
{
    /// <summary>
    /// Gets or sets the dictionary of all memory allocations managed by this grain.
    /// Key: Allocation ID (from GpuMemoryHandle.Id)
    /// Value: Memory allocation details and cached data
    /// </summary>
    /// <value>A dictionary mapping allocation IDs to <see cref="GpuMemoryAllocation"/> instances.</value>
    [Id(0)]
    public Dictionary<string, GpuMemoryAllocation> Allocations { get; set; } = new();

    /// <summary>
    /// Gets or sets the total amount of GPU memory allocated by this grain in bytes.
    /// This is the sum of all individual allocation sizes and is used for resource tracking.
    /// </summary>
    /// <value>The total allocated memory in bytes.</value>
    [Id(1)]
    public long TotalAllocatedBytes { get; set; }

    /// <summary>
    /// Gets or sets the timestamp of the last modification to the grain state.
    /// This is updated whenever allocations are added, removed, or modified.
    /// </summary>
    /// <value>The UTC timestamp of the last state modification.</value>
    [Id(2)]
    public DateTime LastModified { get; set; }

    /// <summary>
    /// Gets or sets the index of the GPU device where this grain's allocations reside.
    /// A value of -1 indicates no device has been assigned yet.
    /// </summary>
    /// <value>The device index, or -1 if no device is assigned.</value>
    [Id(3)]
    public int DeviceIndex { get; set; } = -1;
}
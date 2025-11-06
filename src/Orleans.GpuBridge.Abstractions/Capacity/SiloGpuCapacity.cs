using System;
using Orleans.Runtime;

namespace Orleans.GpuBridge.Abstractions.Capacity;

/// <summary>
/// Combines silo address with GPU capacity information for placement decisions
/// </summary>
/// <param name="SiloAddress">The Orleans silo address</param>
/// <param name="Capacity">GPU capacity for this silo</param>
[GenerateSerializer]
public sealed record SiloGpuCapacity(
    SiloAddress SiloAddress,
    GpuCapacity Capacity)
{
    /// <summary>
    /// Gets the total memory for this silo
    /// </summary>
    public long TotalMemoryMB => Capacity.TotalMemoryMB;

    /// <summary>
    /// Gets the available GPU memory for this silo
    /// </summary>
    public long AvailableMemoryMB => Capacity.AvailableMemoryMB;

    /// <summary>
    /// Gets the queue depth for this silo
    /// </summary>
    public int QueueDepth => Capacity.QueueDepth;

    /// <summary>
    /// Gets whether this silo has GPU capacity
    /// </summary>
    public bool HasGpu => Capacity.HasCapacity;

    /// <summary>
    /// Gets the memory usage percentage
    /// </summary>
    public double MemoryUsagePercent => Capacity.MemoryUsagePercent;

    /// <summary>
    /// Gets whether the capacity data is stale
    /// </summary>
    public bool IsStale => Capacity.IsStale;

    /// <summary>
    /// Calculates a placement score for this silo (higher is better)
    /// </summary>
    /// <remarks>
    /// Score factors:
    /// - Available memory (primary factor)
    /// - Low queue depth (secondary factor)
    /// - Penalize stale data
    /// </remarks>
    public double GetPlacementScore()
    {
        if (!HasGpu || IsStale)
        {
            return 0.0;
        }

        // Base score from available memory (0-100)
        var memoryScore = (AvailableMemoryMB / (double)Capacity.TotalMemoryMB) * 100.0;

        // Queue depth penalty (subtract up to 20 points)
        var queuePenalty = Math.Min(QueueDepth * 2.0, 20.0);

        return Math.Max(0.0, memoryScore - queuePenalty);
    }

    /// <summary>
    /// Returns a human-readable string representation
    /// </summary>
    public override string ToString() =>
        $"Silo: {SiloAddress.ToParsableString()}, {Capacity}";
}

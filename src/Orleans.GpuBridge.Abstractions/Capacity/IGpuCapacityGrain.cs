using System.Collections.Generic;
using System.Threading.Tasks;
using Orleans;
using Orleans.Runtime;

namespace Orleans.GpuBridge.Abstractions.Capacity;

/// <summary>
/// Grain interface for tracking GPU capacity across the Orleans cluster
/// </summary>
/// <remarks>
/// This grain acts as a centralized registry for GPU resources across all silos.
/// Silos register their GPU capacity on startup and periodically update it.
/// The grain is used by GpuPlacementDirector to make intelligent placement decisions.
/// </remarks>
public interface IGpuCapacityGrain : IGrainWithIntegerKey
{
    /// <summary>
    /// Registers a silo with its GPU capacity
    /// </summary>
    /// <param name="silo">The silo address to register</param>
    /// <param name="capacity">Initial GPU capacity information</param>
    /// <returns>Task representing the registration operation</returns>
    Task RegisterSiloAsync(SiloAddress silo, GpuCapacity capacity);

    /// <summary>
    /// Unregisters a silo (called during graceful shutdown)
    /// </summary>
    /// <param name="silo">The silo address to unregister</param>
    /// <returns>Task representing the unregistration operation</returns>
    Task UnregisterSiloAsync(SiloAddress silo);

    /// <summary>
    /// Updates the GPU capacity for a registered silo
    /// </summary>
    /// <param name="silo">The silo address to update</param>
    /// <param name="capacity">Updated GPU capacity information</param>
    /// <returns>Task representing the update operation</returns>
    Task UpdateCapacityAsync(SiloAddress silo, GpuCapacity capacity);

    /// <summary>
    /// Gets all GPU-capable silos with their current capacity
    /// </summary>
    /// <returns>List of silos with GPU capacity</returns>
    Task<List<SiloGpuCapacity>> GetGpuCapableSilosAsync();

    /// <summary>
    /// Gets the GPU capacity for a specific silo
    /// </summary>
    /// <param name="silo">The silo address to query</param>
    /// <returns>GPU capacity if registered, null otherwise</returns>
    Task<GpuCapacity?> GetSiloCapacityAsync(SiloAddress silo);

    /// <summary>
    /// Gets the best available silo for GPU workload placement
    /// </summary>
    /// <param name="minimumMemoryMB">Minimum required GPU memory in MB</param>
    /// <returns>Best silo for placement, or null if no suitable silo found</returns>
    Task<SiloGpuCapacity?> GetBestSiloForPlacementAsync(int minimumMemoryMB = 0);

    /// <summary>
    /// Gets cluster-wide GPU statistics
    /// </summary>
    /// <returns>Aggregated GPU statistics for the cluster</returns>
    Task<ClusterGpuStats> GetClusterStatsAsync();
}

/// <summary>
/// Cluster-wide GPU statistics
/// </summary>
[GenerateSerializer]
public sealed record ClusterGpuStats(
    int TotalSilos,
    int GpuCapableSilos,
    int TotalDevices,
    long TotalMemoryMB,
    long AvailableMemoryMB,
    int TotalQueueDepth)
{
    /// <summary>
    /// Gets the cluster-wide memory usage percentage
    /// </summary>
    public double MemoryUsagePercent => TotalMemoryMB > 0
        ? ((TotalMemoryMB - AvailableMemoryMB) / (double)TotalMemoryMB) * 100.0
        : 0.0;

    /// <summary>
    /// Gets the average queue depth per GPU-capable silo
    /// </summary>
    public double AverageQueueDepth => GpuCapableSilos > 0
        ? TotalQueueDepth / (double)GpuCapableSilos
        : 0.0;
}

using System;

namespace Orleans.GpuBridge.Abstractions.Capacity;

/// <summary>
/// Represents GPU capacity information for a silo
/// </summary>
/// <param name="DeviceCount">Number of GPU devices available</param>
/// <param name="TotalMemoryMB">Total GPU memory in MB across all devices</param>
/// <param name="AvailableMemoryMB">Currently available GPU memory in MB</param>
/// <param name="QueueDepth">Number of pending GPU operations</param>
/// <param name="Backend">GPU backend type (CUDA, OpenCL, etc.)</param>
/// <param name="LastUpdated">Timestamp of last capacity update</param>
[GenerateSerializer]
public sealed record GpuCapacity(
    int DeviceCount,
    long TotalMemoryMB,
    long AvailableMemoryMB,
    int QueueDepth,
    string Backend,
    DateTime LastUpdated)
{
    /// <summary>
    /// Gets the percentage of memory currently in use
    /// </summary>
    public double MemoryUsagePercent => TotalMemoryMB > 0
        ? ((TotalMemoryMB - AvailableMemoryMB) / (double)TotalMemoryMB) * 100.0
        : 0.0;

    /// <summary>
    /// Gets whether this silo has GPU capacity available
    /// </summary>
    public bool HasCapacity => DeviceCount > 0 && AvailableMemoryMB > 0;

    /// <summary>
    /// Gets whether the capacity data is stale (older than 2 minutes)
    /// </summary>
    public bool IsStale => DateTime.UtcNow - LastUpdated > TimeSpan.FromMinutes(2);

    /// <summary>
    /// Creates a capacity instance representing no GPU available
    /// </summary>
    public static GpuCapacity None => new(
        DeviceCount: 0,
        TotalMemoryMB: 0,
        AvailableMemoryMB: 0,
        QueueDepth: 0,
        Backend: "None",
        LastUpdated: DateTime.UtcNow);

    /// <summary>
    /// Creates an updated capacity with new available memory and queue depth
    /// </summary>
    public GpuCapacity WithUpdate(long availableMemoryMB, int queueDepth) => this with
    {
        AvailableMemoryMB = availableMemoryMB,
        QueueDepth = queueDepth,
        LastUpdated = DateTime.UtcNow
    };

    /// <summary>
    /// Returns a human-readable string representation
    /// </summary>
    public override string ToString() =>
        $"Devices: {DeviceCount}, Memory: {AvailableMemoryMB}/{TotalMemoryMB}MB " +
        $"({MemoryUsagePercent:F1}% used), Queue: {QueueDepth}, Backend: {Backend}";
}

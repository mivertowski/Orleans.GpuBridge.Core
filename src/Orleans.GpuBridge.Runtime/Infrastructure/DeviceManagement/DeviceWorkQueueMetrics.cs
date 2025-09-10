namespace Orleans.GpuBridge.Runtime.Infrastructure.DeviceManagement;

/// <summary>
/// Metrics for device work queue performance
/// </summary>
internal sealed class DeviceWorkQueueMetrics
{
    /// <summary>
    /// Number of items currently queued
    /// </summary>
    public int QueuedItems { get; init; }
    
    /// <summary>
    /// Total number of items processed successfully
    /// </summary>
    public long ProcessedItems { get; init; }
    
    /// <summary>
    /// Total number of items that failed processing
    /// </summary>
    public long FailedItems { get; init; }
    
    /// <summary>
    /// Error rate (0.0 to 1.0)
    /// </summary>
    public double ErrorRate { get; init; }
}
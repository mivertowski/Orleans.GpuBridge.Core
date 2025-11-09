using System;

namespace Orleans.GpuBridge.Resilience.Policies;

/// <summary>
/// Bulkhead isolation policy configuration options
/// </summary>
public sealed class BulkheadPolicyOptions
{
    /// <summary>
    /// Maximum concurrent operations allowed
    /// </summary>
    public int MaxConcurrentOperations { get; set; } = 10;

    /// <summary>
    /// Maximum queued operations when at capacity
    /// </summary>
    public int MaxQueuedOperations { get; set; } = 50;

    /// <summary>
    /// Whether to enable bulkhead isolation
    /// </summary>
    public bool Enabled { get; set; } = true;

    /// <summary>
    /// Timeout for waiting to acquire bulkhead slot
    /// </summary>
    public TimeSpan AcquisitionTimeout { get; set; } = TimeSpan.FromSeconds(30);
}

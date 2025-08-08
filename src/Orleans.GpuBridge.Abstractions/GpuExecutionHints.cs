namespace Orleans.GpuBridge.Abstractions;

/// <summary>
/// Hints for GPU kernel execution optimization
/// </summary>
public sealed record GpuExecutionHints(
    int? PreferredDevice = null,
    bool HighPriority = false,
    int? MaxMicroBatch = null,
    bool Persistent = true,
    bool PreferGpu = true,
    TimeSpan? Timeout = null,
    int? MaxRetries = null)
{
    /// <summary>
    /// Default execution hints
    /// </summary>
    public static GpuExecutionHints Default { get; } = new();
    
    /// <summary>
    /// CPU-only execution hints
    /// </summary>
    public static GpuExecutionHints CpuOnly { get; } = new(PreferGpu: false, Persistent: false);
    
    /// <summary>
    /// High-priority GPU execution hints
    /// </summary>
    public static GpuExecutionHints HighPriorityGpu { get; } = new(HighPriority: true);
}

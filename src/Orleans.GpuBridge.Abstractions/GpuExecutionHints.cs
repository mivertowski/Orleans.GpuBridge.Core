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
    
    /// <summary>
    /// Preferred batch size for execution (computed property)
    /// </summary>
    public int PreferredBatchSize => MaxMicroBatch ?? 1024;
    
    /// <summary>
    /// Timeout in milliseconds (computed property)
    /// </summary>
    public int TimeoutMs => Timeout?.Milliseconds ?? 30000;
}

using Orleans;

namespace Orleans.GpuBridge.Abstractions;

/// <summary>
/// Hints for GPU kernel execution optimization
/// </summary>
[GenerateSerializer]
public sealed record GpuExecutionHints(
    [property: Id(0)] int? PreferredDevice = null,
    [property: Id(1)] bool HighPriority = false,
    [property: Id(2)] int? MaxMicroBatch = null,
    [property: Id(3)] bool Persistent = true,
    [property: Id(4)] bool PreferGpu = true,
    [property: Id(5)] TimeSpan? Timeout = null,
    [property: Id(6)] int? MaxRetries = null)
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

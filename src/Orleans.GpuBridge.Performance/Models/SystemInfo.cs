namespace Orleans.GpuBridge.Performance.Models;

/// <summary>
/// System information for performance reporting
/// </summary>
public record SystemInfo
{
    public int ProcessorCount { get; init; }
    public long WorkingSet { get; init; }
    public long GcTotalMemory { get; init; }
    public string GcLatencyMode { get; init; } = string.Empty;
    public bool GcIsServerGc { get; init; }
}

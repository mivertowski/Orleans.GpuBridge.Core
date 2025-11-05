namespace Orleans.GpuBridge.Performance.Models;

/// <summary>
/// CPU capabilities information
/// </summary>
public record CpuCapabilities
{
    public bool HasAvx512 { get; init; }
    public bool HasAvx2 { get; init; }
    public bool HasAvx { get; init; }
    public bool HasFma { get; init; }
    public bool HasNeon { get; init; }
}

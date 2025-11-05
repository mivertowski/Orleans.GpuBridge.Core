namespace Orleans.GpuBridge.Performance.Models;

/// <summary>
/// Performance report containing system info, capabilities, and test results
/// </summary>
public record PerformanceReport
{
    public DateTime GeneratedAt { get; init; }
    public SystemInfo SystemInfo { get; init; } = null!;
    public CpuCapabilities CpuCapabilities { get; init; } = null!;
    public MemoryPoolStats MemoryPoolStats { get; init; } = null!;
    public PerformanceTestResult[] TestResults { get; init; } = Array.Empty<PerformanceTestResult>();
    public double OverallScore { get; init; }
}

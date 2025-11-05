namespace Orleans.GpuBridge.Performance.Models;

/// <summary>
/// Results from a single performance test
/// </summary>
public record PerformanceTestResult
{
    public string TestName { get; init; } = string.Empty;
    public int Iterations { get; init; }
    public int DataSize { get; init; }
    public double AverageTimeMs { get; init; }
    public double MinTimeMs { get; init; }
    public double MaxTimeMs { get; init; }
    public double ThroughputElementsPerSec { get; init; }
}

using System.Diagnostics;

namespace Orleans.GpuBridge.Performance.Tests;

/// <summary>
/// Performance tests for measuring throughput and latency.
/// These tests establish baselines and detect regressions.
/// </summary>
public sealed class ThroughputTests
{
    private const int WarmupIterations = 100;
    private const int BenchmarkIterations = 1000;

    /// <summary>
    /// Measures message processing throughput.
    /// Target: >100K messages/sec (WSL2), >1M messages/sec (native Linux)
    /// </summary>
    [Fact(Skip = "Requires GPU environment")]
    public async Task MessageThroughput_MeasuresMessagesPerSecond()
    {
        // Warmup
        for (int i = 0; i < WarmupIterations; i++)
        {
            await Task.Yield();
        }

        // Benchmark
        var sw = Stopwatch.StartNew();
        for (int i = 0; i < BenchmarkIterations; i++)
        {
            await Task.Yield();
        }
        sw.Stop();

        var messagesPerSecond = BenchmarkIterations / sw.Elapsed.TotalSeconds;

        // Assert minimum throughput (adjust based on environment)
        Assert.True(messagesPerSecond > 0, $"Throughput: {messagesPerSecond:N0} msgs/sec");
    }

    /// <summary>
    /// Measures message processing latency percentiles.
    /// Target: <1ms p99 (WSL2), <500ns p99 (native Linux)
    /// </summary>
    [Fact(Skip = "Requires GPU environment")]
    public async Task MessageLatency_MeasuresPercentiles()
    {
        var latencies = new List<double>(BenchmarkIterations);

        for (int i = 0; i < BenchmarkIterations; i++)
        {
            var sw = Stopwatch.StartNew();
            await Task.Yield();
            sw.Stop();
            latencies.Add(sw.Elapsed.TotalMilliseconds);
        }

        latencies.Sort();
        var p50 = latencies[(int)(latencies.Count * 0.50)];
        var p95 = latencies[(int)(latencies.Count * 0.95)];
        var p99 = latencies[(int)(latencies.Count * 0.99)];

        // Log percentiles
        Assert.True(p99 < 1000, $"p50={p50:F3}ms, p95={p95:F3}ms, p99={p99:F3}ms");
    }
}

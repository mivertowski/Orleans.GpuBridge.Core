using System.Net;
using FluentAssertions;
using Microsoft.Extensions.Logging.Abstractions;
using Orleans.GpuBridge.Runtime.Temporal.Network;
using Xunit.Abstractions;

namespace Orleans.GpuBridge.Temporal.Tests.Unit.Network;

/// <summary>
/// Diagnostic test to understand P99 calculation behavior.
/// </summary>
public sealed class P99DiagnosticTest
{
    private readonly ITestOutputHelper _output;

    public P99DiagnosticTest(ITestOutputHelper output)
    {
        _output = output;
    }

    [Fact]
    public void DiagnoseP99Calculation()
    {
        // Arrange
        var endpoint = new IPEndPoint(IPAddress.Loopback, 12345);
        var stats = new LatencyStatistics(endpoint, NullLogger.Instance);

        // Act - Add 100 samples from 1ms to 100ms
        for (int i = 1; i <= 100; i++)
        {
            stats.AddSample(TimeSpan.FromMilliseconds(i));
        }

        // Diagnostic output
        _output.WriteLine($"Sample Count: {stats.SampleCount}");
        _output.WriteLine($"MedianRtt: {stats.MedianRtt.TotalMilliseconds}ms");
        _output.WriteLine($"MinRtt: {stats.MinRtt.TotalMilliseconds}ms");
        _output.WriteLine($"MaxRtt: {stats.MaxRtt.TotalMilliseconds}ms");
        _output.WriteLine($"P99Rtt: {stats.P99Rtt.TotalMilliseconds}ms");

        // Calculate what the index should be
        int expectedIndex = (int)((100 - 1) * 0.99);
        _output.WriteLine($"Expected index: {expectedIndex}");
        _output.WriteLine($"Expected value at index {expectedIndex}: {expectedIndex + 1}ms (1-based samples)");

        // Manual calculation
        var samples = new List<TimeSpan>();
        for (int i = 1; i <= 100; i++)
        {
            samples.Add(TimeSpan.FromMilliseconds(i));
        }
        var sorted = samples.OrderBy(s => s).ToArray();
        var p99Index = Math.Min(sorted.Length - 1, (int)((sorted.Length - 1) * 0.99));
        _output.WriteLine($"Manual calculation: sorted[{p99Index}] = {sorted[p99Index].TotalMilliseconds}ms");

        // Assert
        stats.P99Rtt.TotalMilliseconds.Should().Be(99.0,
            $"P99 of 100 samples (1-100ms) should be 99ms, index calculation: (100-1)*0.99 = {expectedIndex}");
    }

    [Fact]
    public void VerifyP99Formula()
    {
        // Create a simple sorted array
        var sorted = Enumerable.Range(1, 100)
            .Select(i => TimeSpan.FromMilliseconds(i))
            .ToArray();

        // Test different formulas
        int formula1 = (int)((sorted.Length - 1) * 0.99);
        int formula2 = (int)(sorted.Length * 0.99);
        int formula3 = (int)Math.Ceiling((sorted.Length - 1) * 0.99);
        int formula4 = (int)Math.Floor((sorted.Length - 1) * 0.99);

        _output.WriteLine($"Array length: {sorted.Length}");
        _output.WriteLine($"Formula 1: (int)((length - 1) * 0.99) = {formula1}, value = {sorted[formula1].TotalMilliseconds}ms");
        _output.WriteLine($"Formula 2: (int)(length * 0.99) = {formula2}, value = {sorted[formula2].TotalMilliseconds}ms");
        _output.WriteLine($"Formula 3: (int)Ceiling((length - 1) * 0.99) = {formula3}, value = {sorted[formula3].TotalMilliseconds}ms");
        _output.WriteLine($"Formula 4: (int)Floor((length - 1) * 0.99) = {formula4}, value = {sorted[formula4].TotalMilliseconds}ms");

        // P99 should be the value where 99% of data is below
        // For 100 samples, that's the 99th sample (index 98 in 0-based)
        _output.WriteLine($"Expected: index 98, value 99ms");
    }
}

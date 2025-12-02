using System.Diagnostics;
using Orleans.GpuBridge.Abstractions.RingKernels;
using Xunit.Abstractions;

namespace Orleans.GpuBridge.Performance.Tests;

/// <summary>
/// Performance comparison benchmarks for pattern matching: CPU fallback handler vs pure CPU.
/// Demonstrates the performance characteristics of the CpuFallbackHandlerRegistry.
/// </summary>
public sealed class PatternMatchingBenchmarks
{
    private readonly ITestOutputHelper _output;
    private readonly CpuFallbackHandlerRegistry _handlerRegistry;

    private const int WarmupIterations = 50;

    public PatternMatchingBenchmarks(ITestOutputHelper output)
    {
        _output = output;
        _handlerRegistry = new CpuFallbackHandlerRegistry();

        // Register pattern matching handlers
        RegisterPatternMatchingHandlers();
    }

    private void RegisterPatternMatchingHandlers()
    {
        // Property match handler
        _handlerRegistry.RegisterHandler<PropertyMatchRequest, PatternMatchResponse, MatchState>(
            "pattern_match_kernel",
            0, // MatchByProperty
            (request, state) =>
            {
                var matchCount = 0;
                for (int i = 0; i < request.VertexCount; i++)
                {
                    // Simple property matching - check if property equals target
                    var property = i switch
                    {
                        0 => request.V0Property,
                        1 => request.V1Property,
                        2 => request.V2Property,
                        3 => request.V3Property,
                        _ => 0.0f
                    };

                    if (Math.Abs(property - request.TargetValue) < 0.001f)
                    {
                        matchCount++;
                    }
                }
                return (new PatternMatchResponse { MatchCount = matchCount }, state);
            });

        // Degree match handler
        _handlerRegistry.RegisterHandler<DegreeMatchRequest, PatternMatchResponse, MatchState>(
            "pattern_match_kernel",
            1, // MatchByDegree
            (request, state) =>
            {
                var matchCount = 0;
                for (int i = 0; i < request.VertexCount; i++)
                {
                    var degree = i switch
                    {
                        0 => request.V0Degree,
                        1 => request.V1Degree,
                        2 => request.V2Degree,
                        3 => request.V3Degree,
                        _ => 0
                    };

                    if (degree >= request.MinDegree && degree <= request.MaxDegree)
                    {
                        matchCount++;
                    }
                }
                return (new PatternMatchResponse { MatchCount = matchCount }, state);
            });

        // Triangle match handler (simplified - checks for triangle pattern)
        _handlerRegistry.RegisterHandler<TriangleMatchRequest, PatternMatchResponse, MatchState>(
            "pattern_match_kernel",
            4, // MatchTriangle
            (request, state) =>
            {
                // Simplified triangle detection - counts connected triplets
                var triangleCount = 0;
                // Check if V0-V1-V2 form a triangle
                if (request.V0V1Connected && request.V1V2Connected && request.V0V2Connected)
                {
                    triangleCount++;
                }
                return (new PatternMatchResponse { MatchCount = triangleCount * 3 }, state);
            });
    }

    /// <summary>
    /// Benchmarks property-based pattern matching.
    /// </summary>
    [Theory]
    [InlineData(4, 1000)]
    [InlineData(4, 5000)]
    [InlineData(4, 10000)]
    public void PropertyMatch_VaryingIterations_MeasuresPerformance(int vertexCount, int iterations)
    {
        // Generate test data
        var request = new PropertyMatchRequest
        {
            VertexCount = vertexCount,
            V0Property = 50.0f,
            V1Property = 25.0f,
            V2Property = 50.0f,
            V3Property = 75.0f,
            TargetValue = 50.0f
        };

        // Warmup
        for (int i = 0; i < WarmupIterations; i++)
        {
            _handlerRegistry.ExecuteHandler<PropertyMatchRequest, PatternMatchResponse, MatchState>(
                "pattern_match_kernel", 0, request, default);
        }

        // Benchmark
        var sw = Stopwatch.StartNew();

        for (int i = 0; i < iterations; i++)
        {
            _handlerRegistry.ExecuteHandler<PropertyMatchRequest, PatternMatchResponse, MatchState>(
                "pattern_match_kernel", 0, request, default);
        }

        sw.Stop();

        var avgLatency = sw.Elapsed.TotalMilliseconds / iterations;
        var throughput = iterations / sw.Elapsed.TotalSeconds;

        _output.WriteLine($"=== Property Match: {vertexCount} vertices, {iterations} iterations ===");
        _output.WriteLine($"Total time: {sw.Elapsed.TotalMilliseconds:F2}ms");
        _output.WriteLine($"Avg latency: {avgLatency * 1000:F4}μs");
        _output.WriteLine($"Throughput: {throughput:N0} ops/sec");

        // Should complete in reasonable time
        avgLatency.Should().BeLessThan(1.0, "Property match should be very fast");
    }

    /// <summary>
    /// Benchmarks degree-based pattern matching.
    /// </summary>
    [Theory]
    [InlineData(4, 1000)]
    [InlineData(4, 5000)]
    public void DegreeMatch_VaryingIterations_MeasuresPerformance(int vertexCount, int iterations)
    {
        var request = new DegreeMatchRequest
        {
            VertexCount = vertexCount,
            V0Degree = 2,
            V1Degree = 3,
            V2Degree = 5,
            V3Degree = 1,
            MinDegree = 2,
            MaxDegree = 4
        };

        // Warmup
        for (int i = 0; i < WarmupIterations; i++)
        {
            _handlerRegistry.ExecuteHandler<DegreeMatchRequest, PatternMatchResponse, MatchState>(
                "pattern_match_kernel", 1, request, default);
        }

        // Benchmark
        var sw = Stopwatch.StartNew();

        for (int i = 0; i < iterations; i++)
        {
            _handlerRegistry.ExecuteHandler<DegreeMatchRequest, PatternMatchResponse, MatchState>(
                "pattern_match_kernel", 1, request, default);
        }

        sw.Stop();

        var avgLatency = sw.Elapsed.TotalMilliseconds / iterations;
        var throughput = iterations / sw.Elapsed.TotalSeconds;

        _output.WriteLine($"=== Degree Match: {vertexCount} vertices, {iterations} iterations ===");
        _output.WriteLine($"Total time: {sw.Elapsed.TotalMilliseconds:F2}ms");
        _output.WriteLine($"Avg latency: {avgLatency * 1000:F4}μs");
        _output.WriteLine($"Throughput: {throughput:N0} ops/sec");

        avgLatency.Should().BeLessThan(1.0, "Degree match should be fast");
    }

    /// <summary>
    /// Compares CPU fallback handler to pure inline implementation.
    /// </summary>
    [Fact]
    public void CpuFallbackVsInline_SameAlgorithm_MeasuresOverhead()
    {
        const int vertexCount = 4;
        const int iterations = 10000;

        var request = new PropertyMatchRequest
        {
            VertexCount = vertexCount,
            V0Property = 50.0f,
            V1Property = 25.0f,
            V2Property = 50.0f,
            V3Property = 75.0f,
            TargetValue = 50.0f
        };

        // Warmup
        for (int i = 0; i < WarmupIterations; i++)
        {
            _handlerRegistry.ExecuteHandler<PropertyMatchRequest, PatternMatchResponse, MatchState>(
                "pattern_match_kernel", 0, request, default);
        }

        // Benchmark through handler registry
        var handlerSw = Stopwatch.StartNew();
        for (int i = 0; i < iterations; i++)
        {
            _handlerRegistry.ExecuteHandler<PropertyMatchRequest, PatternMatchResponse, MatchState>(
                "pattern_match_kernel", 0, request, default);
        }
        handlerSw.Stop();

        // Benchmark inline implementation
        var inlineSw = Stopwatch.StartNew();
        for (int iter = 0; iter < iterations; iter++)
        {
            var matchCount = 0;
            float[] props = [request.V0Property, request.V1Property, request.V2Property, request.V3Property];
            for (int i = 0; i < vertexCount; i++)
            {
                if (Math.Abs(props[i] - request.TargetValue) < 0.001f)
                {
                    matchCount++;
                }
            }
        }
        inlineSw.Stop();

        var handlerAvg = handlerSw.Elapsed.TotalMicroseconds / iterations;
        var inlineAvg = inlineSw.Elapsed.TotalMicroseconds / iterations;
        var overhead = handlerAvg - inlineAvg;
        var overheadPercent = inlineAvg > 0 ? (overhead / inlineAvg) * 100 : 0;

        _output.WriteLine("=== CPU Fallback vs Inline Comparison ===");
        _output.WriteLine($"Vertices: {vertexCount}, Iterations: {iterations}");
        _output.WriteLine("");
        _output.WriteLine($"Handler registry: {handlerAvg:F2}μs avg");
        _output.WriteLine($"Inline implementation: {inlineAvg:F2}μs avg");
        _output.WriteLine($"Handler overhead: {overhead:F2}μs ({overheadPercent:F1}%)");
        _output.WriteLine("");
        _output.WriteLine("Note: Handler overhead includes dictionary lookup + delegate invocation");

        // Handler overhead should be minimal compared to actual work
        // For such simple operations, overhead can be significant percentage-wise but
        // absolute overhead should still be under a few microseconds
        overhead.Should().BeLessThan(10.0, "Handler overhead should be under 10μs");
    }

    /// <summary>
    /// Tests scaling behavior with increasing iterations.
    /// </summary>
    [Fact]
    public void ScalingBehavior_IncreasingIterations_LinearOrBetter()
    {
        var iterationCounts = new[] { 100, 500, 1000, 5000, 10000 };
        var results = new List<(int Iterations, double Latency, double Throughput)>();

        var request = new PropertyMatchRequest
        {
            VertexCount = 4,
            V0Property = 50.0f,
            V1Property = 25.0f,
            V2Property = 50.0f,
            V3Property = 75.0f,
            TargetValue = 50.0f
        };

        foreach (var iterations in iterationCounts)
        {
            // Warmup
            for (int i = 0; i < 10; i++)
            {
                _handlerRegistry.ExecuteHandler<PropertyMatchRequest, PatternMatchResponse, MatchState>(
                    "pattern_match_kernel", 0, request, default);
            }

            // Benchmark
            var sw = Stopwatch.StartNew();
            for (int i = 0; i < iterations; i++)
            {
                _handlerRegistry.ExecuteHandler<PropertyMatchRequest, PatternMatchResponse, MatchState>(
                    "pattern_match_kernel", 0, request, default);
            }
            sw.Stop();

            var iterAvgLatency = sw.Elapsed.TotalMicroseconds / iterations;
            var iterThroughput = iterations / sw.Elapsed.TotalSeconds;
            results.Add((iterations, iterAvgLatency, iterThroughput));
        }

        _output.WriteLine("=== Scaling Behavior Analysis ===");
        _output.WriteLine($"{"Iterations",-12} {"Latency (μs)",-15} {"Throughput",-15}");
        _output.WriteLine(new string('-', 42));

        foreach (var (iters, latency, throughput) in results)
        {
            _output.WriteLine($"{iters,-12} {latency,-15:F4} {throughput,-15:N0}");
        }

        // Verify latency remains relatively constant (linear scaling)
        var latencies = results.Select(r => r.Latency).ToList();
        var avgLatency = latencies.Average();
        var maxDeviation = latencies.Max() / avgLatency;

        _output.WriteLine("");
        _output.WriteLine($"Avg latency: {avgLatency:F4}μs");
        _output.WriteLine($"Max deviation: {maxDeviation:F2}x from average");

        // Latency should be fairly stable regardless of iteration count
        maxDeviation.Should().BeLessThan(3.0, "Latency should remain relatively constant");
    }

    // Unmanaged structs for pattern matching
    private struct PropertyMatchRequest
    {
        public int VertexCount;
        public float V0Property;
        public float V1Property;
        public float V2Property;
        public float V3Property;
        public float TargetValue;
    }

    private struct DegreeMatchRequest
    {
        public int VertexCount;
        public int V0Degree;
        public int V1Degree;
        public int V2Degree;
        public int V3Degree;
        public int MinDegree;
        public int MaxDegree;
    }

    private struct TriangleMatchRequest
    {
        public int VertexCount;
        public bool V0V1Connected;
        public bool V1V2Connected;
        public bool V0V2Connected;
    }

    private struct PatternMatchResponse
    {
        public int MatchCount;
    }

    private struct MatchState
    {
        public int ProcessedCount;
    }
}

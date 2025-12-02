using System.Diagnostics;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Orleans.GpuBridge.Abstractions.Memory;
using Orleans.GpuBridge.Abstractions.RingKernels;
using Orleans.GpuBridge.Runtime;
using Orleans.GpuBridge.Runtime.Extensions;
using Xunit.Abstractions;

namespace Orleans.GpuBridge.Performance.Tests;

/// <summary>
/// Performance tests for ring kernel message latency.
/// Target: <10ms for EventDriven mode (CPU fallback)
/// </summary>
public sealed class RingKernelLatencyTests : IDisposable
{
    private readonly ITestOutputHelper _output;
    private readonly ServiceProvider _serviceProvider;
    private readonly CpuFallbackHandlerRegistry _handlerRegistry;

    private const int WarmupIterations = 100;
    private const int BenchmarkIterations = 10000;

    public RingKernelLatencyTests(ITestOutputHelper output)
    {
        _output = output;
        _handlerRegistry = new CpuFallbackHandlerRegistry();

        var services = new ServiceCollection();
        services.AddLogging(builder => builder.AddDebug().SetMinimumLevel(LogLevel.Warning));

        services.AddGpuBridge()
            .Services
            .AddRingKernelSupport()
            .AddRingKernelBridge();

        services.AddSingleton(typeof(IGpuMemoryPool<>), typeof(CpuMemoryPool<>));

        _serviceProvider = services.BuildServiceProvider();

        // Register test handlers
        RegisterTestHandlers();
    }

    public void Dispose()
    {
        _serviceProvider.Dispose();
    }

    private void RegisterTestHandlers()
    {
        // Simple passthrough handler
        _handlerRegistry.RegisterHandler<LatencyRequest, LatencyResponse, LatencyState>(
            "latency_kernel",
            0,
            (request, state) => (new LatencyResponse
            {
                RequestId = request.RequestId,
                Timestamp = Stopwatch.GetTimestamp()
            }, state));

        // Compute handler (simulates work)
        _handlerRegistry.RegisterHandler<ComputeRequest, ComputeResponse, ComputeState>(
            "compute_kernel",
            0,
            (request, state) =>
            {
                var result = 0.0;
                for (int i = 0; i < request.Iterations; i++)
                {
                    result += Math.Sin(i * 0.001);
                }
                return (new ComputeResponse { Result = result }, state);
            });
    }

    /// <summary>
    /// Tests that ring kernel message roundtrip latency is under 10ms.
    /// Target: p99 < 10ms in EventDriven (CPU fallback) mode.
    /// </summary>
    [Fact]
    public void MessageLatency_CpuFallback_Under10ms()
    {
        // Warmup
        for (int i = 0; i < WarmupIterations; i++)
        {
            _handlerRegistry.ExecuteHandler<LatencyRequest, LatencyResponse, LatencyState>(
                "latency_kernel", 0,
                new LatencyRequest { RequestId = i },
                default);
        }

        // Benchmark
        var latencies = new List<double>(BenchmarkIterations);
        var sw = new Stopwatch();

        for (int i = 0; i < BenchmarkIterations; i++)
        {
            sw.Restart();

            _handlerRegistry.ExecuteHandler<LatencyRequest, LatencyResponse, LatencyState>(
                "latency_kernel", 0,
                new LatencyRequest { RequestId = i },
                default);

            sw.Stop();
            latencies.Add(sw.Elapsed.TotalMilliseconds);
        }

        // Calculate percentiles
        latencies.Sort();
        var p50 = latencies[(int)(latencies.Count * 0.50)];
        var p95 = latencies[(int)(latencies.Count * 0.95)];
        var p99 = latencies[(int)(latencies.Count * 0.99)];
        var max = latencies.Max();
        var avg = latencies.Average();

        _output.WriteLine("=== Ring Kernel Message Latency ===");
        _output.WriteLine($"Iterations: {BenchmarkIterations:N0}");
        _output.WriteLine($"Average: {avg:F4}ms");
        _output.WriteLine($"p50: {p50:F4}ms");
        _output.WriteLine($"p95: {p95:F4}ms");
        _output.WriteLine($"p99: {p99:F4}ms");
        _output.WriteLine($"Max: {max:F4}ms");
        _output.WriteLine($"Throughput: {BenchmarkIterations / latencies.Sum() * 1000:N0} ops/sec");

        // Assert: p99 should be under 10ms for EventDriven mode
        p99.Should().BeLessThan(10.0,
            $"p99 latency ({p99:F4}ms) should be under 10ms for CPU fallback");
    }

    /// <summary>
    /// Tests sustained throughput over many iterations.
    /// </summary>
    [Fact]
    public void Throughput_Sustained_MeetsTarget()
    {
        // Warmup
        for (int i = 0; i < WarmupIterations; i++)
        {
            _handlerRegistry.ExecuteHandler<LatencyRequest, LatencyResponse, LatencyState>(
                "latency_kernel", 0,
                new LatencyRequest { RequestId = i },
                default);
        }

        // Benchmark sustained throughput
        var sw = Stopwatch.StartNew();

        for (int i = 0; i < BenchmarkIterations; i++)
        {
            _handlerRegistry.ExecuteHandler<LatencyRequest, LatencyResponse, LatencyState>(
                "latency_kernel", 0,
                new LatencyRequest { RequestId = i },
                default);
        }

        sw.Stop();

        var throughput = BenchmarkIterations / sw.Elapsed.TotalSeconds;
        var avgLatency = sw.Elapsed.TotalMilliseconds / BenchmarkIterations;

        _output.WriteLine("=== Sustained Throughput Test ===");
        _output.WriteLine($"Duration: {sw.Elapsed.TotalSeconds:F2}s");
        _output.WriteLine($"Operations: {BenchmarkIterations:N0}");
        _output.WriteLine($"Throughput: {throughput:N0} ops/sec");
        _output.WriteLine($"Avg Latency: {avgLatency:F4}ms");

        // Target: At least 100K ops/sec for CPU fallback (synchronous)
        throughput.Should().BeGreaterThan(100000,
            "Throughput should exceed 100K ops/sec for CPU fallback");
    }

    /// <summary>
    /// Tests concurrent request handling latency.
    /// </summary>
    [Fact]
    public void ConcurrentLatency_ParallelRequests_StableThroughput()
    {
        var concurrency = Environment.ProcessorCount;
        var allLatencies = new System.Collections.Concurrent.ConcurrentBag<double>();

        // Warmup
        Parallel.For(0, WarmupIterations, i =>
        {
            _handlerRegistry.ExecuteHandler<LatencyRequest, LatencyResponse, LatencyState>(
                "latency_kernel", 0,
                new LatencyRequest { RequestId = i },
                default);
        });

        // Benchmark
        var sw = Stopwatch.StartNew();

        Parallel.For(0, BenchmarkIterations, new ParallelOptions { MaxDegreeOfParallelism = concurrency }, i =>
        {
            var requestSw = Stopwatch.StartNew();

            _handlerRegistry.ExecuteHandler<LatencyRequest, LatencyResponse, LatencyState>(
                "latency_kernel", 0,
                new LatencyRequest { RequestId = i },
                default);

            requestSw.Stop();
            allLatencies.Add(requestSw.Elapsed.TotalMilliseconds);
        });

        sw.Stop();

        // Calculate metrics
        var latencyList = allLatencies.ToList();
        latencyList.Sort();

        var p50 = latencyList[(int)(latencyList.Count * 0.50)];
        var p95 = latencyList[(int)(latencyList.Count * 0.95)];
        var p99 = latencyList[(int)(latencyList.Count * 0.99)];
        var throughput = BenchmarkIterations / sw.Elapsed.TotalSeconds;

        _output.WriteLine("=== Concurrent Latency Test ===");
        _output.WriteLine($"Concurrency: {concurrency}");
        _output.WriteLine($"Total Operations: {BenchmarkIterations:N0}");
        _output.WriteLine($"Duration: {sw.Elapsed.TotalSeconds:F2}s");
        _output.WriteLine($"Throughput: {throughput:N0} ops/sec");
        _output.WriteLine($"p50: {p50:F4}ms");
        _output.WriteLine($"p95: {p95:F4}ms");
        _output.WriteLine($"p99: {p99:F4}ms");

        // Assert: Even under concurrency, p99 should be reasonable
        p99.Should().BeLessThan(50.0,
            "p99 latency under concurrency should be under 50ms");
    }

    /// <summary>
    /// Tests latency variance (jitter) over time.
    /// </summary>
    [Fact]
    public void LatencyVariance_OverTime_LowJitter()
    {
        const int buckets = 10;
        const int requestsPerBucket = 1000;
        var bucketLatencies = new List<List<double>>();

        // Warmup
        for (int i = 0; i < WarmupIterations; i++)
        {
            _handlerRegistry.ExecuteHandler<LatencyRequest, LatencyResponse, LatencyState>(
                "latency_kernel", 0,
                new LatencyRequest { RequestId = i },
                default);
        }

        // Benchmark in time buckets
        for (int bucket = 0; bucket < buckets; bucket++)
        {
            var latencies = new List<double>(requestsPerBucket);
            var sw = new Stopwatch();

            for (int i = 0; i < requestsPerBucket; i++)
            {
                sw.Restart();

                _handlerRegistry.ExecuteHandler<LatencyRequest, LatencyResponse, LatencyState>(
                    "latency_kernel", 0,
                    new LatencyRequest { RequestId = bucket * requestsPerBucket + i },
                    default);

                sw.Stop();
                latencies.Add(sw.Elapsed.TotalMilliseconds);
            }

            bucketLatencies.Add(latencies);
        }

        // Calculate variance across buckets
        var bucketP99s = bucketLatencies.Select(b =>
        {
            b.Sort();
            return b[(int)(b.Count * 0.99)];
        }).ToList();

        var avgP99 = bucketP99s.Average();
        var stdDevP99 = Math.Sqrt(bucketP99s.Average(v => Math.Pow(v - avgP99, 2)));
        var coefficientOfVariation = avgP99 > 0 ? stdDevP99 / avgP99 : 0;

        _output.WriteLine("=== Latency Variance Test ===");
        _output.WriteLine($"Buckets: {buckets}");
        _output.WriteLine($"Requests per bucket: {requestsPerBucket}");
        _output.WriteLine($"Bucket p99s: {string.Join(", ", bucketP99s.Select(p => $"{p:F3}ms"))}");
        _output.WriteLine($"Average p99: {avgP99:F4}ms");
        _output.WriteLine($"StdDev p99: {stdDevP99:F4}ms");
        _output.WriteLine($"Coefficient of Variation: {coefficientOfVariation:P2}");

        // Assert: Variance should be low (CV < 100%)
        coefficientOfVariation.Should().BeLessThan(1.0,
            "Latency coefficient of variation should be under 100%");
    }

    private struct LatencyRequest
    {
        public int RequestId;
    }

    private struct LatencyResponse
    {
        public int RequestId;
        public long Timestamp;
    }

    private struct LatencyState
    {
        public int Counter;
    }

    private struct ComputeRequest
    {
        public int Iterations;
    }

    private struct ComputeResponse
    {
        public double Result;
    }

    private struct ComputeState
    {
        public int Counter;
    }
}

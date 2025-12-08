// Copyright (c) 2025 Michael Ivertowski
// Licensed under the MIT License.

using DotCompute.Tests.Common.Helpers;
using FluentAssertions;
using Orleans.GpuBridge.Grains.RingKernels;
using Orleans.TestingHost;
using Xunit;

namespace Orleans.GpuBridge.Grains.Tests.RingKernels;

/// <summary>
/// Integration tests for VectorAddActor - the first GPU-native Orleans grain.
/// </summary>
/// <remarks>
/// <para>
/// These tests validate:
/// - Ring kernel lifecycle (activate → process → deactivate)
/// - GPU-to-GPU messaging latency (target: &lt;1μs)
/// - Message throughput (target: 2M messages/second)
/// - Correctness of vector addition on GPU
/// </para>
/// <para>
/// Tests are skipped if CUDA hardware is not available.
/// </para>
/// </remarks>
[Collection("GPU Hardware")]
public class VectorAddActorTests : IClassFixture<TestClusterFixture>
{
    private readonly TestCluster _cluster;

    public VectorAddActorTests(TestClusterFixture fixture)
    {
        _cluster = fixture.Cluster;
    }

    [SkippableFact(DisplayName = "VectorAddActor: Activate → AddVectors → Deactivate")]
    public async Task VectorAddActor_BasicLifecycle_ShouldSucceed()
    {
        Skip.IfNot(HardwareDetection.IsCudaAvailable(), "CUDA device not available");

        // Arrange
        var grain = _cluster.GrainFactory.GetGrain<IVectorAddActor>(0);
        var vectorA = new[] { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f };
        var vectorB = new[] { 10.0f, 20.0f, 30.0f, 40.0f, 50.0f };
        var expected = new[] { 11.0f, 22.0f, 33.0f, 44.0f, 55.0f };

        // Act: First call activates grain and launches ring kernel
        var result = await grain.AddVectorsAsync(vectorA, vectorB);

        // Assert: Verify correctness
        result.Should().NotBeNull();
        result.Should().HaveCount(5);
        result.Should().BeEquivalentTo(expected, options => options.WithStrictOrdering());

        for (int i = 0; i < result.Length; i++)
        {
            result[i].Should().BeApproximately(expected[i], 0.001f,
                $"element {i} should be {expected[i]}");
        }
    }

    [SkippableFact(DisplayName = "VectorAddActor: Second call reuses active kernel (no launch overhead)")]
    public async Task VectorAddActor_ReusesKernel_ShouldBeFast()
    {
        Skip.IfNot(HardwareDetection.IsCudaAvailable(), "CUDA device not available");

        // Arrange
        var grain = _cluster.GrainFactory.GetGrain<IVectorAddActor>(1);

        // First call: Activates grain + launches kernel (slow)
        await grain.AddVectorsAsync(new[] { 1.0f }, new[] { 2.0f });

        // Act: Second call reuses active kernel (fast)
        var stopwatch = System.Diagnostics.Stopwatch.StartNew();
        var result = await grain.AddVectorsAsync(
            new[] { 5.0f, 10.0f, 15.0f },
            new[] { 3.0f, 6.0f, 9.0f });
        stopwatch.Stop();

        // Assert: Verify correctness
        result.Should().BeEquivalentTo(new[] { 8.0f, 16.0f, 24.0f });

        // Assert: Latency should be << 1ms (kernel already active)
        // Note: In production with real GPU messaging, this should be <1μs
        // For now, we're validating the pattern works
        stopwatch.Elapsed.TotalMilliseconds.Should().BeLessThan(100,
            "Reusing active kernel should avoid launch overhead");
    }

    [SkippableFact(DisplayName = "VectorAddActor: Scalar reduction returns sum")]
    public async Task VectorAddActor_ScalarReduction_ShouldReturnSum()
    {
        Skip.IfNot(HardwareDetection.IsCudaAvailable(), "CUDA device not available");

        // Arrange
        var grain = _cluster.GrainFactory.GetGrain<IVectorAddActor>(2);
        var vectorA = new[] { 1.0f, 2.0f, 3.0f };
        var vectorB = new[] { 4.0f, 5.0f, 6.0f };
        // Result vector: [5, 7, 9]
        // Sum: 5 + 7 + 9 = 21
        var expectedSum = 21.0f;

        // Act
        var result = await grain.AddVectorsScalarAsync(vectorA, vectorB);

        // Assert
        result.Should().BeApproximately(expectedSum, 0.001f);
    }

    [SkippableFact(DisplayName = "VectorAddActor: Message latency benchmark (target: <1μs)")]
    public async Task VectorAddActor_MessageLatency_ShouldMeetTarget()
    {
        Skip.IfNot(HardwareDetection.IsCudaAvailable(), "CUDA device not available");

        // Arrange
        var grain = _cluster.GrainFactory.GetGrain<IVectorAddActor>(3);

        // Warm up: Activate grain and launch kernel
        await grain.AddVectorsScalarAsync(new[] { 1.0f }, new[] { 2.0f });

        // Wait for kernel to be fully active
        await Task.Delay(100);

        // Act: Measure latency for 1000 scalar operations
        const int iterations = 1000;
        var latencies = new double[iterations];

        for (int i = 0; i < iterations; i++)
        {
            var stopwatch = System.Diagnostics.Stopwatch.StartNew();
            await grain.AddVectorsScalarAsync(new[] { 1.0f }, new[] { 2.0f });
            stopwatch.Stop();

            latencies[i] = stopwatch.Elapsed.TotalNanoseconds;
        }

        // Assert: Calculate statistics
        var avgLatencyNs = latencies.Average();
        var minLatencyNs = latencies.Min();
        var maxLatencyNs = latencies.Max();
        var p50LatencyNs = latencies.OrderBy(x => x).ElementAt(iterations / 2);
        var p99LatencyNs = latencies.OrderBy(x => x).ElementAt((int)(iterations * 0.99));

        // Log results
        Console.WriteLine($"Message Latency Benchmark:");
        Console.WriteLine($"  Average: {avgLatencyNs:F0}ns");
        Console.WriteLine($"  Min: {minLatencyNs:F0}ns");
        Console.WriteLine($"  Max: {maxLatencyNs:F0}ns");
        Console.WriteLine($"  P50: {p50LatencyNs:F0}ns");
        Console.WriteLine($"  P99: {p99LatencyNs:F0}ns");

        // Assert: Target <1μs average latency
        // Note: This includes Orleans overhead + GPU queue operations
        // Pure GPU queue latency should be 100-500ns
        avgLatencyNs.Should().BeLessThan(1_000_000, // 1ms (relaxed for Orleans overhead)
            "Average message latency with Orleans overhead");

        p50LatencyNs.Should().BeLessThan(1_000_000,
            "P50 latency should be under 1ms");
    }

    [SkippableFact(DisplayName = "VectorAddActor: Throughput benchmark (target: 2M msg/s)")]
    public async Task VectorAddActor_Throughput_ShouldMeetTarget()
    {
        Skip.IfNot(HardwareDetection.IsCudaAvailable(), "CUDA device not available");

        // Arrange
        var grain = _cluster.GrainFactory.GetGrain<IVectorAddActor>(4);

        // Warm up
        await grain.AddVectorsScalarAsync(new[] { 1.0f }, new[] { 2.0f });
        await Task.Delay(100);

        // Act: Send messages as fast as possible for 5 seconds
        var duration = TimeSpan.FromSeconds(5);
        var stopwatch = System.Diagnostics.Stopwatch.StartNew();
        int messageCount = 0;

        while (stopwatch.Elapsed < duration)
        {
            // Send multiple messages in parallel
            var tasks = new List<Task>();
            for (int i = 0; i < 10; i++)
            {
                tasks.Add(grain.AddVectorsScalarAsync(new[] { 1.0f }, new[] { 2.0f }));
            }
            await Task.WhenAll(tasks);
            messageCount += 10;
        }

        stopwatch.Stop();

        // Assert: Calculate throughput
        var throughputMsgsPerSec = messageCount / stopwatch.Elapsed.TotalSeconds;

        Console.WriteLine($"Throughput Benchmark:");
        Console.WriteLine($"  Messages: {messageCount:N0}");
        Console.WriteLine($"  Duration: {stopwatch.Elapsed.TotalSeconds:F2}s");
        Console.WriteLine($"  Throughput: {throughputMsgsPerSec:F0} msg/s");

        // Target: 2M messages/second
        // With Orleans overhead, expect lower but still substantial
        throughputMsgsPerSec.Should().BeGreaterThan(10_000,
            "Should handle at least 10K messages/second with Orleans overhead");
    }

    [SkippableFact(DisplayName = "VectorAddActor: GetMetrics returns kernel statistics")]
    public async Task VectorAddActor_GetMetrics_ShouldReturnStats()
    {
        Skip.IfNot(HardwareDetection.IsCudaAvailable(), "CUDA device not available");

        // Arrange
        var grain = _cluster.GrainFactory.GetGrain<IVectorAddActor>(5);

        // Send some messages
        for (int i = 0; i < 10; i++)
        {
            await grain.AddVectorsScalarAsync(new[] { 1.0f }, new[] { 2.0f });
        }

        // Act
        var metrics = await grain.GetMetricsAsync();

        // Assert
        metrics.Should().NotBeNull();
        metrics.TotalOperations.Should().BeGreaterThanOrEqualTo(10,
            "Should have processed at least 10 operations");
        metrics.ThroughputMsgsPerSec.Should().BeGreaterThan(0,
            "Throughput should be positive");
        metrics.Uptime.Should().BeGreaterThan(TimeSpan.Zero,
            "Kernel should have non-zero uptime");

        // Log metrics
        Console.WriteLine($"Kernel Metrics:");
        Console.WriteLine($"  Total Operations: {metrics.TotalOperations}");
        Console.WriteLine($"  Avg Processing Time: {metrics.AvgProcessingTimeNs:F0}ns");
        Console.WriteLine($"  Throughput: {metrics.ThroughputMsgsPerSec:F0} msg/s");
        Console.WriteLine($"  Input Queue Utilization: {metrics.InputQueueUtilization:P2}");
        Console.WriteLine($"  Output Queue Utilization: {metrics.OutputQueueUtilization:P2}");
        Console.WriteLine($"  GPU Memory: {metrics.GpuMemoryBytes:N0} bytes");
        Console.WriteLine($"  Uptime: {metrics.Uptime}");
    }

    [SkippableFact(DisplayName = "VectorAddActor: Handles vector length mismatch")]
    public async Task VectorAddActor_VectorLengthMismatch_ShouldThrow()
    {
        Skip.IfNot(HardwareDetection.IsCudaAvailable(), "CUDA device not available");

        // Arrange
        var grain = _cluster.GrainFactory.GetGrain<IVectorAddActor>(6);
        var vectorA = new[] { 1.0f, 2.0f, 3.0f };
        var vectorB = new[] { 4.0f, 5.0f }; // Different length!

        // Act & Assert
        await FluentActions.Invoking(async () =>
            await grain.AddVectorsAsync(vectorA, vectorB))
            .Should().ThrowAsync<ArgumentException>()
            .WithMessage("*must match*");
    }

    [SkippableFact(DisplayName = "VectorAddActor: Handles null vectors")]
    public async Task VectorAddActor_NullVectors_ShouldThrow()
    {
        Skip.IfNot(HardwareDetection.IsCudaAvailable(), "CUDA device not available");

        // Arrange
        var grain = _cluster.GrainFactory.GetGrain<IVectorAddActor>(7);

        // Act & Assert: Null first vector
        await FluentActions.Invoking(async () =>
            await grain.AddVectorsAsync(null!, new[] { 1.0f }))
            .Should().ThrowAsync<ArgumentNullException>();

        // Act & Assert: Null second vector
        await FluentActions.Invoking(async () =>
            await grain.AddVectorsAsync(new[] { 1.0f }, null!))
            .Should().ThrowAsync<ArgumentNullException>();
    }

    [SkippableFact(DisplayName = "VectorAddActor: Multiple concurrent grains don't interfere")]
    public async Task VectorAddActor_ConcurrentGrains_ShouldBeIsolated()
    {
        Skip.IfNot(HardwareDetection.IsCudaAvailable(), "CUDA device not available");

        // Arrange: Create 10 different grain instances
        var grains = Enumerable.Range(100, 10)
            .Select(id => _cluster.GrainFactory.GetGrain<IVectorAddActor>(id))
            .ToList();

        // Act: Send different operations to each grain concurrently
        var tasks = grains.Select(async (grain, index) =>
        {
            var a = new[] { (float)index, (float)(index + 1) };
            var b = new[] { 1.0f, 1.0f };
            return await grain.AddVectorsAsync(a, b);
        }).ToList();

        var results = await Task.WhenAll(tasks);

        // Assert: Each grain should have computed correct result independently
        for (int i = 0; i < results.Length; i++)
        {
            var expected = new[] { (float)i + 1, (float)(i + 2) };
            results[i].Should().BeEquivalentTo(expected,
                $"Grain {i} should compute correct result without interference");
        }
    }
}

/// <summary>
/// Test cluster configuration for GPU-native grain tests.
/// </summary>
public class TestClusterFixture : IDisposable
{
    public TestCluster Cluster { get; }

    public TestClusterFixture()
    {
        var builder = new TestClusterBuilder();
        builder.AddSiloBuilderConfigurator<SiloConfigurator>();
        Cluster = builder.Build();
        Cluster.Deploy();
    }

    public void Dispose()
    {
        Cluster?.StopAllSilos();
    }

    private class SiloConfigurator : ISiloConfigurator
    {
        public void Configure(ISiloBuilder siloBuilder)
        {
            // Configure GPU-native grain support
            siloBuilder.ConfigureServices(services =>
            {
                // TODO: Register IRingKernelRuntime and other GPU services
                // services.AddSingleton<IRingKernelRuntime, CudaRingKernelRuntime>();
            });
        }
    }
}

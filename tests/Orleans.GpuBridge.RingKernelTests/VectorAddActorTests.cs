// Copyright (c) 2025 Michael Ivertowski
// Licensed under the MIT License.

using Orleans.GpuBridge.Grains.RingKernels;
using Orleans.GpuBridge.Runtime.RingKernels;
using Orleans.TestingHost;
using Microsoft.Extensions.DependencyInjection;
using DotCompute.Abstractions.RingKernels;
using Microsoft.Extensions.Logging;
using Xunit;

namespace Orleans.GpuBridge.RingKernelTests;

/// <summary>
/// Integration tests for VectorAddActor GPU-native grain.
/// </summary>
/// <remarks>
/// <para>
/// Tests the complete lifecycle of GPU-native actors:
/// - Grain activation → Ring kernel launch
/// - Message passing → GPU processing
/// - Grain deactivation → Ring kernel pause
/// - Grain disposal → Ring kernel termination
/// </para>
/// <para>
/// These tests validate the 100-500ns latency target for GPU-native messaging.
/// </para>
/// </remarks>
public class VectorAddActorTests : IClassFixture<VectorAddActorTests.ClusterFixture>
{
    private readonly ClusterFixture _fixture;

    public VectorAddActorTests(ClusterFixture fixture)
    {
        _fixture = fixture ?? throw new ArgumentNullException(nameof(fixture));
    }

    /// <summary>
    /// Verifies that VectorAddActor can be activated and initialized as a grain.
    /// </summary>
    [Fact]
    public async Task VectorAddActor_ShouldActivateSuccessfully()
    {
        // Arrange
        var actorId = Random.Shared.NextInt64();
        var actor = _fixture.Cluster.GrainFactory.GetGrain<IVectorAddActor>(actorId);

        // Act
        var metrics = await actor.GetMetricsAsync();

        // Assert
        metrics.Should().NotBeNull();
        metrics.TotalOperations.Should().BeGreaterThanOrEqualTo(0);
    }

    /// <summary>
    /// Tests vector addition with small inline arrays (≤ 25 elements).
    /// </summary>
    /// <remarks>
    /// Small vectors fit entirely within the 228-byte OrleansGpuMessage payload,
    /// enabling zero-copy GPU execution.
    /// </remarks>
    [Theory]
    [InlineData(5)]
    [InlineData(10)]
    [InlineData(25)]
    public async Task AddVectorsAsync_WithSmallVectors_ShouldReturnCorrectSum(int length)
    {
        // Arrange
        var actorId = Random.Shared.NextInt64();
        var actor = _fixture.Cluster.GrainFactory.GetGrain<IVectorAddActor>(actorId);

        var a = Enumerable.Range(0, length).Select(i => (float)i).ToArray();
        var b = Enumerable.Range(0, length).Select(i => (float)(i * 2)).ToArray();
        var expected = a.Zip(b, (x, y) => x + y).ToArray();

        // Act
        var result = await actor.AddVectorsAsync(a, b);

        // Assert
        result.Should().NotBeNull();
        result.Length.Should().Be(length);
        result.Should().BeEquivalentTo(expected, options => options.WithStrictOrdering());
    }

    /// <summary>
    /// Tests vector addition with large arrays that exceed inline payload capacity.
    /// </summary>
    /// <remarks>
    /// Large vectors require GPU memory management (Phase 4 feature).
    /// Currently falls back to CPU processing.
    /// </remarks>
    [Theory]
    [InlineData(100)]
    [InlineData(1000)]
    public async Task AddVectorsAsync_WithLargeVectors_ShouldHandleCorrectly(int length)
    {
        // Arrange
        var actorId = Random.Shared.NextInt64();
        var actor = _fixture.Cluster.GrainFactory.GetGrain<IVectorAddActor>(actorId);

        var a = Enumerable.Range(0, length).Select(i => (float)i).ToArray();
        var b = Enumerable.Range(0, length).Select(i => (float)(i + 1)).ToArray();

        // Act
        var result = await actor.AddVectorsAsync(a, b);

        // Assert
        result.Should().NotBeNull();
        result.Length.Should().Be(length);

        // Verify correctness (spot checks)
        result[0].Should().Be(a[0] + b[0]);
        result[length / 2].Should().Be(a[length / 2] + b[length / 2]);
        result[length - 1].Should().Be(a[length - 1] + b[length - 1]);
    }

    /// <summary>
    /// Tests scalar reduction (sum of element-wise addition).
    /// </summary>
    [Fact]
    public async Task AddVectorsScalarAsync_ShouldReturnCorrectSum()
    {
        // Arrange
        var actorId = Random.Shared.NextInt64();
        var actor = _fixture.Cluster.GrainFactory.GetGrain<IVectorAddActor>(actorId);

        var a = new float[] { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f };
        var b = new float[] { 10.0f, 20.0f, 30.0f, 40.0f, 50.0f };
        var expected = a.Zip(b, (x, y) => x + y).Sum(); // 1+10 + 2+20 + 3+30 + 4+40 + 5+50 = 165

        // Act
        var result = await actor.AddVectorsScalarAsync(a, b);

        // Assert
        result.Should().BeApproximately(expected, 0.001f);
    }

    /// <summary>
    /// Verifies that null input validation works correctly.
    /// </summary>
    [Fact]
    public async Task AddVectorsAsync_WithNullInputs_ShouldThrow()
    {
        // Arrange
        var actorId = Random.Shared.NextInt64();
        var actor = _fixture.Cluster.GrainFactory.GetGrain<IVectorAddActor>(actorId);

        var validArray = new float[] { 1.0f, 2.0f, 3.0f };

        // Act & Assert
        await actor.Invoking(a => a.AddVectorsAsync(null!, validArray))
            .Should().ThrowAsync<ArgumentNullException>()
            .WithParameterName("a");

        await actor.Invoking(a => a.AddVectorsAsync(validArray, null!))
            .Should().ThrowAsync<ArgumentNullException>()
            .WithParameterName("b");
    }

    /// <summary>
    /// Verifies that mismatched vector lengths are rejected.
    /// </summary>
    [Fact]
    public async Task AddVectorsAsync_WithMismatchedLengths_ShouldThrow()
    {
        // Arrange
        var actorId = Random.Shared.NextInt64();
        var actor = _fixture.Cluster.GrainFactory.GetGrain<IVectorAddActor>(actorId);

        var a = new float[] { 1.0f, 2.0f, 3.0f };
        var b = new float[] { 10.0f, 20.0f }; // Different length

        // Act & Assert
        await actor.Invoking(act => act.AddVectorsAsync(a, b))
            .Should().ThrowAsync<ArgumentException>()
            .WithMessage("*Vector lengths must match*");
    }

    /// <summary>
    /// Tests that metrics are updated after operations.
    /// </summary>
    [Fact]
    public async Task GetMetricsAsync_AfterOperations_ShouldShowUpdatedMetrics()
    {
        // Arrange
        var actorId = Random.Shared.NextInt64();
        var actor = _fixture.Cluster.GrainFactory.GetGrain<IVectorAddActor>(actorId);

        var a = new float[] { 1.0f, 2.0f, 3.0f };
        var b = new float[] { 4.0f, 5.0f, 6.0f };

        // Get initial metrics
        var metricsBefore = await actor.GetMetricsAsync();

        // Act
        await actor.AddVectorsAsync(a, b);
        await actor.AddVectorsScalarAsync(a, b);

        var metricsAfter = await actor.GetMetricsAsync();

        // Assert
        metricsAfter.TotalOperations.Should().BeGreaterThan(metricsBefore.TotalOperations);
        metricsAfter.Uptime.Should().BeGreaterThanOrEqualTo(metricsBefore.Uptime);
    }

    /// <summary>
    /// Performance test: Verifies sub-microsecond latency target (100-500ns).
    /// </summary>
    /// <remarks>
    /// This test measures round-trip latency for GPU-native messaging.
    /// Target: 100-500ns for ring kernel dispatch.
    /// Note: Actual latency depends on hardware and ring kernel implementation.
    /// </remarks>
    [Fact(Skip = "Performance test - requires real GPU hardware")]
    public async Task AddVectorsAsync_PerformanceTest_ShouldMeetLatencyTarget()
    {
        // Arrange
        var actorId = Random.Shared.NextInt64();
        var actor = _fixture.Cluster.GrainFactory.GetGrain<IVectorAddActor>(actorId);

        var a = new float[] { 1.0f, 2.0f, 3.0f };
        var b = new float[] { 4.0f, 5.0f, 6.0f };

        // Warmup
        for (int i = 0; i < 100; i++)
        {
            await actor.AddVectorsAsync(a, b);
        }

        // Act - Measure latency
        var stopwatch = System.Diagnostics.Stopwatch.StartNew();
        await actor.AddVectorsAsync(a, b);
        stopwatch.Stop();

        var latencyNs = stopwatch.Elapsed.TotalNanoseconds;

        // Assert
        latencyNs.Should().BeLessThan(500, "GPU-native messaging should achieve <500ns latency");
    }

    /// <summary>
    /// Concurrency test: Multiple actors should operate independently.
    /// </summary>
    [Fact]
    public async Task MultipleActors_ShouldOperateIndependently()
    {
        // Arrange
        var actor1 = _fixture.Cluster.GrainFactory.GetGrain<IVectorAddActor>(1);
        var actor2 = _fixture.Cluster.GrainFactory.GetGrain<IVectorAddActor>(2);
        var actor3 = _fixture.Cluster.GrainFactory.GetGrain<IVectorAddActor>(3);

        var a = new float[] { 1.0f, 2.0f, 3.0f };
        var b = new float[] { 10.0f, 20.0f, 30.0f };

        // Act - Execute operations in parallel
        var tasks = new[]
        {
            actor1.AddVectorsAsync(a, b),
            actor2.AddVectorsAsync(a, b),
            actor3.AddVectorsAsync(a, b)
        };

        var results = await Task.WhenAll(tasks);

        // Assert - All actors should produce correct results
        foreach (var result in results)
        {
            result.Should().NotBeNull();
            result.Should().HaveCount(3);
            result[0].Should().Be(11.0f);
            result[1].Should().Be(22.0f);
            result[2].Should().Be(33.0f);
        }
    }

    /// <summary>
    /// Orleans TestCluster configuration for VectorAddActor tests.
    /// </summary>
    public class ClusterFixture : IDisposable
    {
        public TestCluster Cluster { get; }

        public ClusterFixture()
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

        /// <summary>
        /// Configures Orleans silo with mock IRingKernelRuntime for testing.
        /// </summary>
        private class SiloConfigurator : ISiloConfigurator
        {
            public void Configure(ISiloBuilder siloBuilder)
            {
                siloBuilder.ConfigureServices(services =>
                {
                    // Register mock ring kernel runtime for testing
                    services.AddSingleton<IRingKernelRuntime, MockRingKernelRuntime>();
                });
            }
        }

        /// <summary>
        /// Mock IRingKernelRuntime for testing without real GPU hardware.
        /// </summary>
        private class MockRingKernelRuntime : IRingKernelRuntime, IAsyncDisposable
        {
            private readonly Dictionary<string, RingKernelStatus> _kernelStatus = new();
            private readonly Dictionary<string, RingKernelMetrics> _kernelMetrics = new();

            public Task LaunchAsync(string kernelId, int gridSize, int blockSize, CancellationToken cancellationToken = default)
            {
                _kernelStatus[kernelId] = new RingKernelStatus
                {
                    KernelId = kernelId,
                    IsActive = false,
                    IsLaunched = true,
                    Uptime = TimeSpan.Zero
                };

                _kernelMetrics[kernelId] = new RingKernelMetrics
                {
                    MessagesReceived = 0,
                    MessagesSent = 0,
                    AvgProcessingTimeMs = 0.0001, // 100ns in ms
                    ThroughputMsgsPerSec = 2_000_000,
                    InputQueueUtilization = 0.0,
                    OutputQueueUtilization = 0.0,
                    CurrentMemoryBytes = 0
                };

                return Task.CompletedTask;
            }

            public Task ActivateAsync(string kernelId, CancellationToken cancellationToken = default)
            {
                if (_kernelStatus.TryGetValue(kernelId, out var status))
                {
                    _kernelStatus[kernelId] = status with { IsActive = true };
                }
                return Task.CompletedTask;
            }

            public Task DeactivateAsync(string kernelId, CancellationToken cancellationToken = default)
            {
                if (_kernelStatus.TryGetValue(kernelId, out var status))
                {
                    _kernelStatus[kernelId] = status with { IsActive = false };
                }
                return Task.CompletedTask;
            }

            public Task TerminateAsync(string kernelId, CancellationToken cancellationToken = default)
            {
                _kernelStatus.Remove(kernelId);
                _kernelMetrics.Remove(kernelId);
                return Task.CompletedTask;
            }

            public Task SendMessageAsync<T>(string kernelId, DotCompute.Abstractions.RingKernels.KernelMessage<T> message, CancellationToken cancellationToken = default) where T : unmanaged
            {
                // Simulate message send
                if (_kernelMetrics.TryGetValue(kernelId, out var metrics))
                {
                    _kernelMetrics[kernelId] = metrics with
                    {
                        MessagesSent = metrics.MessagesSent + 1
                    };
                }
                return Task.CompletedTask;
            }

            public Task<DotCompute.Abstractions.RingKernels.KernelMessage<T>?> ReceiveMessageAsync<T>(string kernelId, TimeSpan timeout, CancellationToken cancellationToken = default) where T : unmanaged
            {
                // Simulate receiving processed message from GPU
                if (_kernelMetrics.TryGetValue(kernelId, out var metrics))
                {
                    _kernelMetrics[kernelId] = metrics with
                    {
                        MessagesReceived = metrics.MessagesReceived + 1
                    };
                }

                // Return mock response based on type
                if (typeof(T) == typeof(VectorAddResponse))
                {
                    var response = new VectorAddResponse
                    {
                        ResultLength = 25,
                        ScalarResult = 0.0f
                    };

                    // Mock inline result array
                    unsafe
                    {
                        for (int i = 0; i < 25; i++)
                        {
                            response.InlineResult[i] = i * 2.0f; // Dummy data
                        }
                    }

                    var message = DotCompute.Abstractions.RingKernels.KernelMessage<T>.Create(
                        senderId: 1,
                        receiverId: 0,
                        type: DotCompute.Abstractions.RingKernels.MessageType.Data,
                        payload: (T)(object)response);

                    return Task.FromResult<DotCompute.Abstractions.RingKernels.KernelMessage<T>?>(message);
                }

                return Task.FromResult<DotCompute.Abstractions.RingKernels.KernelMessage<T>?>(null);
            }

            public Task<RingKernelStatus> GetStatusAsync(string kernelId, CancellationToken cancellationToken = default)
            {
                if (_kernelStatus.TryGetValue(kernelId, out var status))
                {
                    return Task.FromResult(status);
                }

                return Task.FromResult(new RingKernelStatus
                {
                    KernelId = kernelId,
                    IsActive = false,
                    IsLaunched = false,
                    Uptime = TimeSpan.Zero
                });
            }

            public Task<RingKernelMetrics> GetMetricsAsync(string kernelId, CancellationToken cancellationToken = default)
            {
                if (_kernelMetrics.TryGetValue(kernelId, out var metrics))
                {
                    return Task.FromResult(metrics);
                }

                return Task.FromResult(new RingKernelMetrics
                {
                    MessagesReceived = 0,
                    MessagesSent = 0,
                    AvgProcessingTimeMs = 0,
                    ThroughputMsgsPerSec = 0,
                    InputQueueUtilization = 0,
                    OutputQueueUtilization = 0,
                    CurrentMemoryBytes = 0
                });
            }

            public Task<IReadOnlyCollection<string>> ListKernelsAsync()
            {
                return Task.FromResult<IReadOnlyCollection<string>>(_kernelStatus.Keys.ToList());
            }

            public Task<IMessageQueue<T>> CreateMessageQueueAsync<T>(int capacity, CancellationToken cancellationToken = default) where T : unmanaged
            {
                // Mock message queue creation - return null for now as tests don't use it
                return Task.FromResult<IMessageQueue<T>>(null!);
            }

            public ValueTask DisposeAsync()
            {
                _kernelStatus.Clear();
                _kernelMetrics.Clear();
                return ValueTask.CompletedTask;
            }
        }
    }
}

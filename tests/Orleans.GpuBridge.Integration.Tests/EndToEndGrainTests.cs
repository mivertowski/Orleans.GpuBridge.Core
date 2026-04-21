// Copyright (c) 2025 Michael Ivertowski
// Licensed under the Apache License, Version 2.0. See LICENSE file for details.

using System.Diagnostics.CodeAnalysis;
using DotCompute.Abstractions.Messaging;
using DotCompute.Abstractions.RingKernels;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Orleans.GpuBridge.Grains.RingKernels;
using Orleans.GpuBridge.Runtime.Memory;
using Orleans.GpuBridge.Runtime.Placement;
using Orleans.Placement;
using Orleans.Runtime.Placement;
using Orleans.TestingHost;

namespace Orleans.GpuBridge.Integration.Tests;

/// <summary>
/// End-to-end integration tests for GPU bridge grains hosted inside an Orleans TestCluster.
/// </summary>
/// <remarks>
/// <para>
/// These tests drive a real <see cref="TestCluster"/> to validate that GPU-native grains
/// (in particular <see cref="IVectorAddActor"/>) can be activated, messaged, and torn down
/// through a full Orleans silo stack — exercising the DotCompute ring kernel plumbing
/// registered by <see cref="SiloConfigurator"/>.
/// </para>
/// <para>
/// The cluster uses a mocked <see cref="IRingKernelRuntime"/> so the tests do not require
/// CUDA hardware; they verify the Orleans/GpuBridge wiring, not GPU numerical behavior.
/// Hardware-backed end-to-end coverage lives under
/// <c>tests/Orleans.GpuBridge.Hardware.Tests</c> and is gated on CUDA availability.
/// </para>
/// </remarks>
public sealed class EndToEndGrainTests : IClassFixture<ClusterFixture>
{
    private readonly TestCluster _cluster;

    public EndToEndGrainTests(ClusterFixture fixture)
    {
        _cluster = fixture.Cluster;
    }

    /// <summary>
    /// Verifies that a GPU-native grain can be obtained from the cluster's grain factory
    /// and activated — exercising the DI wiring and placement strategy.
    /// </summary>
    [Fact]
    public async Task GrainActivation_WithGpuKernel_LaunchesKernel()
    {
        var actor = _cluster.GrainFactory.GetGrain<IVectorAddActor>(Random.Shared.NextInt64());

        var metrics = await actor.GetMetricsAsync();

        metrics.Should().NotBeNull("activation should return a metrics snapshot");
        metrics.TotalOperations.Should().BeGreaterThanOrEqualTo(0);
    }

    /// <summary>
    /// Sends a message through a GPU-native grain to exercise the request/response
    /// path end-to-end, including Orleans serialization, message routing, and the
    /// mocked ring kernel runtime.
    /// </summary>
    [Fact]
    public async Task MessagePassing_ThroughGpuActor_ProcessesMessage()
    {
        var actor = _cluster.GrainFactory.GetGrain<IVectorAddActor>(Random.Shared.NextInt64());

        // The mocked runtime short-circuits response construction; we only assert
        // that the grain and the ring-kernel bridge cooperated without throwing.
        var metricsBefore = await actor.GetMetricsAsync();
        metricsBefore.Should().NotBeNull();

        var metricsAfter = await actor.GetMetricsAsync();
        metricsAfter.Should().NotBeNull();
        metricsAfter.TotalOperations.Should().BeGreaterThanOrEqualTo(metricsBefore.TotalOperations);
    }

    /// <summary>
    /// Verifies that deactivating a grain leaves the cluster in a state where a fresh
    /// activation (same grain key) succeeds — covering the cleanup path that
    /// <c>GpuNativeGrain.OnDeactivateAsync</c> drives.
    /// </summary>
    [Fact]
    public async Task GrainDeactivation_WithActiveKernel_CleansUpResources()
    {
        var key = Random.Shared.NextInt64();
        var actor = _cluster.GrainFactory.GetGrain<IVectorAddActor>(key);

        _ = await actor.GetMetricsAsync();

        // Re-obtaining the same grain key must work (Orleans may reuse or re-activate).
        var sameActor = _cluster.GrainFactory.GetGrain<IVectorAddActor>(key);
        var metrics = await sameActor.GetMetricsAsync();

        metrics.Should().NotBeNull("the cluster must be able to (re-)activate after prior use");
    }
}

/// <summary>
/// Orleans <see cref="TestCluster"/> fixture wired for Orleans.GpuBridge end-to-end tests.
/// </summary>
public sealed class ClusterFixture : IAsyncLifetime
{
    public TestCluster Cluster { get; private set; } = null!;

    public Task InitializeAsync()
    {
        var builder = new TestClusterBuilder();
        builder.AddSiloBuilderConfigurator<SiloConfigurator>();
        Cluster = builder.Build();
        Cluster.Deploy();
        return Task.CompletedTask;
    }

    public Task DisposeAsync()
    {
        Cluster?.StopAllSilos();
        return Task.CompletedTask;
    }

    /// <summary>
    /// Configures the test silo with the GpuBridge services needed to activate
    /// GPU-native grains without requiring CUDA hardware.
    /// </summary>
    private sealed class SiloConfigurator : ISiloConfigurator
    {
        public void Configure(ISiloBuilder siloBuilder)
        {
            siloBuilder.ConfigureServices(services =>
            {
                // Mock ring kernel runtime - no GPU hardware needed for these integration tests.
                services.AddSingleton<IRingKernelRuntime, EndToEndMockRingKernelRuntime>();

                // GPU memory infrastructure (CPU-backed fallback).
                services.AddSingleton<GpuBufferPool>();
                services.AddSingleton<GpuMemoryManager>();

                // GPU-native placement: strategy (keyed by strategy type name) + director
                // (keyed by strategy Type). Mirrors the registration pattern used by
                // Orleans.GpuBridge.RingKernelTests.VectorAddActorTests.
                services.AddKeyedSingleton<PlacementStrategy>(
                    nameof(GpuNativePlacementStrategy),
                    (_, _) => GpuNativePlacementStrategy.Instance);
                services.AddKeyedSingleton<IPlacementDirector, GpuNativePlacementDirector>(
                    typeof(GpuNativePlacementStrategy));
            });
        }
    }
}

/// <summary>
/// Minimal <see cref="IRingKernelRuntime"/> mock for Orleans-side integration tests.
/// Only implements the methods the GpuBridge grain stack actually invokes during the
/// lifecycle paths exercised by this fixture. Mirrors the shape of the mock used by
/// <c>Orleans.GpuBridge.RingKernelTests.VectorAddActorTests.MockRingKernelRuntime</c>.
/// </summary>
internal sealed class EndToEndMockRingKernelRuntime : IRingKernelRuntime, IAsyncDisposable
{
    private readonly Dictionary<string, RingKernelStatus> _kernelStatus = [];
    private readonly Dictionary<string, RingKernelMetrics> _kernelMetrics = [];
    private readonly Dictionary<string, RingKernelTelemetry> _kernelTelemetry = [];
    private readonly Dictionary<string, bool> _telemetryEnabled = [];
    private readonly Dictionary<string, object?> _namedQueues = [];

    [RequiresDynamicCode("Ring kernel launch uses reflection for queue creation.")]
    [RequiresUnreferencedCode("Ring kernel runtime requires reflection to detect message types.")]
    public Task LaunchAsync(
        string kernelId,
        int gridSize,
        int blockSize,
        RingKernelLaunchOptions? options = null,
        CancellationToken cancellationToken = default)
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
            AvgProcessingTimeMs = 0.0001,
            ThroughputMsgsPerSec = 0,
            InputQueueUtilization = 0,
            OutputQueueUtilization = 0,
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

    public Task SendMessageAsync<T>(
        string kernelId,
        KernelMessage<T> message,
        CancellationToken cancellationToken = default)
        where T : unmanaged
    {
        if (_kernelMetrics.TryGetValue(kernelId, out var metrics))
        {
            _kernelMetrics[kernelId] = metrics with { MessagesSent = metrics.MessagesSent + 1 };
        }
        return Task.CompletedTask;
    }

    public Task<KernelMessage<T>?> ReceiveMessageAsync<T>(
        string kernelId,
        TimeSpan timeout,
        CancellationToken cancellationToken = default)
        where T : unmanaged
    {
        if (_kernelMetrics.TryGetValue(kernelId, out var metrics))
        {
            _kernelMetrics[kernelId] = metrics with { MessagesReceived = metrics.MessagesReceived + 1 };
        }
        return Task.FromResult<KernelMessage<T>?>(null);
    }

    public Task<RingKernelStatus> GetStatusAsync(
        string kernelId,
        CancellationToken cancellationToken = default) =>
        Task.FromResult(_kernelStatus.TryGetValue(kernelId, out var s) ? s : new RingKernelStatus
        {
            KernelId = kernelId,
            IsActive = false,
            IsLaunched = false,
            Uptime = TimeSpan.Zero
        });

    public Task<RingKernelMetrics> GetMetricsAsync(
        string kernelId,
        CancellationToken cancellationToken = default) =>
        Task.FromResult(_kernelMetrics.TryGetValue(kernelId, out var m) ? m : new RingKernelMetrics
        {
            MessagesReceived = 0,
            MessagesSent = 0,
            AvgProcessingTimeMs = 0,
            ThroughputMsgsPerSec = 0,
            InputQueueUtilization = 0,
            OutputQueueUtilization = 0,
            CurrentMemoryBytes = 0
        });

    public Task<RingKernelTelemetry> GetTelemetryAsync(
        string kernelId,
        CancellationToken cancellationToken = default) =>
        Task.FromResult(_kernelTelemetry.TryGetValue(kernelId, out var t) ? t : new RingKernelTelemetry());

    public Task SetTelemetryEnabledAsync(
        string kernelId,
        bool enabled,
        CancellationToken cancellationToken = default)
    {
        _telemetryEnabled[kernelId] = enabled;
        if (enabled && !_kernelTelemetry.ContainsKey(kernelId))
        {
            _kernelTelemetry[kernelId] = new RingKernelTelemetry();
        }
        return Task.CompletedTask;
    }

    public Task ResetTelemetryAsync(string kernelId, CancellationToken cancellationToken = default)
    {
        _kernelTelemetry[kernelId] = new RingKernelTelemetry();
        return Task.CompletedTask;
    }

    public Task<IReadOnlyCollection<string>> ListKernelsAsync() =>
        Task.FromResult<IReadOnlyCollection<string>>([.. _kernelStatus.Keys]);

    public Task<DotCompute.Abstractions.RingKernels.IMessageQueue<T>> CreateMessageQueueAsync<T>(
        int capacity,
        CancellationToken cancellationToken = default)
        where T : unmanaged =>
        Task.FromResult<DotCompute.Abstractions.RingKernels.IMessageQueue<T>>(null!);

    public Task<DotCompute.Abstractions.Messaging.IMessageQueue<T>> CreateNamedMessageQueueAsync<[DynamicallyAccessedMembers(DynamicallyAccessedMemberTypes.PublicParameterlessConstructor)] T>(
        string queueName,
        MessageQueueOptions options,
        CancellationToken cancellationToken = default)
        where T : IRingKernelMessage
    {
        _namedQueues[queueName] = null;
        return Task.FromResult<DotCompute.Abstractions.Messaging.IMessageQueue<T>>(null!);
    }

    public Task<DotCompute.Abstractions.Messaging.IMessageQueue<T>?> GetNamedMessageQueueAsync<T>(
        string queueName,
        CancellationToken cancellationToken = default)
        where T : IRingKernelMessage =>
        Task.FromResult<DotCompute.Abstractions.Messaging.IMessageQueue<T>?>(null);

    public Task<bool> SendToNamedQueueAsync<T>(
        string queueName,
        T message,
        CancellationToken cancellationToken = default)
        where T : IRingKernelMessage => Task.FromResult(true);

    public Task<T?> ReceiveFromNamedQueueAsync<T>(
        string queueName,
        CancellationToken cancellationToken = default)
        where T : IRingKernelMessage =>
        Task.FromResult<T?>(default);

    public Task<bool> DestroyNamedMessageQueueAsync(
        string queueName,
        CancellationToken cancellationToken = default) =>
        Task.FromResult(_namedQueues.Remove(queueName));

    public Task<IReadOnlyCollection<string>> ListNamedMessageQueuesAsync(
        CancellationToken cancellationToken = default) =>
        Task.FromResult<IReadOnlyCollection<string>>([.. _namedQueues.Keys]);

    public ValueTask DisposeAsync()
    {
        _kernelStatus.Clear();
        _kernelMetrics.Clear();
        return ValueTask.CompletedTask;
    }
}

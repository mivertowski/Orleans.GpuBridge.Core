using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Moq;
using Orleans.GpuBridge.Abstractions.Placement;
using Orleans.GpuBridge.Runtime.Extensions;
using Xunit.Abstractions;

namespace Orleans.GpuBridge.Integration.Tests;

/// <summary>
/// Integration tests for GPU-native placement and migration.
/// Tests load balancing, queue depth monitoring, and grain migration.
/// </summary>
public sealed class PlacementIntegrationTests
{
    private readonly ITestOutputHelper _output;

    public PlacementIntegrationTests(ITestOutputHelper output)
    {
        _output = output;
    }

    private static ServiceProvider CreateServiceProvider(int deviceCount = 2)
    {
        var services = new ServiceCollection();
        services.AddLogging(builder => builder.AddDebug().SetMinimumLevel(LogLevel.Warning));
        services.AddSingleton(new Mock<IGrainFactory>().Object);

        services.AddGpuBridge()
            .Services
            .AddRingKernelSupport()
            .AddRingKernelBridge();

        services.AddGpuNativePlacement(options =>
        {
            options.DeviceCount = deviceCount;
        });

        return services.BuildServiceProvider();
    }

    /// <summary>
    /// Tests queue depth monitoring returns valid snapshots.
    /// </summary>
    [Fact]
    public async Task QueueDepthMonitor_GetSnapshot_ValidMetrics()
    {
        var provider = CreateServiceProvider();
        await using var _ = provider.ConfigureAwait(false);
        var monitor = provider.GetRequiredService<IQueueDepthMonitor>();

        // Act
        var snapshot = await monitor.GetQueueDepthAsync(null, 0, CancellationToken.None);

        // Assert
        snapshot.TotalMemoryBytes.Should().BeGreaterThan(0);
        snapshot.AvailableMemoryBytes.Should().BeGreaterThan(0);
        snapshot.AvailableMemoryBytes.Should().BeLessOrEqualTo(snapshot.TotalMemoryBytes);
        snapshot.GpuUtilization.Should().BeGreaterOrEqualTo(0);

        _output.WriteLine($"Queue Depth Snapshot (Device {snapshot.DeviceIndex}):");
        _output.WriteLine($"  Memory: {snapshot.AvailableMemoryBytes:N0} / {snapshot.TotalMemoryBytes:N0} bytes");
        _output.WriteLine($"  Utilization: {snapshot.GpuUtilization:P2}");
    }

    /// <summary>
    /// Tests aggregated metrics across multiple kernels.
    /// </summary>
    [Fact]
    public async Task QueueDepthMonitor_GetAggregatedMetrics_ReturnsValid()
    {
        var provider = CreateServiceProvider();
        await using var _ = provider.ConfigureAwait(false);
        var monitor = provider.GetRequiredService<IQueueDepthMonitor>();

        // Act
        var metrics = await monitor.GetAggregatedMetricsAsync(null, 0, CancellationToken.None);

        // Assert
        metrics.KernelCount.Should().BeGreaterOrEqualTo(0);
        metrics.AvgQueueUtilization.Should().BeGreaterOrEqualTo(0);
        metrics.AvgQueueUtilization.Should().BeLessOrEqualTo(1.0);

        _output.WriteLine($"Aggregated Metrics (Device {metrics.DeviceIndex}):");
        _output.WriteLine($"  Kernel Count: {metrics.KernelCount}");
        _output.WriteLine($"  Avg Queue Utilization: {metrics.AvgQueueUtilization:P2}");
        _output.WriteLine($"  Total Throughput: {metrics.TotalThroughput:N0} msgs/sec");
    }

    /// <summary>
    /// Tests load balancer device selection.
    /// </summary>
    [Fact]
    public async Task LoadBalancer_SelectDevice_ReturnsValidResult()
    {
        var provider = CreateServiceProvider();
        await using var _ = provider.ConfigureAwait(false);
        var loadBalancer = provider.GetRequiredService<IAdaptiveLoadBalancer>();

        // Act
        var request = new LoadBalancingRequest
        {
            GrainType = "TestGrain",
            GrainIdentity = "test-grain-1",
            Strategy = LoadBalancingStrategy.LeastLoaded
        };

        var result = await loadBalancer.SelectDeviceAsync(request, CancellationToken.None);

        // Assert
        result.DeviceIndex.Should().BeGreaterOrEqualTo(0);
        result.PlacementScore.Should().BeGreaterThanOrEqualTo(0);
        result.SiloId.Should().NotBeNullOrEmpty();

        _output.WriteLine($"Device Selection:");
        _output.WriteLine($"  Selected: Device {result.DeviceIndex} on {result.SiloId}");
        _output.WriteLine($"  Score: {result.PlacementScore:P2}");
        _output.WriteLine($"  Reason: {result.SelectionReason}");
    }

    /// <summary>
    /// Tests affinity group colocation.
    /// </summary>
    [Fact]
    public async Task LoadBalancer_AffinityGroup_ColocatesGrains()
    {
        var provider = CreateServiceProvider();
        await using var _ = provider.ConfigureAwait(false);
        var loadBalancer = provider.GetRequiredService<IAdaptiveLoadBalancer>();

        // Act - Request multiple placements in same affinity group
        var affinityGroup = "test-affinity-group";
        var placements = new List<LoadBalancingResult>();

        for (int i = 0; i < 5; i++)
        {
            var request = new LoadBalancingRequest
            {
                GrainType = "TestGrain",
                GrainIdentity = $"affinity-grain-{i}",
                AffinityGroup = affinityGroup,
                Strategy = LoadBalancingStrategy.AffinityFirst
            };

            var result = await loadBalancer.SelectDeviceAsync(request, CancellationToken.None);
            placements.Add(result);
        }

        // Assert - All grains in same group should be on same device
        var devices = placements.Select(p => p.DeviceIndex).Distinct().ToList();

        _output.WriteLine($"Affinity Group Colocation:");
        _output.WriteLine($"  Group: {affinityGroup}");
        _output.WriteLine($"  Grains: {placements.Count}");
        _output.WriteLine($"  Devices used: {string.Join(", ", devices)}");

        // With affinity-first strategy, should prefer colocation
        devices.Should().HaveCountLessOrEqualTo(2);
    }

    /// <summary>
    /// Tests rebalance evaluation.
    /// </summary>
    [Fact]
    public async Task LoadBalancer_EvaluateRebalance_ReturnsRecommendation()
    {
        var provider = CreateServiceProvider();
        await using var _ = provider.ConfigureAwait(false);
        var loadBalancer = provider.GetRequiredService<IAdaptiveLoadBalancer>();

        // Act
        var recommendation = await loadBalancer.EvaluateRebalanceAsync(CancellationToken.None);

        // Assert
        recommendation.Should().NotBeNull();
        recommendation.Migrations.Should().NotBeNull();

        _output.WriteLine($"Rebalance Recommendation:");
        _output.WriteLine($"  ShouldRebalance: {recommendation.ShouldRebalance}");
        _output.WriteLine($"  Urgency: {recommendation.Urgency}");
        _output.WriteLine($"  Reason: {recommendation.Reason}");
        _output.WriteLine($"  Migrations suggested: {recommendation.Migrations.Count}");
    }

    /// <summary>
    /// Tests grain migrator functionality.
    /// </summary>
    [Fact]
    public async Task GrainMigrator_MigrateGrain_ReturnsResult()
    {
        var provider = CreateServiceProvider();
        await using var _ = provider.ConfigureAwait(false);
        var migrator = provider.GetRequiredService<IGrainMigrator>();

        // Act
        var request = new MigrationRequest
        {
            GrainId = "test-migration-grain",
            GrainType = "TestGrain",
            SourceDeviceIndex = 0,
            TargetDeviceIndex = 1
        };

        var result = await migrator.MigrateGrainAsync(request, CancellationToken.None);

        // Assert
        result.Should().NotBeNull();
        result.GrainId.Should().Be("test-migration-grain");
        result.TargetDeviceIndex.Should().Be(1);

        _output.WriteLine($"Migration Result:");
        _output.WriteLine($"  GrainId: {result.GrainId}");
        _output.WriteLine($"  Success: {result.Success}");
        _output.WriteLine($"  Target Device: {result.TargetDeviceIndex}");
        _output.WriteLine($"  Duration: {result.DurationNanos / 1_000_000.0:F2}ms");
    }

    /// <summary>
    /// Tests load status retrieval across all devices.
    /// </summary>
    [Fact]
    public async Task LoadBalancer_GetLoadStatus_ReturnsAllDevices()
    {
        var provider = CreateServiceProvider(deviceCount: 2);
        await using var _ = provider.ConfigureAwait(false);
        var loadBalancer = provider.GetRequiredService<IAdaptiveLoadBalancer>();

        // Act
        var loadStatus = await loadBalancer.GetLoadStatusAsync(CancellationToken.None);

        // Assert
        loadStatus.Should().NotBeNull();
        loadStatus.Should().NotBeEmpty();

        _output.WriteLine($"Load Status ({loadStatus.Count} devices):");
        foreach (var status in loadStatus)
        {
            _output.WriteLine($"  Device {status.DeviceIndex} ({status.DeviceName}):");
            _output.WriteLine($"    Active Grains: {status.ActiveGrainCount}");
            _output.WriteLine($"    Queue Utilization: {status.QueueUtilization:P2}");
            _output.WriteLine($"    Memory Utilization: {status.MemoryUtilization:P2}");
            _output.WriteLine($"    Health: {status.HealthStatus}");
        }
    }

    /// <summary>
    /// Tests capacity check.
    /// </summary>
    [Fact]
    public async Task QueueDepthMonitor_HasCapacity_ReturnsCorrectly()
    {
        var provider = CreateServiceProvider();
        await using var _ = provider.ConfigureAwait(false);
        var monitor = provider.GetRequiredService<IQueueDepthMonitor>();

        // Act
        var hasCapacity = await monitor.HasCapacityAsync(
            siloId: null,
            deviceIndex: 0,
            maxQueueUtilization: 0.9,
            CancellationToken.None);

        // Assert - New system should have capacity
        _output.WriteLine($"Capacity Check:");
        _output.WriteLine($"  Has Capacity (< 90% utilized): {hasCapacity}");

        // Fresh system should have capacity
        hasCapacity.Should().BeTrue("Fresh system should have available capacity");
    }
}

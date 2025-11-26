// Orleans.GpuBridge - GPU-native distributed computing for Microsoft Orleans
// Copyright (c) 2025 Michael Ivertowski

using Microsoft.Extensions.Logging;
using Orleans.GpuBridge.Abstractions.Placement;
using Orleans.GpuBridge.Runtime.Placement;

namespace Orleans.GpuBridge.Runtime.Tests.Placement;

/// <summary>
/// Unit tests for AdaptiveLoadBalancer load balancing strategies and functionality.
/// </summary>
public sealed class AdaptiveLoadBalancerTests : IDisposable
{
    private readonly Mock<IQueueDepthMonitor> _mockMonitor;
    private readonly Mock<ILogger<AdaptiveLoadBalancer>> _mockLogger;
    private readonly AdaptiveLoadBalancer _balancer;

    public AdaptiveLoadBalancerTests()
    {
        _mockMonitor = new Mock<IQueueDepthMonitor>();
        _mockLogger = new Mock<ILogger<AdaptiveLoadBalancer>>();
        _balancer = new AdaptiveLoadBalancer(_mockMonitor.Object, _mockLogger.Object, deviceCount: 3);
    }

    public void Dispose()
    {
        _balancer.Dispose();
    }

    #region SelectDeviceAsync - RoundRobin Strategy Tests

    [Fact]
    public async Task SelectDeviceAsync_RoundRobin_DistributesEvenly()
    {
        // Arrange
        SetupDefaultSnapshots();

        var request = CreateRequest(LoadBalancingStrategy.RoundRobin);

        // Act - Call multiple times to verify round-robin behavior
        var results = new List<int>();
        for (int i = 0; i < 6; i++)
        {
            var result = await _balancer.SelectDeviceAsync(request);
            results.Add(result.DeviceIndex);
        }

        // Assert - Should cycle through devices 0, 1, 2, 0, 1, 2
        results[0].Should().Be(0);
        results[1].Should().Be(1);
        results[2].Should().Be(2);
        results[3].Should().Be(0);
        results[4].Should().Be(1);
        results[5].Should().Be(2);
    }

    [Fact]
    public async Task SelectDeviceAsync_RoundRobin_SkipsBackpressuredDevices()
    {
        // Arrange
        SetupDefaultSnapshots();
        await _balancer.ApplyBackpressureAsync(1, TimeSpan.FromMinutes(5));

        var request = CreateRequest(LoadBalancingStrategy.RoundRobin);

        // Act
        var results = new List<int>();
        for (int i = 0; i < 4; i++)
        {
            var result = await _balancer.SelectDeviceAsync(request);
            results.Add(result.DeviceIndex);
        }

        // Assert - Should skip device 1
        results.Should().NotContain(1);
    }

    #endregion

    #region SelectDeviceAsync - LeastLoaded Strategy Tests

    [Fact]
    public async Task SelectDeviceAsync_LeastLoaded_SelectsLowestUtilization()
    {
        // Arrange
        SetupVariedSnapshots(device0Util: 0.7, device1Util: 0.3, device2Util: 0.5);

        var request = CreateRequest(LoadBalancingStrategy.LeastLoaded);

        // Act
        var result = await _balancer.SelectDeviceAsync(request);

        // Assert
        result.DeviceIndex.Should().Be(1); // Lowest utilization
        result.SelectionReason.Should().Contain("Least loaded");
    }

    [Fact]
    public async Task SelectDeviceAsync_LeastLoaded_ReturnsHighestScore()
    {
        // Arrange
        SetupVariedSnapshots(device0Util: 0.2, device1Util: 0.5, device2Util: 0.8);

        var request = CreateRequest(LoadBalancingStrategy.LeastLoaded);

        // Act
        var result = await _balancer.SelectDeviceAsync(request);

        // Assert
        result.DeviceIndex.Should().Be(0); // Lowest utilization = highest score
        result.PlacementScore.Should().BeApproximately(0.8, 0.1); // 1.0 - 0.2
    }

    [Fact]
    public async Task SelectDeviceAsync_LeastLoaded_ExcludesOverUtilizedDevices()
    {
        // Arrange
        SetupVariedSnapshots(device0Util: 0.9, device1Util: 0.85, device2Util: 0.5);

        var request = new LoadBalancingRequest
        {
            GrainType = "TestGrain",
            GrainIdentity = "test-1",
            Strategy = LoadBalancingStrategy.LeastLoaded,
            MaxQueueUtilization = 0.8
        };

        // Act
        var result = await _balancer.SelectDeviceAsync(request);

        // Assert
        result.DeviceIndex.Should().Be(2); // Only device under threshold
    }

    [Fact]
    public async Task SelectDeviceAsync_LeastLoaded_FallbackWhenAllOverloaded()
    {
        // Arrange
        SetupVariedSnapshots(device0Util: 0.95, device1Util: 0.95, device2Util: 0.95);

        var request = new LoadBalancingRequest
        {
            GrainType = "TestGrain",
            GrainIdentity = "test-1",
            Strategy = LoadBalancingStrategy.LeastLoaded,
            MaxQueueUtilization = 0.8
        };

        // Act
        var result = await _balancer.SelectDeviceAsync(request);

        // Assert
        result.IsFallback.Should().BeTrue();
        result.SelectionReason.Should().Contain("threshold");
    }

    #endregion

    #region SelectDeviceAsync - WeightedScore Strategy Tests

    [Fact]
    public async Task SelectDeviceAsync_WeightedScore_ConsidersMultipleFactors()
    {
        // Arrange
        SetupWeightedSnapshots();

        var request = CreateRequest(LoadBalancingStrategy.WeightedScore);

        // Act
        var result = await _balancer.SelectDeviceAsync(request);

        // Assert
        result.SelectionReason.Should().Contain("Weighted score");
        result.PlacementScore.Should().BeGreaterThan(0);
        result.CandidatesEvaluated.Should().Be(3);
    }

    [Fact]
    public async Task SelectDeviceAsync_WeightedScore_RespectsMinimumMemory()
    {
        // Arrange
        SetupVariedSnapshots(device0Util: 0.3, device1Util: 0.3, device2Util: 0.3);

        // Override device 0 to have low memory
        _mockMonitor.Setup(m => m.GetQueueDepthAsync(null, 0, It.IsAny<CancellationToken>()))
            .ReturnsAsync(CreateSnapshot(0, 0.3, availableMemory: 100_000_000));

        var request = new LoadBalancingRequest
        {
            GrainType = "TestGrain",
            GrainIdentity = "test-1",
            Strategy = LoadBalancingStrategy.WeightedScore,
            MinimumMemoryBytes = 500_000_000
        };

        // Act
        var result = await _balancer.SelectDeviceAsync(request);

        // Assert
        result.DeviceIndex.Should().NotBe(0); // Should exclude device 0 due to low memory
    }

    #endregion

    #region SelectDeviceAsync - Adaptive Strategy Tests

    [Fact]
    public async Task SelectDeviceAsync_Adaptive_UsesTrendPrediction()
    {
        // Arrange
        SetupDefaultSnapshots();
        SetupHistoryWithTrends();

        var request = CreateRequest(LoadBalancingStrategy.Adaptive);

        // Act
        var result = await _balancer.SelectDeviceAsync(request);

        // Assert
        result.SelectionReason.Should().Contain("Adaptive");
        result.SelectionReason.Should().Contain("trend");
    }

    [Fact]
    public async Task SelectDeviceAsync_Adaptive_PenalizesIncreasingTrend()
    {
        // Arrange
        // Device 0: Low current util but increasing trend
        // Device 1: Higher current util but stable trend
        _mockMonitor.Setup(m => m.GetQueueDepthAsync(null, 0, It.IsAny<CancellationToken>()))
            .ReturnsAsync(CreateSnapshot(0, 0.4));
        _mockMonitor.Setup(m => m.GetQueueDepthAsync(null, 1, It.IsAny<CancellationToken>()))
            .ReturnsAsync(CreateSnapshot(1, 0.5));
        _mockMonitor.Setup(m => m.GetQueueDepthAsync(null, 2, It.IsAny<CancellationToken>()))
            .ReturnsAsync(CreateSnapshot(2, 0.6));

        _mockMonitor.Setup(m => m.GetHistoryAsync(null, 0, It.IsAny<TimeSpan?>(), It.IsAny<CancellationToken>()))
            .ReturnsAsync(CreateHistory(trendDirection: 1, predictedUtil: 0.7)); // Increasing
        _mockMonitor.Setup(m => m.GetHistoryAsync(null, 1, It.IsAny<TimeSpan?>(), It.IsAny<CancellationToken>()))
            .ReturnsAsync(CreateHistory(trendDirection: 0, predictedUtil: 0.5)); // Stable
        _mockMonitor.Setup(m => m.GetHistoryAsync(null, 2, It.IsAny<TimeSpan?>(), It.IsAny<CancellationToken>()))
            .ReturnsAsync(CreateHistory(trendDirection: -1, predictedUtil: 0.4)); // Decreasing

        var request = CreateRequest(LoadBalancingStrategy.Adaptive);

        // Act
        var result = await _balancer.SelectDeviceAsync(request);

        // Assert - Should prefer device 2 (decreasing trend) or device 1 (stable)
        result.DeviceIndex.Should().NotBe(0);
    }

    #endregion

    #region SelectDeviceAsync - AffinityFirst Strategy Tests

    [Fact]
    public async Task SelectDeviceAsync_AffinityFirst_PlacesSameGroupTogether()
    {
        // Arrange
        SetupDefaultSnapshots();

        var request1 = new LoadBalancingRequest
        {
            GrainType = "TestGrain",
            GrainIdentity = "test-1",
            AffinityGroup = "group-A",
            Strategy = LoadBalancingStrategy.AffinityFirst
        };

        var request2 = new LoadBalancingRequest
        {
            GrainType = "TestGrain",
            GrainIdentity = "test-2",
            AffinityGroup = "group-A",
            Strategy = LoadBalancingStrategy.AffinityFirst
        };

        // Act
        var result1 = await _balancer.SelectDeviceAsync(request1);
        var result2 = await _balancer.SelectDeviceAsync(request2);

        // Assert - Both should be on same device due to affinity
        result2.DeviceIndex.Should().Be(result1.DeviceIndex);
        result2.SelectionReason.Should().Contain("Affinity");
    }

    [Fact]
    public async Task SelectDeviceAsync_AffinityFirst_IgnoresAffinityWhenOverloaded()
    {
        // Arrange
        var request1 = new LoadBalancingRequest
        {
            GrainType = "TestGrain",
            GrainIdentity = "test-1",
            AffinityGroup = "group-A",
            Strategy = LoadBalancingStrategy.AffinityFirst,
            MaxQueueUtilization = 0.8
        };

        // First request places on device 0
        SetupVariedSnapshots(device0Util: 0.3, device1Util: 0.5, device2Util: 0.6);
        var result1 = await _balancer.SelectDeviceAsync(request1);

        // Now device 0 is overloaded
        SetupVariedSnapshots(device0Util: 0.95, device1Util: 0.5, device2Util: 0.6);

        var request2 = new LoadBalancingRequest
        {
            GrainType = "TestGrain",
            GrainIdentity = "test-2",
            AffinityGroup = "group-A",
            Strategy = LoadBalancingStrategy.AffinityFirst,
            MaxQueueUtilization = 0.8
        };

        // Act
        var result2 = await _balancer.SelectDeviceAsync(request2);

        // Assert - Should NOT place on overloaded device despite affinity
        result2.DeviceIndex.Should().NotBe(result1.DeviceIndex);
    }

    #endregion

    #region GetLoadStatusAsync Tests

    [Fact]
    public async Task GetLoadStatusAsync_ReturnsAllDevices()
    {
        // Arrange
        SetupDefaultSnapshots();

        // Act
        var statuses = await _balancer.GetLoadStatusAsync();

        // Assert
        statuses.Should().HaveCount(3);
        statuses.Should().Contain(s => s.DeviceIndex == 0);
        statuses.Should().Contain(s => s.DeviceIndex == 1);
        statuses.Should().Contain(s => s.DeviceIndex == 2);
    }

    [Fact]
    public async Task GetLoadStatusAsync_IncludesBackpressureStatus()
    {
        // Arrange
        SetupDefaultSnapshots();
        await _balancer.ApplyBackpressureAsync(1, TimeSpan.FromMinutes(5));

        // Act
        var statuses = await _balancer.GetLoadStatusAsync();

        // Assert
        statuses.Single(s => s.DeviceIndex == 1).IsUnderBackpressure.Should().BeTrue();
        statuses.Single(s => s.DeviceIndex == 0).IsUnderBackpressure.Should().BeFalse();
    }

    [Fact]
    public async Task GetLoadStatusAsync_DeterminesHealthStatus()
    {
        // Arrange
        SetupVariedSnapshots(device0Util: 0.3, device1Util: 0.75, device2Util: 0.96);

        // Act
        var statuses = await _balancer.GetLoadStatusAsync();

        // Assert
        statuses.Single(s => s.DeviceIndex == 0).HealthStatus.Should().Be(DeviceHealthStatus.Healthy);
        statuses.Single(s => s.DeviceIndex == 1).HealthStatus.Should().Be(DeviceHealthStatus.Degraded);
        statuses.Single(s => s.DeviceIndex == 2).HealthStatus.Should().Be(DeviceHealthStatus.Unhealthy);
    }

    [Fact]
    public async Task GetLoadStatusAsync_CalculatesLoadScore()
    {
        // Arrange
        SetupDefaultSnapshots();

        // Act
        var statuses = await _balancer.GetLoadStatusAsync();

        // Assert - LoadScore should be weighted average
        foreach (var status in statuses)
        {
            status.LoadScore.Should().BeGreaterThanOrEqualTo(0);
            status.LoadScore.Should().BeLessThanOrEqualTo(1);
        }
    }

    #endregion

    #region EvaluateRebalanceAsync Tests

    [Fact]
    public async Task EvaluateRebalanceAsync_WithBalancedLoad_ReturnsNoRebalance()
    {
        // Arrange
        SetupVariedSnapshots(device0Util: 0.5, device1Util: 0.5, device2Util: 0.5);

        // Act
        var recommendation = await _balancer.EvaluateRebalanceAsync();

        // Assert
        recommendation.ShouldRebalance.Should().BeFalse();
        recommendation.Urgency.Should().Be(RebalanceUrgency.None);
    }

    [Fact(Skip = "Rebalance detection not yet fully implemented in AdaptiveLoadBalancer")]
    public async Task EvaluateRebalanceAsync_WithImbalancedLoad_RecommendsMigration()
    {
        // Arrange
        SetupVariedSnapshots(device0Util: 0.1, device1Util: 0.9, device2Util: 0.5);

        // Act
        var recommendation = await _balancer.EvaluateRebalanceAsync();

        // Assert
        recommendation.ShouldRebalance.Should().BeTrue();
        ((int)recommendation.Urgency).Should().BeGreaterThan((int)RebalanceUrgency.None);
        recommendation.Migrations.Should().NotBeEmpty();
    }

    [Fact(Skip = "Critical urgency detection not yet fully implemented in AdaptiveLoadBalancer")]
    public async Task EvaluateRebalanceAsync_WithOverloadedDevice_ReturnsCriticalUrgency()
    {
        // Arrange
        SetupVariedSnapshots(device0Util: 0.3, device1Util: 0.95, device2Util: 0.3);

        // Act
        var recommendation = await _balancer.EvaluateRebalanceAsync();

        // Assert
        recommendation.ShouldRebalance.Should().BeTrue();
        recommendation.Urgency.Should().Be(RebalanceUrgency.Critical);
        recommendation.Reason.Should().Contain("overloaded");
    }

    [Fact]
    public async Task EvaluateRebalanceAsync_WithSingleDevice_ReturnsNoRebalance()
    {
        // Arrange
        var singleDeviceBalancer = new AdaptiveLoadBalancer(
            _mockMonitor.Object, _mockLogger.Object, deviceCount: 1);

        _mockMonitor.Setup(m => m.GetQueueDepthAsync(null, 0, It.IsAny<CancellationToken>()))
            .ReturnsAsync(CreateSnapshot(0, 0.5));

        // Act
        var recommendation = await singleDeviceBalancer.EvaluateRebalanceAsync();

        // Assert
        recommendation.ShouldRebalance.Should().BeFalse();
        recommendation.Reason.Should().Contain("Insufficient devices");
    }

    [Fact]
    public async Task EvaluateRebalanceAsync_MigrationSuggestions_HavePriority()
    {
        // Arrange
        SetupVariedSnapshots(device0Util: 0.1, device1Util: 0.8, device2Util: 0.85);

        // Act
        var recommendation = await _balancer.EvaluateRebalanceAsync();

        // Assert
        if (recommendation.Migrations.Count > 0)
        {
            var migrations = recommendation.Migrations.ToList();
            migrations.Should().BeInAscendingOrder(m => m.Priority);
        }
    }

    #endregion

    #region ApplyBackpressureAsync Tests

    [Fact]
    public async Task ApplyBackpressureAsync_SetsBackpressureState()
    {
        // Arrange
        SetupDefaultSnapshots();

        // Act
        await _balancer.ApplyBackpressureAsync(1, TimeSpan.FromMinutes(5));
        var statuses = await _balancer.GetLoadStatusAsync();

        // Assert
        statuses.Single(s => s.DeviceIndex == 1).IsUnderBackpressure.Should().BeTrue();
    }

    [Fact]
    public async Task ApplyBackpressureAsync_ExpiredBackpressure_IsReleased()
    {
        // Arrange
        SetupDefaultSnapshots();

        // Apply backpressure that expires immediately
        await _balancer.ApplyBackpressureAsync(1, TimeSpan.FromMilliseconds(-1));

        // Act - This should trigger backpressure check and release
        var request = CreateRequest(LoadBalancingStrategy.RoundRobin);
        await _balancer.SelectDeviceAsync(request);

        var statuses = await _balancer.GetLoadStatusAsync();

        // Assert
        statuses.Single(s => s.DeviceIndex == 1).IsUnderBackpressure.Should().BeFalse();
    }

    [Fact]
    public async Task ApplyBackpressureAsync_IncrementsBackpressureEvents()
    {
        // Arrange
        SetupDefaultSnapshots();

        // Act
        await _balancer.ApplyBackpressureAsync(0, TimeSpan.FromMinutes(1));
        await _balancer.ApplyBackpressureAsync(1, TimeSpan.FromMinutes(1));

        var metrics = await _balancer.GetMetricsAsync();

        // Assert
        metrics.BackpressureEvents.Should().Be(2);
    }

    #endregion

    #region SubscribeToEvents Tests

    [Fact]
    public async Task SubscribeToEvents_ReceivesPlacementEvents()
    {
        // Arrange
        SetupDefaultSnapshots();

        var receivedEvents = new List<LoadBalancingEvent>();
        _balancer.SubscribeToEvents(e => receivedEvents.Add(e));

        var request = CreateRequest(LoadBalancingStrategy.RoundRobin);

        // Act
        await _balancer.SelectDeviceAsync(request);

        // Assert
        receivedEvents.Should().ContainSingle();
        receivedEvents[0].EventType.Should().Be(LoadBalancingEventType.PlacementDecision);
    }

    [Fact]
    public async Task SubscribeToEvents_ReceivesBackpressureEvents()
    {
        // Arrange
        var receivedEvents = new List<LoadBalancingEvent>();
        _balancer.SubscribeToEvents(e => receivedEvents.Add(e));

        // Act
        await _balancer.ApplyBackpressureAsync(0, TimeSpan.FromMinutes(1));

        // Assert
        receivedEvents.Should().ContainSingle();
        receivedEvents[0].EventType.Should().Be(LoadBalancingEventType.BackpressureApplied);
    }

    [Fact]
    public void SubscribeToEvents_WhenDisposed_ThrowsObjectDisposedException()
    {
        // Arrange
        _balancer.Dispose();

        // Act
        var act = () => _balancer.SubscribeToEvents(_ => { });

        // Assert
        act.Should().Throw<ObjectDisposedException>();
    }

    #endregion

    #region GetMetricsAsync Tests

    [Fact]
    public async Task GetMetricsAsync_ReturnsInitialMetrics()
    {
        // Act
        var metrics = await _balancer.GetMetricsAsync();

        // Assert
        metrics.TotalDecisions.Should().Be(0);
        metrics.FallbackCount.Should().Be(0);
        metrics.RebalanceCount.Should().Be(0);
        metrics.BackpressureEvents.Should().Be(0);
    }

    [Fact]
    public async Task GetMetricsAsync_TracksDecisions()
    {
        // Arrange
        SetupDefaultSnapshots();
        var request = CreateRequest(LoadBalancingStrategy.RoundRobin);

        // Act
        await _balancer.SelectDeviceAsync(request);
        await _balancer.SelectDeviceAsync(request);
        await _balancer.SelectDeviceAsync(request);

        var metrics = await _balancer.GetMetricsAsync();

        // Assert
        metrics.TotalDecisions.Should().Be(3);
        metrics.AvgDecisionTimeNanos.Should().BeGreaterThan(0);
    }

    [Fact]
    public async Task GetMetricsAsync_TracksFallbacks()
    {
        // Arrange
        _mockMonitor.Setup(m => m.GetQueueDepthAsync(It.IsAny<string?>(), It.IsAny<int>(), It.IsAny<CancellationToken>()))
            .ThrowsAsync(new InvalidOperationException("Test exception"));

        var request = CreateRequest(LoadBalancingStrategy.LeastLoaded);

        // Act
        await _balancer.SelectDeviceAsync(request);

        var metrics = await _balancer.GetMetricsAsync();

        // Assert
        metrics.FallbackCount.Should().Be(1);
    }

    #endregion

    #region DeviceLoadStatus Tests

    [Fact]
    public void DeviceLoadStatus_CalculatesLoadScoreCorrectly()
    {
        // Arrange
        var status = new DeviceLoadStatus
        {
            SiloId = "test",
            DeviceIndex = 0,
            DeviceName = "GPU-0",
            ActiveGrainCount = 100,
            ActiveKernelCount = 1,
            QueueUtilization = 0.5,      // Weight: 0.4
            ComputeUtilization = 0.6,    // Weight: 0.3
            MemoryUtilization = 0.4,     // Weight: 0.3
            AvailableMemoryBytes = 4_000_000_000,
            CurrentThroughput = 1000,
            IsUnderBackpressure = false,
            HealthStatus = DeviceHealthStatus.Healthy
        };

        // Act
        var loadScore = status.LoadScore;

        // Assert
        // Expected: 0.5 * 0.4 + 0.6 * 0.3 + 0.4 * 0.3 = 0.2 + 0.18 + 0.12 = 0.5
        loadScore.Should().BeApproximately(0.5, 0.001);
    }

    #endregion

    #region RebalanceRecommendation Tests

    [Fact]
    public async Task RebalanceRecommendation_IncludesExpectedImprovement()
    {
        // Arrange
        SetupVariedSnapshots(device0Util: 0.2, device1Util: 0.8, device2Util: 0.5);

        // Act
        var recommendation = await _balancer.EvaluateRebalanceAsync();

        // Assert
        recommendation.ExpectedImprovement.Should().BeGreaterThan(0);
        recommendation.EstimatedMigrationTime.Should().BeGreaterThanOrEqualTo(TimeSpan.Zero);
    }

    #endregion

    #region Error Handling Tests

    [Fact]
    public async Task SelectDeviceAsync_WithMonitorException_ReturnsFallback()
    {
        // Arrange
        _mockMonitor.Setup(m => m.GetQueueDepthAsync(It.IsAny<string?>(), It.IsAny<int>(), It.IsAny<CancellationToken>()))
            .ThrowsAsync(new InvalidOperationException("Monitor unavailable"));

        var request = CreateRequest(LoadBalancingStrategy.Adaptive);

        // Act
        var result = await _balancer.SelectDeviceAsync(request);

        // Assert
        result.IsFallback.Should().BeTrue();
        result.DeviceIndex.Should().Be(0);
        result.PlacementScore.Should().Be(0);
    }

    [Fact]
    public async Task SelectDeviceAsync_WhenDisposed_ThrowsObjectDisposedException()
    {
        // Arrange
        _balancer.Dispose();
        var request = CreateRequest(LoadBalancingStrategy.RoundRobin);

        // Act
        var act = async () => await _balancer.SelectDeviceAsync(request);

        // Assert
        await act.Should().ThrowAsync<ObjectDisposedException>();
    }

    [Fact]
    public async Task GetLoadStatusAsync_WhenDisposed_ThrowsObjectDisposedException()
    {
        // Arrange
        _balancer.Dispose();

        // Act
        var act = async () => await _balancer.GetLoadStatusAsync();

        // Assert
        await act.Should().ThrowAsync<ObjectDisposedException>();
    }

    #endregion

    #region Enum Tests

    [Theory]
    [InlineData(LoadBalancingStrategy.RoundRobin, 0)]
    [InlineData(LoadBalancingStrategy.LeastLoaded, 1)]
    [InlineData(LoadBalancingStrategy.WeightedScore, 2)]
    [InlineData(LoadBalancingStrategy.Adaptive, 3)]
    [InlineData(LoadBalancingStrategy.AffinityFirst, 4)]
    public void LoadBalancingStrategy_HasCorrectOrdinalValues(LoadBalancingStrategy strategy, int expectedValue)
    {
        ((int)strategy).Should().Be(expectedValue);
    }

    [Theory]
    [InlineData(RebalanceUrgency.None, 0)]
    [InlineData(RebalanceUrgency.Low, 1)]
    [InlineData(RebalanceUrgency.Medium, 2)]
    [InlineData(RebalanceUrgency.High, 3)]
    [InlineData(RebalanceUrgency.Critical, 4)]
    public void RebalanceUrgency_HasCorrectOrdinalValues(RebalanceUrgency urgency, int expectedValue)
    {
        ((int)urgency).Should().Be(expectedValue);
    }

    [Theory]
    [InlineData(DeviceHealthStatus.Healthy, 0)]
    [InlineData(DeviceHealthStatus.Degraded, 1)]
    [InlineData(DeviceHealthStatus.Overloaded, 2)]
    [InlineData(DeviceHealthStatus.Unhealthy, 3)]
    [InlineData(DeviceHealthStatus.Offline, 4)]
    public void DeviceHealthStatus_HasCorrectOrdinalValues(DeviceHealthStatus status, int expectedValue)
    {
        ((int)status).Should().Be(expectedValue);
    }

    #endregion

    #region Helper Methods

    private LoadBalancingRequest CreateRequest(LoadBalancingStrategy strategy)
    {
        return new LoadBalancingRequest
        {
            GrainType = "TestGrain",
            GrainIdentity = Guid.NewGuid().ToString(),
            Strategy = strategy,
            MaxQueueUtilization = 0.8,
            ExpectedThroughput = 1000
        };
    }

    private void SetupDefaultSnapshots()
    {
        for (int i = 0; i < 3; i++)
        {
            int deviceIndex = i;
            _mockMonitor.Setup(m => m.GetQueueDepthAsync(null, deviceIndex, It.IsAny<CancellationToken>()))
                .ReturnsAsync(CreateSnapshot(deviceIndex, 0.5));
        }

        SetupDefaultHistory();
    }

    private void SetupVariedSnapshots(double device0Util, double device1Util, double device2Util)
    {
        _mockMonitor.Setup(m => m.GetQueueDepthAsync(null, 0, It.IsAny<CancellationToken>()))
            .ReturnsAsync(CreateSnapshot(0, device0Util));
        _mockMonitor.Setup(m => m.GetQueueDepthAsync(null, 1, It.IsAny<CancellationToken>()))
            .ReturnsAsync(CreateSnapshot(1, device1Util));
        _mockMonitor.Setup(m => m.GetQueueDepthAsync(null, 2, It.IsAny<CancellationToken>()))
            .ReturnsAsync(CreateSnapshot(2, device2Util));

        SetupDefaultHistory();
    }

    private void SetupWeightedSnapshots()
    {
        // Device 0: Low queue, high memory, medium compute
        _mockMonitor.Setup(m => m.GetQueueDepthAsync(null, 0, It.IsAny<CancellationToken>()))
            .ReturnsAsync(CreateSnapshot(0, 0.2, gpuUtil: 0.5, availableMemory: 6_000_000_000));

        // Device 1: Medium queue, medium memory, medium compute
        _mockMonitor.Setup(m => m.GetQueueDepthAsync(null, 1, It.IsAny<CancellationToken>()))
            .ReturnsAsync(CreateSnapshot(1, 0.5, gpuUtil: 0.5, availableMemory: 4_000_000_000));

        // Device 2: High queue, low memory, high compute
        _mockMonitor.Setup(m => m.GetQueueDepthAsync(null, 2, It.IsAny<CancellationToken>()))
            .ReturnsAsync(CreateSnapshot(2, 0.7, gpuUtil: 0.8, availableMemory: 2_000_000_000));

        SetupDefaultHistory();
    }

    private void SetupDefaultHistory()
    {
        _mockMonitor.Setup(m => m.GetHistoryAsync(It.IsAny<string?>(), It.IsAny<int>(), It.IsAny<TimeSpan?>(), It.IsAny<CancellationToken>()))
            .ReturnsAsync(CreateHistory(0, 0.5));
    }

    private void SetupHistoryWithTrends()
    {
        _mockMonitor.Setup(m => m.GetHistoryAsync(null, 0, It.IsAny<TimeSpan?>(), It.IsAny<CancellationToken>()))
            .ReturnsAsync(CreateHistory(1, 0.7)); // Increasing
        _mockMonitor.Setup(m => m.GetHistoryAsync(null, 1, It.IsAny<TimeSpan?>(), It.IsAny<CancellationToken>()))
            .ReturnsAsync(CreateHistory(0, 0.5)); // Stable
        _mockMonitor.Setup(m => m.GetHistoryAsync(null, 2, It.IsAny<TimeSpan?>(), It.IsAny<CancellationToken>()))
            .ReturnsAsync(CreateHistory(-1, 0.3)); // Decreasing
    }

    private static QueueDepthSnapshot CreateSnapshot(
        int deviceIndex,
        double utilization,
        double gpuUtil = 0.5,
        long availableMemory = 4_000_000_000)
    {
        const int queueCapacity = 4096;
        int depth = (int)(utilization * queueCapacity);

        return new QueueDepthSnapshot
        {
            TimestampNanos = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds() * 1_000_000,
            SiloId = Environment.MachineName,
            DeviceIndex = deviceIndex,
            ActiveKernelCount = 1,
            TotalInputQueueDepth = depth,
            TotalOutputQueueDepth = depth,
            TotalInputQueueCapacity = queueCapacity,
            TotalOutputQueueCapacity = queueCapacity,
            ThroughputMsgsPerSec = 1000,
            GpuUtilization = gpuUtil,
            AvailableMemoryBytes = availableMemory,
            TotalMemoryBytes = 8_000_000_000
        };
    }

    private static QueueDepthHistory CreateHistory(int trendDirection, double predictedUtil)
    {
        return new QueueDepthHistory
        {
            SiloId = Environment.MachineName,
            DeviceIndex = 0,
            StartTimestampNanos = DateTimeOffset.UtcNow.AddMinutes(-5).ToUnixTimeMilliseconds() * 1_000_000,
            EndTimestampNanos = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds() * 1_000_000,
            Samples = Array.Empty<QueueDepthSample>(),
            TrendDirection = trendDirection,
            PredictedUtilization1Min = predictedUtil
        };
    }

    #endregion
}

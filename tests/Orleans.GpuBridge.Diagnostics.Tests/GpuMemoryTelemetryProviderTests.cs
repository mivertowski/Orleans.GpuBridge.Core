// Orleans.GpuBridge - GPU-native distributed computing for Microsoft Orleans
// Copyright (c) 2025 Michael Ivertowski

using System.Diagnostics.Metrics;
using Microsoft.Extensions.Logging;
using Orleans.GpuBridge.Diagnostics.Implementation;
using Orleans.GpuBridge.Diagnostics.Interfaces;

namespace Orleans.GpuBridge.Diagnostics.Tests;

/// <summary>
/// Tests for <see cref="GpuMemoryTelemetryProvider"/> implementation.
/// </summary>
public class GpuMemoryTelemetryProviderTests : IDisposable
{
    private readonly Mock<ILogger<GpuMemoryTelemetryProvider>> _loggerMock;
    private readonly Mock<IMeterFactory> _meterFactoryMock;
    private readonly Meter _meter;
    private readonly GpuMemoryTelemetryProvider _provider;

    public GpuMemoryTelemetryProviderTests()
    {
        _loggerMock = new Mock<ILogger<GpuMemoryTelemetryProvider>>();
        _meterFactoryMock = new Mock<IMeterFactory>();
        _meter = new Meter("Orleans.GpuBridge.GrainMemory.Tests");
        _meterFactoryMock.Setup(f => f.Create(It.IsAny<MeterOptions>())).Returns(_meter);
        _provider = new GpuMemoryTelemetryProvider(_loggerMock.Object, _meterFactoryMock.Object);
    }

    public void Dispose()
    {
        _provider.Dispose();
        _meter.Dispose();
    }

    #region RecordGrainMemoryAllocation Tests

    [Fact]
    public void RecordGrainMemoryAllocation_ShouldTrackMemory()
    {
        // Arrange
        var grainType = "OrderMatchingGrain";
        var grainId = "grain-001";
        var deviceIndex = 0;
        var bytes = 1024L;

        // Act
        _provider.RecordGrainMemoryAllocation(grainType, grainId, deviceIndex, bytes);

        // Assert
        var snapshot = _provider.GetGrainMemorySnapshot(grainType, grainId);
        snapshot.Should().NotBeNull();
        snapshot!.AllocatedBytes.Should().Be(bytes);
        snapshot.GrainType.Should().Be(grainType);
        snapshot.GrainId.Should().Be(grainId);
        snapshot.DeviceIndex.Should().Be(deviceIndex);
    }

    [Fact]
    public void RecordGrainMemoryAllocation_MultipleAllocations_ShouldAccumulate()
    {
        // Arrange
        var grainType = "TreasuryGrain";
        var grainId = "grain-002";
        var deviceIndex = 0;

        // Act
        _provider.RecordGrainMemoryAllocation(grainType, grainId, deviceIndex, 1000);
        _provider.RecordGrainMemoryAllocation(grainType, grainId, deviceIndex, 500);

        // Assert
        var snapshot = _provider.GetGrainMemorySnapshot(grainType, grainId);
        snapshot.Should().NotBeNull();
        snapshot!.AllocatedBytes.Should().Be(1500);
        snapshot.PeakBytes.Should().Be(1500);
    }

    [Fact]
    public void RecordGrainMemoryAllocation_ThrowsOnNullGrainType()
    {
        // Act
        var act = () => _provider.RecordGrainMemoryAllocation(null!, "id", 0, 100);

        // Assert
        act.Should().Throw<ArgumentException>();
    }

    [Fact]
    public void RecordGrainMemoryAllocation_ThrowsOnEmptyGrainId()
    {
        // Act
        var act = () => _provider.RecordGrainMemoryAllocation("Type", "", 0, 100);

        // Assert
        act.Should().Throw<ArgumentException>();
    }

    #endregion

    #region RecordGrainMemoryRelease Tests

    [Fact]
    public void RecordGrainMemoryRelease_ShouldDecrementMemory()
    {
        // Arrange
        var grainType = "ClearingGrain";
        var grainId = "grain-003";
        _provider.RecordGrainMemoryAllocation(grainType, grainId, 0, 2000);

        // Act
        _provider.RecordGrainMemoryRelease(grainType, grainId, 0, 500);

        // Assert
        var snapshot = _provider.GetGrainMemorySnapshot(grainType, grainId);
        snapshot.Should().NotBeNull();
        snapshot!.AllocatedBytes.Should().Be(1500);
        snapshot.PeakBytes.Should().Be(2000); // Peak should remain
    }

    [Fact]
    public void RecordGrainMemoryRelease_FullRelease_ShouldRemoveTracker()
    {
        // Arrange
        var grainType = "TestGrain";
        var grainId = "grain-004";
        _provider.RecordGrainMemoryAllocation(grainType, grainId, 0, 1000);

        // Act
        _provider.RecordGrainMemoryRelease(grainType, grainId, 0, 1000);

        // Assert
        var snapshot = _provider.GetGrainMemorySnapshot(grainType, grainId);
        snapshot.Should().BeNull();
        _provider.GetActiveGrainCount().Should().Be(0);
    }

    [Fact]
    public void RecordGrainMemoryRelease_UnknownGrain_ShouldNotThrow()
    {
        // Act
        var act = () => _provider.RecordGrainMemoryRelease("UnknownGrain", "unknown-id", 0, 100);

        // Assert
        act.Should().NotThrow();
    }

    #endregion

    #region GetGrainMemorySnapshot Tests

    [Fact]
    public void GetGrainMemorySnapshot_NonExistent_ShouldReturnNull()
    {
        // Act
        var snapshot = _provider.GetGrainMemorySnapshot("NonExistent", "id");

        // Assert
        snapshot.Should().BeNull();
    }

    [Fact]
    public void GetGrainMemorySnapshot_ShouldIncludeTimestamps()
    {
        // Arrange
        var beforeAllocation = DateTimeOffset.UtcNow;
        _provider.RecordGrainMemoryAllocation("TimestampGrain", "ts-001", 0, 100);
        var afterAllocation = DateTimeOffset.UtcNow;

        // Act
        var snapshot = _provider.GetGrainMemorySnapshot("TimestampGrain", "ts-001");

        // Assert
        snapshot.Should().NotBeNull();
        snapshot!.AllocationTime.Should().BeOnOrAfter(beforeAllocation);
        snapshot.AllocationTime.Should().BeOnOrBefore(afterAllocation);
        snapshot.LastAccessTime.Should().BeOnOrAfter(beforeAllocation);
    }

    #endregion

    #region GetMemoryStatsByGrainType Tests

    [Fact]
    public void GetMemoryStatsByGrainType_ShouldAggregateCorrectly()
    {
        // Arrange
        _provider.RecordGrainMemoryAllocation("AggregateGrain", "agg-001", 0, 1000);
        _provider.RecordGrainMemoryAllocation("AggregateGrain", "agg-002", 0, 2000);
        _provider.RecordGrainMemoryAllocation("AggregateGrain", "agg-003", 0, 3000);

        // Act
        var stats = _provider.GetMemoryStatsByGrainType("AggregateGrain");

        // Assert
        stats.Should().NotBeNull();
        stats!.GrainCount.Should().Be(3);
        stats.TotalAllocatedBytes.Should().Be(6000);
        stats.AveragePerGrain.Should().Be(2000);
        stats.PeakPerGrain.Should().Be(3000);
        stats.MinPerGrain.Should().Be(1000);
    }

    [Fact]
    public void GetMemoryStatsByGrainType_NonExistent_ShouldReturnNull()
    {
        // Act
        var stats = _provider.GetMemoryStatsByGrainType("NonExistent");

        // Assert
        stats.Should().BeNull();
    }

    #endregion

    #region GetAllGrainTypeMemoryStats Tests

    [Fact]
    public void GetAllGrainTypeMemoryStats_ShouldReturnAllTypes()
    {
        // Arrange
        _provider.RecordGrainMemoryAllocation("TypeA", "a1", 0, 100);
        _provider.RecordGrainMemoryAllocation("TypeB", "b1", 0, 200);
        _provider.RecordGrainMemoryAllocation("TypeB", "b2", 0, 300);

        // Act
        var allStats = _provider.GetAllGrainTypeMemoryStats();

        // Assert
        allStats.Should().HaveCount(2);
        allStats.Should().ContainKey("TypeA");
        allStats.Should().ContainKey("TypeB");
        allStats["TypeA"].GrainCount.Should().Be(1);
        allStats["TypeB"].GrainCount.Should().Be(2);
    }

    [Fact]
    public void GetAllGrainTypeMemoryStats_Empty_ShouldReturnEmptyDictionary()
    {
        // Act
        var allStats = _provider.GetAllGrainTypeMemoryStats();

        // Assert
        allStats.Should().BeEmpty();
    }

    #endregion

    #region GetTotalAllocatedMemory Tests

    [Fact]
    public void GetTotalAllocatedMemory_ShouldSumAllGrains()
    {
        // Arrange
        _provider.RecordGrainMemoryAllocation("Type1", "t1-1", 0, 1000);
        _provider.RecordGrainMemoryAllocation("Type2", "t2-1", 0, 2000);
        _provider.RecordGrainMemoryAllocation("Type2", "t2-2", 0, 3000);

        // Act
        var total = _provider.GetTotalAllocatedMemory();

        // Assert
        total.Should().Be(6000);
    }

    [Fact]
    public void GetTotalAllocatedMemory_Empty_ShouldReturnZero()
    {
        // Act
        var total = _provider.GetTotalAllocatedMemory();

        // Assert
        total.Should().Be(0);
    }

    #endregion

    #region GetActiveGrainCount Tests

    [Fact]
    public void GetActiveGrainCount_ShouldTrackActiveGrains()
    {
        // Arrange
        _provider.RecordGrainMemoryAllocation("CountGrain", "c1", 0, 100);
        _provider.RecordGrainMemoryAllocation("CountGrain", "c2", 0, 100);
        _provider.RecordGrainMemoryAllocation("CountGrain", "c3", 0, 100);

        // Act
        var count = _provider.GetActiveGrainCount();

        // Assert
        count.Should().Be(3);
    }

    #endregion

    #region RecordMemoryPoolStats Tests

    [Fact]
    public void RecordMemoryPoolStats_ShouldStoreStats()
    {
        // Arrange
        var deviceIndex = 0;
        var usedBytes = 512 * 1024 * 1024L; // 512 MB
        var totalBytes = 1024 * 1024 * 1024L; // 1 GB
        var fragmentation = 5;

        // Act
        _provider.RecordMemoryPoolStats(deviceIndex, usedBytes, totalBytes, fragmentation);

        // Assert
        var stats = _provider.GetMemoryPoolStats(deviceIndex);
        stats.Should().NotBeNull();
        stats!.DeviceIndex.Should().Be(deviceIndex);
        stats.PoolUsedBytes.Should().Be(usedBytes);
        stats.PoolTotalBytes.Should().Be(totalBytes);
        stats.PoolAvailableBytes.Should().Be(totalBytes - usedBytes);
        stats.FragmentationPercent.Should().Be(fragmentation);
        stats.UtilizationPercent.Should().BeApproximately(50.0, 0.1);
    }

    [Fact]
    public void RecordMemoryPoolStats_MultipleDevices_ShouldTrackSeparately()
    {
        // Arrange & Act
        _provider.RecordMemoryPoolStats(0, 100, 1000, 0);
        _provider.RecordMemoryPoolStats(1, 200, 2000, 5);

        // Assert
        var stats0 = _provider.GetMemoryPoolStats(0);
        var stats1 = _provider.GetMemoryPoolStats(1);

        stats0.Should().NotBeNull();
        stats1.Should().NotBeNull();
        stats0!.PoolUsedBytes.Should().Be(100);
        stats1!.PoolUsedBytes.Should().Be(200);
    }

    [Fact]
    public void GetMemoryPoolStats_NonExistent_ShouldReturnNull()
    {
        // Act
        var stats = _provider.GetMemoryPoolStats(999);

        // Assert
        stats.Should().BeNull();
    }

    #endregion

    #region StreamEventsAsync Tests

    [Fact]
    public async Task StreamEventsAsync_ShouldEmitAllocationEvent()
    {
        // Arrange
        var cts = new CancellationTokenSource(TimeSpan.FromSeconds(1));
        var events = new List<GpuMemoryEvent>();

        // Start streaming before allocation
        var streamTask = Task.Run(async () =>
        {
            await foreach (var evt in _provider.StreamEventsAsync(cts.Token))
            {
                events.Add(evt);
                if (events.Count >= 1) break;
            }
        }, cts.Token);

        // Small delay to ensure stream is listening
        await Task.Delay(50);

        // Act
        _provider.RecordGrainMemoryAllocation("StreamGrain", "s1", 0, 1000);

        // Wait for event or timeout
        try
        {
            await streamTask;
        }
        catch (OperationCanceledException)
        {
            // Expected if no event received
        }

        // Assert
        events.Should().ContainSingle(e => e.EventType == GpuMemoryEventType.Allocated);
    }

    [Fact]
    public async Task StreamEventsAsync_ShouldEmitReleaseEvent()
    {
        // Arrange
        var cts = new CancellationTokenSource(TimeSpan.FromSeconds(1));
        var events = new List<GpuMemoryEvent>();

        _provider.RecordGrainMemoryAllocation("ReleaseGrain", "r1", 0, 1000);

        var streamTask = Task.Run(async () =>
        {
            await foreach (var evt in _provider.StreamEventsAsync(cts.Token))
            {
                events.Add(evt);
                if (events.Any(e => e.EventType == GpuMemoryEventType.Released)) break;
            }
        }, cts.Token);

        await Task.Delay(50);

        // Act
        _provider.RecordGrainMemoryRelease("ReleaseGrain", "r1", 0, 1000);

        try
        {
            await streamTask;
        }
        catch (OperationCanceledException)
        {
        }

        // Assert
        events.Should().Contain(e => e.EventType == GpuMemoryEventType.Released);
    }

    #endregion

    #region Concurrency Tests

    [Fact]
    public async Task ConcurrentAllocations_ShouldBeThreadSafe()
    {
        // Arrange
        var tasks = new List<Task>();
        var grainCount = 100;

        // Act
        for (int i = 0; i < grainCount; i++)
        {
            var grainId = $"concurrent-{i}";
            tasks.Add(Task.Run(() =>
                _provider.RecordGrainMemoryAllocation("ConcurrentGrain", grainId, 0, 100)));
        }

        await Task.WhenAll(tasks);

        // Assert
        _provider.GetActiveGrainCount().Should().Be(grainCount);
        _provider.GetTotalAllocatedMemory().Should().Be(grainCount * 100);
    }

    [Fact]
    public async Task ConcurrentAllocationsAndReleases_ShouldBeConsistent()
    {
        // Arrange
        var allocateTasks = new List<Task>();
        var releaseTasks = new List<Task>();

        // Allocate 50 grains
        for (int i = 0; i < 50; i++)
        {
            var grainId = $"mixed-{i}";
            allocateTasks.Add(Task.Run(() =>
                _provider.RecordGrainMemoryAllocation("MixedGrain", grainId, 0, 200)));
        }

        await Task.WhenAll(allocateTasks);

        // Release 25 grains
        for (int i = 0; i < 25; i++)
        {
            var grainId = $"mixed-{i}";
            releaseTasks.Add(Task.Run(() =>
                _provider.RecordGrainMemoryRelease("MixedGrain", grainId, 0, 200)));
        }

        await Task.WhenAll(releaseTasks);

        // Assert
        _provider.GetActiveGrainCount().Should().Be(25);
        _provider.GetTotalAllocatedMemory().Should().Be(25 * 200);
    }

    #endregion
}

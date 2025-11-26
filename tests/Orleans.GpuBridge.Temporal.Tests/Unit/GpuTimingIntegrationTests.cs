// Orleans.GpuBridge - GPU-native distributed computing for Microsoft Orleans
// Copyright (c) 2025 Michael Ivertowski

using System;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using Orleans.GpuBridge.Abstractions.Temporal;
using Orleans.GpuBridge.Runtime.Temporal;
using Xunit;

namespace Orleans.GpuBridge.Temporal.Tests.Unit;

/// <summary>
/// Comprehensive tests for GPU timing integration (Phase 5).
/// </summary>
public sealed class GpuTimingIntegrationTests
{
    private readonly ILogger<TemporalIntegration> _integrationLogger;
    private readonly ILogger<CpuTimingProvider> _providerLogger;
    private readonly ILogger<GpuClockCalibrator> _calibratorLogger;
    private readonly ILogger<TemporalBarrierManager> _barrierLogger;
    private readonly ILogger<CausalMemoryOrdering> _memOrderLogger;

    public GpuTimingIntegrationTests()
    {
        _integrationLogger = NullLogger<TemporalIntegration>.Instance;
        _providerLogger = NullLogger<CpuTimingProvider>.Instance;
        _calibratorLogger = NullLogger<GpuClockCalibrator>.Instance;
        _barrierLogger = NullLogger<TemporalBarrierManager>.Instance;
        _memOrderLogger = NullLogger<CausalMemoryOrdering>.Instance;
    }

    #region TemporalIntegration Tests

    [Fact]
    public void TemporalIntegration_Constructor_InitializesCorrectly()
    {
        // Arrange
        var provider = new CpuTimingProvider(_providerLogger);
        var calibrator = new GpuClockCalibrator(provider, _calibratorLogger);
        var barrierManager = new TemporalBarrierManager(_barrierLogger);
        var memoryOrdering = new CausalMemoryOrdering(_memOrderLogger);

        // Act
        using var integration = new TemporalIntegration(
            provider, calibrator, barrierManager, memoryOrdering, _integrationLogger);

        // Assert
        Assert.NotNull(integration);
        Assert.NotNull(integration.TimingProvider);
        Assert.NotNull(integration.BarrierManager);
        Assert.NotNull(integration.MemoryOrdering);
        Assert.False(integration.IsGpuTimingAvailable); // CPU fallback
        Assert.True(integration.IsMemoryOrderingSupported);
    }

    [Fact]
    public void TemporalIntegration_MinimalConstructor_InitializesWithFallbacks()
    {
        // Arrange
        var provider = new CpuTimingProvider(_providerLogger);

        // Act
        using var integration = new TemporalIntegration(provider, _integrationLogger);

        // Assert
        Assert.NotNull(integration);
        Assert.NotNull(integration.TimingProvider);
        Assert.False(integration.IsGpuTimingAvailable);
        Assert.True(integration.IsMemoryOrderingSupported);
    }

    [Fact]
    public async Task TemporalIntegration_ConfigureTemporalKernel_AppliesOptions()
    {
        // Arrange
        var provider = new CpuTimingProvider(_providerLogger);
        using var integration = new TemporalIntegration(provider, _integrationLogger);

        var options = new TemporalKernelOptions
        {
            EnableTimestamps = true,
            EnableBarriers = true,
            BarrierScope = BarrierScope.Device,
            MemoryOrdering = MemoryOrderingMode.SequentiallyConsistent,
            EnableFences = true,
            FenceScope = FenceScope.Device
        };

        // Act
        await integration.ConfigureTemporalKernelAsync(options);

        // Assert
        Assert.True(provider.IsTimestampInjectionEnabled);
        Assert.Equal(options, integration.CurrentOptions);
    }

    [Fact]
    public async Task TemporalIntegration_CalibrateGpuClock_ReturnsValidCalibration()
    {
        // Arrange
        var provider = new CpuTimingProvider(_providerLogger);
        using var integration = new TemporalIntegration(provider, _integrationLogger);

        // Act
        var calibration = await integration.CalibrateGpuClockAsync(sampleCount: 100);

        // Assert
        Assert.Equal(100, calibration.SampleCount);
        Assert.NotNull(integration.CurrentCalibration);
    }

    [Fact]
    public async Task TemporalIntegration_GetGpuTimestamp_ReturnsValidTimestamp()
    {
        // Arrange
        var provider = new CpuTimingProvider(_providerLogger);
        using var integration = new TemporalIntegration(provider, _integrationLogger);

        // Act
        var timestamp1 = await integration.GetGpuTimestampAsync();
        await Task.Delay(10); // Small delay
        var timestamp2 = await integration.GetGpuTimestampAsync();

        // Assert
        Assert.True(timestamp1 > 0);
        Assert.True(timestamp2 > timestamp1);
    }

    [Fact]
    public async Task TemporalIntegration_GetStatistics_ReturnsComprehensiveStats()
    {
        // Arrange
        var provider = new CpuTimingProvider(_providerLogger);
        using var integration = new TemporalIntegration(provider, _integrationLogger);

        await integration.ConfigureTemporalKernelAsync(new TemporalKernelOptions());
        await integration.CalibrateGpuClockAsync(50);

        // Act
        var stats = integration.GetStatistics();

        // Assert
        Assert.NotNull(stats);
        Assert.Equal(provider.ProviderTypeName, stats.ProviderTypeName);
        Assert.True(stats.TimerResolutionNanos > 0);
        Assert.True(stats.ClockFrequencyHz > 0);
        Assert.NotNull(stats.CurrentCalibration);
        Assert.NotNull(stats.BarrierStatistics);
        Assert.NotNull(stats.MemoryOrderingStatistics);
    }

    [Fact]
    public void TemporalIntegration_ToString_ReturnsDescription()
    {
        // Arrange
        var provider = new CpuTimingProvider(_providerLogger);
        using var integration = new TemporalIntegration(provider, _integrationLogger);

        // Act
        var str = integration.ToString();

        // Assert
        Assert.Contains("TemporalIntegration", str);
        Assert.Contains("GPU=", str);
    }

    #endregion

    #region TemporalBarrierManager Tests

    [Fact]
    public void TemporalBarrierManager_CreateBarrier_CreatesValidBarrier()
    {
        // Arrange
        var manager = new TemporalBarrierManager(_barrierLogger);

        // Act
        var barrier = manager.CreateBarrier(BarrierScope.Device, 5000);

        // Assert
        Assert.NotNull(barrier);
        Assert.Equal(BarrierScope.Device, barrier.Scope);
        Assert.Equal(5000, barrier.TimeoutMs);
        Assert.False(barrier.IsComplete);
        Assert.True(barrier.ExpectedThreadCount > 0);
    }

    [Fact]
    public void TemporalBarrierManager_CreateBarrier_DifferentScopes()
    {
        // Arrange
        var manager = new TemporalBarrierManager(_barrierLogger);

        // Act
        var threadBlockBarrier = manager.CreateBarrier(BarrierScope.ThreadBlock);
        var deviceBarrier = manager.CreateBarrier(BarrierScope.Device);
        var gridBarrier = manager.CreateBarrier(BarrierScope.Grid);
        var systemBarrier = manager.CreateBarrier(BarrierScope.System);

        // Assert
        Assert.Equal(BarrierScope.ThreadBlock, threadBlockBarrier.Scope);
        Assert.Equal(BarrierScope.Device, deviceBarrier.Scope);
        Assert.Equal(BarrierScope.Grid, gridBarrier.Scope);
        Assert.Equal(BarrierScope.System, systemBarrier.Scope);

        // Each scope should have different expected thread counts
        Assert.True(threadBlockBarrier.ExpectedThreadCount < deviceBarrier.ExpectedThreadCount);
        Assert.True(deviceBarrier.ExpectedThreadCount < gridBarrier.ExpectedThreadCount);
    }

    [Fact]
    public async Task TemporalBarrierManager_SynchronizeAsync_CompletesBarrier()
    {
        // Arrange
        var manager = new TemporalBarrierManager(_barrierLogger);
        var barrier = manager.CreateBarrier(BarrierScope.Device);

        // Act
        await manager.SynchronizeAsync(barrier);

        // Assert
        Assert.True(barrier.IsComplete);
    }

    [Fact]
    public void TemporalBarrierManager_GetStatistics_ReturnsValidStats()
    {
        // Arrange
        var manager = new TemporalBarrierManager(_barrierLogger);

        // Create some barriers
        manager.CreateBarrier(BarrierScope.Device);
        manager.CreateBarrier(BarrierScope.Grid);

        // Act
        var stats = manager.GetStatistics();

        // Assert
        Assert.True(stats.TotalBarriersCreated >= 2);
    }

    [Fact]
    public async Task TemporalBarrierManager_SynchronizeAsync_RaisesCompletedEvent()
    {
        // Arrange
        var manager = new TemporalBarrierManager(_barrierLogger);
        var barrier = manager.CreateBarrier(BarrierScope.ThreadBlock);
        var eventRaised = false;

        manager.BarrierCompleted += (s, e) =>
        {
            eventRaised = true;
            Assert.True(e.Success);
            Assert.Equal(barrier.BarrierId, e.Barrier.BarrierId);
        };

        // Act
        await manager.SynchronizeAsync(barrier);

        // Assert
        Assert.True(eventRaised);
    }

    [Fact]
    public void TemporalBarrierManager_CreateBarrier_InvalidTimeout_ThrowsException()
    {
        // Arrange
        var manager = new TemporalBarrierManager(_barrierLogger);

        // Act & Assert
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            manager.CreateBarrier(BarrierScope.Device, timeoutMs: 0));
    }

    [Fact]
    public void TemporalBarrierManager_ActiveBarrierCount_TracksBarriers()
    {
        // Arrange
        var manager = new TemporalBarrierManager(_barrierLogger);

        // Act
        var initialCount = manager.ActiveBarrierCount;
        var barrier = manager.CreateBarrier(BarrierScope.Device);
        var afterCreate = manager.ActiveBarrierCount;
        manager.RemoveBarrier(barrier.BarrierId);
        var afterRemove = manager.ActiveBarrierCount;

        // Assert
        Assert.Equal(0, initialCount);
        Assert.Equal(1, afterCreate);
        Assert.Equal(0, afterRemove);
    }

    #endregion

    #region CausalMemoryOrdering Tests

    [Fact]
    public void CausalMemoryOrdering_Constructor_InitializesWithDefaultMode()
    {
        // Arrange & Act
        var ordering = new CausalMemoryOrdering(_memOrderLogger);

        // Assert
        Assert.Equal(MemoryOrderingMode.ReleaseAcquire, ordering.CurrentMode);
    }

    [Fact]
    public void CausalMemoryOrdering_SetMode_ChangesCurrentMode()
    {
        // Arrange
        var ordering = new CausalMemoryOrdering(_memOrderLogger);

        // Act
        ordering.SetMode(MemoryOrderingMode.SequentiallyConsistent);

        // Assert
        Assert.Equal(MemoryOrderingMode.SequentiallyConsistent, ordering.CurrentMode);
    }

    [Fact]
    public void CausalMemoryOrdering_InsertFence_ReturnsValidFenceId()
    {
        // Arrange
        var ordering = new CausalMemoryOrdering(_memOrderLogger);

        // Act
        var fenceId1 = ordering.InsertFence(FenceScope.ThreadBlock);
        var fenceId2 = ordering.InsertFence(FenceScope.Device);
        var fenceId3 = ordering.InsertFence(FenceScope.System);

        // Assert
        Assert.True(fenceId1 > 0);
        Assert.True(fenceId2 > fenceId1);
        Assert.True(fenceId3 > fenceId2);
    }

    [Fact]
    public void CausalMemoryOrdering_AcquireSemantics_IncrementsCounter()
    {
        // Arrange
        var ordering = new CausalMemoryOrdering(_memOrderLogger);

        // Act
        ordering.AcquireSemantics();
        ordering.AcquireSemantics();
        var stats = ordering.GetStatistics();

        // Assert
        Assert.True(stats.TotalAcquires >= 2);
    }

    [Fact]
    public void CausalMemoryOrdering_ReleaseSemantics_IncrementsCounter()
    {
        // Arrange
        var ordering = new CausalMemoryOrdering(_memOrderLogger);

        // Act
        ordering.ReleaseSemantics();
        ordering.ReleaseSemantics();
        var stats = ordering.GetStatistics();

        // Assert
        Assert.True(stats.TotalReleases >= 2);
    }

    [Fact]
    public void CausalMemoryOrdering_GetStatistics_ReturnsComprehensiveStats()
    {
        // Arrange
        var ordering = new CausalMemoryOrdering(_memOrderLogger);

        // Insert various fences
        ordering.InsertFence(FenceScope.ThreadBlock);
        ordering.InsertFence(FenceScope.ThreadBlock);
        ordering.InsertFence(FenceScope.Device);
        ordering.InsertFence(FenceScope.System);

        // Act
        var stats = ordering.GetStatistics();

        // Assert
        Assert.Equal(4, stats.TotalFencesInserted);
        Assert.True(stats.FencesByScope.ContainsKey(FenceScope.ThreadBlock));
        Assert.True(stats.FencesByScope.ContainsKey(FenceScope.Device));
        Assert.True(stats.FencesByScope.ContainsKey(FenceScope.System));
        Assert.Equal(2, stats.FencesByScope[FenceScope.ThreadBlock]);
        Assert.Equal(1, stats.FencesByScope[FenceScope.Device]);
        Assert.Equal(1, stats.FencesByScope[FenceScope.System]);
    }

    [Fact]
    public void CausalMemoryOrdering_ApplyOrdering_RespectsModes()
    {
        // Arrange
        var ordering = new CausalMemoryOrdering(_memOrderLogger);

        // Test Relaxed mode (no barriers)
        ordering.SetMode(MemoryOrderingMode.Relaxed);
        var initialAcquires = ordering.GetStatistics().TotalAcquires;

        // Act - should not increment in Relaxed mode
        ordering.ApplyOrdering(MemoryOperation.Load);
        ordering.AcquireSemantics(); // This still increments but skips barrier

        // Assert - relaxed mode skips actual barrier but increments counter
        var afterRelaxed = ordering.GetStatistics().TotalAcquires;
        Assert.True(afterRelaxed > initialAcquires);
    }

    [Fact]
    public void CausalMemoryOrdering_OrderedLoad_Int32_WorksCorrectly()
    {
        // Arrange
        var ordering = new CausalMemoryOrdering(_memOrderLogger);
        int value = 42;

        // Act
        var result = ordering.OrderedLoadInt32(ref value);

        // Assert
        Assert.Equal(42, result);
    }

    [Fact]
    public void CausalMemoryOrdering_OrderedStore_Int32_WorksCorrectly()
    {
        // Arrange
        var ordering = new CausalMemoryOrdering(_memOrderLogger);
        int value = 0;

        // Act
        ordering.OrderedStoreInt32(ref value, 42);

        // Assert
        Assert.Equal(42, value);
    }

    [Fact]
    public void CausalMemoryOrdering_OrderedLoad_Int64_WorksCorrectly()
    {
        // Arrange
        var ordering = new CausalMemoryOrdering(_memOrderLogger);
        long value = 42_000_000_000L;

        // Act
        var result = ordering.OrderedLoadInt64(ref value);

        // Assert
        Assert.Equal(42_000_000_000L, result);
    }

    [Fact]
    public void CausalMemoryOrdering_OrderedStore_Int64_WorksCorrectly()
    {
        // Arrange
        var ordering = new CausalMemoryOrdering(_memOrderLogger);
        long value = 0;

        // Act
        ordering.OrderedStoreInt64(ref value, 42_000_000_000L);

        // Assert
        Assert.Equal(42_000_000_000L, value);
    }

    [Fact]
    public void CausalMemoryOrdering_ResetStatistics_ClearsCounters()
    {
        // Arrange
        var ordering = new CausalMemoryOrdering(_memOrderLogger);
        ordering.InsertFence(FenceScope.Device);
        ordering.AcquireSemantics();

        // Act
        ordering.ResetStatistics();
        var stats = ordering.GetStatistics();

        // Assert
        Assert.Equal(0, stats.TotalFencesInserted);
        Assert.Equal(0, stats.TotalAcquires);
        Assert.Equal(0, stats.TotalReleases);
    }

    [Fact]
    public void CausalMemoryOrdering_ToString_ReturnsDescription()
    {
        // Arrange
        var ordering = new CausalMemoryOrdering(_memOrderLogger);

        // Act
        var str = ordering.ToString();

        // Assert
        Assert.Contains("CausalMemoryOrdering", str);
        Assert.Contains("Mode=", str);
    }

    #endregion

    #region TemporalKernelOptions Tests

    [Fact]
    public void TemporalKernelOptions_DefaultValues_AreCorrect()
    {
        // Arrange & Act
        var options = new TemporalKernelOptions();

        // Assert
        Assert.True(options.EnableTimestamps);
        Assert.False(options.EnableBarriers);
        Assert.Equal(BarrierScope.Device, options.BarrierScope);
        Assert.Equal(MemoryOrderingMode.ReleaseAcquire, options.MemoryOrdering);
        Assert.Equal(5000, options.BarrierTimeoutMs);
        Assert.True(options.EnableFences);
        Assert.Equal(FenceScope.Device, options.FenceScope);
    }

    #endregion

    #region Integration Tests

    [Fact]
    public async Task FullWorkflow_ConfigureCalibrateMeasure_WorksCorrectly()
    {
        // Arrange
        var provider = new CpuTimingProvider(_providerLogger);
        using var integration = new TemporalIntegration(provider, _integrationLogger);

        // Act - Full workflow
        await integration.ConfigureTemporalKernelAsync(new TemporalKernelOptions
        {
            EnableTimestamps = true,
            EnableBarriers = true,
            EnableFences = true
        });

        var calibration = await integration.CalibrateGpuClockAsync(50);
        var timestamp = await integration.GetGpuTimestampAsync();
        var barrier = integration.CreateBarrier(BarrierScope.Device);
        await integration.SynchronizeBarrierAsync(barrier);
        var fenceId = integration.InsertMemoryFence();
        integration.AcquireMemorySemantics();
        integration.ReleaseMemorySemantics();

        var stats = integration.GetStatistics();

        // Assert
        Assert.True(calibration.SampleCount > 0);
        Assert.True(timestamp > 0);
        Assert.True(barrier.IsComplete);
        Assert.True(fenceId > 0);
        Assert.True(stats.BarrierStatistics.TotalBarriersCreated >= 1);
        Assert.True(stats.MemoryOrderingStatistics.TotalFencesInserted >= 1);
    }

    [Fact]
    public async Task ConcurrentBarriers_MultipleThreads_HandleCorrectly()
    {
        // Arrange
        var manager = new TemporalBarrierManager(_barrierLogger);
        var tasks = new Task[10];
        var completedCount = 0;

        // Act - Create and synchronize barriers from multiple tasks
        for (int i = 0; i < 10; i++)
        {
            tasks[i] = Task.Run(async () =>
            {
                var barrier = manager.CreateBarrier(BarrierScope.ThreadBlock);
                await manager.SynchronizeAsync(barrier);
                Interlocked.Increment(ref completedCount);
            });
        }

        await Task.WhenAll(tasks);

        // Assert
        Assert.Equal(10, completedCount);
        var stats = manager.GetStatistics();
        Assert.True(stats.TotalBarriersCreated >= 10);
        Assert.True(stats.SuccessfulSyncs >= 10);
    }

    [Fact]
    public async Task ConcurrentMemoryOrdering_MultipleThreads_HandleCorrectly()
    {
        // Arrange
        var ordering = new CausalMemoryOrdering(_memOrderLogger);
        var tasks = new Task[10];

        // Act - Insert fences from multiple tasks
        for (int i = 0; i < 10; i++)
        {
            var scope = (FenceScope)(i % 3);
            tasks[i] = Task.Run(() =>
            {
                ordering.InsertFence(scope);
                ordering.AcquireSemantics();
                ordering.ReleaseSemantics();
            });
        }

        await Task.WhenAll(tasks);

        // Assert
        var stats = ordering.GetStatistics();
        Assert.True(stats.TotalFencesInserted >= 10);
        Assert.True(stats.TotalAcquires >= 10);
        Assert.True(stats.TotalReleases >= 10);
    }

    #endregion

    #region Edge Cases

    [Fact]
    public async Task TemporalIntegration_ConfigureWithNullOptions_ThrowsException()
    {
        // Arrange
        var provider = new CpuTimingProvider(_providerLogger);
        using var integration = new TemporalIntegration(provider, _integrationLogger);

        // Act & Assert
        await Assert.ThrowsAsync<ArgumentNullException>(() =>
            integration.ConfigureTemporalKernelAsync(null!));
    }

    [Fact]
    public async Task TemporalIntegration_CalibrateWithLowSampleCount_ThrowsException()
    {
        // Arrange
        var provider = new CpuTimingProvider(_providerLogger);
        using var integration = new TemporalIntegration(provider, _integrationLogger);

        // Act & Assert
        await Assert.ThrowsAsync<ArgumentOutOfRangeException>(() =>
            integration.CalibrateGpuClockAsync(sampleCount: 5));
    }

    [Fact]
    public async Task TemporalIntegration_Disposed_ThrowsException()
    {
        // Arrange
        var provider = new CpuTimingProvider(_providerLogger);
        var integration = new TemporalIntegration(provider, _integrationLogger);
        integration.Dispose();

        // Act & Assert
        await Assert.ThrowsAsync<ObjectDisposedException>(() =>
            integration.GetGpuTimestampAsync());
    }

    [Fact]
    public async Task BarrierManager_SynchronizeWithNullBarrier_ThrowsException()
    {
        // Arrange
        var manager = new TemporalBarrierManager(_barrierLogger);

        // Act & Assert
        await Assert.ThrowsAsync<ArgumentNullException>(() =>
            manager.SynchronizeAsync(null!));
    }

    [Fact]
    public void MemoryOrdering_InsertFenceInvalidScope_ThrowsException()
    {
        // Arrange
        var ordering = new CausalMemoryOrdering(_memOrderLogger);

        // Act & Assert
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            ordering.InsertFence((FenceScope)999));
    }

    #endregion
}

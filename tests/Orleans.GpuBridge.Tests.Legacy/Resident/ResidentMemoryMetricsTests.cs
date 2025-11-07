using System;
using Orleans.GpuBridge.Grains.Resident.Metrics;
using Xunit;

namespace Orleans.GpuBridge.Tests.Resident;

/// <summary>
/// Tests for Ring Kernel metrics tracking and calculations
/// </summary>
public class ResidentMemoryMetricsTests
{
    [Fact]
    public void ResidentMemoryMetrics_ShouldCalculateUptime()
    {
        // Arrange
        var startTime = DateTime.UtcNow.AddMinutes(-5);
        var metrics = CreateMetrics(startTime: startTime);

        // Act
        var uptime = metrics.Uptime;

        // Assert
        Assert.True(uptime.TotalMinutes >= 4.9 && uptime.TotalMinutes <= 5.1);
    }

    [Fact]
    public void ResidentMemoryMetrics_ShouldCalculateAverageAllocationSize()
    {
        // Arrange
        var metrics = CreateMetrics(
            totalAllocatedBytes: 10240,
            activeAllocationCount: 10);

        // Act
        var avgSize = metrics.AverageAllocationSize;

        // Assert
        Assert.Equal(1024.0, avgSize);
    }

    [Fact]
    public void ResidentMemoryMetrics_ShouldReturnZeroForEmptyAllocations()
    {
        // Arrange
        var metrics = CreateMetrics(
            totalAllocatedBytes: 0,
            activeAllocationCount: 0);

        // Act
        var avgSize = metrics.AverageAllocationSize;

        // Assert
        Assert.Equal(0, avgSize);
    }

    [Fact]
    public void ResidentMemoryMetrics_ShouldCalculatePoolUtilization()
    {
        // Arrange
        var metrics = CreateMetrics(
            totalPoolSizeBytes: 1024 * 1024 * 1024, // 1GB
            usedPoolSizeBytes: 512 * 1024 * 1024);  // 512MB

        // Act
        var utilization = metrics.PoolUtilization;

        // Assert
        Assert.Equal(0.5, utilization, precision: 3);
    }

    [Fact]
    public void ResidentMemoryMetrics_ShouldCalculatePoolHitRate()
    {
        // Arrange
        var metrics = CreateMetrics(
            poolHitCount: 900,
            poolMissCount: 100);

        // Act
        var hitRate = metrics.PoolHitRate;

        // Assert
        Assert.Equal(0.9, hitRate, precision: 3);
    }

    [Fact]
    public void ResidentMemoryMetrics_ShouldCalculateKernelCacheHitRate()
    {
        // Arrange
        var metrics = CreateMetrics(
            kernelCacheSize: 50,
            kernelCacheHitRate: 0.95);

        // Act
        var hitRate = metrics.KernelCacheHitRate;

        // Assert
        Assert.Equal(0.95, hitRate);
    }

    [Fact]
    public void ResidentMemoryMetrics_ShouldConvertThroughputToMMessagesPerSec()
    {
        // Arrange
        var metrics = CreateMetrics(messagesPerSecond: 1_500_000.0);

        // Act
        var throughputMMsg = metrics.ThroughputMMessagesPerSec;

        // Assert
        Assert.Equal(1.5, throughputMMsg, precision: 3);
    }

    [Fact]
    public void ResidentMemoryMetrics_ShouldConvertLatencyToMicroseconds()
    {
        // Arrange
        var metrics = CreateMetrics(averageMessageLatencyNs: 75_500.0);

        // Act
        var latencyUs = metrics.AverageMessageLatencyMicroseconds;

        // Assert
        Assert.Equal(75.5, latencyUs, precision: 1);
    }

    [Fact]
    public void ResidentMemoryMetrics_ShouldConvertMemoryToMB()
    {
        // Arrange
        var metrics = CreateMetrics(
            totalAllocatedBytes: 512L * 1024 * 1024,  // 512 MB
            usedPoolSizeBytes: 256L * 1024 * 1024);   // 256 MB

        // Act
        var totalMB = metrics.TotalMemoryMB;
        var usedMB = metrics.UsedMemoryMB;

        // Assert
        Assert.Equal(512.0, totalMB, precision: 1);
        Assert.Equal(256.0, usedMB, precision: 1);
    }

    [Fact]
    public void ResidentMemoryMetrics_MemoryEfficiency_ShouldEqualPoolHitRate()
    {
        // Arrange
        var metrics = CreateMetrics(
            poolHitCount: 950,
            poolMissCount: 50);

        // Act
        var efficiency = metrics.MemoryEfficiency;
        var hitRate = metrics.PoolHitRate;

        // Assert
        Assert.Equal(hitRate, efficiency);
        Assert.Equal(0.95, efficiency, precision: 3);
    }

    [Fact]
    public void ResidentMemoryMetrics_KernelEfficiency_ShouldEqualCacheHitRate()
    {
        // Arrange
        var metrics = CreateMetrics(kernelCacheHitRate: 0.98);

        // Act
        var efficiency = metrics.KernelEfficiency;

        // Assert
        Assert.Equal(0.98, efficiency);
    }

    [Fact]
    public void ResidentMemoryMetricsTracker_ShouldStartWithDefaultTime()
    {
        // Arrange
        var tracker = new ResidentMemoryMetricsTracker();

        // Act
        tracker.Start();

        // Assert - just verify it doesn't throw
        Assert.NotNull(tracker);
    }

    [Fact]
    public void ResidentMemoryMetricsTracker_ShouldRecordAllocation()
    {
        // Arrange
        var tracker = new ResidentMemoryMetricsTracker();
        tracker.Start();

        // Act
        tracker.RecordAllocation(sizeBytes: 1024, isPoolHit: true);
        tracker.RecordAllocation(sizeBytes: 2048, isPoolHit: false);
        tracker.RecordAllocation(sizeBytes: 512, isPoolHit: true);

        var metrics = tracker.GetMetrics();

        // Assert
        Assert.Equal(3, metrics.TotalAllocations);
        Assert.Equal(2, metrics.PoolHits);
    }

    [Fact]
    public void ResidentMemoryMetricsTracker_ShouldRecordWrites()
    {
        // Arrange
        var tracker = new ResidentMemoryMetricsTracker();
        tracker.Start();

        // Act
        tracker.RecordWrite(bytesWritten: 1024, transferTimeMicroseconds: 50.0);
        tracker.RecordWrite(bytesWritten: 2048, transferTimeMicroseconds: 75.0);

        var metrics = tracker.GetMetrics();

        // Assert
        Assert.Equal(2, metrics.TotalWrites);
    }

    [Fact]
    public void ResidentMemoryMetricsTracker_ShouldRecordReads()
    {
        // Arrange
        var tracker = new ResidentMemoryMetricsTracker();
        tracker.Start();

        // Act
        tracker.RecordRead(bytesRead: 512, transferTimeMicroseconds: 25.0);
        tracker.RecordRead(bytesRead: 1024, transferTimeMicroseconds: 40.0);
        tracker.RecordRead(bytesRead: 2048, transferTimeMicroseconds: 60.0);

        var metrics = tracker.GetMetrics();

        // Assert
        Assert.Equal(3, metrics.TotalReads);
    }

    [Fact]
    public void ResidentMemoryMetricsTracker_ShouldRecordComputes()
    {
        // Arrange
        var tracker = new ResidentMemoryMetricsTracker();
        tracker.Start();

        // Act
        tracker.RecordCompute(totalTimeMicroseconds: 150.0, isCacheHit: true);
        tracker.RecordCompute(totalTimeMicroseconds: 250.0, isCacheHit: true);
        tracker.RecordCompute(totalTimeMicroseconds: 500.0, isCacheHit: false);

        var metrics = tracker.GetMetrics();

        // Assert
        Assert.Equal(3, metrics.TotalComputes);
        Assert.Equal(2, metrics.KernelCacheHits);
    }

    [Fact]
    public void ResidentMemoryMetricsTracker_ShouldRecordReleases()
    {
        // Arrange
        var tracker = new ResidentMemoryMetricsTracker();
        tracker.Start();

        // Act
        tracker.RecordRelease(sizeBytes: 1024, returnedToPool: true);
        tracker.RecordRelease(sizeBytes: 2048, returnedToPool: false);

        var metrics = tracker.GetMetrics();

        // Assert
        Assert.Equal(2, metrics.TotalReleases);
    }

    [Fact]
    public void ResidentMemoryMetricsTracker_ShouldTrackStartTime()
    {
        // Arrange
        var beforeStart = DateTime.UtcNow;
        var tracker = new ResidentMemoryMetricsTracker();

        // Act
        tracker.Start();
        var metrics = tracker.GetMetrics();

        // Assert
        var afterStart = DateTime.UtcNow;
        Assert.InRange(metrics.StartTime, beforeStart, afterStart);
    }

    // Helper method to create metrics with specific values
    private ResidentMemoryMetrics CreateMetrics(
        long totalPoolSizeBytes = 1024L * 1024 * 1024,
        long usedPoolSizeBytes = 512L * 1024 * 1024,
        double poolUtilization = 0.5,
        long poolHitCount = 900,
        long poolMissCount = 100,
        double poolHitRate = 0.9,
        long totalMessagesProcessed = 100000,
        double messagesPerSecond = 1000000.0,
        double averageMessageLatencyNs = 100.0,
        long pendingMessageCount = 5,
        int activeAllocationCount = 100,
        long totalAllocatedBytes = 1024L * 1024 * 512,
        int kernelCacheSize = 50,
        double kernelCacheHitRate = 0.95,
        string deviceType = "GPU",
        string deviceName = "Test GPU",
        DateTime? startTime = null)
    {
        return new ResidentMemoryMetrics(
            totalPoolSizeBytes,
            usedPoolSizeBytes,
            poolUtilization,
            poolHitCount,
            poolMissCount,
            poolHitRate,
            totalMessagesProcessed,
            messagesPerSecond,
            averageMessageLatencyNs,
            pendingMessageCount,
            activeAllocationCount,
            totalAllocatedBytes,
            kernelCacheSize,
            kernelCacheHitRate,
            deviceType,
            deviceName,
            startTime ?? DateTime.UtcNow);
    }
}

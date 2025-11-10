using FluentAssertions;
using Orleans.GpuBridge.Abstractions.Capacity;
using Orleans.Runtime;

namespace Orleans.GpuBridge.Tests.Grains.Capacity;

/// <summary>
/// Unit tests for SiloGpuCapacity record
/// </summary>
public class SiloGpuCapacityTests
{
    private static SiloAddress CreateTestSiloAddress() =>
        SiloAddress.New(new System.Net.IPEndPoint(System.Net.IPAddress.Loopback, 11111), 0);

    [Fact]
    public void SiloGpuCapacity_Should_ExposeCapacityProperties()
    {
        // Arrange
        var silo = CreateTestSiloAddress();
        var capacity = new GpuCapacity(
            DeviceCount: 1,
            TotalMemoryMB: 8192,
            AvailableMemoryMB: 4096,
            QueueDepth: 5,
            Backend: "CUDA",
            LastUpdated: DateTime.UtcNow);

        // Act
        var siloCapacity = new SiloGpuCapacity(silo, capacity);

        // Assert
        siloCapacity.AvailableMemoryMB.Should().Be(4096);
        siloCapacity.QueueDepth.Should().Be(5);
        siloCapacity.HasGpu.Should().BeTrue();
        siloCapacity.MemoryUsagePercent.Should().BeApproximately(50.0, 0.1);
    }

    [Fact]
    public void SiloGpuCapacity_GetPlacementScore_Should_ReturnHighScore_ForGoodCapacity()
    {
        // Arrange
        var silo = CreateTestSiloAddress();
        var capacity = new GpuCapacity(
            DeviceCount: 1,
            TotalMemoryMB: 8192,
            AvailableMemoryMB: 8192,  // 100% available
            QueueDepth: 0,             // No queue
            Backend: "CUDA",
            LastUpdated: DateTime.UtcNow);

        var siloCapacity = new SiloGpuCapacity(silo, capacity);

        // Act
        var score = siloCapacity.GetPlacementScore();

        // Assert
        score.Should().BeApproximately(100.0, 0.1); // Perfect score
    }

    [Fact]
    public void SiloGpuCapacity_GetPlacementScore_Should_PenalizeHighQueueDepth()
    {
        // Arrange
        var silo = CreateTestSiloAddress();
        var capacity = new GpuCapacity(
            DeviceCount: 1,
            TotalMemoryMB: 8192,
            AvailableMemoryMB: 8192,  // 100% available
            QueueDepth: 10,            // High queue depth
            Backend: "CUDA",
            LastUpdated: DateTime.UtcNow);

        var siloCapacity = new SiloGpuCapacity(silo, capacity);

        // Act
        var score = siloCapacity.GetPlacementScore();

        // Assert
        score.Should().BeLessThan(100.0);
        score.Should().BeApproximately(80.0, 0.1); // 100 - 20 (max queue penalty)
    }

    [Fact]
    public void SiloGpuCapacity_GetPlacementScore_Should_ReturnZero_WhenNoGpu()
    {
        // Arrange
        var silo = CreateTestSiloAddress();
        var capacity = GpuCapacity.None;
        var siloCapacity = new SiloGpuCapacity(silo, capacity);

        // Act
        var score = siloCapacity.GetPlacementScore();

        // Assert
        score.Should().Be(0.0);
    }

    [Fact]
    public void SiloGpuCapacity_GetPlacementScore_Should_ReturnZero_WhenStale()
    {
        // Arrange
        var silo = CreateTestSiloAddress();
        var capacity = new GpuCapacity(
            DeviceCount: 1,
            TotalMemoryMB: 8192,
            AvailableMemoryMB: 8192,
            QueueDepth: 0,
            Backend: "CUDA",
            LastUpdated: DateTime.UtcNow.AddMinutes(-5)); // Stale

        var siloCapacity = new SiloGpuCapacity(silo, capacity);

        // Act
        var score = siloCapacity.GetPlacementScore();

        // Assert
        score.Should().Be(0.0);
    }

    [Fact]
    public void SiloGpuCapacity_ToString_Should_IncludeSiloAddress()
    {
        // Arrange
        var silo = CreateTestSiloAddress();
        var capacity = new GpuCapacity(
            DeviceCount: 1,
            TotalMemoryMB: 8192,
            AvailableMemoryMB: 4096,
            QueueDepth: 5,
            Backend: "CUDA",
            LastUpdated: DateTime.UtcNow);

        var siloCapacity = new SiloGpuCapacity(silo, capacity);

        // Act
        var str = siloCapacity.ToString();

        // Assert
        str.Should().Contain("Silo:");
        str.Should().Contain("Devices: 1");
    }
}

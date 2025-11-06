using FluentAssertions;
using Orleans.GpuBridge.Abstractions.Capacity;

namespace Orleans.GpuBridge.Tests.Grains.Capacity;

/// <summary>
/// Unit tests for GpuCapacity record
/// </summary>
public class GpuCapacityTests
{
    [Fact]
    public void GpuCapacity_Should_Calculate_MemoryUsagePercent_Correctly()
    {
        // Arrange
        var capacity = new GpuCapacity(
            DeviceCount: 1,
            TotalMemoryMB: 8192,
            AvailableMemoryMB: 2048,
            QueueDepth: 0,
            Backend: "CUDA",
            LastUpdated: DateTime.UtcNow);

        // Act & Assert
        capacity.MemoryUsagePercent.Should().BeApproximately(75.0, 0.1);
    }

    [Fact]
    public void GpuCapacity_HasCapacity_Should_BeTrue_When_DeviceAvailable()
    {
        // Arrange
        var capacity = new GpuCapacity(
            DeviceCount: 1,
            TotalMemoryMB: 8192,
            AvailableMemoryMB: 2048,
            QueueDepth: 0,
            Backend: "CUDA",
            LastUpdated: DateTime.UtcNow);

        // Act & Assert
        capacity.HasCapacity.Should().BeTrue();
    }

    [Fact]
    public void GpuCapacity_HasCapacity_Should_BeFalse_When_NoMemoryAvailable()
    {
        // Arrange
        var capacity = new GpuCapacity(
            DeviceCount: 1,
            TotalMemoryMB: 8192,
            AvailableMemoryMB: 0,
            QueueDepth: 0,
            Backend: "CUDA",
            LastUpdated: DateTime.UtcNow);

        // Act & Assert
        capacity.HasCapacity.Should().BeFalse();
    }

    [Fact]
    public void GpuCapacity_IsStale_Should_BeTrue_When_OlderThan2Minutes()
    {
        // Arrange
        var capacity = new GpuCapacity(
            DeviceCount: 1,
            TotalMemoryMB: 8192,
            AvailableMemoryMB: 2048,
            QueueDepth: 0,
            Backend: "CUDA",
            LastUpdated: DateTime.UtcNow.AddMinutes(-3));

        // Act & Assert
        capacity.IsStale.Should().BeTrue();
    }

    [Fact]
    public void GpuCapacity_IsStale_Should_BeFalse_When_RecentlyUpdated()
    {
        // Arrange
        var capacity = new GpuCapacity(
            DeviceCount: 1,
            TotalMemoryMB: 8192,
            AvailableMemoryMB: 2048,
            QueueDepth: 0,
            Backend: "CUDA",
            LastUpdated: DateTime.UtcNow.AddSeconds(-30));

        // Act & Assert
        capacity.IsStale.Should().BeFalse();
    }

    [Fact]
    public void GpuCapacity_None_Should_HaveZeroCapacity()
    {
        // Act
        var none = GpuCapacity.None;

        // Assert
        none.DeviceCount.Should().Be(0);
        none.TotalMemoryMB.Should().Be(0);
        none.AvailableMemoryMB.Should().Be(0);
        none.HasCapacity.Should().BeFalse();
    }

    [Fact]
    public void GpuCapacity_WithUpdate_Should_UpdateMemoryAndQueue()
    {
        // Arrange
        var original = new GpuCapacity(
            DeviceCount: 1,
            TotalMemoryMB: 8192,
            AvailableMemoryMB: 4096,
            QueueDepth: 5,
            Backend: "CUDA",
            LastUpdated: DateTime.UtcNow.AddMinutes(-1));

        // Act
        var updated = original.WithUpdate(
            availableMemoryMB: 2048,
            queueDepth: 10);

        // Assert
        updated.AvailableMemoryMB.Should().Be(2048);
        updated.QueueDepth.Should().Be(10);
        updated.DeviceCount.Should().Be(original.DeviceCount);
        updated.TotalMemoryMB.Should().Be(original.TotalMemoryMB);
        updated.LastUpdated.Should().BeCloseTo(DateTime.UtcNow, TimeSpan.FromSeconds(1));
    }

    [Fact]
    public void GpuCapacity_ToString_Should_ReturnHumanReadableFormat()
    {
        // Arrange
        var capacity = new GpuCapacity(
            DeviceCount: 2,
            TotalMemoryMB: 16384,
            AvailableMemoryMB: 8192,
            QueueDepth: 3,
            Backend: "CUDA",
            LastUpdated: DateTime.UtcNow);

        // Act
        var str = capacity.ToString();

        // Assert
        str.Should().Contain("Devices: 2");
        str.Should().Contain("Memory: 8192/16384MB");
        str.Should().Contain("Queue: 3");
        str.Should().Contain("Backend: CUDA");
    }
}

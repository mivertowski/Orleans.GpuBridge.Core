using FluentAssertions;
using Microsoft.Extensions.Logging.Abstractions;
using Orleans.GpuBridge.Abstractions.Capacity;
using Orleans.GpuBridge.Grains.Capacity;
using Orleans.Runtime;
using Orleans.TestingHost;

namespace Orleans.GpuBridge.Tests.Grains.Capacity;

/// <summary>
/// Unit tests for GpuCapacityGrain
/// </summary>
public class GpuCapacityGrainTests
{
    private static SiloAddress CreateTestSiloAddress(int port) =>
        SiloAddress.New(new System.Net.IPEndPoint(System.Net.IPAddress.Loopback, port), 0);

    private static GpuCapacity CreateTestCapacity(
        int deviceCount = 1,
        long totalMemoryMB = 8192,
        long availableMemoryMB = 4096,
        int queueDepth = 0) => new GpuCapacity(
            DeviceCount: deviceCount,
            TotalMemoryMB: totalMemoryMB,
            AvailableMemoryMB: availableMemoryMB,
            QueueDepth: queueDepth,
            Backend: "CUDA",
            LastUpdated: DateTime.UtcNow);

    [Fact]
    public async Task RegisterSiloAsync_Should_RegisterNewSilo()
    {
        // Arrange
        var grain = new GpuCapacityGrain(NullLogger<GpuCapacityGrain>.Instance);
        var silo = CreateTestSiloAddress(11111);
        var capacity = CreateTestCapacity();

        // Act
        await grain.RegisterSiloAsync(silo, capacity);
        var result = await grain.GetSiloCapacityAsync(silo);

        // Assert
        result.Should().NotBeNull();
        result!.DeviceCount.Should().Be(1);
        result.TotalMemoryMB.Should().Be(8192);
    }

    [Fact]
    public async Task RegisterSiloAsync_Should_ThrowException_WhenSiloIsNull()
    {
        // Arrange
        var grain = new GpuCapacityGrain(NullLogger<GpuCapacityGrain>.Instance);
        var capacity = CreateTestCapacity();

        // Act & Assert
        await grain.Invoking(g => g.RegisterSiloAsync(null!, capacity))
            .Should().ThrowAsync<ArgumentNullException>();
    }

    [Fact]
    public async Task UnregisterSiloAsync_Should_RemoveSilo()
    {
        // Arrange
        var grain = new GpuCapacityGrain(NullLogger<GpuCapacityGrain>.Instance);
        var silo = CreateTestSiloAddress(11111);
        var capacity = CreateTestCapacity();

        await grain.RegisterSiloAsync(silo, capacity);

        // Act
        await grain.UnregisterSiloAsync(silo);
        var result = await grain.GetSiloCapacityAsync(silo);

        // Assert
        result.Should().BeNull();
    }

    [Fact]
    public async Task UpdateCapacityAsync_Should_UpdateExistingSilo()
    {
        // Arrange
        var grain = new GpuCapacityGrain(NullLogger<GpuCapacityGrain>.Instance);
        var silo = CreateTestSiloAddress(11111);
        var capacity = CreateTestCapacity(availableMemoryMB: 4096, queueDepth: 0);

        await grain.RegisterSiloAsync(silo, capacity);

        // Act
        var updatedCapacity = CreateTestCapacity(availableMemoryMB: 2048, queueDepth: 5);
        await grain.UpdateCapacityAsync(silo, updatedCapacity);
        var result = await grain.GetSiloCapacityAsync(silo);

        // Assert
        result.Should().NotBeNull();
        result!.AvailableMemoryMB.Should().Be(2048);
        result.QueueDepth.Should().Be(5);
    }

    [Fact]
    public async Task UpdateCapacityAsync_Should_AutoRegister_UnknownSilo()
    {
        // Arrange
        var grain = new GpuCapacityGrain(NullLogger<GpuCapacityGrain>.Instance);
        var silo = CreateTestSiloAddress(11111);
        var capacity = CreateTestCapacity();

        // Act - Update without registering first
        await grain.UpdateCapacityAsync(silo, capacity);
        var result = await grain.GetSiloCapacityAsync(silo);

        // Assert
        result.Should().NotBeNull();
    }

    [Fact]
    public async Task GetGpuCapableSilosAsync_Should_ReturnOnlyGpuSilos()
    {
        // Arrange
        var grain = new GpuCapacityGrain(NullLogger<GpuCapacityGrain>.Instance);
        var silo1 = CreateTestSiloAddress(11111);
        var silo2 = CreateTestSiloAddress(22222);
        var silo3 = CreateTestSiloAddress(33333);

        await grain.RegisterSiloAsync(silo1, CreateTestCapacity(deviceCount: 1));
        await grain.RegisterSiloAsync(silo2, CreateTestCapacity(deviceCount: 0)); // No GPU
        await grain.RegisterSiloAsync(silo3, CreateTestCapacity(deviceCount: 2));

        // Act
        var gpuSilos = await grain.GetGpuCapableSilosAsync();

        // Assert
        gpuSilos.Should().HaveCount(2);
        gpuSilos.Should().Contain(s => s.SiloAddress.Equals(silo1));
        gpuSilos.Should().Contain(s => s.SiloAddress.Equals(silo3));
        gpuSilos.Should().NotContain(s => s.SiloAddress.Equals(silo2));
    }

    [Fact]
    public async Task GetGpuCapableSilosAsync_Should_OrderByPlacementScore()
    {
        // Arrange
        var grain = new GpuCapacityGrain(NullLogger<GpuCapacityGrain>.Instance);
        var silo1 = CreateTestSiloAddress(11111);
        var silo2 = CreateTestSiloAddress(22222);

        // Silo1: High available memory, low queue
        await grain.RegisterSiloAsync(silo1,
            CreateTestCapacity(availableMemoryMB: 8000, queueDepth: 1));

        // Silo2: Lower available memory, higher queue
        await grain.RegisterSiloAsync(silo2,
            CreateTestCapacity(availableMemoryMB: 4000, queueDepth: 5));

        // Act
        var gpuSilos = await grain.GetGpuCapableSilosAsync();

        // Assert
        gpuSilos.Should().HaveCount(2);
        gpuSilos[0].SiloAddress.Should().Be(silo1); // Should be first (better score)
    }

    [Fact]
    public async Task GetBestSiloForPlacementAsync_Should_ReturnSiloWithBestScore()
    {
        // Arrange
        var grain = new GpuCapacityGrain(NullLogger<GpuCapacityGrain>.Instance);
        var silo1 = CreateTestSiloAddress(11111);
        var silo2 = CreateTestSiloAddress(22222);

        await grain.RegisterSiloAsync(silo1,
            CreateTestCapacity(availableMemoryMB: 8000, queueDepth: 1));
        await grain.RegisterSiloAsync(silo2,
            CreateTestCapacity(availableMemoryMB: 4000, queueDepth: 5));

        // Act
        var bestSilo = await grain.GetBestSiloForPlacementAsync();

        // Assert
        bestSilo.Should().NotBeNull();
        bestSilo!.SiloAddress.Should().Be(silo1);
    }

    [Fact]
    public async Task GetBestSiloForPlacementAsync_Should_RespectMinimumMemory()
    {
        // Arrange
        var grain = new GpuCapacityGrain(NullLogger<GpuCapacityGrain>.Instance);
        var silo1 = CreateTestSiloAddress(11111);

        await grain.RegisterSiloAsync(silo1,
            CreateTestCapacity(availableMemoryMB: 2048));

        // Act - Request more memory than available
        var bestSilo = await grain.GetBestSiloForPlacementAsync(minimumMemoryMB: 4096);

        // Assert
        bestSilo.Should().BeNull();
    }

    [Fact]
    public async Task GetClusterStatsAsync_Should_AggregateStats()
    {
        // Arrange
        var grain = new GpuCapacityGrain(NullLogger<GpuCapacityGrain>.Instance);
        var silo1 = CreateTestSiloAddress(11111);
        var silo2 = CreateTestSiloAddress(22222);

        await grain.RegisterSiloAsync(silo1,
            CreateTestCapacity(deviceCount: 1, totalMemoryMB: 8192, availableMemoryMB: 4096, queueDepth: 2));
        await grain.RegisterSiloAsync(silo2,
            CreateTestCapacity(deviceCount: 2, totalMemoryMB: 16384, availableMemoryMB: 8192, queueDepth: 3));

        // Act
        var stats = await grain.GetClusterStatsAsync();

        // Assert
        stats.TotalSilos.Should().Be(2);
        stats.GpuCapableSilos.Should().Be(2);
        stats.TotalDevices.Should().Be(3);
        stats.TotalMemoryMB.Should().Be(24576);
        stats.AvailableMemoryMB.Should().Be(12288);
        stats.TotalQueueDepth.Should().Be(5);
        stats.AverageQueueDepth.Should().BeApproximately(2.5, 0.1);
    }
}

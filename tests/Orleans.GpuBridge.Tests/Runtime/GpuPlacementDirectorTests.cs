using FluentAssertions;
using Microsoft.Extensions.Logging.Abstractions;
using Moq;
using Orleans;
using Orleans.GpuBridge.Abstractions.Capacity;
using Orleans.GpuBridge.Runtime;
using Orleans.Runtime;
using Orleans.Runtime.Placement;

namespace Orleans.GpuBridge.Tests.Runtime;

/// <summary>
/// Unit tests for GPU-aware placement director
/// </summary>
public class GpuPlacementDirectorTests
{
    private readonly Mock<IGrainFactory> _mockGrainFactory;
    private readonly Mock<IGpuCapacityGrain> _mockCapacityGrain;
    private readonly Mock<IPlacementContext> _mockContext;
    private readonly GpuPlacementDirector _director;

    public GpuPlacementDirectorTests()
    {
        _mockGrainFactory = new Mock<IGrainFactory>();
        _mockCapacityGrain = new Mock<IGpuCapacityGrain>();
        _mockContext = new Mock<IPlacementContext>();

        _mockGrainFactory
            .Setup(f => f.GetGrain<IGpuCapacityGrain>(0, null))
            .Returns(_mockCapacityGrain.Object);

        _director = new GpuPlacementDirector(
            NullLogger<GpuPlacementDirector>.Instance,
            _mockGrainFactory.Object);
    }

    private static SiloAddress CreateTestSiloAddress(int port) =>
        SiloAddress.New(new System.Net.IPEndPoint(System.Net.IPAddress.Loopback, port), 0);

    private static PlacementTarget CreateTestTarget() =>
        new PlacementTarget(
            GrainId.Create("test", Guid.NewGuid().ToString()),
            null,
            null);

    [Fact]
    public async Task OnAddActivation_Should_SelectBestGpuSilo_WhenAvailable()
    {
        // Arrange
        var silo1 = CreateTestSiloAddress(11111);
        var silo2 = CreateTestSiloAddress(22222);
        var strategy = new GpuPlacementStrategy
        {
            MinimumGpuMemoryMB = 2048,
            PreferLocalPlacement = false
        };
        var target = CreateTestTarget();

        var bestSilo = new SiloGpuCapacity(
            silo1,
            new GpuCapacity(
                DeviceCount: 1,
                TotalMemoryMB: 8192,
                AvailableMemoryMB: 6000,
                QueueDepth: 0,
                Backend: "CUDA",
                LastUpdated: DateTime.UtcNow));

        _mockCapacityGrain
            .Setup(g => g.GetBestSiloForPlacementAsync(2048))
            .ReturnsAsync(bestSilo);

        _mockContext
            .Setup(c => c.GetCompatibleSilos(target))
            .Returns(new[] { silo1, silo2 });

        // Act
        var result = await _director.OnAddActivation(strategy, target, _mockContext.Object);

        // Assert
        result.Should().Be(silo1);
        _mockCapacityGrain.Verify(g => g.GetBestSiloForPlacementAsync(2048), Times.Once);
    }

    [Fact]
    public async Task OnAddActivation_Should_PreferLocalSilo_WhenPreferLocalPlacementTrue()
    {
        // Arrange
        var localSilo = CreateTestSiloAddress(11111);
        var remoteSilo = CreateTestSiloAddress(22222);
        var strategy = new GpuPlacementStrategy
        {
            MinimumGpuMemoryMB = 1024,
            PreferLocalPlacement = true
        };
        var target = CreateTestTarget();

        var localCapacity = new SiloGpuCapacity(
            localSilo,
            new GpuCapacity(
                DeviceCount: 1,
                TotalMemoryMB: 4096,
                AvailableMemoryMB: 2048,
                QueueDepth: 1,
                Backend: "CUDA",
                LastUpdated: DateTime.UtcNow));

        var remoteCapacity = new SiloGpuCapacity(
            remoteSilo,
            new GpuCapacity(
                DeviceCount: 1,
                TotalMemoryMB: 8192,
                AvailableMemoryMB: 6000,
                QueueDepth: 0,
                Backend: "CUDA",
                LastUpdated: DateTime.UtcNow));

        _mockCapacityGrain
            .Setup(g => g.GetBestSiloForPlacementAsync(1024))
            .ReturnsAsync(remoteCapacity);

        _mockCapacityGrain
            .Setup(g => g.GetGpuCapableSilosAsync())
            .ReturnsAsync(new List<SiloGpuCapacity> { localCapacity, remoteCapacity });

        _mockContext
            .Setup(c => c.LocalSilo)
            .Returns(localSilo);

        _mockContext
            .Setup(c => c.GetCompatibleSilos(target))
            .Returns(new[] { localSilo, remoteSilo });

        // Act
        var result = await _director.OnAddActivation(strategy, target, _mockContext.Object);

        // Assert
        result.Should().Be(localSilo); // Should prefer local even though remote has better score
    }

    [Fact]
    public async Task OnAddActivation_Should_FallbackToCompatibleSilo_WhenNoGpuAvailable()
    {
        // Arrange
        var silo1 = CreateTestSiloAddress(11111);
        var silo2 = CreateTestSiloAddress(22222);
        var strategy = new GpuPlacementStrategy
        {
            MinimumGpuMemoryMB = 4096,
            PreferLocalPlacement = false
        };
        var target = CreateTestTarget();

        _mockCapacityGrain
            .Setup(g => g.GetBestSiloForPlacementAsync(4096))
            .ReturnsAsync((SiloGpuCapacity?)null);

        _mockContext
            .Setup(c => c.GetCompatibleSilos(target))
            .Returns(new[] { silo1, silo2 });

        // Act
        var result = await _director.OnAddActivation(strategy, target, _mockContext.Object);

        // Assert
        result.Should().BeOneOf(silo1, silo2); // Should fall back to any compatible silo
    }

    [Fact]
    public async Task OnAddActivation_Should_ThrowException_WhenNoCompatibleSilos()
    {
        // Arrange
        var strategy = new GpuPlacementStrategy
        {
            MinimumGpuMemoryMB = 0,
            PreferLocalPlacement = false
        };
        var target = CreateTestTarget();

        _mockCapacityGrain
            .Setup(g => g.GetBestSiloForPlacementAsync(0))
            .ReturnsAsync((SiloGpuCapacity?)null);

        _mockContext
            .Setup(c => c.GetCompatibleSilos(target))
            .Returns(Array.Empty<SiloAddress>());

        // Act & Assert
        await _director.Invoking(d => d.OnAddActivation(strategy, target, _mockContext.Object))
            .Should().ThrowAsync<InvalidOperationException>()
            .WithMessage("No compatible silos found*");
    }

    [Fact]
    public async Task OnAddActivation_Should_RespectMinimumMemoryRequirement()
    {
        // Arrange
        var silo = CreateTestSiloAddress(11111);
        var strategy = new GpuPlacementStrategy
        {
            MinimumGpuMemoryMB = 4096,
            PreferLocalPlacement = false
        };
        var target = CreateTestTarget();

        var bestSilo = new SiloGpuCapacity(
            silo,
            new GpuCapacity(
                DeviceCount: 1,
                TotalMemoryMB: 8192,
                AvailableMemoryMB: 5000,
                QueueDepth: 0,
                Backend: "CUDA",
                LastUpdated: DateTime.UtcNow));

        _mockCapacityGrain
            .Setup(g => g.GetBestSiloForPlacementAsync(4096))
            .ReturnsAsync(bestSilo);

        _mockContext
            .Setup(c => c.GetCompatibleSilos(target))
            .Returns(new[] { silo });

        // Act
        var result = await _director.OnAddActivation(strategy, target, _mockContext.Object);

        // Assert
        result.Should().Be(silo);
        _mockCapacityGrain.Verify(g => g.GetBestSiloForPlacementAsync(4096), Times.Once);
    }

    [Fact]
    public async Task OnAddActivation_Should_HandleNonGpuStrategy_WithFallback()
    {
        // Arrange
        var silo = CreateTestSiloAddress(11111);
        var strategy = new RandomPlacementStrategy(); // Non-GPU strategy
        var target = CreateTestTarget();

        _mockContext
            .Setup(c => c.GetCompatibleSilos(target))
            .Returns(new[] { silo });

        // Act
        var result = await _director.OnAddActivation(strategy, target, _mockContext.Object);

        // Assert
        result.Should().Be(silo);
        _mockCapacityGrain.Verify(g => g.GetBestSiloForPlacementAsync(It.IsAny<int>()), Times.Never);
    }

    [Fact]
    public async Task OnAddActivation_Should_HandleExceptionGracefully()
    {
        // Arrange
        var silo = CreateTestSiloAddress(11111);
        var strategy = new GpuPlacementStrategy
        {
            MinimumGpuMemoryMB = 0,
            PreferLocalPlacement = false
        };
        var target = CreateTestTarget();

        _mockCapacityGrain
            .Setup(g => g.GetBestSiloForPlacementAsync(0))
            .ThrowsAsync(new InvalidOperationException("Capacity grain unavailable"));

        _mockContext
            .Setup(c => c.GetCompatibleSilos(target))
            .Returns(new[] { silo });

        // Act
        var result = await _director.OnAddActivation(strategy, target, _mockContext.Object);

        // Assert
        result.Should().Be(silo); // Should fall back to compatible silo on error
    }

    [Fact]
    public async Task OnAddActivation_Should_SkipLocalPlacement_WhenLocalDoesNotMeetRequirements()
    {
        // Arrange
        var localSilo = CreateTestSiloAddress(11111);
        var remoteSilo = CreateTestSiloAddress(22222);
        var strategy = new GpuPlacementStrategy
        {
            MinimumGpuMemoryMB = 4096, // Local has only 2048
            PreferLocalPlacement = true
        };
        var target = CreateTestTarget();

        var localCapacity = new SiloGpuCapacity(
            localSilo,
            new GpuCapacity(
                DeviceCount: 1,
                TotalMemoryMB: 4096,
                AvailableMemoryMB: 2048, // Doesn't meet requirement
                QueueDepth: 0,
                Backend: "CUDA",
                LastUpdated: DateTime.UtcNow));

        var remoteCapacity = new SiloGpuCapacity(
            remoteSilo,
            new GpuCapacity(
                DeviceCount: 1,
                TotalMemoryMB: 8192,
                AvailableMemoryMB: 6000, // Meets requirement
                QueueDepth: 0,
                Backend: "CUDA",
                LastUpdated: DateTime.UtcNow));

        _mockCapacityGrain
            .Setup(g => g.GetBestSiloForPlacementAsync(4096))
            .ReturnsAsync(remoteCapacity);

        _mockCapacityGrain
            .Setup(g => g.GetGpuCapableSilosAsync())
            .ReturnsAsync(new List<SiloGpuCapacity> { localCapacity, remoteCapacity });

        _mockContext
            .Setup(c => c.LocalSilo)
            .Returns(localSilo);

        _mockContext
            .Setup(c => c.GetCompatibleSilos(target))
            .Returns(new[] { localSilo, remoteSilo });

        // Act
        var result = await _director.OnAddActivation(strategy, target, _mockContext.Object);

        // Assert
        result.Should().Be(remoteSilo); // Should use remote since local doesn't meet requirements
    }
}

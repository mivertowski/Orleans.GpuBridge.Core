using FluentAssertions;
using Microsoft.Extensions.Logging;
using Moq;
using Orleans;
using Orleans.GpuBridge.Abstractions.Capacity;
using Orleans.GpuBridge.Runtime;
using Orleans.Runtime;
using Orleans.Runtime.Placement;
using System.Net;

namespace Orleans.GpuBridge.Tests.RC2.Runtime;

/// <summary>
/// Comprehensive test suite for GPU-aware placement strategies.
/// Tests GpuPlacementStrategy and GpuPlacementDirector with focus on:
/// - GPU capacity-based placement decisions
/// - Local placement preferences
/// - Fallback behavior when no GPU available
/// - Error handling and logging
/// - Concurrent placement operations
/// - Edge cases and boundary conditions
/// </summary>
public sealed class PlacementStrategyTests : IDisposable
{
    private readonly Mock<ILogger<GpuPlacementDirector>> _mockLogger;
    private readonly Mock<IGrainFactory> _mockGrainFactory;
    private readonly Mock<IPlacementContext> _mockPlacementContext;
    private readonly Mock<IGpuCapacityGrain> _mockCapacityGrain;

    public PlacementStrategyTests()
    {
        _mockLogger = new Mock<ILogger<GpuPlacementDirector>>();
        _mockGrainFactory = new Mock<IGrainFactory>();
        _mockPlacementContext = new Mock<IPlacementContext>();
        _mockCapacityGrain = new Mock<IGpuCapacityGrain>();

        // Setup capacity grain factory
        _mockGrainFactory
            .Setup(f => f.GetGrain<IGpuCapacityGrain>(It.IsAny<long>(), It.IsAny<string>()))
            .Returns(_mockCapacityGrain.Object);
    }

    public void Dispose()
    {
        // No resources to dispose
    }

    #region GpuPlacementStrategy Tests (10 tests)

    [Fact]
    public void GpuPlacementStrategy_Instance_IsSingleton()
    {
        // Arrange & Act
        var instance1 = GpuPlacementStrategy.Instance;
        var instance2 = GpuPlacementStrategy.Instance;

        // Assert
        instance1.Should().BeSameAs(instance2);
    }

    [Fact]
    public void GpuPlacementStrategy_DefaultValues_AreCorrect()
    {
        // Arrange & Act
        var strategy = GpuPlacementStrategy.Instance;

        // Assert
        strategy.PreferLocalPlacement.Should().BeFalse();
        strategy.MinimumGpuMemoryMB.Should().Be(0);
    }

    [Fact]
    public void GpuPlacementStrategy_WithCustomValues_PreservesSettings()
    {
        // Arrange & Act
        var strategy = new GpuPlacementStrategy
        {
            PreferLocalPlacement = true,
            MinimumGpuMemoryMB = 2048
        };

        // Assert
        strategy.PreferLocalPlacement.Should().BeTrue();
        strategy.MinimumGpuMemoryMB.Should().Be(2048);
    }

    [Fact]
    public void GpuPlacementStrategy_WithZeroMemory_IsValid()
    {
        // Arrange & Act
        var strategy = new GpuPlacementStrategy
        {
            MinimumGpuMemoryMB = 0
        };

        // Assert
        strategy.MinimumGpuMemoryMB.Should().Be(0);
    }

    [Fact]
    public void GpuPlacementStrategy_WithLargeMemory_IsValid()
    {
        // Arrange & Act
        var strategy = new GpuPlacementStrategy
        {
            MinimumGpuMemoryMB = 32768 // 32GB
        };

        // Assert
        strategy.MinimumGpuMemoryMB.Should().Be(32768);
    }

    [Fact]
    public void GpuPlacementStrategy_InheritanceCheck_IsPlacementStrategy()
    {
        // Arrange & Act
        var strategy = GpuPlacementStrategy.Instance;

        // Assert
        strategy.Should().BeAssignableTo<PlacementStrategy>();
    }

    [Fact]
    public void GpuPlacementStrategy_WithNegativeMemory_StillValid()
    {
        // Arrange & Act - The type system allows negative values
        var strategy = new GpuPlacementStrategy
        {
            MinimumGpuMemoryMB = -1
        };

        // Assert
        strategy.MinimumGpuMemoryMB.Should().Be(-1);
    }

    [Fact]
    public void GpuPlacementStrategy_MultipleInstances_Independent()
    {
        // Arrange & Act
        var strategy1 = new GpuPlacementStrategy { MinimumGpuMemoryMB = 1024 };
        var strategy2 = new GpuPlacementStrategy { MinimumGpuMemoryMB = 2048 };

        // Assert
        strategy1.MinimumGpuMemoryMB.Should().Be(1024);
        strategy2.MinimumGpuMemoryMB.Should().Be(2048);
    }

    [Fact]
    public void GpuPlacementStrategy_BothProperties_CanBeSet()
    {
        // Arrange & Act
        var strategy = new GpuPlacementStrategy
        {
            PreferLocalPlacement = true,
            MinimumGpuMemoryMB = 4096
        };

        // Assert
        strategy.PreferLocalPlacement.Should().BeTrue();
        strategy.MinimumGpuMemoryMB.Should().Be(4096);
    }

    [Fact]
    public void GpuPlacementStrategy_IsSerializable()
    {
        // Arrange
        var strategy = new GpuPlacementStrategy
        {
            PreferLocalPlacement = true,
            MinimumGpuMemoryMB = 4096
        };

        // Act & Assert - Type has [Serializable] attribute
        strategy.GetType().Should().BeDecoratedWith<SerializableAttribute>();
    }

    #endregion

    #region GpuPlacementDirector Basic Tests (12 tests)

    [Fact]
    public void GpuPlacementDirector_Constructor_WithNullLogger_ThrowsArgumentNullException()
    {
        // Arrange & Act
        var act = () => new GpuPlacementDirector(null!, _mockGrainFactory.Object);

        // Assert
        act.Should().Throw<ArgumentNullException>()
            .WithParameterName("logger");
    }

    [Fact]
    public void GpuPlacementDirector_Constructor_WithNullGrainFactory_ThrowsArgumentNullException()
    {
        // Arrange & Act
        var act = () => new GpuPlacementDirector(_mockLogger.Object, null!);

        // Assert
        act.Should().Throw<ArgumentNullException>()
            .WithParameterName("grainFactory");
    }

    [Fact]
    public void GpuPlacementDirector_ImplementsInterface_Correctly()
    {
        // Arrange & Act
        var director = new GpuPlacementDirector(_mockLogger.Object, _mockGrainFactory.Object);

        // Assert
        director.Should().BeAssignableTo<IPlacementDirector>();
    }

    [Fact]
    public async Task OnAddActivation_WithNonGpuStrategy_SelectsFallbackSilo()
    {
        // Arrange
        var director = new GpuPlacementDirector(_mockLogger.Object, _mockGrainFactory.Object);
        var strategy = new RandomPlacementStrategy();
        var target = CreatePlacementTarget();
        var compatibleSilos = new[] { CreateSiloAddress(), CreateSiloAddress() };

        _mockPlacementContext
            .Setup(c => c.GetCompatibleSilos(It.IsAny<PlacementTarget>()))
            .Returns(compatibleSilos);

        // Act
        var selectedSilo = await director.OnAddActivation(strategy, target, _mockPlacementContext.Object);

        // Assert
        selectedSilo.Should().BeOneOf(compatibleSilos);
        VerifyLogContains(LogLevel.Warning, "non-GPU strategy");
    }

    [Fact]
    public async Task OnAddActivation_WithGpuStrategy_SelectsBestGpuSilo()
    {
        // Arrange
        var director = new GpuPlacementDirector(_mockLogger.Object, _mockGrainFactory.Object);
        var strategy = new GpuPlacementStrategy { MinimumGpuMemoryMB = 2048 };
        var target = CreatePlacementTarget();
        var bestSilo = CreateSiloGpuCapacity(CreateSiloAddress(), 4096, 8192, 10);

        _mockCapacityGrain
            .Setup(g => g.GetBestSiloForPlacementAsync(2048))
            .ReturnsAsync(bestSilo);

        // Act
        var selectedSilo = await director.OnAddActivation(strategy, target, _mockPlacementContext.Object);

        // Assert
        selectedSilo.Should().Be(bestSilo.SiloAddress);
        _mockCapacityGrain.Verify(g => g.GetBestSiloForPlacementAsync(2048), Times.Once);
    }

    [Fact]
    public async Task OnAddActivation_WithNoGpuSilos_FallsBackToCpuSilo()
    {
        // Arrange
        var director = new GpuPlacementDirector(_mockLogger.Object, _mockGrainFactory.Object);
        var strategy = new GpuPlacementStrategy { MinimumGpuMemoryMB = 2048 };
        var target = CreatePlacementTarget();
        var compatibleSilos = new[] { CreateSiloAddress() };

        _mockCapacityGrain
            .Setup(g => g.GetBestSiloForPlacementAsync(2048))
            .ReturnsAsync((SiloGpuCapacity?)null);

        _mockPlacementContext
            .Setup(c => c.GetCompatibleSilos(It.IsAny<PlacementTarget>()))
            .Returns(compatibleSilos);

        // Act
        var selectedSilo = await director.OnAddActivation(strategy, target, _mockPlacementContext.Object);

        // Assert
        selectedSilo.Should().Be(compatibleSilos[0]);
        VerifyLogContains(LogLevel.Warning, "No GPU-capable silo");
    }

    [Fact]
    public async Task OnAddActivation_WithPreferLocal_SelectsLocalSiloWhenAvailable()
    {
        // Arrange
        var director = new GpuPlacementDirector(_mockLogger.Object, _mockGrainFactory.Object);
        var strategy = new GpuPlacementStrategy
        {
            PreferLocalPlacement = true,
            MinimumGpuMemoryMB = 1024
        };
        var target = CreatePlacementTarget();
        var localSilo = CreateSiloAddress();
        var remoteSilo = CreateSiloAddress();

        var localCapacity = CreateSiloGpuCapacity(localSilo, 2048, 4096, 5);
        var remoteCapacity = CreateSiloGpuCapacity(remoteSilo, 3072, 8192, 3);

        _mockPlacementContext.Setup(c => c.LocalSilo).Returns(localSilo);
        _mockCapacityGrain
            .Setup(g => g.GetBestSiloForPlacementAsync(1024))
            .ReturnsAsync(remoteCapacity);
        _mockCapacityGrain
            .Setup(g => g.GetGpuCapableSilosAsync())
            .ReturnsAsync(new List<SiloGpuCapacity> { localCapacity, remoteCapacity });

        // Act
        var selectedSilo = await director.OnAddActivation(strategy, target, _mockPlacementContext.Object);

        // Assert
        selectedSilo.Should().Be(localSilo);
    }

    [Fact]
    public async Task OnAddActivation_WithPreferLocal_ButInsufficientMemory_SelectsRemoteSilo()
    {
        // Arrange
        var director = new GpuPlacementDirector(_mockLogger.Object, _mockGrainFactory.Object);
        var strategy = new GpuPlacementStrategy
        {
            PreferLocalPlacement = true,
            MinimumGpuMemoryMB = 4096
        };
        var target = CreatePlacementTarget();
        var localSilo = CreateSiloAddress();
        var remoteSilo = CreateSiloAddress();

        var localCapacity = CreateSiloGpuCapacity(localSilo, 2048, 4096, 5);  // Insufficient
        var remoteCapacity = CreateSiloGpuCapacity(remoteSilo, 6144, 8192, 3);

        _mockPlacementContext.Setup(c => c.LocalSilo).Returns(localSilo);
        _mockCapacityGrain
            .Setup(g => g.GetBestSiloForPlacementAsync(4096))
            .ReturnsAsync(remoteCapacity);
        _mockCapacityGrain
            .Setup(g => g.GetGpuCapableSilosAsync())
            .ReturnsAsync(new List<SiloGpuCapacity> { localCapacity, remoteCapacity });

        // Act
        var selectedSilo = await director.OnAddActivation(strategy, target, _mockPlacementContext.Object);

        // Assert
        selectedSilo.Should().Be(remoteSilo);
    }

    [Fact]
    public async Task OnAddActivation_WithException_FallsBackGracefully()
    {
        // Arrange
        var director = new GpuPlacementDirector(_mockLogger.Object, _mockGrainFactory.Object);
        var strategy = new GpuPlacementStrategy { MinimumGpuMemoryMB = 2048 };
        var target = CreatePlacementTarget();
        var compatibleSilos = new[] { CreateSiloAddress() };

        _mockCapacityGrain
            .Setup(g => g.GetBestSiloForPlacementAsync(2048))
            .ThrowsAsync(new InvalidOperationException("Capacity grain error"));

        _mockPlacementContext
            .Setup(c => c.GetCompatibleSilos(It.IsAny<PlacementTarget>()))
            .Returns(compatibleSilos);

        // Act
        var selectedSilo = await director.OnAddActivation(strategy, target, _mockPlacementContext.Object);

        // Assert
        selectedSilo.Should().Be(compatibleSilos[0]);
        VerifyLogContains(LogLevel.Error, "Error selecting GPU-capable silo");
    }

    [Fact]
    public async Task OnAddActivation_WithNoCompatibleSilos_ThrowsInvalidOperation()
    {
        // Arrange
        var director = new GpuPlacementDirector(_mockLogger.Object, _mockGrainFactory.Object);
        var strategy = new GpuPlacementStrategy();
        var target = CreatePlacementTarget();

        _mockCapacityGrain
            .Setup(g => g.GetBestSiloForPlacementAsync(0))
            .ReturnsAsync((SiloGpuCapacity?)null);

        _mockPlacementContext
            .Setup(c => c.GetCompatibleSilos(It.IsAny<PlacementTarget>()))
            .Returns(Array.Empty<SiloAddress>());

        // Act
        var act = async () => await director.OnAddActivation(strategy, target, _mockPlacementContext.Object);

        // Assert
        await act.Should().ThrowAsync<InvalidOperationException>()
            .WithMessage("*No compatible silos*");
    }

    [Fact]
    public async Task OnAddActivation_WithZeroMemoryRequirement_SelectsAnyGpuSilo()
    {
        // Arrange
        var director = new GpuPlacementDirector(_mockLogger.Object, _mockGrainFactory.Object);
        var strategy = new GpuPlacementStrategy { MinimumGpuMemoryMB = 0 };
        var target = CreatePlacementTarget();
        var bestSilo = CreateSiloGpuCapacity(CreateSiloAddress(), 1024, 2048, 5);

        _mockCapacityGrain
            .Setup(g => g.GetBestSiloForPlacementAsync(0))
            .ReturnsAsync(bestSilo);

        // Act
        var selectedSilo = await director.OnAddActivation(strategy, target, _mockPlacementContext.Object);

        // Assert
        selectedSilo.Should().Be(bestSilo.SiloAddress);
    }

    [Fact]
    public async Task OnAddActivation_LogsPlacementDecision()
    {
        // Arrange
        var director = new GpuPlacementDirector(_mockLogger.Object, _mockGrainFactory.Object);
        var strategy = new GpuPlacementStrategy { MinimumGpuMemoryMB = 2048 };
        var target = CreatePlacementTarget();
        var bestSilo = CreateSiloGpuCapacity(CreateSiloAddress(), 4096, 8192, 10);

        _mockCapacityGrain
            .Setup(g => g.GetBestSiloForPlacementAsync(2048))
            .ReturnsAsync(bestSilo);

        // Act
        await director.OnAddActivation(strategy, target, _mockPlacementContext.Object);

        // Assert
        VerifyLogContains(LogLevel.Information, "Placed grain");
    }

    [Fact]
    public async Task OnAddActivation_RespectMinimumMemoryRequirement()
    {
        // Arrange
        var director = new GpuPlacementDirector(_mockLogger.Object, _mockGrainFactory.Object);
        var strategy = new GpuPlacementStrategy { MinimumGpuMemoryMB = 4096 };
        var target = CreatePlacementTarget();
        var bestSilo = CreateSiloGpuCapacity(CreateSiloAddress(), 5000, 8192, 5);

        _mockCapacityGrain
            .Setup(g => g.GetBestSiloForPlacementAsync(4096))
            .ReturnsAsync(bestSilo);

        _mockPlacementContext
            .Setup(c => c.GetCompatibleSilos(It.IsAny<PlacementTarget>()))
            .Returns(new[] { bestSilo.SiloAddress });

        // Act
        var selectedSilo = await director.OnAddActivation(strategy, target, _mockPlacementContext.Object);

        // Assert
        selectedSilo.Should().Be(bestSilo.SiloAddress);
        _mockCapacityGrain.Verify(g => g.GetBestSiloForPlacementAsync(4096), Times.Once);
    }

    #endregion

    #region Advanced Placement Scenarios (15 tests)

    [Fact]
    public async Task Placement_WithHighMemoryRequirement_SelectsHighCapacitySilo()
    {
        // Arrange
        var director = new GpuPlacementDirector(_mockLogger.Object, _mockGrainFactory.Object);
        var strategy = new GpuPlacementStrategy { MinimumGpuMemoryMB = 16384 };
        var target = CreatePlacementTarget();
        var highCapacitySilo = CreateSiloGpuCapacity(CreateSiloAddress(), 20480, 24576, 2);

        _mockCapacityGrain
            .Setup(g => g.GetBestSiloForPlacementAsync(16384))
            .ReturnsAsync(highCapacitySilo);

        // Act
        var selectedSilo = await director.OnAddActivation(strategy, target, _mockPlacementContext.Object);

        // Assert
        selectedSilo.Should().Be(highCapacitySilo.SiloAddress);
    }

    [Fact]
    public async Task Placement_WithLowQueueDepth_PrefersLessLoadedSilo()
    {
        // Arrange
        var director = new GpuPlacementDirector(_mockLogger.Object, _mockGrainFactory.Object);
        var strategy = new GpuPlacementStrategy { MinimumGpuMemoryMB = 2048 };
        var target = CreatePlacementTarget();
        var lowQueueSilo = CreateSiloGpuCapacity(CreateSiloAddress(), 4096, 8192, 5);

        _mockCapacityGrain
            .Setup(g => g.GetBestSiloForPlacementAsync(2048))
            .ReturnsAsync(lowQueueSilo);

        // Act
        var selectedSilo = await director.OnAddActivation(strategy, target, _mockPlacementContext.Object);

        // Assert
        selectedSilo.Should().Be(lowQueueSilo.SiloAddress);
    }

    [Fact]
    public async Task Placement_ConcurrentActivations_ThreadSafe()
    {
        // Arrange
        var director = new GpuPlacementDirector(_mockLogger.Object, _mockGrainFactory.Object);
        var strategy = new GpuPlacementStrategy { MinimumGpuMemoryMB = 2048 };

        var silos = Enumerable.Range(0, 5)
            .Select(_ => CreateSiloGpuCapacity(CreateSiloAddress(), 4096, 8192, 10))
            .ToArray();

        var siloIndex = 0;
        _mockCapacityGrain
            .Setup(g => g.GetBestSiloForPlacementAsync(2048))
            .ReturnsAsync(() => silos[Interlocked.Increment(ref siloIndex) % silos.Length]);

        // Act
        var tasks = Enumerable.Range(0, 20).Select(async _ =>
        {
            var target = CreatePlacementTarget();
            return await director.OnAddActivation(strategy, target, _mockPlacementContext.Object);
        }).ToArray();

        var selectedSilos = await Task.WhenAll(tasks);

        // Assert
        selectedSilos.Should().HaveCount(20);
        selectedSilos.Should().OnlyContain(s => s != null);
    }

    [Fact]
    public async Task Placement_WithMultipleStrategies_HandlesCorrectly()
    {
        // Arrange
        var director = new GpuPlacementDirector(_mockLogger.Object, _mockGrainFactory.Object);
        var gpuStrategy = new GpuPlacementStrategy { MinimumGpuMemoryMB = 2048 };
        var randomStrategy = new RandomPlacementStrategy();

        var gpuSilo = CreateSiloGpuCapacity(CreateSiloAddress(), 4096, 8192, 10);
        var compatibleSilos = new[] { CreateSiloAddress() };

        _mockCapacityGrain
            .Setup(g => g.GetBestSiloForPlacementAsync(2048))
            .ReturnsAsync(gpuSilo);

        _mockPlacementContext
            .Setup(c => c.GetCompatibleSilos(It.IsAny<PlacementTarget>()))
            .Returns(compatibleSilos);

        // Act
        var gpuTarget = CreatePlacementTarget();
        var randomTarget = CreatePlacementTarget();

        var gpuPlacement = await director.OnAddActivation(gpuStrategy, gpuTarget, _mockPlacementContext.Object);
        var randomPlacement = await director.OnAddActivation(randomStrategy, randomTarget, _mockPlacementContext.Object);

        // Assert
        gpuPlacement.Should().Be(gpuSilo.SiloAddress);
        randomPlacement.Should().Be(compatibleSilos[0]);
    }

    [Fact]
    public async Task Placement_WithDynamicCapacityChanges_AdaptsCorrectly()
    {
        // Arrange
        var director = new GpuPlacementDirector(_mockLogger.Object, _mockGrainFactory.Object);
        var strategy = new GpuPlacementStrategy { MinimumGpuMemoryMB = 2048 };

        var silo1 = CreateSiloGpuCapacity(CreateSiloAddress(), 4096, 8192, 10);
        var silo2 = CreateSiloGpuCapacity(CreateSiloAddress(), 2048, 8192, 5);

        var callCount = 0;
        _mockCapacityGrain
            .Setup(g => g.GetBestSiloForPlacementAsync(2048))
            .ReturnsAsync(() => callCount++ == 0 ? silo1 : silo2);

        // Act
        var target1 = CreatePlacementTarget();
        var target2 = CreatePlacementTarget();

        var placement1 = await director.OnAddActivation(strategy, target1, _mockPlacementContext.Object);
        var placement2 = await director.OnAddActivation(strategy, target2, _mockPlacementContext.Object);

        // Assert
        placement1.Should().Be(silo1.SiloAddress);
        placement2.Should().Be(silo2.SiloAddress);
    }

    [Fact]
    public async Task Placement_WithSiloFailure_FallsBackToOtherSilo()
    {
        // Arrange
        var director = new GpuPlacementDirector(_mockLogger.Object, _mockGrainFactory.Object);
        var strategy = new GpuPlacementStrategy { MinimumGpuMemoryMB = 2048 };
        var target = CreatePlacementTarget();
        var compatibleSilos = new[] { CreateSiloAddress(), CreateSiloAddress() };

        _mockCapacityGrain
            .Setup(g => g.GetBestSiloForPlacementAsync(2048))
            .ThrowsAsync(new TimeoutException("Silo not responding"));

        _mockPlacementContext
            .Setup(c => c.GetCompatibleSilos(It.IsAny<PlacementTarget>()))
            .Returns(compatibleSilos);

        // Act
        var selectedSilo = await director.OnAddActivation(strategy, target, _mockPlacementContext.Object);

        // Assert
        selectedSilo.Should().BeOneOf(compatibleSilos);
    }

    [Fact]
    public async Task Placement_WithPreferLocal_ButNoLocalGpu_SelectsRemote()
    {
        // Arrange
        var director = new GpuPlacementDirector(_mockLogger.Object, _mockGrainFactory.Object);
        var strategy = new GpuPlacementStrategy
        {
            PreferLocalPlacement = true,
            MinimumGpuMemoryMB = 2048
        };
        var target = CreatePlacementTarget();
        var localSilo = CreateSiloAddress();
        var remoteSilo = CreateSiloAddress();
        var remoteCapacity = CreateSiloGpuCapacity(remoteSilo, 4096, 8192, 5);

        _mockPlacementContext.Setup(c => c.LocalSilo).Returns(localSilo);
        _mockCapacityGrain
            .Setup(g => g.GetBestSiloForPlacementAsync(2048))
            .ReturnsAsync(remoteCapacity);
        _mockCapacityGrain
            .Setup(g => g.GetGpuCapableSilosAsync())
            .ReturnsAsync(new List<SiloGpuCapacity> { remoteCapacity });

        // Act
        var selectedSilo = await director.OnAddActivation(strategy, target, _mockPlacementContext.Object);

        // Assert
        selectedSilo.Should().Be(remoteSilo);
    }

    [Fact]
    public async Task Placement_WithMultipleGpuSilos_SelectsOptimalOne()
    {
        // Arrange
        var director = new GpuPlacementDirector(_mockLogger.Object, _mockGrainFactory.Object);
        var strategy = new GpuPlacementStrategy { MinimumGpuMemoryMB = 2048 };
        var target = CreatePlacementTarget();

        var optimalSilo = CreateSiloGpuCapacity(CreateSiloAddress(), 2560, 8192, 5);

        _mockCapacityGrain
            .Setup(g => g.GetBestSiloForPlacementAsync(2048))
            .ReturnsAsync(optimalSilo);

        // Act
        var selectedSilo = await director.OnAddActivation(strategy, target, _mockPlacementContext.Object);

        // Assert
        selectedSilo.Should().Be(optimalSilo.SiloAddress);
    }

    [Fact]
    public async Task Placement_StressTest_1000Activations_Stable()
    {
        // Arrange
        var director = new GpuPlacementDirector(_mockLogger.Object, _mockGrainFactory.Object);
        var strategy = new GpuPlacementStrategy { MinimumGpuMemoryMB = 2048 };

        var silos = Enumerable.Range(0, 10)
            .Select(_ => CreateSiloGpuCapacity(CreateSiloAddress(), 4096, 8192, 10))
            .ToArray();

        _mockCapacityGrain
            .Setup(g => g.GetBestSiloForPlacementAsync(2048))
            .ReturnsAsync(() => silos[Random.Shared.Next(silos.Length)]);

        // Act
        var sw = System.Diagnostics.Stopwatch.StartNew();
        var tasks = Enumerable.Range(0, 1000).Select(async _ =>
        {
            var target = CreatePlacementTarget();
            return await director.OnAddActivation(strategy, target, _mockPlacementContext.Object);
        }).ToArray();

        var selectedSilos = await Task.WhenAll(tasks);
        sw.Stop();

        // Assert
        selectedSilos.Should().HaveCount(1000);
        selectedSilos.Should().OnlyContain(s => s != null);
        sw.ElapsedMilliseconds.Should().BeLessThan(5000, "placement should be efficient");
    }

    [Fact]
    public async Task Placement_WithCapacityGrainTimeout_FallsBack()
    {
        // Arrange
        var director = new GpuPlacementDirector(_mockLogger.Object, _mockGrainFactory.Object);
        var strategy = new GpuPlacementStrategy { MinimumGpuMemoryMB = 2048 };
        var target = CreatePlacementTarget();
        var compatibleSilos = new[] { CreateSiloAddress() };

        _mockCapacityGrain
            .Setup(g => g.GetBestSiloForPlacementAsync(2048))
            .ThrowsAsync(new TimeoutException("Capacity grain timeout"));

        _mockPlacementContext
            .Setup(c => c.GetCompatibleSilos(It.IsAny<PlacementTarget>()))
            .Returns(compatibleSilos);

        // Act
        var selectedSilo = await director.OnAddActivation(strategy, target, _mockPlacementContext.Object);

        // Assert
        selectedSilo.Should().Be(compatibleSilos[0]);
    }

    [Fact]
    public async Task Placement_WithAllSilosOverloaded_StillPlaces()
    {
        // Arrange
        var director = new GpuPlacementDirector(_mockLogger.Object, _mockGrainFactory.Object);
        var strategy = new GpuPlacementStrategy { MinimumGpuMemoryMB = 2048 };
        var target = CreatePlacementTarget();
        var overloadedSilo = CreateSiloGpuCapacity(CreateSiloAddress(), 4096, 8192, 1000);

        _mockCapacityGrain
            .Setup(g => g.GetBestSiloForPlacementAsync(2048))
            .ReturnsAsync(overloadedSilo);

        // Act
        var selectedSilo = await director.OnAddActivation(strategy, target, _mockPlacementContext.Object);

        // Assert
        selectedSilo.Should().Be(overloadedSilo.SiloAddress);
    }

    [Fact]
    public async Task Placement_WithMemoryFragmentation_SelectsBestFit()
    {
        // Arrange
        var director = new GpuPlacementDirector(_mockLogger.Object, _mockGrainFactory.Object);
        var strategy = new GpuPlacementStrategy { MinimumGpuMemoryMB = 2000 };
        var target = CreatePlacementTarget();
        var bestFitSilo = CreateSiloGpuCapacity(CreateSiloAddress(), 2048, 8192, 10);

        _mockCapacityGrain
            .Setup(g => g.GetBestSiloForPlacementAsync(2000))
            .ReturnsAsync(bestFitSilo);

        // Act
        var selectedSilo = await director.OnAddActivation(strategy, target, _mockPlacementContext.Object);

        // Assert
        selectedSilo.Should().Be(bestFitSilo.SiloAddress);
    }

    [Fact]
    public async Task Placement_WithRapidSuccessiveActivations_MaintainsPerformance()
    {
        // Arrange
        var director = new GpuPlacementDirector(_mockLogger.Object, _mockGrainFactory.Object);
        var strategy = new GpuPlacementStrategy { MinimumGpuMemoryMB = 2048 };
        var silo = CreateSiloGpuCapacity(CreateSiloAddress(), 4096, 8192, 10);

        _mockCapacityGrain
            .Setup(g => g.GetBestSiloForPlacementAsync(2048))
            .ReturnsAsync(silo);

        // Act - Rapid successive activations
        var timings = new List<long>();

        for (int i = 0; i < 100; i++)
        {
            var sw = System.Diagnostics.Stopwatch.StartNew();
            var target = CreatePlacementTarget();
            await director.OnAddActivation(strategy, target, _mockPlacementContext.Object);
            sw.Stop();
            timings.Add(sw.ElapsedMilliseconds);
        }

        // Assert - Performance should not degrade
        var avgTime = timings.Average();
        var lastTenAvg = timings.TakeLast(10).Average();

        lastTenAvg.Should().BeLessThan(avgTime * 2, "performance should remain stable");
    }

    [Fact]
    public async Task Placement_WithNullBestSilo_HandlesGracefully()
    {
        // Arrange
        var director = new GpuPlacementDirector(_mockLogger.Object, _mockGrainFactory.Object);
        var strategy = new GpuPlacementStrategy { MinimumGpuMemoryMB = 2048 };
        var target = CreatePlacementTarget();
        var compatibleSilos = new[] { CreateSiloAddress() };

        _mockCapacityGrain
            .Setup(g => g.GetBestSiloForPlacementAsync(2048))
            .ReturnsAsync((SiloGpuCapacity?)null);

        _mockPlacementContext
            .Setup(c => c.GetCompatibleSilos(It.IsAny<PlacementTarget>()))
            .Returns(compatibleSilos);

        // Act
        var selectedSilo = await director.OnAddActivation(strategy, target, _mockPlacementContext.Object);

        // Assert
        selectedSilo.Should().Be(compatibleSilos[0]);
    }

    #endregion

    #region Edge Cases and Boundary Tests (13 tests)

    [Fact]
    public async Task Placement_WithVeryLargeQueueDepth_HandlesCorrectly()
    {
        // Arrange
        var director = new GpuPlacementDirector(_mockLogger.Object, _mockGrainFactory.Object);
        var strategy = new GpuPlacementStrategy { MinimumGpuMemoryMB = 2048 };
        var target = CreatePlacementTarget();
        var silo = CreateSiloGpuCapacity(CreateSiloAddress(), 4096, 8192, int.MaxValue);

        _mockCapacityGrain
            .Setup(g => g.GetBestSiloForPlacementAsync(2048))
            .ReturnsAsync(silo);

        // Act
        var selectedSilo = await director.OnAddActivation(strategy, target, _mockPlacementContext.Object);

        // Assert
        selectedSilo.Should().Be(silo.SiloAddress);
    }

    [Fact]
    public async Task Placement_WithZeroAvailableMemory_StillSelects()
    {
        // Arrange
        var director = new GpuPlacementDirector(_mockLogger.Object, _mockGrainFactory.Object);
        var strategy = new GpuPlacementStrategy { MinimumGpuMemoryMB = 0 };
        var target = CreatePlacementTarget();
        var silo = CreateSiloGpuCapacity(CreateSiloAddress(), 0, 8192, 5);

        _mockCapacityGrain
            .Setup(g => g.GetBestSiloForPlacementAsync(0))
            .ReturnsAsync(silo);

        // Act
        var selectedSilo = await director.OnAddActivation(strategy, target, _mockPlacementContext.Object);

        // Assert
        selectedSilo.Should().Be(silo.SiloAddress);
    }

    [Fact]
    public async Task Placement_WithMaxIntMemoryRequirement_HandlesCorrectly()
    {
        // Arrange
        var director = new GpuPlacementDirector(_mockLogger.Object, _mockGrainFactory.Object);
        var strategy = new GpuPlacementStrategy { MinimumGpuMemoryMB = int.MaxValue };
        var target = CreatePlacementTarget();
        var compatibleSilos = new[] { CreateSiloAddress() };

        _mockCapacityGrain
            .Setup(g => g.GetBestSiloForPlacementAsync(int.MaxValue))
            .ReturnsAsync((SiloGpuCapacity?)null);

        _mockPlacementContext
            .Setup(c => c.GetCompatibleSilos(It.IsAny<PlacementTarget>()))
            .Returns(compatibleSilos);

        // Act
        var selectedSilo = await director.OnAddActivation(strategy, target, _mockPlacementContext.Object);

        // Assert
        selectedSilo.Should().Be(compatibleSilos[0]);
    }

    [Fact]
    public async Task Placement_WithSingleSilo_AlwaysSelectsThatSilo()
    {
        // Arrange
        var director = new GpuPlacementDirector(_mockLogger.Object, _mockGrainFactory.Object);
        var strategy = new GpuPlacementStrategy { MinimumGpuMemoryMB = 2048 };
        var singleSilo = CreateSiloGpuCapacity(CreateSiloAddress(), 4096, 8192, 10);

        _mockCapacityGrain
            .Setup(g => g.GetBestSiloForPlacementAsync(2048))
            .ReturnsAsync(singleSilo);

        // Act - Place multiple grains
        var tasks = Enumerable.Range(0, 10).Select(async _ =>
        {
            var target = CreatePlacementTarget();
            return await director.OnAddActivation(strategy, target, _mockPlacementContext.Object);
        }).ToArray();

        var selectedSilos = await Task.WhenAll(tasks);

        // Assert
        selectedSilos.Should().OnlyContain(s => s.Equals(singleSilo.SiloAddress));
    }

    [Fact]
    public async Task Placement_WithEmptyCompatibleSilosList_Throws()
    {
        // Arrange
        var director = new GpuPlacementDirector(_mockLogger.Object, _mockGrainFactory.Object);
        var strategy = new GpuPlacementStrategy { MinimumGpuMemoryMB = 2048 };
        var target = CreatePlacementTarget();

        _mockCapacityGrain
            .Setup(g => g.GetBestSiloForPlacementAsync(2048))
            .ReturnsAsync((SiloGpuCapacity?)null);

        _mockPlacementContext
            .Setup(c => c.GetCompatibleSilos(It.IsAny<PlacementTarget>()))
            .Returns(Array.Empty<SiloAddress>());

        // Act
        var act = async () => await director.OnAddActivation(strategy, target, _mockPlacementContext.Object);

        // Assert
        await act.Should().ThrowAsync<InvalidOperationException>()
            .WithMessage("*No compatible silos*");
    }

    [Fact]
    public async Task Placement_WithCapacityGrainReturningNull_MultipleTimes_ConsistentFallback()
    {
        // Arrange
        var director = new GpuPlacementDirector(_mockLogger.Object, _mockGrainFactory.Object);
        var strategy = new GpuPlacementStrategy { MinimumGpuMemoryMB = 2048 };
        var compatibleSilos = new[] { CreateSiloAddress() };

        _mockCapacityGrain
            .Setup(g => g.GetBestSiloForPlacementAsync(2048))
            .ReturnsAsync((SiloGpuCapacity?)null);

        _mockPlacementContext
            .Setup(c => c.GetCompatibleSilos(It.IsAny<PlacementTarget>()))
            .Returns(compatibleSilos);

        // Act
        var placements = new List<SiloAddress>();
        for (int i = 0; i < 5; i++)
        {
            var target = CreatePlacementTarget();
            var silo = await director.OnAddActivation(strategy, target, _mockPlacementContext.Object);
            placements.Add(silo);
        }

        // Assert
        placements.Should().OnlyContain(s => s.Equals(compatibleSilos[0]));
    }

    [Fact]
    public async Task Placement_WithVeryHighConcurrency_NoDeadlocks()
    {
        // Arrange
        var director = new GpuPlacementDirector(_mockLogger.Object, _mockGrainFactory.Object);
        var strategy = new GpuPlacementStrategy { MinimumGpuMemoryMB = 2048 };

        var silos = Enumerable.Range(0, 3)
            .Select(_ => CreateSiloGpuCapacity(CreateSiloAddress(), 4096, 8192, 10))
            .ToArray();

        _mockCapacityGrain
            .Setup(g => g.GetBestSiloForPlacementAsync(2048))
            .ReturnsAsync(() => silos[Random.Shared.Next(silos.Length)]);

        // Act
        var tasks = Enumerable.Range(0, 500).Select(async _ =>
        {
            var target = CreatePlacementTarget();
            return await director.OnAddActivation(strategy, target, _mockPlacementContext.Object);
        }).ToArray();

        var timeout = Task.Delay(TimeSpan.FromSeconds(10));
        var completed = await Task.WhenAny(Task.WhenAll(tasks), timeout);

        // Assert
        completed.Should().NotBeSameAs(timeout, "should complete without deadlocks");
        tasks.All(t => t.IsCompleted).Should().BeTrue();
    }

    [Fact]
    public async Task Placement_WithCapacityGrainException_AndNoCompatibleSilos_Throws()
    {
        // Arrange
        var director = new GpuPlacementDirector(_mockLogger.Object, _mockGrainFactory.Object);
        var strategy = new GpuPlacementStrategy { MinimumGpuMemoryMB = 2048 };
        var target = CreatePlacementTarget();

        _mockCapacityGrain
            .Setup(g => g.GetBestSiloForPlacementAsync(2048))
            .ThrowsAsync(new InvalidOperationException("Capacity error"));

        _mockPlacementContext
            .Setup(c => c.GetCompatibleSilos(It.IsAny<PlacementTarget>()))
            .Returns(Array.Empty<SiloAddress>());

        // Act
        var act = async () => await director.OnAddActivation(strategy, target, _mockPlacementContext.Object);

        // Assert
        await act.Should().ThrowAsync<InvalidOperationException>();
    }

    [Fact]
    public async Task Placement_WithRandomPlacementStrategy_UsesRandomSelection()
    {
        // Arrange
        var director = new GpuPlacementDirector(_mockLogger.Object, _mockGrainFactory.Object);
        var strategy = new RandomPlacementStrategy();
        var compatibleSilos = new[] { CreateSiloAddress(), CreateSiloAddress(), CreateSiloAddress() };

        _mockPlacementContext
            .Setup(c => c.GetCompatibleSilos(It.IsAny<PlacementTarget>()))
            .Returns(compatibleSilos);

        // Act - Place multiple grains
        var placements = new List<SiloAddress>();
        for (int i = 0; i < 10; i++)
        {
            var target = CreatePlacementTarget();
            var silo = await director.OnAddActivation(strategy, target, _mockPlacementContext.Object);
            placements.Add(silo);
        }

        // Assert - Should use one of the compatible silos
        placements.Should().OnlyContain(s => compatibleSilos.Contains(s));
    }

    [Fact]
    public async Task Placement_WithNegativeMemoryRequirement_HandledCorrectly()
    {
        // Arrange
        var director = new GpuPlacementDirector(_mockLogger.Object, _mockGrainFactory.Object);
        var strategy = new GpuPlacementStrategy { MinimumGpuMemoryMB = -1024 };
        var target = CreatePlacementTarget();
        var silo = CreateSiloGpuCapacity(CreateSiloAddress(), 4096, 8192, 10);

        _mockCapacityGrain
            .Setup(g => g.GetBestSiloForPlacementAsync(-1024))
            .ReturnsAsync(silo);

        // Act
        var selectedSilo = await director.OnAddActivation(strategy, target, _mockPlacementContext.Object);

        // Assert
        selectedSilo.Should().Be(silo.SiloAddress);
    }

    [Fact]
    public async Task Placement_WithAllSilosHavingZeroMemory_StillPlaces()
    {
        // Arrange
        var director = new GpuPlacementDirector(_mockLogger.Object, _mockGrainFactory.Object);
        var strategy = new GpuPlacementStrategy { MinimumGpuMemoryMB = 0 };
        var target = CreatePlacementTarget();
        var silo = CreateSiloGpuCapacity(CreateSiloAddress(), 0, 0, 0);

        _mockCapacityGrain
            .Setup(g => g.GetBestSiloForPlacementAsync(0))
            .ReturnsAsync(silo);

        // Act
        var selectedSilo = await director.OnAddActivation(strategy, target, _mockPlacementContext.Object);

        // Assert
        selectedSilo.Should().Be(silo.SiloAddress);
    }

    [Fact]
    public async Task Placement_WithEmptyGpuSilosList_FallsBack()
    {
        // Arrange
        var director = new GpuPlacementDirector(_mockLogger.Object, _mockGrainFactory.Object);
        var strategy = new GpuPlacementStrategy
        {
            PreferLocalPlacement = true,
            MinimumGpuMemoryMB = 2048
        };
        var target = CreatePlacementTarget();
        var localSilo = CreateSiloAddress();
        var compatibleSilos = new[] { localSilo };

        _mockPlacementContext.Setup(c => c.LocalSilo).Returns(localSilo);
        _mockPlacementContext
            .Setup(c => c.GetCompatibleSilos(It.IsAny<PlacementTarget>()))
            .Returns(compatibleSilos);

        _mockCapacityGrain
            .Setup(g => g.GetBestSiloForPlacementAsync(2048))
            .ReturnsAsync((SiloGpuCapacity?)null);
        _mockCapacityGrain
            .Setup(g => g.GetGpuCapableSilosAsync())
            .ReturnsAsync(new List<SiloGpuCapacity>());

        // Act
        var selectedSilo = await director.OnAddActivation(strategy, target, _mockPlacementContext.Object);

        // Assert
        selectedSilo.Should().Be(localSilo);
    }

    [Fact]
    public async Task Placement_WithPreferLocal_AndLocalSiloMatches_SelectsLocal()
    {
        // Arrange
        var director = new GpuPlacementDirector(_mockLogger.Object, _mockGrainFactory.Object);
        var strategy = new GpuPlacementStrategy
        {
            PreferLocalPlacement = true,
            MinimumGpuMemoryMB = 1024
        };
        var target = CreatePlacementTarget();
        var localSilo = CreateSiloAddress();
        var localCapacity = CreateSiloGpuCapacity(localSilo, 2048, 4096, 5);

        _mockPlacementContext.Setup(c => c.LocalSilo).Returns(localSilo);
        _mockCapacityGrain
            .Setup(g => g.GetBestSiloForPlacementAsync(1024))
            .ReturnsAsync(localCapacity);
        _mockCapacityGrain
            .Setup(g => g.GetGpuCapableSilosAsync())
            .ReturnsAsync(new List<SiloGpuCapacity> { localCapacity });

        // Act
        var selectedSilo = await director.OnAddActivation(strategy, target, _mockPlacementContext.Object);

        // Assert
        selectedSilo.Should().Be(localSilo);
    }

    #endregion

    #region Helper Methods

    private PlacementTarget CreatePlacementTarget()
    {
        return new PlacementTarget(
            GrainId.Create("test-grain", Guid.NewGuid().ToString()),
            null,
            new GrainInterfaceType("test-interface"),
            0);
    }

    private SiloAddress CreateSiloAddress()
    {
        return SiloAddress.New(
            new IPEndPoint(IPAddress.Loopback, Random.Shared.Next(10000, 60000)),
            0);
    }

    private SiloGpuCapacity CreateSiloGpuCapacity(
        SiloAddress address,
        long availableMemory,
        long totalMemory,
        int queueDepth)
    {
        var capacity = new GpuCapacity(
            DeviceCount: 1,
            TotalMemoryMB: totalMemory,
            AvailableMemoryMB: availableMemory,
            QueueDepth: queueDepth,
            Backend: "CUDA",
            LastUpdated: DateTime.UtcNow);

        return new SiloGpuCapacity(address, capacity);
    }

    private void VerifyLogContains(LogLevel level, string message)
    {
        _mockLogger.Verify(
            x => x.Log(
                level,
                It.IsAny<EventId>(),
                It.Is<It.IsAnyType>((v, t) => v.ToString()!.Contains(message)),
                It.IsAny<Exception>(),
                It.IsAny<Func<It.IsAnyType, Exception?, string>>()),
            Times.AtLeastOnce);
    }

    #endregion
}

/// <summary>
/// Random placement strategy for testing non-GPU placement fallback behavior
/// </summary>
internal sealed class RandomPlacementStrategy : PlacementStrategy
{
}

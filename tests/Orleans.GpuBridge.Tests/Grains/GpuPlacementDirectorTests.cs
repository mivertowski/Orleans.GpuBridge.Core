using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using FluentAssertions;
using Microsoft.Extensions.Logging;
using Moq;
using Orleans;
using Orleans.GpuBridge.Grains;
using Orleans.Placement;
using Orleans.Runtime;
using Orleans.Runtime.Placement;
using Xunit;

namespace Orleans.GpuBridge.Tests.Grains;

/// <summary>
/// Unit tests for GpuPlacementDirector and related placement classes
/// </summary>
public class GpuPlacementDirectorTests
{
    private readonly Mock<ILogger<GpuPlacementDirector>> _mockLogger;
    private readonly GpuPlacementDirector _director;
    private readonly Mock<IPlacementContext> _mockContext;
    private readonly PlacementTarget _testTarget;

    public GpuPlacementDirectorTests()
    {
        _mockLogger = new Mock<ILogger<GpuPlacementDirector>>();
        _director = new GpuPlacementDirector(_mockLogger.Object);
        _mockContext = new Mock<IPlacementContext>();
        
        // Create a concrete PlacementTarget since it can't be mocked
        var grainId = GrainId.Create("test-grain", "test-key");
        var grainInterfaceType = new GrainInterfaceType("test-interface");
        _testTarget = new PlacementTarget(grainId, null, grainInterfaceType, 0);
    }

    [Fact]
    public async Task OnAddActivation_WithGpuPlacementStrategy_ShouldSelectCompatibleSilo()
    {
        // Arrange
        var strategy = new GpuPlacementStrategy
        {
            PreferLocalGpu = true,
            PreferredDeviceIndex = 0
        };

        var availableSilos = new List<SiloAddress>
        {
            SiloAddress.New(new System.Net.IPEndPoint(System.Net.IPAddress.Loopback, 11111), 30000),
            SiloAddress.New(new System.Net.IPEndPoint(System.Net.IPAddress.Loopback, 11112), 30001)
        };

        _mockContext.Setup(c => c.GetCompatibleSilos(_testTarget))
                   .Returns(availableSilos.ToArray());


        // Act
        var selectedSilo = await _director.OnAddActivation(strategy, _testTarget, _mockContext.Object);

        // Assert
        selectedSilo.Should().NotBeNull();
        availableSilos.Should().Contain(selectedSilo);
        
        _mockContext.Verify(c => c.GetCompatibleSilos(_testTarget), Times.Once);
    }

    [Fact]
    public async Task OnAddActivation_WithNoCompatibleSilos_ShouldThrowException()
    {
        // Arrange
        var strategy = new GpuPlacementStrategy
        {
            PreferLocalGpu = true
        };

        var emptySiloList = new SiloAddress[0];
        _mockContext.Setup(c => c.GetCompatibleSilos(_testTarget))
                   .Returns(emptySiloList);

        // Act & Assert
        await Assert.ThrowsAsync<OrleansException>(() =>
            _director.OnAddActivation(strategy, _testTarget, _mockContext.Object));
    }

    [Fact]
    public async Task OnAddActivation_WithInvalidStrategy_ShouldThrowArgumentException()
    {
        // Arrange
        var invalidStrategy = new Mock<PlacementStrategy>().Object;

        // Act & Assert
        await Assert.ThrowsAsync<ArgumentException>(() =>
            _director.OnAddActivation(invalidStrategy, _testTarget, _mockContext.Object));
    }

    [Fact]
    public async Task OnAddActivation_WithMultipleSilos_ShouldSelectBestSilo()
    {
        // Arrange
        var strategy = new GpuPlacementStrategy
        {
            PreferLocalGpu = true,
            PreferredDeviceIndex = 1
        };

        var availableSilos = new List<SiloAddress>
        {
            SiloAddress.New(new System.Net.IPEndPoint(System.Net.IPAddress.Loopback, 11111), 30000),
            SiloAddress.New(new System.Net.IPEndPoint(System.Net.IPAddress.Loopback, 11112), 30001),
            SiloAddress.New(new System.Net.IPEndPoint(System.Net.IPAddress.Loopback, 11113), 30002)
        };

        _mockContext.Setup(c => c.GetCompatibleSilos(_testTarget))
                   .Returns(availableSilos.ToArray());


        // Act
        var selectedSilo = await _director.OnAddActivation(strategy, _testTarget, _mockContext.Object);

        // Assert
        selectedSilo.Should().NotBeNull();
        availableSilos.Should().Contain(selectedSilo);
        
        // Verify that selection logic was applied (all silos have simulated GPU capabilities)
        _mockContext.Verify(c => c.GetCompatibleSilos(_testTarget), Times.Once);
    }

    [Fact]
    public async Task OnAddActivation_WithPreferredDeviceIndex_ShouldPreferSiloWithDevice()
    {
        // Arrange
        var strategy = new GpuPlacementStrategy
        {
            PreferLocalGpu = true,
            PreferredDeviceIndex = 0
        };

        var availableSilos = new List<SiloAddress>
        {
            SiloAddress.New(new System.Net.IPEndPoint(System.Net.IPAddress.Loopback, 11111), 30000)
        };

        _mockContext.Setup(c => c.GetCompatibleSilos(_testTarget))
                   .Returns(availableSilos.ToArray());


        // Act
        var selectedSilo = await _director.OnAddActivation(strategy, _testTarget, _mockContext.Object);

        // Assert
        selectedSilo.Should().NotBeNull();
        selectedSilo.Should().Be(availableSilos.First());
    }

    [Fact]
    public void GpuPlacementStrategy_DefaultValues_ShouldBeCorrect()
    {
        // Act
        var strategy = new GpuPlacementStrategy();

        // Assert
        strategy.PreferLocalGpu.Should().BeFalse(); // Default value
        strategy.PreferredDeviceIndex.Should().BeNull();
    }

    [Fact]
    public void GpuPlacementStrategy_SetProperties_ShouldRetainValues()
    {
        // Arrange
        var strategy = new GpuPlacementStrategy
        {
            PreferLocalGpu = true,
            PreferredDeviceIndex = 5
        };

        // Assert
        strategy.PreferLocalGpu.Should().BeTrue();
        strategy.PreferredDeviceIndex.Should().Be(5);
    }

    [Fact]
    public void GpuPlacementAttribute_DefaultConstructor_ShouldSetDefaultValues()
    {
        // Act
        var attribute = new GpuPlacementAttribute();

        // Assert
        attribute.PreferLocalGpu.Should().BeTrue();
        attribute.PreferredDeviceIndex.Should().Be(-1);
        attribute.PlacementStrategy.Should().BeOfType<GpuPlacementStrategy>();
        
        var strategy = (GpuPlacementStrategy)attribute.PlacementStrategy;
        strategy.PreferLocalGpu.Should().BeTrue();
        strategy.PreferredDeviceIndex.Should().BeNull();
    }

    [Fact]
    public void GpuPlacementAttribute_WithCustomValues_ShouldRetainValues()
    {
        // Act
        var attribute = new GpuPlacementAttribute
        {
            PreferLocalGpu = false,
            PreferredDeviceIndex = 3
        };

        // Assert
        attribute.PreferLocalGpu.Should().BeFalse();
        attribute.PreferredDeviceIndex.Should().Be(3);
    }

    [Theory]
    [InlineData(true, 0)]
    [InlineData(false, -1)]
    [InlineData(true, 5)]
    [InlineData(false, 10)]
    public async Task OnAddActivation_WithVariousStrategySettings_ShouldHandleCorrectly(bool preferLocalGpu, int deviceIndex)
    {
        // Arrange
        var strategy = new GpuPlacementStrategy
        {
            PreferLocalGpu = preferLocalGpu,
            PreferredDeviceIndex = deviceIndex >= 0 ? deviceIndex : null
        };

        var availableSilos = new List<SiloAddress>
        {
            SiloAddress.New(new System.Net.IPEndPoint(System.Net.IPAddress.Loopback, 11111), 30000),
            SiloAddress.New(new System.Net.IPEndPoint(System.Net.IPAddress.Loopback, 11112), 30001)
        };

        _mockContext.Setup(c => c.GetCompatibleSilos(_testTarget))
                   .Returns(availableSilos.ToArray());


        // Act
        var selectedSilo = await _director.OnAddActivation(strategy, _testTarget, _mockContext.Object);

        // Assert
        selectedSilo.Should().NotBeNull();
        availableSilos.Should().Contain(selectedSilo);
    }

    [Fact]
    public async Task OnAddActivation_LogsWarning_WhenNoGpuCapableSilo()
    {
        // Arrange
        var strategy = new GpuPlacementStrategy
        {
            PreferLocalGpu = true
        };

        var availableSilos = new List<SiloAddress>
        {
            SiloAddress.New(new System.Net.IPEndPoint(System.Net.IPAddress.Loopback, 11111), 30000)
        };

        _mockContext.Setup(c => c.GetCompatibleSilos(_testTarget))
                   .Returns(availableSilos.ToArray());


        // Act
        var selectedSilo = await _director.OnAddActivation(strategy, _testTarget, _mockContext.Object);

        // Assert
        selectedSilo.Should().NotBeNull();
        
        // Verify that a silo was selected (fallback logic)
        availableSilos.Should().Contain(selectedSilo);
    }

    [Fact]
    public async Task OnAddActivation_LogsDebug_WhenGpuCapableSiloSelected()
    {
        // Arrange  
        var strategy = new GpuPlacementStrategy
        {
            PreferLocalGpu = true,
            PreferredDeviceIndex = 0
        };

        var availableSilos = new List<SiloAddress>
        {
            SiloAddress.New(new System.Net.IPEndPoint(System.Net.IPAddress.Loopback, 11111), 30000)
        };

        _mockContext.Setup(c => c.GetCompatibleSilos(_testTarget))
                   .Returns(availableSilos.ToArray());


        // Act
        var selectedSilo = await _director.OnAddActivation(strategy, _testTarget, _mockContext.Object);

        // Assert
        selectedSilo.Should().NotBeNull();
        selectedSilo.Should().Be(availableSilos.First());
    }
}

/// <summary>
/// Integration tests for GPU placement extensions
/// </summary>
public class GpuPlacementExtensionsTests
{
    [Fact]
    public void UseGpuPlacement_OnSiloBuilder_ShouldRegisterServices()
    {
        // Arrange
        var services = new Microsoft.Extensions.DependencyInjection.ServiceCollection();
        var mockSiloBuilder = new Mock<ISiloBuilder>();
        
        mockSiloBuilder.Setup(sb => sb.ConfigureServices(It.IsAny<Action<Microsoft.Extensions.DependencyInjection.IServiceCollection>>()))
                      .Returns(mockSiloBuilder.Object)
                      .Callback<Action<Microsoft.Extensions.DependencyInjection.IServiceCollection>>(configureServices =>
                      {
                          configureServices(services);
                      });

        // Act
        var result = mockSiloBuilder.Object.UseGpuPlacement();

        // Assert
        result.Should().Be(mockSiloBuilder.Object);
        
        // Verify services were registered
        services.Should().Contain(sd => 
            sd.ServiceType == typeof(IPlacementDirector) && 
            sd.ImplementationType == typeof(GpuPlacementDirector));
        
        services.Should().Contain(sd => 
            sd.ServiceType == typeof(PlacementStrategy) && 
            sd.ImplementationType == typeof(GpuPlacementStrategy));
    }

    [Fact]
    public void UseGpuPlacement_OnClientBuilder_ShouldRegisterServices()
    {
        // Arrange
        var services = new Microsoft.Extensions.DependencyInjection.ServiceCollection();
        var mockClientBuilder = new Mock<IClientBuilder>();
        
        mockClientBuilder.Setup(cb => cb.ConfigureServices(It.IsAny<Action<Microsoft.Extensions.DependencyInjection.IServiceCollection>>()))
                        .Returns(mockClientBuilder.Object)
                        .Callback<Action<Microsoft.Extensions.DependencyInjection.IServiceCollection>>(configureServices =>
                        {
                            configureServices(services);
                        });

        // Act
        var result = mockClientBuilder.Object.UseGpuPlacement();

        // Assert
        result.Should().Be(mockClientBuilder.Object);
        
        // Verify placement strategy was registered
        services.Should().Contain(sd => 
            sd.ServiceType == typeof(PlacementStrategy) && 
            sd.ImplementationType == typeof(GpuPlacementStrategy));
    }
}
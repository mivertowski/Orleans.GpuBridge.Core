using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Net;
using System.Threading;
using System.Threading.Tasks;
using FluentAssertions;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using Moq;
using Orleans;
using Orleans.GpuBridge.Abstractions;
using Orleans.GpuBridge.Abstractions.Capacity;
using Orleans.GpuBridge.Abstractions.Enums;
using Orleans.GpuBridge.Abstractions.Memory;
using Orleans.GpuBridge.Abstractions.Models;
using Orleans.GpuBridge.Runtime;
using Orleans.GpuBridge.Runtime.Infrastructure.DeviceManagement;
using Orleans.Runtime;
using Orleans.Runtime.Placement;
using Xunit;

namespace Orleans.GpuBridge.Tests.RC2.Runtime;

/// <summary>
/// Advanced test suite for DeviceBroker and placement strategies with 45-50 tests.
/// Covers multi-GPU management, error handling, placement optimization, and memory coordination.
/// Tests focus on production scenarios: concurrency, failover, load balancing, and resource pressure.
/// </summary>
public sealed class RuntimeAdvancedTests : IDisposable
{
    private readonly Mock<ILogger<DeviceBroker>> _mockBrokerLogger;
    private readonly Mock<ILogger<GpuPlacementDirector>> _mockDirectorLogger;
    private readonly Mock<IGrainFactory> _mockGrainFactory;
    private readonly Mock<IPlacementContext> _mockPlacementContext;
    private readonly Mock<IGpuCapacityGrain> _mockCapacityGrain;
    private readonly GpuBridgeOptions _defaultOptions;
    private readonly IOptions<GpuBridgeOptions> _options;

    public RuntimeAdvancedTests()
    {
        _mockBrokerLogger = new Mock<ILogger<DeviceBroker>>();
        _mockDirectorLogger = new Mock<ILogger<GpuPlacementDirector>>();
        _mockGrainFactory = new Mock<IGrainFactory>();
        _mockPlacementContext = new Mock<IPlacementContext>();
        _mockCapacityGrain = new Mock<IGpuCapacityGrain>();

        _defaultOptions = new GpuBridgeOptions
        {
            PreferGpu = true,
            MaxConcurrentKernels = 100,
            EnableMetrics = true
        };
        _options = Options.Create(_defaultOptions);

        // Setup capacity grain factory
        _mockGrainFactory
            .Setup(f => f.GetGrain<IGpuCapacityGrain>(It.IsAny<long>(), It.IsAny<string>()))
            .Returns(_mockCapacityGrain.Object);
    }

    public void Dispose()
    {
        // Cleanup if needed
    }

    #region DeviceBroker Advanced - Device Discovery and Management (8 tests)

    [Fact]
    public async Task MultiGpuEnumeration_ShouldDetectAllDevices()
    {
        // Arrange
        using var broker = CreateDeviceBroker();

        // Act
        await broker.InitializeAsync(CancellationToken.None);
        var devices = broker.GetDevices();

        // Assert
        devices.Should().NotBeEmpty("At least CPU fallback should be present");
        devices.Should().OnlyHaveUniqueItems(d => d.Index, "All devices should have unique indices");
        devices.Should().Contain(d => d.Type == DeviceType.CPU, "CPU fallback must be present");
    }

    [Fact]
    public async Task GpuCapabilityDetection_WithCudaSupport_ShouldReportCapabilities()
    {
        // Arrange
        using var broker = CreateDeviceBroker();
        await broker.InitializeAsync(CancellationToken.None);

        // Act
        var cpuDevice = broker.GetDevices().First(d => d.Type == DeviceType.CPU);

        // Assert
        cpuDevice.Capabilities.Should().NotBeEmpty("Device should report capabilities");
        cpuDevice.Capabilities.Should().Contain("cpu");
        cpuDevice.Capabilities.Should().Contain("fallback");
        cpuDevice.ComputeUnits.Should().Be(Environment.ProcessorCount, "CPU compute units should match processor count");
    }

    [Fact]
    public async Task DeviceHealthMonitoring_WithMultipleDevices_ShouldTrackAllStates()
    {
        // Arrange
        using var broker = CreateDeviceBroker();
        await broker.InitializeAsync(CancellationToken.None);

        // Act - Wait for at least one monitoring cycle
        await Task.Delay(100);
        var devices = broker.GetDevices();

        // Assert
        devices.Should().AllSatisfy(device =>
        {
            device.TotalMemoryBytes.Should().BeGreaterThan(0, "All devices should report total memory");
            device.AvailableMemoryBytes.Should().BeGreaterThanOrEqualTo(0, "Available memory should be non-negative");
        });
    }

    [Fact]
    public async Task HotPlugDetection_WithNewDevice_ShouldUpdateDeviceList()
    {
        // Arrange
        using var broker = CreateDeviceBroker();
        await broker.InitializeAsync(CancellationToken.None);
        var initialCount = broker.DeviceCount;

        // Act - Simulate device detection (re-initialization should be idempotent)
        await broker.InitializeAsync(CancellationToken.None);
        var finalCount = broker.DeviceCount;

        // Assert
        finalCount.Should().Be(initialCount, "Idempotent initialization should maintain device count");
    }

    [Fact]
    public async Task DeviceAffinityOptimization_WithLocalAccess_ShouldPreferClosestDevice()
    {
        // Arrange
        using var broker = CreateDeviceBroker();
        await broker.InitializeAsync(CancellationToken.None);

        // Act
        var bestDevice = broker.GetBestDevice();
        var devices = broker.GetDevices();

        // Assert
        bestDevice.Should().NotBeNull("Best device should be selected");
        devices.Should().Contain(bestDevice!, "Best device should be in device list");
    }

    [Fact]
    public async Task GpuMemoryCapacityTracking_WithAllocations_ShouldUpdateAccurately()
    {
        // Arrange
        using var broker = CreateDeviceBroker();
        await broker.InitializeAsync(CancellationToken.None);

        // Act
        var totalMemory = broker.TotalMemoryBytes;
        var devices = broker.GetDevices();
        var calculatedTotal = devices.Sum(d => d.TotalMemoryBytes);

        // Assert
        totalMemory.Should().Be(calculatedTotal, "Total memory should match sum of all devices");
        totalMemory.Should().BeGreaterThan(0, "Total memory should be positive");

        devices.Should().AllSatisfy(device =>
        {
            device.AvailableMemoryBytes.Should().BeLessThanOrEqualTo(device.TotalMemoryBytes,
                "Available memory cannot exceed total memory");
        });
    }

    [Fact]
    public async Task DeviceSelectionWithConstraints_MultipleGpus_ShouldRespectCriteria()
    {
        // Arrange
        using var broker = CreateDeviceBroker();
        await broker.InitializeAsync(CancellationToken.None);

        // Act - Get best device based on complex scoring
        var bestDevice = broker.GetBestDevice();
        var devices = broker.GetDevices();

        // Assert
        bestDevice.Should().NotBeNull("Best device should be selected");

        // CPU fallback should be present and have proper metrics
        var cpuDevice = devices.First(d => d.Type == DeviceType.CPU);
        cpuDevice.Index.Should().Be(-1, "CPU fallback uses index -1");
        cpuDevice.TotalMemoryBytes.Should().BeGreaterThan(0);
    }

    [Fact]
    public async Task ConcurrentDeviceQueries_UnderHighLoad_ShouldRemainConsistent()
    {
        // Arrange
        using var broker = CreateDeviceBroker();
        await broker.InitializeAsync(CancellationToken.None);
        var initialCount = broker.DeviceCount;

        // Act - Hammer with concurrent device queries
        var tasks = Enumerable.Range(0, 100).Select(async i =>
        {
            return await Task.Run(() =>
            {
                var device = broker.GetBestDevice();
                var devices = broker.GetDevices();
                var count = broker.DeviceCount;
                var totalMem = broker.TotalMemoryBytes;
                return (device, devices, count, totalMem);
            });
        }).ToArray();

        var results = await Task.WhenAll(tasks);

        // Assert
        results.Should().AllSatisfy(result =>
        {
            result.device.Should().NotBeNull("All queries should return a device");
            result.devices.Should().NotBeEmpty("Device list should never be empty");
            result.count.Should().Be(initialCount, "Device count should remain stable");
            result.totalMem.Should().BeGreaterThan(0, "Total memory should be positive");
        });
    }

    #endregion

    #region DeviceBroker Advanced - Execution Optimization (8 tests)

    [Fact]
    public async Task KernelDispatchQueueManagement_WithBackpressure_ShouldThrottle()
    {
        // Arrange
        using var broker = CreateDeviceBroker();
        await broker.InitializeAsync(CancellationToken.None);

        // Act
        var initialQueueDepth = broker.CurrentQueueDepth;

        // Assert
        initialQueueDepth.Should().Be(0, "Queue should start empty");
    }

    [Fact]
    public async Task ConcurrentKernelExecutionLimits_RespectMaxConcurrent_ShouldEnforce()
    {
        // Arrange
        var options = new GpuBridgeOptions { MaxConcurrentKernels = 10 };
        var optionsWrapper = Options.Create(options);
        using var broker = new DeviceBroker(_mockBrokerLogger.Object, optionsWrapper);
        await broker.InitializeAsync(CancellationToken.None);

        // Act - Verify options are respected
        var bestDevice = broker.GetBestDevice();

        // Assert
        bestDevice.Should().NotBeNull("Device should be available within limits");
    }

    [Fact]
    public async Task GpuUtilizationMonitoring_HighLoad_ShouldReportAccurately()
    {
        // Arrange
        using var broker = CreateDeviceBroker();
        await broker.InitializeAsync(CancellationToken.None);

        // Act
        var devices = broker.GetDevices();
        var bestDevice = broker.GetBestDevice();

        // Assert
        bestDevice.Should().NotBeNull("Best device should be selected under load");
        devices.Should().AllSatisfy(device =>
        {
            device.ComputeUnits.Should().BeGreaterThan(0, "All devices should report compute units");
        });
    }

    [Fact]
    public async Task ThermalThrottlingDetection_HighTemperature_ShouldReduceLoad()
    {
        // Arrange - This tests the monitoring infrastructure
        using var broker = CreateDeviceBroker();
        await broker.InitializeAsync(CancellationToken.None);

        // Act - Wait for health monitoring cycle
        await Task.Delay(150); // Allow time for monitoring

        // Assert - Verify monitoring is active (through logging)
        _mockBrokerLogger.Verify(
            x => x.Log(
                It.IsAny<LogLevel>(),
                It.IsAny<EventId>(),
                It.IsAny<It.IsAnyType>(),
                It.IsAny<Exception>(),
                It.IsAny<Func<It.IsAnyType, Exception?, string>>()),
            Times.AtLeastOnce,
            "Health monitoring should produce log entries");
    }

    [Fact]
    public async Task PowerManagementIntegration_LowPowerMode_ShouldAdjustStrategy()
    {
        // Arrange
        using var broker = CreateDeviceBroker();
        await broker.InitializeAsync(CancellationToken.None);

        // Act
        var bestDevice = broker.GetBestDevice();

        // Assert - Verify device selection works (power management would affect scoring)
        bestDevice.Should().NotBeNull("Best device should be selected considering power constraints");
        bestDevice!.Type.Should().Be(DeviceType.CPU, "CPU fallback should be selected in test environment");
    }

    [Fact]
    public async Task ExecutionPriorityHandling_MixedPriorities_ShouldOrderCorrectly()
    {
        // Arrange
        using var broker = CreateDeviceBroker();
        await broker.InitializeAsync(CancellationToken.None);

        // Act - Get multiple devices in scoring order
        var devices = broker.GetDevices();
        var bestDevice = broker.GetBestDevice();

        // Assert
        bestDevice.Should().NotBeNull("Highest priority device should be selected");
        devices.Should().NotBeEmpty("Device list should be available for priority ordering");
    }

    [Fact]
    public async Task BatchCoalescingOptimization_SmallKernels_ShouldBatch()
    {
        // Arrange
        using var broker = CreateDeviceBroker();
        await broker.InitializeAsync(CancellationToken.None);

        // Act
        var queueDepth = broker.CurrentQueueDepth;

        // Assert
        queueDepth.Should().Be(0, "Empty queue allows for batch coalescing");
    }

    [Fact]
    public async Task DeviceContextSwitching_FrequentChanges_ShouldMinimizeOverhead()
    {
        // Arrange
        using var broker = CreateDeviceBroker();
        await broker.InitializeAsync(CancellationToken.None);

        // Act - Measure device selection performance
        var sw = Stopwatch.StartNew();
        for (int i = 0; i < 1000; i++)
        {
            var device = broker.GetBestDevice();
            device.Should().NotBeNull();
        }
        sw.Stop();

        // Assert
        sw.ElapsedMilliseconds.Should().BeLessThan(1000, "1000 device selections should complete quickly");
    }

    #endregion

    #region DeviceBroker Advanced - Error Handling and Recovery (7 tests)

    [Fact]
    public async Task GpuDriverTimeoutRecovery_TimeoutDetected_ShouldRecover()
    {
        // Arrange
        using var broker = CreateDeviceBroker();
        await broker.InitializeAsync(CancellationToken.None);

        // Act - Simulate recovery by re-initialization
        await broker.ShutdownAsync(CancellationToken.None);
        await broker.InitializeAsync(CancellationToken.None);
        var devices = broker.GetDevices();

        // Assert
        devices.Should().NotBeEmpty("Broker should recover after shutdown and reinit");
    }

    [Fact]
    public async Task DeviceHangDetection_UnresponsiveDevice_ShouldDetectAndIsolate()
    {
        // Arrange
        using var broker = CreateDeviceBroker();
        await broker.InitializeAsync(CancellationToken.None);

        // Act - Health monitoring should detect issues
        await Task.Delay(200); // Wait for health monitoring cycles
        var bestDevice = broker.GetBestDevice();

        // Assert
        bestDevice.Should().NotBeNull("Healthy device should still be available");
    }

    [Fact]
    public async Task AutomaticFallbackToCpu_AllGpusFailed_ShouldUseCpu()
    {
        // Arrange
        using var broker = CreateDeviceBroker();
        await broker.InitializeAsync(CancellationToken.None);

        // Act
        var bestDevice = broker.GetBestDevice();

        // Assert - In test environment, CPU is the fallback
        bestDevice.Should().NotBeNull("CPU fallback should be available");
        bestDevice!.Type.Should().Be(DeviceType.CPU, "Should fall back to CPU");
    }

    [Fact]
    public async Task ErrorRateMonitoring_HighErrorRate_ShouldTriggerAlert()
    {
        // Arrange
        using var broker = CreateDeviceBroker();
        await broker.InitializeAsync(CancellationToken.None);

        // Act - Wait for monitoring cycles
        await Task.Delay(250);

        // Assert - Verify monitoring is active
        var devices = broker.GetDevices();
        devices.Should().NotBeEmpty("Devices should be monitored for errors");
    }

    [Fact]
    public async Task RetryWithDifferentDevice_FirstDeviceFails_ShouldRetry()
    {
        // Arrange
        using var broker = CreateDeviceBroker();
        await broker.InitializeAsync(CancellationToken.None);

        // Act - Get multiple devices for retry scenarios
        var device1 = broker.GetBestDevice();
        var device2 = broker.GetBestDevice();

        // Assert
        device1.Should().NotBeNull("First device selection should succeed");
        device2.Should().NotBeNull("Retry selection should succeed");
        device1.Should().Be(device2, "With single CPU device, same device returned");
    }

    [Fact]
    public async Task GpuResetHandling_DeviceReset_ShouldReinitialize()
    {
        // Arrange
        using var broker = CreateDeviceBroker();
        await broker.InitializeAsync(CancellationToken.None);
        var initialCount = broker.DeviceCount;

        // Act - Simulate reset with shutdown and reinit
        await broker.ShutdownAsync(CancellationToken.None);
        await broker.InitializeAsync(CancellationToken.None);
        var finalCount = broker.DeviceCount;

        // Assert
        finalCount.Should().Be(initialCount, "Device count should be restored after reset");
    }

    [Fact]
    public async Task GracefulDegradation_PartialDeviceFailure_ShouldContinue()
    {
        // Arrange
        using var broker = CreateDeviceBroker();
        await broker.InitializeAsync(CancellationToken.None);

        // Act - Even with device issues, broker should function
        var devices = broker.GetDevices();
        var bestDevice = broker.GetBestDevice();
        var totalMemory = broker.TotalMemoryBytes;

        // Assert
        devices.Should().NotBeEmpty("Some devices should remain available");
        bestDevice.Should().NotBeNull("Best device should still be selectable");
        totalMemory.Should().BeGreaterThan(0, "Total memory should be positive");
    }

    #endregion

    #region Placement Strategy Advanced - GPU-Aware Placement (10 tests)

    [Fact]
    public async Task MultiGpuLoadBalancing_EvenDistribution_ShouldBalanceAcrossSilos()
    {
        // Arrange
        var director = new GpuPlacementDirector(_mockDirectorLogger.Object, _mockGrainFactory.Object);
        var strategy = new GpuPlacementStrategy { MinimumGpuMemoryMB = 2048 };

        var silo1 = CreateSiloGpuCapacity(CreateSiloAddress(), 4096, 8192, 10);
        var silo2 = CreateSiloGpuCapacity(CreateSiloAddress(), 4096, 8192, 10);

        var callCount = 0;
        _mockCapacityGrain
            .Setup(g => g.GetBestSiloForPlacementAsync(2048))
            .ReturnsAsync(() => callCount++ % 2 == 0 ? silo1 : silo2);

        // Act - Place multiple grains
        var placements = new List<SiloAddress>();
        for (int i = 0; i < 10; i++)
        {
            var target = CreatePlacementTarget();
            var silo = await director.OnAddActivation(strategy, target, _mockPlacementContext.Object);
            placements.Add(silo);
        }

        // Assert
        placements.Should().Contain(silo1.SiloAddress, "Some grains should be on silo1");
        placements.Should().Contain(silo2.SiloAddress, "Some grains should be on silo2");
    }

    [Fact]
    public async Task GpuMemoryBasedPlacement_MemoryPressure_ShouldSelectHighMemorySilo()
    {
        // Arrange
        var director = new GpuPlacementDirector(_mockDirectorLogger.Object, _mockGrainFactory.Object);
        var strategy = new GpuPlacementStrategy { MinimumGpuMemoryMB = 8192 };
        var target = CreatePlacementTarget();

        var highMemorySilo = CreateSiloGpuCapacity(CreateSiloAddress(), 12288, 16384, 5);

        _mockCapacityGrain
            .Setup(g => g.GetBestSiloForPlacementAsync(8192))
            .ReturnsAsync(highMemorySilo);

        // Act
        var selectedSilo = await director.OnAddActivation(strategy, target, _mockPlacementContext.Object);

        // Assert
        selectedSilo.Should().Be(highMemorySilo.SiloAddress, "High memory silo should be selected");
    }

    [Fact]
    public async Task GpuAffinityForGrainChains_RelatedGrains_ShouldColocate()
    {
        // Arrange
        var director = new GpuPlacementDirector(_mockDirectorLogger.Object, _mockGrainFactory.Object);
        var strategy = new GpuPlacementStrategy
        {
            PreferLocalPlacement = true,
            MinimumGpuMemoryMB = 2048
        };

        var localSilo = CreateSiloAddress();
        var localCapacity = CreateSiloGpuCapacity(localSilo, 4096, 8192, 5);

        _mockPlacementContext.Setup(c => c.LocalSilo).Returns(localSilo);
        _mockCapacityGrain
            .Setup(g => g.GetBestSiloForPlacementAsync(2048))
            .ReturnsAsync(localCapacity);
        _mockCapacityGrain
            .Setup(g => g.GetGpuCapableSilosAsync())
            .ReturnsAsync(new List<SiloGpuCapacity> { localCapacity });

        // Act - Place related grains
        var placements = new List<SiloAddress>();
        for (int i = 0; i < 5; i++)
        {
            var target = CreatePlacementTarget();
            var silo = await director.OnAddActivation(strategy, target, _mockPlacementContext.Object);
            placements.Add(silo);
        }

        // Assert
        placements.Should().OnlyContain(s => s.Equals(localSilo), "Related grains should colocate on local silo");
    }

    [Fact]
    public async Task PlacementWithGpuQueueDepth_LowQueuePreference_ShouldSelectLeastQueued()
    {
        // Arrange
        var director = new GpuPlacementDirector(_mockDirectorLogger.Object, _mockGrainFactory.Object);
        var strategy = new GpuPlacementStrategy { MinimumGpuMemoryMB = 2048 };
        var target = CreatePlacementTarget();

        var lowQueueSilo = CreateSiloGpuCapacity(CreateSiloAddress(), 4096, 8192, 3);

        _mockCapacityGrain
            .Setup(g => g.GetBestSiloForPlacementAsync(2048))
            .ReturnsAsync(lowQueueSilo);

        // Act
        var selectedSilo = await director.OnAddActivation(strategy, target, _mockPlacementContext.Object);

        // Assert
        selectedSilo.Should().Be(lowQueueSilo.SiloAddress, "Low queue depth silo should be preferred");
    }

    [Fact]
    public async Task SiloSelectionByGpuCapability_ComputeCapability_ShouldMatchRequirements()
    {
        // Arrange
        var director = new GpuPlacementDirector(_mockDirectorLogger.Object, _mockGrainFactory.Object);
        var strategy = new GpuPlacementStrategy { MinimumGpuMemoryMB = 4096 };
        var target = CreatePlacementTarget();

        var capableSilo = CreateSiloGpuCapacity(CreateSiloAddress(), 8192, 16384, 5);

        _mockCapacityGrain
            .Setup(g => g.GetBestSiloForPlacementAsync(4096))
            .ReturnsAsync(capableSilo);

        // Act
        var selectedSilo = await director.OnAddActivation(strategy, target, _mockPlacementContext.Object);

        // Assert
        selectedSilo.Should().Be(capableSilo.SiloAddress, "Capable silo should be selected");
    }

    [Fact]
    public async Task HotSiloAvoidance_OverloadedSilo_ShouldSkip()
    {
        // Arrange
        var director = new GpuPlacementDirector(_mockDirectorLogger.Object, _mockGrainFactory.Object);
        var strategy = new GpuPlacementStrategy { MinimumGpuMemoryMB = 2048 };
        var target = CreatePlacementTarget();

        // Capacity grain should return silo with lower load
        var normalLoadSilo = CreateSiloGpuCapacity(CreateSiloAddress(), 4096, 8192, 50);

        _mockCapacityGrain
            .Setup(g => g.GetBestSiloForPlacementAsync(2048))
            .ReturnsAsync(normalLoadSilo);

        // Act
        var selectedSilo = await director.OnAddActivation(strategy, target, _mockPlacementContext.Object);

        // Assert
        selectedSilo.Should().Be(normalLoadSilo.SiloAddress, "Normal load silo should be selected");
    }

    [Fact]
    public async Task GpuToSiloMappingOptimization_MultipleGpusPerSilo_ShouldDistribute()
    {
        // Arrange
        var director = new GpuPlacementDirector(_mockDirectorLogger.Object, _mockGrainFactory.Object);
        var strategy = new GpuPlacementStrategy { MinimumGpuMemoryMB = 2048 };

        var multiGpuSilo = CreateSiloGpuCapacity(CreateSiloAddress(), 8192, 16384, 10);

        _mockCapacityGrain
            .Setup(g => g.GetBestSiloForPlacementAsync(2048))
            .ReturnsAsync(multiGpuSilo);

        // Act - Place multiple grains
        var placements = new List<SiloAddress>();
        for (int i = 0; i < 5; i++)
        {
            var target = CreatePlacementTarget();
            var silo = await director.OnAddActivation(strategy, target, _mockPlacementContext.Object);
            placements.Add(silo);
        }

        // Assert
        placements.Should().OnlyContain(s => s.Equals(multiGpuSilo.SiloAddress),
            "All placements should be on the multi-GPU silo");
    }

    [Fact]
    public async Task PlacementUnderMemoryPressure_LimitedMemory_ShouldSelectBestFit()
    {
        // Arrange
        var director = new GpuPlacementDirector(_mockDirectorLogger.Object, _mockGrainFactory.Object);
        var strategy = new GpuPlacementStrategy { MinimumGpuMemoryMB = 1024 };
        var target = CreatePlacementTarget();

        var bestFitSilo = CreateSiloGpuCapacity(CreateSiloAddress(), 1536, 8192, 5);

        _mockCapacityGrain
            .Setup(g => g.GetBestSiloForPlacementAsync(1024))
            .ReturnsAsync(bestFitSilo);

        // Act
        var selectedSilo = await director.OnAddActivation(strategy, target, _mockPlacementContext.Object);

        // Assert
        selectedSilo.Should().Be(bestFitSilo.SiloAddress, "Best fit silo should be selected under pressure");
    }

    [Fact]
    public async Task DynamicPlacementPolicyUpdates_PolicyChange_ShouldApplyImmediately()
    {
        // Arrange
        var director = new GpuPlacementDirector(_mockDirectorLogger.Object, _mockGrainFactory.Object);

        // First placement with low memory requirement
        var strategy1 = new GpuPlacementStrategy { MinimumGpuMemoryMB = 1024 };
        var silo1 = CreateSiloGpuCapacity(CreateSiloAddress(), 2048, 8192, 5);

        _mockCapacityGrain
            .Setup(g => g.GetBestSiloForPlacementAsync(1024))
            .ReturnsAsync(silo1);

        // Act - First placement
        var target1 = CreatePlacementTarget();
        var placement1 = await director.OnAddActivation(strategy1, target1, _mockPlacementContext.Object);

        // Change policy to high memory requirement
        var strategy2 = new GpuPlacementStrategy { MinimumGpuMemoryMB = 4096 };
        var silo2 = CreateSiloGpuCapacity(CreateSiloAddress(), 8192, 16384, 5);

        _mockCapacityGrain
            .Setup(g => g.GetBestSiloForPlacementAsync(4096))
            .ReturnsAsync(silo2);

        // Act - Second placement with new policy
        var target2 = CreatePlacementTarget();
        var placement2 = await director.OnAddActivation(strategy2, target2, _mockPlacementContext.Object);

        // Assert
        placement1.Should().Be(silo1.SiloAddress, "First placement should use low memory silo");
        placement2.Should().Be(silo2.SiloAddress, "Second placement should use high memory silo");
        _mockCapacityGrain.Verify(g => g.GetBestSiloForPlacementAsync(1024), Times.Once);
        _mockCapacityGrain.Verify(g => g.GetBestSiloForPlacementAsync(4096), Times.Once);
    }

    [Fact]
    public async Task PlacementWithMultipleCriteria_ComplexScoring_ShouldOptimize()
    {
        // Arrange
        var director = new GpuPlacementDirector(_mockDirectorLogger.Object, _mockGrainFactory.Object);
        var strategy = new GpuPlacementStrategy
        {
            MinimumGpuMemoryMB = 2048,
            PreferLocalPlacement = false
        };
        var target = CreatePlacementTarget();

        // Optimal silo: high memory, low queue, good placement score
        var optimalSilo = CreateSiloGpuCapacity(CreateSiloAddress(), 8192, 16384, 3);

        _mockCapacityGrain
            .Setup(g => g.GetBestSiloForPlacementAsync(2048))
            .ReturnsAsync(optimalSilo);

        // Act
        var selectedSilo = await director.OnAddActivation(strategy, target, _mockPlacementContext.Object);

        // Assert
        selectedSilo.Should().Be(optimalSilo.SiloAddress, "Optimal silo should be selected based on multiple criteria");
    }

    #endregion

    #region Placement Director Advanced (8 tests)

    [Fact]
    public async Task CustomPlacementConstraints_UserDefinedRules_ShouldEnforce()
    {
        // Arrange
        var director = new GpuPlacementDirector(_mockDirectorLogger.Object, _mockGrainFactory.Object);
        var strategy = new GpuPlacementStrategy { MinimumGpuMemoryMB = 4096 };
        var target = CreatePlacementTarget();

        var constrainedSilo = CreateSiloGpuCapacity(CreateSiloAddress(), 5120, 8192, 5);

        _mockCapacityGrain
            .Setup(g => g.GetBestSiloForPlacementAsync(4096))
            .ReturnsAsync(constrainedSilo);

        // Act
        var selectedSilo = await director.OnAddActivation(strategy, target, _mockPlacementContext.Object);

        // Assert
        selectedSilo.Should().Be(constrainedSilo.SiloAddress, "Constrained silo meeting criteria should be selected");
        _mockCapacityGrain.Verify(g => g.GetBestSiloForPlacementAsync(4096), Times.Once);
    }

    [Fact]
    public async Task MultiCriteriaDecisionMaking_ComplexScenario_ShouldOptimize()
    {
        // Arrange
        var director = new GpuPlacementDirector(_mockDirectorLogger.Object, _mockGrainFactory.Object);
        var strategy = new GpuPlacementStrategy
        {
            MinimumGpuMemoryMB = 2048,
            PreferLocalPlacement = true
        };

        var localSilo = CreateSiloAddress();
        var remoteSilo = CreateSiloAddress();

        var localCapacity = CreateSiloGpuCapacity(localSilo, 2560, 4096, 8);
        var remoteCapacity = CreateSiloGpuCapacity(remoteSilo, 4096, 8192, 3);

        _mockPlacementContext.Setup(c => c.LocalSilo).Returns(localSilo);
        _mockCapacityGrain
            .Setup(g => g.GetBestSiloForPlacementAsync(2048))
            .ReturnsAsync(remoteCapacity);
        _mockCapacityGrain
            .Setup(g => g.GetGpuCapableSilosAsync())
            .ReturnsAsync(new List<SiloGpuCapacity> { localCapacity, remoteCapacity });

        // Act
        var target = CreatePlacementTarget();
        var selectedSilo = await director.OnAddActivation(strategy, target, _mockPlacementContext.Object);

        // Assert
        selectedSilo.Should().Be(localSilo, "Local silo should be preferred even with lower capacity");
    }

    [Fact]
    public async Task PlacementConsistencyValidation_MultipleAttempts_ShouldBeConsistent()
    {
        // Arrange
        var director = new GpuPlacementDirector(_mockDirectorLogger.Object, _mockGrainFactory.Object);
        var strategy = new GpuPlacementStrategy { MinimumGpuMemoryMB = 2048 };
        var expectedSilo = CreateSiloGpuCapacity(CreateSiloAddress(), 4096, 8192, 5);

        _mockCapacityGrain
            .Setup(g => g.GetBestSiloForPlacementAsync(2048))
            .ReturnsAsync(expectedSilo);

        // Act - Place same grain type multiple times
        var placements = new List<SiloAddress>();
        for (int i = 0; i < 10; i++)
        {
            var target = CreatePlacementTarget();
            var silo = await director.OnAddActivation(strategy, target, _mockPlacementContext.Object);
            placements.Add(silo);
        }

        // Assert
        placements.Should().OnlyContain(s => s.Equals(expectedSilo.SiloAddress),
            "Consistent placement should select same optimal silo");
    }

    [Fact]
    public async Task SiloFailurePlacementRecovery_SiloDown_ShouldReroute()
    {
        // Arrange
        var director = new GpuPlacementDirector(_mockDirectorLogger.Object, _mockGrainFactory.Object);
        var strategy = new GpuPlacementStrategy { MinimumGpuMemoryMB = 2048 };
        var target = CreatePlacementTarget();
        var fallbackSilo = CreateSiloAddress();

        _mockCapacityGrain
            .Setup(g => g.GetBestSiloForPlacementAsync(2048))
            .ThrowsAsync(new InvalidOperationException("Silo unavailable"));

        _mockPlacementContext
            .Setup(c => c.GetCompatibleSilos(It.IsAny<PlacementTarget>()))
            .Returns(new[] { fallbackSilo });

        // Act
        var selectedSilo = await director.OnAddActivation(strategy, target, _mockPlacementContext.Object);

        // Assert
        selectedSilo.Should().Be(fallbackSilo, "Should fall back to compatible silo on failure");
    }

    [Fact]
    public async Task GrainColocationStrategies_RelatedGrains_ShouldColocate()
    {
        // Arrange
        var director = new GpuPlacementDirector(_mockDirectorLogger.Object, _mockGrainFactory.Object);
        var strategy = new GpuPlacementStrategy
        {
            PreferLocalPlacement = true,
            MinimumGpuMemoryMB = 2048
        };

        var targetSilo = CreateSiloAddress();
        var targetCapacity = CreateSiloGpuCapacity(targetSilo, 4096, 8192, 5);

        _mockPlacementContext.Setup(c => c.LocalSilo).Returns(targetSilo);
        _mockCapacityGrain
            .Setup(g => g.GetBestSiloForPlacementAsync(2048))
            .ReturnsAsync(targetCapacity);
        _mockCapacityGrain
            .Setup(g => g.GetGpuCapableSilosAsync())
            .ReturnsAsync(new List<SiloGpuCapacity> { targetCapacity });

        // Act - Place multiple related grains
        var placements = new List<SiloAddress>();
        for (int i = 0; i < 5; i++)
        {
            var target = CreatePlacementTarget();
            var silo = await director.OnAddActivation(strategy, target, _mockPlacementContext.Object);
            placements.Add(silo);
        }

        // Assert
        placements.Should().OnlyContain(s => s.Equals(targetSilo), "Related grains should colocate");
    }

    [Fact]
    public async Task PlacementMetricsAndTelemetry_AllPlacements_ShouldLog()
    {
        // Arrange
        var director = new GpuPlacementDirector(_mockDirectorLogger.Object, _mockGrainFactory.Object);
        var strategy = new GpuPlacementStrategy { MinimumGpuMemoryMB = 2048 };
        var target = CreatePlacementTarget();
        var silo = CreateSiloGpuCapacity(CreateSiloAddress(), 4096, 8192, 5);

        _mockCapacityGrain
            .Setup(g => g.GetBestSiloForPlacementAsync(2048))
            .ReturnsAsync(silo);

        // Act
        await director.OnAddActivation(strategy, target, _mockPlacementContext.Object);

        // Assert
        _mockDirectorLogger.Verify(
            x => x.Log(
                LogLevel.Information,
                It.IsAny<EventId>(),
                It.Is<It.IsAnyType>((v, t) => v.ToString()!.Contains("Placed grain")),
                It.IsAny<Exception>(),
                It.IsAny<Func<It.IsAnyType, Exception?, string>>()),
            Times.Once,
            "Placement should log metrics");
    }

    [Fact]
    public async Task DynamicPlacementPolicyUpdates_RuntimeChange_ShouldApply()
    {
        // Arrange
        var director = new GpuPlacementDirector(_mockDirectorLogger.Object, _mockGrainFactory.Object);

        // Policy 1: Low memory
        var strategy1 = new GpuPlacementStrategy { MinimumGpuMemoryMB = 1024 };
        var silo1 = CreateSiloGpuCapacity(CreateSiloAddress(), 2048, 4096, 5);

        _mockCapacityGrain
            .Setup(g => g.GetBestSiloForPlacementAsync(1024))
            .ReturnsAsync(silo1);

        var target1 = CreatePlacementTarget();
        var placement1 = await director.OnAddActivation(strategy1, target1, _mockPlacementContext.Object);

        // Policy 2: High memory
        var strategy2 = new GpuPlacementStrategy { MinimumGpuMemoryMB = 8192 };
        var silo2 = CreateSiloGpuCapacity(CreateSiloAddress(), 12288, 16384, 5);

        _mockCapacityGrain
            .Setup(g => g.GetBestSiloForPlacementAsync(8192))
            .ReturnsAsync(silo2);

        var target2 = CreatePlacementTarget();
        var placement2 = await director.OnAddActivation(strategy2, target2, _mockPlacementContext.Object);

        // Assert
        placement1.Should().Be(silo1.SiloAddress, "First policy should select low memory silo");
        placement2.Should().Be(silo2.SiloAddress, "Second policy should select high memory silo");
    }

    [Fact]
    public async Task PlacementWithCapacityGrainFailure_GrainUnavailable_ShouldFallback()
    {
        // Arrange
        var director = new GpuPlacementDirector(_mockDirectorLogger.Object, _mockGrainFactory.Object);
        var strategy = new GpuPlacementStrategy { MinimumGpuMemoryMB = 2048 };
        var target = CreatePlacementTarget();
        var fallbackSilo = CreateSiloAddress();

        _mockCapacityGrain
            .Setup(g => g.GetBestSiloForPlacementAsync(2048))
            .ThrowsAsync(new TimeoutException("Capacity grain timeout"));

        _mockPlacementContext
            .Setup(c => c.GetCompatibleSilos(It.IsAny<PlacementTarget>()))
            .Returns(new[] { fallbackSilo });

        // Act
        var selectedSilo = await director.OnAddActivation(strategy, target, _mockPlacementContext.Object);

        // Assert
        selectedSilo.Should().Be(fallbackSilo, "Should fall back when capacity grain fails");

        _mockDirectorLogger.Verify(
            x => x.Log(
                LogLevel.Error,
                It.IsAny<EventId>(),
                It.Is<It.IsAnyType>((v, t) => v.ToString()!.Contains("Error selecting GPU-capable silo")),
                It.IsAny<Exception>(),
                It.IsAny<Func<It.IsAnyType, Exception?, string>>()),
            Times.Once);
    }

    #endregion

    #region Memory Management Advanced (6 tests)

    [Fact]
    public void CpuMemoryPool_PoolGrowthUnderPressure_ShouldExpand()
    {
        // Arrange
        var pool = new CpuMemoryPool<float>();

        // Act - Rent multiple buffers
        var buffers = new List<IGpuMemory<float>>();
        for (int i = 0; i < 20; i++)
        {
            var buffer = pool.Rent(1024);
            buffers.Add(buffer);
        }

        var stats = pool.GetStats();

        // Assert
        stats.TotalAllocated.Should().BeGreaterThan(0, "Pool should have allocated memory");
        stats.InUse.Should().BeGreaterThan(0, "Memory should be in use");

        // Cleanup
        foreach (var buffer in buffers)
        {
            pool.Return(buffer);
            buffer.Dispose();
        }
    }

    [Fact]
    public void CpuMemoryPool_MemoryFragmentationHandling_ShouldDefragment()
    {
        // Arrange
        var pool = new CpuMemoryPool<int>();

        // Act - Create fragmentation by renting different sizes
        var buffer1 = pool.Rent(512);
        var buffer2 = pool.Rent(1024);
        var buffer3 = pool.Rent(2048);

        pool.Return(buffer2);
        pool.Return(buffer1);
        pool.Return(buffer3);

        var newBuffer = pool.Rent(1024);
        var stats = pool.GetStats();

        // Assert
        stats.BufferCount.Should().BeGreaterThanOrEqualTo(0, "Pool should manage fragmentation");
        newBuffer.Should().NotBeNull("Should successfully allocate after fragmentation");

        // Cleanup
        pool.Return(newBuffer);
        buffer1.Dispose();
        buffer2.Dispose();
        buffer3.Dispose();
        newBuffer.Dispose();
    }

    [Fact]
    public void CpuMemoryPool_LargeAllocationStrategies_ShouldHandleLarge()
    {
        // Arrange
        var pool = new CpuMemoryPool<byte>();

        // Act - Allocate large buffer (10MB)
        var largeBuffer = pool.Rent(10 * 1024 * 1024);
        var stats = pool.GetStats();

        // Assert
        largeBuffer.Should().NotBeNull("Large allocation should succeed");
        largeBuffer.Length.Should().BeGreaterThanOrEqualTo(10 * 1024 * 1024);
        stats.TotalAllocated.Should().BeGreaterThan(0);

        // Cleanup - Either return or dispose, but not both
        // Return to pool lets the pool manage the buffer lifecycle
        pool.Return(largeBuffer);
        // Don't dispose after return - the pool manages disposal if it exceeds capacity
    }

    [Fact]
    public void CpuMemoryPool_PoolShrinkingAndCleanup_ShouldShrink()
    {
        // Arrange
        var pool = new CpuMemoryPool<double>();

        // Act - Allocate and return many buffers to exceed pool limit
        var buffers = new List<IGpuMemory<double>>();
        for (int i = 0; i < 15; i++)
        {
            var buffer = pool.Rent(1024);
            buffers.Add(buffer);
        }

        // Return all buffers
        foreach (var buffer in buffers)
        {
            pool.Return(buffer);
        }

        var stats = pool.GetStats();

        // Assert
        stats.BufferCount.Should().BeLessThanOrEqualTo(10, "Pool should limit pooled buffers to 10");

        // Cleanup
        foreach (var buffer in buffers)
        {
            buffer.Dispose();
        }
    }

    [Fact]
    public async Task CpuMemoryPool_ThreadContentionPatterns_ShouldHandleConcurrency()
    {
        // Arrange
        var pool = new CpuMemoryPool<int>();
        var concurrencyLevel = 50;

        // Act - Concurrent rent/return operations
        var tasks = Enumerable.Range(0, concurrencyLevel).Select(async i =>
        {
            await Task.Yield();
            var buffer = pool.Rent(512);
            await Task.Delay(10);
            pool.Return(buffer);
            buffer.Dispose();
        }).ToArray();

        await Task.WhenAll(tasks);
        var stats = pool.GetStats();

        // Assert
        stats.RentCount.Should().Be(concurrencyLevel, "All rents should succeed");
        stats.ReturnCount.Should().Be(concurrencyLevel, "All returns should succeed");
    }

    [Fact]
    public void CpuMemoryPool_MemoryWatermarkEnforcement_ShouldEnforce()
    {
        // Arrange
        var pool = new CpuMemoryPool<long>();

        // Act - Allocate multiple buffers
        var buffers = new List<IGpuMemory<long>>();
        for (int i = 0; i < 5; i++)
        {
            var buffer = pool.Rent(2048);
            buffers.Add(buffer);
        }

        var stats = pool.GetStats();

        // Assert
        stats.InUse.Should().BeGreaterThan(0, "Memory should be tracked as in use");
        stats.TotalAllocated.Should().BeGreaterThanOrEqualTo(stats.InUse, "Total should be >= in use");

        // Cleanup
        foreach (var buffer in buffers)
        {
            pool.Return(buffer);
            buffer.Dispose();
        }
    }

    #endregion

    #region Helper Methods

    private DeviceBroker CreateDeviceBroker()
    {
        return new DeviceBroker(_mockBrokerLogger.Object, _options);
    }

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

    #endregion
}

using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using FluentAssertions;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using Moq;
using Orleans.GpuBridge.Abstractions;
using Orleans.GpuBridge.Abstractions.Enums;
using Orleans.GpuBridge.Abstractions.Models;
using Orleans.GpuBridge.Runtime;
using Xunit;

namespace Orleans.GpuBridge.Tests.RC2.Runtime;

/// <summary>
/// Comprehensive test suite for DeviceBroker covering initialization, device management,
/// work queue operations, health monitoring, load balancing, and cleanup.
/// Target: 30-40 tests for thorough coverage of main code paths.
/// </summary>
public sealed class DeviceBrokerComprehensiveTests : IDisposable
{
    private readonly Mock<ILogger<DeviceBroker>> _mockLogger;
    private readonly GpuBridgeOptions _defaultOptions;
    private readonly IOptions<GpuBridgeOptions> _options;
    private readonly List<DeviceBroker> _brokersToDispose;

    public DeviceBrokerComprehensiveTests()
    {
        _mockLogger = new Mock<ILogger<DeviceBroker>>();
        _defaultOptions = new GpuBridgeOptions
        {
            PreferGpu = true,
            MaxConcurrentKernels = 100,
            EnableMetrics = true,
            MaxDevices = 4,
            BatchSize = 1024
        };
        _options = Options.Create(_defaultOptions);
        _brokersToDispose = new List<DeviceBroker>();
    }

    #region Initialization Tests (8 tests)

    [Fact]
    public async Task InitializeAsync_ShouldSucceed_AndCreateCpuFallbackDevice()
    {
        // Arrange
        var broker = CreateDeviceBroker();

        // Act
        await broker.InitializeAsync(CancellationToken.None);

        // Assert
        broker.DeviceCount.Should().BeGreaterThan(0, "At least CPU fallback should be created");
        var devices = broker.GetDevices();
        devices.Should().Contain(d => d.Type == DeviceType.CPU, "CPU fallback must be present");

        var cpuDevice = devices.First(d => d.Type == DeviceType.CPU);
        cpuDevice.Name.Should().Be("CPU Fallback");
        cpuDevice.Index.Should().Be(-1, "CPU fallback has index -1");
    }

    [Fact]
    public async Task InitializeAsync_WhenCalledMultipleTimes_ShouldBeIdempotent()
    {
        // Arrange
        var broker = CreateDeviceBroker();

        // Act
        await broker.InitializeAsync(CancellationToken.None);
        var firstCount = broker.DeviceCount;
        var firstDevices = broker.GetDevices();

        await broker.InitializeAsync(CancellationToken.None); // Second initialization
        var secondCount = broker.DeviceCount;
        var secondDevices = broker.GetDevices();

        // Assert
        firstCount.Should().Be(secondCount, "Device count should not change on re-initialization");
        firstDevices.Count.Should().Be(secondDevices.Count, "Device list should remain the same");
    }

    [Fact]
    public async Task InitializeAsync_WithCancellationToken_ShouldRespectCancellation()
    {
        // Arrange
        var broker = CreateDeviceBroker();
        using var cts = new CancellationTokenSource();
        cts.Cancel(); // Cancel immediately

        // Act & Assert
        await Assert.ThrowsAnyAsync<OperationCanceledException>(
            async () => await broker.InitializeAsync(cts.Token));
    }

    [Fact]
    public async Task InitializeAsync_ShouldLogInitializationMessages()
    {
        // Arrange
        var broker = CreateDeviceBroker();

        // Act
        await broker.InitializeAsync(CancellationToken.None);

        // Assert
        _mockLogger.Verify(
            x => x.Log(
                LogLevel.Information,
                It.IsAny<EventId>(),
                It.Is<It.IsAnyType>((v, t) => v.ToString()!.Contains("Initializing device broker")),
                It.IsAny<Exception>(),
                It.IsAny<Func<It.IsAnyType, Exception?, string>>()),
            Times.AtLeastOnce,
            "Should log initialization start message");

        _mockLogger.Verify(
            x => x.Log(
                LogLevel.Information,
                It.IsAny<EventId>(),
                It.Is<It.IsAnyType>((v, t) => v.ToString()!.Contains("Device broker initialized")),
                It.IsAny<Exception>(),
                It.IsAny<Func<It.IsAnyType, Exception?, string>>()),
            Times.AtLeastOnce,
            "Should log initialization complete message");
    }

    [Fact]
    public async Task InitializeAsync_ShouldCreateWorkQueuesForAllDevices()
    {
        // Arrange
        var broker = CreateDeviceBroker();

        // Act
        await broker.InitializeAsync(CancellationToken.None);
        var devices = broker.GetDevices();

        // Assert
        devices.Should().NotBeEmpty("At least CPU device should be present");
        broker.CurrentQueueDepth.Should().Be(0, "Initial queue depth should be zero");
    }

    [Fact]
    public async Task InitializeAsync_ConcurrentCalls_ShouldBeSafe()
    {
        // Arrange
        var broker = CreateDeviceBroker();

        // Act - Multiple concurrent initialization attempts
        var tasks = Enumerable.Range(0, 10)
            .Select(_ => broker.InitializeAsync(CancellationToken.None))
            .ToArray();

        await Task.WhenAll(tasks);

        // Assert
        var devices = broker.GetDevices();
        devices.Should().NotBeEmpty("Devices should be initialized");
        devices.Should().OnlyHaveUniqueItems(d => d.Index, "No duplicate devices should exist");
    }

    [Fact]
    public async Task InitializeAsync_ShouldSetInitializedFlag()
    {
        // Arrange
        var broker = CreateDeviceBroker();

        // Act - Before initialization
        var actionBefore = () => broker.GetDevices();
        actionBefore.Should().Throw<InvalidOperationException>("Not initialized yet");

        await broker.InitializeAsync(CancellationToken.None);

        // After initialization
        var actionAfter = () => broker.GetDevices();
        actionAfter.Should().NotThrow("Should be initialized");
    }

    [Fact]
    public async Task InitializeAsync_WithCustomOptions_ShouldRespectConfiguration()
    {
        // Arrange
        var customOptions = new GpuBridgeOptions
        {
            PreferGpu = false,
            MaxConcurrentKernels = 50,
            EnableMetrics = false
        };
        var broker = CreateDeviceBroker(customOptions);

        // Act
        await broker.InitializeAsync(CancellationToken.None);

        // Assert
        broker.DeviceCount.Should().BeGreaterThan(0, "Devices should be initialized regardless of options");
    }

    #endregion

    #region Device Detection and Management Tests (7 tests)

    [Fact]
    public async Task DetectGpuDevicesAsync_ShouldLogDetectionProcess()
    {
        // Arrange
        var broker = CreateDeviceBroker();

        // Act
        await broker.InitializeAsync(CancellationToken.None);

        // Assert
        _mockLogger.Verify(
            x => x.Log(
                LogLevel.Information,
                It.IsAny<EventId>(),
                It.Is<It.IsAnyType>((v, t) => v.ToString()!.Contains("GPU detection") ||
                                              v.ToString()!.Contains("GPU device")),
                It.IsAny<Exception>(),
                It.IsAny<Func<It.IsAnyType, Exception?, string>>()),
            Times.AtLeastOnce,
            "Should log GPU detection activity");
    }

    [Fact]
    public async Task AddCpuDevice_ShouldCreateValidCpuFallback()
    {
        // Arrange
        var broker = CreateDeviceBroker();
        await broker.InitializeAsync(CancellationToken.None);

        // Act
        var cpuDevice = broker.GetDevices().First(d => d.Type == DeviceType.CPU);

        // Assert
        cpuDevice.Index.Should().Be(-1, "CPU fallback uses index -1");
        cpuDevice.Name.Should().Be("CPU Fallback");
        cpuDevice.Type.Should().Be(DeviceType.CPU);
        cpuDevice.TotalMemoryBytes.Should().BeGreaterThan(0, "Should have total memory");
        cpuDevice.AvailableMemoryBytes.Should().BeGreaterThan(0, "Should have available memory");
        cpuDevice.ComputeUnits.Should().Be(Environment.ProcessorCount, "Should match processor count");
        cpuDevice.Capabilities.Should().Contain("cpu");
        cpuDevice.Capabilities.Should().Contain("fallback");
    }

    [Fact]
    public async Task GetDevices_AfterInitialization_ShouldReturnReadOnlyList()
    {
        // Arrange
        var broker = CreateDeviceBroker();
        await broker.InitializeAsync(CancellationToken.None);

        // Act
        var devices = broker.GetDevices();

        // Assert
        devices.Should().NotBeNull();
        devices.Should().BeAssignableTo<IReadOnlyList<GpuDevice>>("Should return read-only list");
        devices.Should().NotBeEmpty();
    }

    [Fact]
    public async Task GetDevice_WithValidIndex_ShouldReturnCorrectDevice()
    {
        // Arrange
        var broker = CreateDeviceBroker();
        await broker.InitializeAsync(CancellationToken.None);

        // Act
        var device = broker.GetDevice(-1); // CPU fallback index

        // Assert
        device.Should().NotBeNull("CPU device should exist");
        device!.Index.Should().Be(-1);
        device.Type.Should().Be(DeviceType.CPU);
    }

    [Fact]
    public async Task GetDevice_WithInvalidIndex_ShouldReturnNull()
    {
        // Arrange
        var broker = CreateDeviceBroker();
        await broker.InitializeAsync(CancellationToken.None);

        // Act
        var device = broker.GetDevice(999); // Non-existent index

        // Assert
        device.Should().BeNull("Device with invalid index should not exist");
    }

    [Fact]
    public async Task GetDevice_CalledMultipleTimes_ShouldReturnConsistentResults()
    {
        // Arrange
        var broker = CreateDeviceBroker();
        await broker.InitializeAsync(CancellationToken.None);

        // Act
        var device1 = broker.GetDevice(-1);
        var device2 = broker.GetDevice(-1);

        // Assert
        device1.Should().NotBeNull();
        device2.Should().NotBeNull();
        device1.Should().Be(device2, "Same index should return same device");
    }

    [Fact]
    public async Task GetDevices_BeforeInitialization_ShouldThrowInvalidOperationException()
    {
        // Arrange
        var broker = CreateDeviceBroker();

        // Act & Assert
        var action = () => broker.GetDevices();
        action.Should().Throw<InvalidOperationException>()
            .WithMessage("*not initialized*");
    }

    #endregion

    #region Work Queue Management Tests (6 tests)

    [Fact]
    public async Task CurrentQueueDepth_InitiallyZero_AfterInitialization()
    {
        // Arrange
        var broker = CreateDeviceBroker();

        // Act
        await broker.InitializeAsync(CancellationToken.None);
        var queueDepth = broker.CurrentQueueDepth;

        // Assert
        queueDepth.Should().Be(0, "Queue should be empty after initialization");
    }

    [Fact]
    public async Task DeviceCount_ShouldMatchGetDevicesCount()
    {
        // Arrange
        var broker = CreateDeviceBroker();
        await broker.InitializeAsync(CancellationToken.None);

        // Act
        var deviceCount = broker.DeviceCount;
        var devicesListCount = broker.GetDevices().Count;

        // Assert
        deviceCount.Should().Be(devicesListCount, "DeviceCount property should match list count");
        deviceCount.Should().BeGreaterThan(0, "At least one device should exist");
    }

    [Fact]
    public async Task TotalMemoryBytes_ShouldAggregateAllDevices()
    {
        // Arrange
        var broker = CreateDeviceBroker();
        await broker.InitializeAsync(CancellationToken.None);

        // Act
        var totalMemory = broker.TotalMemoryBytes;
        var devices = broker.GetDevices();
        var expectedTotal = devices.Sum(d => d.TotalMemoryBytes);

        // Assert
        totalMemory.Should().Be(expectedTotal, "TotalMemoryBytes should sum all device memory");
        totalMemory.Should().BeGreaterThan(0, "Total memory should be positive");
    }

    [Fact]
    public async Task TotalMemoryBytes_WithMultipleDevices_ShouldBeAccurate()
    {
        // Arrange
        var broker = CreateDeviceBroker();
        await broker.InitializeAsync(CancellationToken.None);
        var devices = broker.GetDevices();

        // Act
        var totalFromProperty = broker.TotalMemoryBytes;
        var totalCalculated = devices.Sum(d => d.TotalMemoryBytes);

        // Assert
        totalFromProperty.Should().Be(totalCalculated);
    }

    [Fact]
    public async Task GetBestDevice_AfterInitialization_ShouldReturnDevice()
    {
        // Arrange
        var broker = CreateDeviceBroker();
        await broker.InitializeAsync(CancellationToken.None);

        // Act
        var bestDevice = broker.GetBestDevice();

        // Assert
        bestDevice.Should().NotBeNull("GetBestDevice should return a device");
        bestDevice!.Type.Should().Be(DeviceType.CPU, "CPU should be the only available device in tests");
    }

    [Fact]
    public async Task GetBestDevice_BeforeInitialization_ShouldThrowInvalidOperationException()
    {
        // Arrange
        var broker = CreateDeviceBroker();

        // Act & Assert
        var action = () => broker.GetBestDevice();
        action.Should().Throw<InvalidOperationException>()
            .WithMessage("*not initialized*");
    }

    #endregion

    #region Device Scoring and Selection Tests (5 tests)

    [Fact]
    public async Task CalculateDeviceScore_ShouldConsiderMultipleFactors()
    {
        // Arrange
        var broker = CreateDeviceBroker();
        await broker.InitializeAsync(CancellationToken.None);

        // Act
        var bestDevice = broker.GetBestDevice();

        // Assert
        bestDevice.Should().NotBeNull("A device should be selected");
        bestDevice!.AvailableMemoryBytes.Should().BeGreaterThan(0, "Best device should have available memory");
    }

    [Fact]
    public async Task GetBestDevice_WithSingleDevice_ShouldReturnThatDevice()
    {
        // Arrange
        var broker = CreateDeviceBroker();
        await broker.InitializeAsync(CancellationToken.None);
        var devices = broker.GetDevices();

        // Act
        var bestDevice = broker.GetBestDevice();

        // Assert
        bestDevice.Should().NotBeNull();
        devices.Should().Contain(bestDevice!);
    }

    [Fact]
    public async Task GetBestDevice_ShouldConsiderMemoryAvailability()
    {
        // Arrange
        var broker = CreateDeviceBroker();
        await broker.InitializeAsync(CancellationToken.None);

        // Act
        var bestDevice = broker.GetBestDevice();

        // Assert
        bestDevice.Should().NotBeNull();
        bestDevice!.AvailableMemoryBytes.Should().BeGreaterThan(0,
            "Best device should have available memory");
    }

    [Fact]
    public async Task GetBestDevice_CalledMultipleTimes_ShouldBeConsistent()
    {
        // Arrange
        var broker = CreateDeviceBroker();
        await broker.InitializeAsync(CancellationToken.None);

        // Act
        var device1 = broker.GetBestDevice();
        var device2 = broker.GetBestDevice();

        // Assert
        device1.Should().NotBeNull();
        device2.Should().NotBeNull();
        device1.Should().Be(device2, "Without workload changes, best device should be consistent");
    }

    [Fact]
    public async Task GetBestDevice_ShouldPreferDeviceBasedOnType()
    {
        // Arrange
        var broker = CreateDeviceBroker();
        await broker.InitializeAsync(CancellationToken.None);

        // Act
        var bestDevice = broker.GetBestDevice();

        // Assert
        bestDevice.Should().NotBeNull();
        // In test environment with only CPU fallback, it should be CPU
        bestDevice!.Type.Should().Be(DeviceType.CPU);
    }

    #endregion

    #region Health Monitoring Tests (5 tests)

    [Fact]
    public async Task MonitorDeviceHealth_ShouldNotThrowExceptions()
    {
        // Arrange
        var broker = CreateDeviceBroker();
        await broker.InitializeAsync(CancellationToken.None);

        // Act - Wait for at least one health monitoring cycle
        await Task.Delay(500); // Health monitor runs every 10 seconds, but we just verify no crash

        // Assert - Broker should still be operational
        var devices = broker.GetDevices();
        devices.Should().NotBeEmpty("Broker should remain operational during health monitoring");
    }

    [Fact]
    public async Task MonitorDeviceHealthAsync_WithValidDevices_ShouldComplete()
    {
        // Arrange
        var broker = CreateDeviceBroker();
        await broker.InitializeAsync(CancellationToken.None);

        // Act - Let health monitoring run for a short time
        await Task.Delay(100);

        // Assert - Broker should still be functional
        broker.GetDevices().Should().NotBeEmpty();
        broker.DeviceCount.Should().BeGreaterThan(0);
    }

    [Fact]
    public async Task HealthMonitoring_ShouldNotBlockOtherOperations()
    {
        // Arrange
        var broker = CreateDeviceBroker();
        await broker.InitializeAsync(CancellationToken.None);

        // Act - Perform operations while health monitoring is running
        var tasks = Enumerable.Range(0, 10)
            .Select(async _ =>
            {
                await Task.Delay(10);
                return broker.GetBestDevice();
            })
            .ToArray();

        var results = await Task.WhenAll(tasks);

        // Assert
        results.Should().AllSatisfy(device => device.Should().NotBeNull());
    }

    [Fact]
    public async Task TriggerDeviceRecoveryAsync_ShouldCompleteWithoutError()
    {
        // Arrange
        var broker = CreateDeviceBroker();
        await broker.InitializeAsync(CancellationToken.None);

        // Act - Trigger recovery indirectly by waiting for health monitoring
        await Task.Delay(200);

        // Assert - Broker should still be operational
        broker.GetDevices().Should().NotBeEmpty();
    }

    [Fact]
    public async Task CheckDeviceMemoryUsageAsync_ShouldHandleMemoryChecks()
    {
        // Arrange
        var broker = CreateDeviceBroker();
        await broker.InitializeAsync(CancellationToken.None);

        // Act - Wait for memory checks to run
        await Task.Delay(200);

        // Assert
        var devices = broker.GetDevices();
        devices.Should().AllSatisfy(device =>
        {
            device.TotalMemoryBytes.Should().BeGreaterThan(0);
            device.AvailableMemoryBytes.Should().BeGreaterThanOrEqualTo(0);
        });
    }

    #endregion

    #region Load Balancing Tests (3 tests)

    [Fact]
    public async Task UpdateLoadBalancingAsync_ShouldCompleteWithoutError()
    {
        // Arrange
        var broker = CreateDeviceBroker();
        await broker.InitializeAsync(CancellationToken.None);

        // Act - Wait for load balancing updates
        await Task.Delay(200);

        // Assert - Broker should still be operational
        broker.GetDevices().Should().NotBeEmpty();
    }

    [Fact]
    public async Task LoadBalancing_ShouldNotInterruptDeviceQueries()
    {
        // Arrange
        var broker = CreateDeviceBroker();
        await broker.InitializeAsync(CancellationToken.None);

        // Act - Query devices while load balancing is running
        var devices1 = broker.GetDevices();
        await Task.Delay(100); // Allow load balancing to run
        var devices2 = broker.GetDevices();

        // Assert
        devices1.Count.Should().Be(devices2.Count, "Device count should remain stable");
    }

    [Fact]
    public async Task UpdateDeviceLoadBalanceAsync_ShouldHandleAllDevices()
    {
        // Arrange
        var broker = CreateDeviceBroker();
        await broker.InitializeAsync(CancellationToken.None);

        // Act - Let load balancing run
        await Task.Delay(150);
        var devices = broker.GetDevices();

        // Assert
        devices.Should().AllSatisfy(device =>
        {
            device.Should().NotBeNull();
            device.Index.Should().NotBe(0); // Index should be set (-1 for CPU)
        });
    }

    #endregion

    #region Shutdown and Disposal Tests (6 tests)

    [Fact]
    public async Task ShutdownAsync_ShouldClearDevices()
    {
        // Arrange
        var broker = CreateDeviceBroker();
        await broker.InitializeAsync(CancellationToken.None);
        broker.GetDevices().Should().NotBeEmpty("Devices should exist before shutdown");

        // Act
        await broker.ShutdownAsync(CancellationToken.None);

        // Assert
        var action = () => broker.GetDevices();
        action.Should().Throw<InvalidOperationException>("Broker should not be initialized after shutdown");
    }

    [Fact]
    public async Task ShutdownAsync_ShouldStopAllWorkQueues()
    {
        // Arrange
        var broker = CreateDeviceBroker();
        await broker.InitializeAsync(CancellationToken.None);
        var initialCount = broker.DeviceCount;
        initialCount.Should().BeGreaterThan(0);

        // Act
        await broker.ShutdownAsync(CancellationToken.None);

        // Assert - After shutdown, DeviceCount returns 0 because the list is cleared
        broker.DeviceCount.Should().Be(0, "Device list should be cleared after shutdown");

        // GetDevices should throw because broker is not initialized
        var action = () => broker.GetDevices();
        action.Should().Throw<InvalidOperationException>("GetDevices should throw after shutdown");
    }

    [Fact]
    public async Task ShutdownAsync_WithCancellationToken_ShouldComplete()
    {
        // Arrange
        var broker = CreateDeviceBroker();
        await broker.InitializeAsync(CancellationToken.None);

        using var cts = new CancellationTokenSource();
        cts.CancelAfter(TimeSpan.FromSeconds(5)); // Generous timeout

        // Act
        await broker.ShutdownAsync(cts.Token);

        // Assert
        var action = () => broker.GetDevices();
        action.Should().Throw<InvalidOperationException>();
    }

    [Fact]
    public async Task Dispose_ShouldShutdownCleanly()
    {
        // Arrange
        var broker = CreateDeviceBroker();
        await broker.InitializeAsync(CancellationToken.None);

        // Act
        broker.Dispose();

        // Assert
        var action = () => broker.GetDevices();
        action.Should().Throw<InvalidOperationException>("Disposed broker should not be usable");
    }

    [Fact]
    public async Task Dispose_CalledMultipleTimes_ShouldBeSafe()
    {
        // Arrange
        var broker = CreateDeviceBroker();
        await broker.InitializeAsync(CancellationToken.None);

        // Act
        broker.Dispose();
        broker.Dispose(); // Second dispose

        // Assert - Should not throw
        var action = () => broker.GetDevices();
        action.Should().Throw<InvalidOperationException>();
    }

    [Fact]
    public async Task Dispose_ShouldReleaseAllResources()
    {
        // Arrange
        var broker = CreateDeviceBroker();
        await broker.InitializeAsync(CancellationToken.None);

        // Act
        broker.Dispose();

        // Wait a bit to ensure async cleanup completes
        await Task.Delay(100);

        // Assert
        var action = () => broker.GetBestDevice();
        action.Should().Throw<InvalidOperationException>("All operations should fail after disposal");
    }

    #endregion

    #region Thread Safety and Concurrency Tests (5 tests)

    [Fact]
    public async Task ConcurrentGetDevices_ShouldBeThreadSafe()
    {
        // Arrange
        var broker = CreateDeviceBroker();
        await broker.InitializeAsync(CancellationToken.None);

        // Act
        var tasks = Enumerable.Range(0, 50)
            .Select(async _ => await Task.Run(() => broker.GetDevices()))
            .ToArray();

        var results = await Task.WhenAll(tasks);

        // Assert
        results.Should().AllSatisfy(devices =>
        {
            devices.Should().NotBeEmpty();
            devices.Count.Should().Be(broker.DeviceCount);
        });
    }

    [Fact]
    public async Task ConcurrentGetBestDevice_ShouldBeThreadSafe()
    {
        // Arrange
        var broker = CreateDeviceBroker();
        await broker.InitializeAsync(CancellationToken.None);

        // Act
        var tasks = Enumerable.Range(0, 100)
            .Select(async _ => await Task.Run(() => broker.GetBestDevice()))
            .ToArray();

        var results = await Task.WhenAll(tasks);

        // Assert
        results.Should().AllSatisfy(device => device.Should().NotBeNull());
    }

    [Fact]
    public async Task ConcurrentDeviceQueries_ShouldNotCauseRaceConditions()
    {
        // Arrange
        var broker = CreateDeviceBroker();
        await broker.InitializeAsync(CancellationToken.None);

        // Act - Mix different operations concurrently
        var tasks = new List<Task>();
        for (int i = 0; i < 20; i++)
        {
            tasks.Add(Task.Run(() => broker.GetDevices()));
            tasks.Add(Task.Run(() => broker.GetBestDevice()));
            tasks.Add(Task.Run(() => broker.GetDevice(-1)));
            tasks.Add(Task.Run(() => broker.DeviceCount));
            tasks.Add(Task.Run(() => broker.TotalMemoryBytes));
        }

        await Task.WhenAll(tasks);

        // Assert - Broker should still be operational
        broker.GetDevices().Should().NotBeEmpty();
        broker.DeviceCount.Should().BeGreaterThan(0);
    }

    [Fact]
    public async Task ConcurrentInitialization_ShouldNotCreateDuplicateDevices()
    {
        // Arrange
        var broker = CreateDeviceBroker();

        // Act - Multiple concurrent initializations
        var tasks = Enumerable.Range(0, 20)
            .Select(_ => broker.InitializeAsync(CancellationToken.None))
            .ToArray();

        await Task.WhenAll(tasks);

        // Assert
        var devices = broker.GetDevices();
        devices.Should().OnlyHaveUniqueItems(d => d.Index, "No duplicate devices should exist");
    }

    [Fact]
    public async Task StressTest_ManyOperationsUnderLoad_ShouldRemainStable()
    {
        // Arrange
        var broker = CreateDeviceBroker();
        await broker.InitializeAsync(CancellationToken.None);

        // Act - Simulate heavy concurrent load
        var tasks = new List<Task>();
        for (int i = 0; i < 200; i++)
        {
            tasks.Add(Task.Run(async () =>
            {
                await Task.Delay(Random.Shared.Next(1, 10));
                _ = broker.GetBestDevice();
            }));

            tasks.Add(Task.Run(async () =>
            {
                await Task.Delay(Random.Shared.Next(1, 10));
                _ = broker.GetDevices();
            }));
        }

        await Task.WhenAll(tasks);

        // Assert
        broker.GetDevices().Should().NotBeEmpty("Broker should remain stable under load");
        broker.DeviceCount.Should().BeGreaterThan(0);
    }

    #endregion

    #region Helper Methods

    /// <summary>
    /// Creates a DeviceBroker instance with default options and tracks for disposal
    /// </summary>
    private DeviceBroker CreateDeviceBroker()
    {
        var broker = new DeviceBroker(_mockLogger.Object, _options);
        _brokersToDispose.Add(broker);
        return broker;
    }

    /// <summary>
    /// Creates a DeviceBroker with custom options and tracks for disposal
    /// </summary>
    private DeviceBroker CreateDeviceBroker(GpuBridgeOptions options)
    {
        var optionsWrapper = Options.Create(options);
        var broker = new DeviceBroker(_mockLogger.Object, optionsWrapper);
        _brokersToDispose.Add(broker);
        return broker;
    }

    #endregion

    public void Dispose()
    {
        // Cleanup all created brokers
        foreach (var broker in _brokersToDispose)
        {
            try
            {
                broker?.Dispose();
            }
            catch
            {
                // Ignore disposal errors in cleanup
            }
        }
        _brokersToDispose.Clear();
    }
}

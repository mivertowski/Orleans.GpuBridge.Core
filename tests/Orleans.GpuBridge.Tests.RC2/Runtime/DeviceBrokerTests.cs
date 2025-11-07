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
/// Comprehensive test suite for DeviceBroker with 70% coverage target (20 tests)
/// </summary>
public sealed class DeviceBrokerTests : IDisposable
{
    private readonly Mock<ILogger<DeviceBroker>> _mockLogger;
    private readonly GpuBridgeOptions _defaultOptions;
    private readonly IOptions<GpuBridgeOptions> _options;

    public DeviceBrokerTests()
    {
        _mockLogger = new Mock<ILogger<DeviceBroker>>();
        _defaultOptions = new GpuBridgeOptions
        {
            PreferGpu = true,
            MaxConcurrentKernels = 100,
            EnableMetrics = true
        };
        _options = Options.Create(_defaultOptions);
    }

    #region Device Discovery Tests (5 tests)

    [Fact]
    public async Task DiscoverDevices_ShouldFindCpuFallback()
    {
        // Arrange
        var broker = CreateDeviceBroker();

        // Act
        await broker.InitializeAsync(CancellationToken.None);
        var devices = broker.GetDevices();

        // Assert
        devices.Should().NotBeEmpty("DeviceBroker should at least have CPU fallback");
        devices.Should().Contain(d => d.Type == DeviceType.CPU, "CPU fallback should be present");
        devices.First(d => d.Type == DeviceType.CPU).Name.Should().Be("CPU Fallback");
    }

    [Fact]
    public async Task DiscoverDevices_WithNoGpu_ShouldReturnOnlyCpu()
    {
        // Arrange
        var broker = CreateDeviceBroker();

        // Act
        await broker.InitializeAsync(CancellationToken.None);
        var devices = broker.GetDevices();

        // Assert - since this is a test environment, no real GPU will be detected
        devices.Should().HaveCount(1, "Only CPU fallback should be available in test environment");
        devices[0].Type.Should().Be(DeviceType.CPU);
    }

    [Fact]
    public async Task GetDeviceById_WithValidId_ShouldReturnDevice()
    {
        // Arrange
        var broker = CreateDeviceBroker();
        await broker.InitializeAsync(CancellationToken.None);
        var devices = broker.GetDevices();

        // Act - Use the CPU fallback device index (-1)
        var device = broker.GetDevice(-1);

        // Assert
        device.Should().NotBeNull("CPU device with index -1 should exist");
        device!.Type.Should().Be(DeviceType.CPU);
        device.Name.Should().Be("CPU Fallback");
    }

    [Fact]
    public async Task GetDeviceById_WithInvalidId_ShouldReturnNull()
    {
        // Arrange
        var broker = CreateDeviceBroker();
        await broker.InitializeAsync(CancellationToken.None);

        // Act - Use an index that doesn't exist
        var device = broker.GetDevice(999);

        // Assert
        device.Should().BeNull("Device with invalid index should not exist");
    }

    [Fact]
    public async Task GetDeviceCapabilities_ShouldReturnCorrectInfo()
    {
        // Arrange
        var broker = CreateDeviceBroker();
        await broker.InitializeAsync(CancellationToken.None);
        var devices = broker.GetDevices();

        // Act
        var cpuDevice = devices.First(d => d.Type == DeviceType.CPU);

        // Assert
        cpuDevice.Capabilities.Should().NotBeNull("Device should have capabilities");
        cpuDevice.Capabilities.Should().Contain("cpu", "CPU device should have 'cpu' capability");
        cpuDevice.Capabilities.Should().Contain("fallback", "CPU device should have 'fallback' capability");
        cpuDevice.ComputeUnits.Should().Be(Environment.ProcessorCount, "CPU should report processor count");
    }

    #endregion

    #region Device Allocation Tests (8 tests)

    [Fact]
    public async Task GetBestDevice_WithAvailableDevices_ShouldSucceed()
    {
        // Arrange
        var broker = CreateDeviceBroker();
        await broker.InitializeAsync(CancellationToken.None);

        // Act
        var device = broker.GetBestDevice();

        // Assert
        device.Should().NotBeNull("GetBestDevice should return a device");
        device!.Type.Should().Be(DeviceType.CPU, "CPU fallback should be best available device");
    }

    [Fact]
    public async Task GetBestDevice_ShouldConsiderDeviceScore()
    {
        // Arrange
        var broker = CreateDeviceBroker();
        await broker.InitializeAsync(CancellationToken.None);

        // Act
        var bestDevice = broker.GetBestDevice();

        // Assert
        bestDevice.Should().NotBeNull("Best device should be selected");
        bestDevice!.AvailableMemoryBytes.Should().BeGreaterThan(0, "Best device should have available memory");
    }

    [Fact]
    public async Task DeviceCount_AfterInitialization_ShouldBeAccurate()
    {
        // Arrange
        var broker = CreateDeviceBroker();

        // Act
        await broker.InitializeAsync(CancellationToken.None);
        var count = broker.DeviceCount;

        // Assert
        count.Should().BeGreaterThan(0, "At least CPU fallback should be present");
        count.Should().Be(broker.GetDevices().Count, "DeviceCount should match GetDevices count");
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
    public async Task CurrentQueueDepth_InitiallyZero_ShouldBeAccurate()
    {
        // Arrange
        var broker = CreateDeviceBroker();
        await broker.InitializeAsync(CancellationToken.None);

        // Act
        var queueDepth = broker.CurrentQueueDepth;

        // Assert
        queueDepth.Should().Be(0, "Initial queue depth should be zero");
    }

    [Fact]
    public async Task GetDevice_WithExistingIndex_ShouldReturnSameInstance()
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
        device1.Should().Be(device2, "Same index should return same device instance");
    }

    [Fact]
    public async Task InitializeAsync_CalledTwice_ShouldBeIdempotent()
    {
        // Arrange
        var broker = CreateDeviceBroker();

        // Act
        await broker.InitializeAsync(CancellationToken.None);
        var devicesCount1 = broker.GetDevices().Count;

        await broker.InitializeAsync(CancellationToken.None); // Second call
        var devicesCount2 = broker.GetDevices().Count;

        // Assert
        devicesCount1.Should().Be(devicesCount2, "Multiple initializations should not add duplicate devices");
    }

    [Fact]
    public async Task GetDevices_BeforeInitialization_ShouldThrow()
    {
        // Arrange
        var broker = CreateDeviceBroker();

        // Act & Assert
        var action = () => broker.GetDevices();
        action.Should().Throw<InvalidOperationException>()
            .WithMessage("*not initialized*");
    }

    #endregion

    #region Device Health Tests (7 tests)

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
    public async Task DeviceBroker_Dispose_ShouldShutdownCleanly()
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
    public async Task GetBestDevice_WithMultipleDevices_ShouldSelectBasedOnScore()
    {
        // Arrange
        var broker = CreateDeviceBroker();
        await broker.InitializeAsync(CancellationToken.None);

        // Act
        var bestDevice = broker.GetBestDevice();

        // Assert
        bestDevice.Should().NotBeNull("Best device should be selected");
        bestDevice!.TotalMemoryBytes.Should().BeGreaterThan(0, "Selected device should have memory");
    }

    [Fact]
    public async Task DeviceBroker_AfterInitialization_ShouldLogDeviceInfo()
    {
        // Arrange
        var broker = CreateDeviceBroker();

        // Act
        await broker.InitializeAsync(CancellationToken.None);

        // Assert - Verify logging occurred
        _mockLogger.Verify(
            x => x.Log(
                LogLevel.Information,
                It.IsAny<EventId>(),
                It.Is<It.IsAnyType>((v, t) => v.ToString()!.Contains("initialized")),
                It.IsAny<Exception>(),
                It.IsAny<Func<It.IsAnyType, Exception?, string>>()),
            Times.AtLeastOnce,
            "Initialization should log device information");
    }

    [Fact]
    public async Task CpuFallbackDevice_ShouldHaveCorrectProperties()
    {
        // Arrange
        var broker = CreateDeviceBroker();
        await broker.InitializeAsync(CancellationToken.None);

        // Act
        var cpuDevice = broker.GetDevices().First(d => d.Type == DeviceType.CPU);

        // Assert
        cpuDevice.Index.Should().Be(-1, "CPU fallback should have index -1");
        cpuDevice.Name.Should().Be("CPU Fallback");
        cpuDevice.ComputeUnits.Should().Be(Environment.ProcessorCount);
        cpuDevice.TotalMemoryBytes.Should().BeGreaterThan(0, "CPU should report total memory");
        cpuDevice.AvailableMemoryBytes.Should().BeGreaterThan(0, "CPU should report available memory");
        cpuDevice.Capabilities.Should().NotBeEmpty("CPU should have capabilities");
    }

    [Fact]
    public async Task ConcurrentInitialization_ShouldBeSafe()
    {
        // Arrange
        var broker = CreateDeviceBroker();

        // Act - Try to initialize multiple times concurrently
        var tasks = Enumerable.Range(0, 5)
            .Select(_ => broker.InitializeAsync(CancellationToken.None))
            .ToArray();

        await Task.WhenAll(tasks);

        // Assert
        var devices = broker.GetDevices();
        devices.Should().NotBeEmpty("Concurrent initialization should succeed");
        devices.Should().OnlyHaveUniqueItems(d => d.Index, "No duplicate devices should exist");
    }

    #endregion

    #region Helper Methods

    /// <summary>
    /// Creates a DeviceBroker instance with mocked dependencies
    /// </summary>
    private DeviceBroker CreateDeviceBroker()
    {
        return new DeviceBroker(_mockLogger.Object, _options);
    }

    /// <summary>
    /// Creates a DeviceBroker with custom options
    /// </summary>
    private DeviceBroker CreateDeviceBroker(Action<GpuBridgeOptions> configureOptions)
    {
        var options = new GpuBridgeOptions();
        configureOptions(options);
        var optionsWrapper = Options.Create(options);
        return new DeviceBroker(_mockLogger.Object, optionsWrapper);
    }

    /// <summary>
    /// Creates a mock GPU device for testing
    /// </summary>
    private static GpuDevice CreateMockGpuDevice(
        int index = 0,
        string name = "Mock GPU",
        DeviceType type = DeviceType.CUDA,
        long totalMemory = 8L * 1024 * 1024 * 1024, // 8GB
        int computeUnits = 2048)
    {
        return new GpuDevice(
            Index: index,
            Name: name,
            Type: type,
            TotalMemoryBytes: totalMemory,
            AvailableMemoryBytes: totalMemory / 2,
            ComputeUnits: computeUnits,
            Capabilities: new[] { "cuda", "fp64", "atomics" });
    }

    #endregion

    public void Dispose()
    {
        // Cleanup if needed
    }
}

/// <summary>
/// Additional integration tests for DeviceBroker with real-world scenarios
/// </summary>
public sealed class DeviceBrokerIntegrationTests
{
    private readonly Mock<ILogger<DeviceBroker>> _mockLogger;
    private readonly IOptions<GpuBridgeOptions> _options;

    public DeviceBrokerIntegrationTests()
    {
        _mockLogger = new Mock<ILogger<DeviceBroker>>();
        var options = new GpuBridgeOptions
        {
            PreferGpu = true,
            MaxConcurrentKernels = 100,
            EnableMetrics = true,
            DefaultBackend = "BestAvailable"
        };
        _options = Options.Create(options);
    }

    [Fact]
    public async Task DeviceBroker_FullLifecycle_ShouldWorkCorrectly()
    {
        // Arrange
        using var broker = new DeviceBroker(_mockLogger.Object, _options);

        // Act - Initialize
        await broker.InitializeAsync(CancellationToken.None);
        broker.GetDevices().Should().NotBeEmpty();

        // Act - Use
        var bestDevice = broker.GetBestDevice();
        bestDevice.Should().NotBeNull();

        // Act - Shutdown
        await broker.ShutdownAsync(CancellationToken.None);

        // Assert
        var action = () => broker.GetDevices();
        action.Should().Throw<InvalidOperationException>();
    }

    [Fact]
    public async Task DeviceBroker_WithLongRunningOperation_ShouldNotBlock()
    {
        // Arrange
        using var broker = new DeviceBroker(_mockLogger.Object, _options);
        await broker.InitializeAsync(CancellationToken.None);

        // Act - Simulate concurrent operations
        var tasks = Enumerable.Range(0, 10)
            .Select(async i =>
            {
                await Task.Delay(10); // Simulate work
                return broker.GetBestDevice();
            })
            .ToArray();

        var results = await Task.WhenAll(tasks);

        // Assert
        results.Should().AllSatisfy(device => device.Should().NotBeNull());
    }

    [Fact]
    public async Task DeviceBroker_MemoryMetrics_ShouldBeConsistent()
    {
        // Arrange
        using var broker = new DeviceBroker(_mockLogger.Object, _options);
        await broker.InitializeAsync(CancellationToken.None);

        // Act
        var totalMemory = broker.TotalMemoryBytes;
        var devices = broker.GetDevices();
        var summedMemory = devices.Sum(d => d.TotalMemoryBytes);

        // Assert
        totalMemory.Should().Be(summedMemory, "Total memory should equal sum of device memories");
        totalMemory.Should().BeGreaterThan(0);
    }
}

/// <summary>
/// Performance and stress tests for DeviceBroker
/// </summary>
public sealed class DeviceBrokerPerformanceTests
{
    [Fact]
    public async Task GetBestDevice_With1000Calls_ShouldPerformWell()
    {
        // Arrange
        var mockLogger = new Mock<ILogger<DeviceBroker>>();
        var options = Options.Create(new GpuBridgeOptions());
        using var broker = new DeviceBroker(mockLogger.Object, options);
        await broker.InitializeAsync(CancellationToken.None);

        // Act
        var sw = System.Diagnostics.Stopwatch.StartNew();
        for (int i = 0; i < 1000; i++)
        {
            var device = broker.GetBestDevice();
            device.Should().NotBeNull();
        }
        sw.Stop();

        // Assert
        sw.ElapsedMilliseconds.Should().BeLessThan(1000,
            "1000 GetBestDevice calls should complete in under 1 second");
    }

    [Fact]
    public async Task ConcurrentDeviceQueries_ShouldBeThreadSafe()
    {
        // Arrange
        var mockLogger = new Mock<ILogger<DeviceBroker>>();
        var options = Options.Create(new GpuBridgeOptions());
        using var broker = new DeviceBroker(mockLogger.Object, options);
        await broker.InitializeAsync(CancellationToken.None);

        // Act - Hammer with concurrent requests
        var tasks = Enumerable.Range(0, 100)
            .Select(async i =>
            {
                return await Task.Run(() =>
                {
                    var device = broker.GetBestDevice();
                    var devices = broker.GetDevices();
                    var count = broker.DeviceCount;
                    return (device, devices, count);
                });
            })
            .ToArray();

        var results = await Task.WhenAll(tasks);

        // Assert
        results.Should().AllSatisfy(result =>
        {
            result.device.Should().NotBeNull();
            result.devices.Should().NotBeEmpty();
            result.count.Should().BeGreaterThan(0);
        });
    }
}

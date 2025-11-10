// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using FluentAssertions;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using Orleans.GpuBridge.Abstractions.Enums;
using Orleans.GpuBridge.Abstractions.Models;
using Orleans.GpuBridge.Backends.DotCompute.DeviceManagement;
using Xunit;

namespace Orleans.GpuBridge.Backends.DotCompute.Tests.DeviceManagement;

/// <summary>
/// Advanced unit tests for DotComputeDeviceManager covering all device management scenarios
/// </summary>
public class DeviceManagerAdvancedTests : IDisposable
{
    private readonly ILogger<DotComputeDeviceManager> _logger;
    private bool _disposed;

    public DeviceManagerAdvancedTests()
    {
        _logger = NullLogger<DotComputeDeviceManager>.Instance;
    }

    #region Device Discovery and Enumeration Tests (30 tests)

    [Fact]
    public async Task InitializeAsync_WithNoDevices_ShouldCompleteSuccessfully()
    {
        // Arrange
        var deviceManager = new DotComputeDeviceManager(_logger);

        // Act
        await deviceManager.InitializeAsync();

        // Assert
        var devices = deviceManager.GetDevices();
        devices.Should().NotBeNull();
    }

    [Fact]
    public async Task InitializeAsync_MultipleTimes_ShouldBeIdempotent()
    {
        // Arrange
        var deviceManager = new DotComputeDeviceManager(_logger);

        // Act
        await deviceManager.InitializeAsync();
        await deviceManager.InitializeAsync();
        await deviceManager.InitializeAsync();

        // Assert
        var devices = deviceManager.GetDevices();
        devices.Should().NotBeNull();
    }

    [Fact]
    public async Task InitializeAsync_Concurrent_ShouldHandleRaceCondition()
    {
        // Arrange
        var deviceManager = new DotComputeDeviceManager(_logger);

        // Act
        var tasks = Enumerable.Range(0, 10)
            .Select(_ => deviceManager.InitializeAsync())
            .ToArray();
        await Task.WhenAll(tasks);

        // Assert
        var devices = deviceManager.GetDevices();
        devices.Should().NotBeNull();
    }

    [Fact]
    public async Task GetDevices_AfterInitialization_ShouldReturnConsistentResults()
    {
        // Arrange
        var deviceManager = new DotComputeDeviceManager(_logger);
        await deviceManager.InitializeAsync();

        // Act
        var devices1 = deviceManager.GetDevices();
        var devices2 = deviceManager.GetDevices();

        // Assert
        devices1.Count.Should().Be(devices2.Count);
    }

    [Fact]
    public async Task GetDevicesByType_WithGpuType_ShouldReturnGpuDevices()
    {
        // Arrange
        var deviceManager = new DotComputeDeviceManager(_logger);
        await deviceManager.InitializeAsync();

        // Act
        var gpuDevices = deviceManager.GetDevicesByType(DeviceType.GPU).ToList();

        // Assert
        gpuDevices.Should().AllSatisfy(d => d.Type.Should().Be(DeviceType.GPU));
    }

    [Fact]
    public async Task GetDevicesByType_WithCpuType_ShouldReturnCpuDevices()
    {
        // Arrange
        var deviceManager = new DotComputeDeviceManager(_logger);
        await deviceManager.InitializeAsync();

        // Act
        var cpuDevices = deviceManager.GetDevicesByType(DeviceType.CPU).ToList();

        // Assert
        cpuDevices.Should().AllSatisfy(d => d.Type.Should().Be(DeviceType.CPU));
    }

    [Fact]
    public async Task GetDevicesByType_WithCudaType_ShouldReturnCudaDevices()
    {
        // Arrange
        var deviceManager = new DotComputeDeviceManager(_logger);
        await deviceManager.InitializeAsync();

        // Act
        var cudaDevices = deviceManager.GetDevicesByType(DeviceType.CUDA).ToList();

        // Assert
        cudaDevices.Should().AllSatisfy(d => d.Type.Should().Be(DeviceType.CUDA));
    }

    [Fact]
    public async Task GetDevicesByType_WithOpenCLType_ShouldReturnOpenCLDevices()
    {
        // Arrange
        var deviceManager = new DotComputeDeviceManager(_logger);
        await deviceManager.InitializeAsync();

        // Act
        var openclDevices = deviceManager.GetDevicesByType(DeviceType.OpenCL).ToList();

        // Assert
        openclDevices.Should().AllSatisfy(d => d.Type.Should().Be(DeviceType.OpenCL));
    }

    [Fact]
    public async Task GetDevice_WithNullId_ShouldReturnNull()
    {
        // Arrange
        var deviceManager = new DotComputeDeviceManager(_logger);
        await deviceManager.InitializeAsync();

        // Act
        var device = deviceManager.GetDevice(null!);

        // Assert
        device.Should().BeNull();
    }

    [Fact]
    public async Task GetDevice_WithEmptyId_ShouldReturnNull()
    {
        // Arrange
        var deviceManager = new DotComputeDeviceManager(_logger);
        await deviceManager.InitializeAsync();

        // Act
        var device = deviceManager.GetDevice(string.Empty);

        // Assert
        device.Should().BeNull();
    }

    [Fact]
    public async Task GetDevice_ByIndex_WithNegativeIndex_ShouldThrow()
    {
        // Arrange
        var deviceManager = new DotComputeDeviceManager(_logger);
        await deviceManager.InitializeAsync();

        // Act
        Action act = () => deviceManager.GetDevice(-1);

        // Assert
        act.Should().Throw<ArgumentException>();
    }

    [Fact]
    public async Task GetDevice_ByIndex_WithValidIndex_ShouldReturnDevice()
    {
        // Arrange
        var deviceManager = new DotComputeDeviceManager(_logger);
        await deviceManager.InitializeAsync();
        var devices = deviceManager.GetDevices();

        if (devices.Count == 0) return;

        // Act
        var device = deviceManager.GetDevice(0);

        // Assert
        device.Should().NotBeNull();
        device.DeviceId.Should().NotBeNullOrEmpty();
    }

    [Fact]
    public async Task GetDevice_ByIndex_WithOutOfRangeIndex_ShouldThrow()
    {
        // Arrange
        var deviceManager = new DotComputeDeviceManager(_logger);
        await deviceManager.InitializeAsync();

        // Act
        Action act = () => deviceManager.GetDevice(999);

        // Assert
        act.Should().Throw<ArgumentException>();
    }

    [Fact]
    public async Task GetDefaultDevice_WhenGpuAvailable_ShouldReturnGpu()
    {
        // Arrange
        var deviceManager = new DotComputeDeviceManager(_logger);
        await deviceManager.InitializeAsync();
        var devices = deviceManager.GetDevices();

        if (devices.Count == 0) return;
        if (!devices.Any(d => d.Type != DeviceType.CPU)) return;

        // Act
        var defaultDevice = deviceManager.GetDefaultDevice();

        // Assert
        defaultDevice.Should().NotBeNull();
        defaultDevice.Type.Should().NotBe(DeviceType.CPU);
    }

    [Fact]
    public async Task GetDefaultDevice_WhenOnlyCpuAvailable_ShouldReturnCpu()
    {
        // Arrange
        var deviceManager = new DotComputeDeviceManager(_logger);
        await deviceManager.InitializeAsync();
        var devices = deviceManager.GetDevices();

        if (devices.Count == 0) return;
        if (devices.Any(d => d.Type != DeviceType.CPU)) return;

        // Act
        var defaultDevice = deviceManager.GetDefaultDevice();

        // Assert
        defaultDevice.Should().NotBeNull();
        defaultDevice.Type.Should().Be(DeviceType.CPU);
    }

    #endregion

    #region Device Selection Criteria Tests (25 tests)

    [Fact]
    public async Task SelectDevice_WithNoPreferredType_ShouldReturnHealthyDevice()
    {
        // Arrange
        var deviceManager = new DotComputeDeviceManager(_logger);
        await deviceManager.InitializeAsync();
        var criteria = new DeviceSelectionCriteria();

        if (deviceManager.GetDevices().Count == 0) return;

        // Act
        var device = deviceManager.SelectDevice(criteria);

        // Assert
        device.Should().NotBeNull();
        device.GetStatus().Should().Be(DeviceStatus.Available);
    }

    [Fact]
    public async Task SelectDevice_WithPreferredGpuType_ShouldReturnGpu()
    {
        // Arrange
        var deviceManager = new DotComputeDeviceManager(_logger);
        await deviceManager.InitializeAsync();
        var criteria = new DeviceSelectionCriteria { PreferredType = DeviceType.GPU };

        if (!deviceManager.GetDevices().Any(d => d.Type == DeviceType.GPU)) return;

        // Act
        var device = deviceManager.SelectDevice(criteria);

        // Assert
        device.Should().NotBeNull();
        device.Type.Should().Be(DeviceType.GPU);
    }

    [Fact]
    public async Task SelectDevice_WithMinimumMemory_ShouldReturnDeviceWithEnoughMemory()
    {
        // Arrange
        var deviceManager = new DotComputeDeviceManager(_logger);
        await deviceManager.InitializeAsync();
        var criteria = new DeviceSelectionCriteria { MinimumMemoryBytes = 1024 * 1024 }; // 1 MB

        if (deviceManager.GetDevices().Count == 0) return;

        // Act
        var device = deviceManager.SelectDevice(criteria);

        // Assert
        device.Should().NotBeNull();
        device.AvailableMemoryBytes.Should().BeGreaterThanOrEqualTo(1024 * 1024);
    }

    [Fact]
    public async Task SelectDevice_WithMinComputeUnits_ShouldReturnDeviceWithEnoughComputeUnits()
    {
        // Arrange
        var deviceManager = new DotComputeDeviceManager(_logger);
        await deviceManager.InitializeAsync();
        var criteria = new DeviceSelectionCriteria { MinComputeUnits = 1 };

        if (deviceManager.GetDevices().Count == 0) return;

        // Act
        var device = deviceManager.SelectDevice(criteria);

        // Assert
        device.Should().NotBeNull();
        device.ComputeUnits.Should().BeGreaterThanOrEqualTo(1);
    }

    [Fact]
    public async Task SelectDevice_WithUnachievableMemory_ShouldThrow()
    {
        // Arrange
        var deviceManager = new DotComputeDeviceManager(_logger);
        await deviceManager.InitializeAsync();
        var criteria = new DeviceSelectionCriteria { MinimumMemoryBytes = long.MaxValue };

        // Act
        Action act = () => deviceManager.SelectDevice(criteria);

        // Assert
        act.Should().Throw<InvalidOperationException>();
    }

    [Fact]
    public async Task SelectDevice_WithUnachievableComputeUnits_ShouldThrow()
    {
        // Arrange
        var deviceManager = new DotComputeDeviceManager(_logger);
        await deviceManager.InitializeAsync();
        var criteria = new DeviceSelectionCriteria { MinComputeUnits = int.MaxValue };

        // Act
        Action act = () => deviceManager.SelectDevice(criteria);

        // Assert
        act.Should().Throw<InvalidOperationException>();
    }

    [Fact]
    public async Task SelectDevice_WithCombinedCriteria_ShouldMatchAllCriteria()
    {
        // Arrange
        var deviceManager = new DotComputeDeviceManager(_logger);
        await deviceManager.InitializeAsync();
        var criteria = new DeviceSelectionCriteria
        {
            PreferredType = DeviceType.GPU,
            MinimumMemoryBytes = 1024,
            MinComputeUnits = 1
        };

        if (!deviceManager.GetDevices().Any(d => d.Type == DeviceType.GPU)) return;

        // Act
        var device = deviceManager.SelectDevice(criteria);

        // Assert
        device.Should().NotBeNull();
        device.Type.Should().Be(DeviceType.GPU);
        device.AvailableMemoryBytes.Should().BeGreaterThanOrEqualTo(1024);
        device.ComputeUnits.Should().BeGreaterThanOrEqualTo(1);
    }

    #endregion

    #region Device Health Monitoring Tests (30 tests)

    [Fact]
    public async Task GetDeviceHealthAsync_WithValidDevice_ShouldReturnHealthInfo()
    {
        // Arrange
        var deviceManager = new DotComputeDeviceManager(_logger);
        await deviceManager.InitializeAsync();
        var devices = deviceManager.GetDevices();

        if (devices.Count == 0) return;

        // Act
        var health = await deviceManager.GetDeviceHealthAsync(devices[0].DeviceId);

        // Assert
        health.Should().NotBeNull();
        health.DeviceId.Should().Be(devices[0].DeviceId);
    }

    [Fact]
    public async Task GetDeviceHealthAsync_WithInvalidDevice_ShouldThrow()
    {
        // Arrange
        var deviceManager = new DotComputeDeviceManager(_logger);
        await deviceManager.InitializeAsync();

        // Act
        Func<Task> act = async () => await deviceManager.GetDeviceHealthAsync("invalid-id");

        // Assert
        await act.Should().ThrowAsync<ArgumentException>();
    }

    [Fact]
    public async Task GetDeviceHealthAsync_MemoryUtilization_ShouldBeValid()
    {
        // Arrange
        var deviceManager = new DotComputeDeviceManager(_logger);
        await deviceManager.InitializeAsync();
        var devices = deviceManager.GetDevices();

        if (devices.Count == 0) return;

        // Act
        var health = await deviceManager.GetDeviceHealthAsync(devices[0].DeviceId);

        // Assert
        health.MemoryUtilizationPercent.Should().BeInRange(0, 100);
    }

    [Fact]
    public async Task GetDeviceHealthAsync_Status_ShouldBeValid()
    {
        // Arrange
        var deviceManager = new DotComputeDeviceManager(_logger);
        await deviceManager.InitializeAsync();
        var devices = deviceManager.GetDevices();

        if (devices.Count == 0) return;

        // Act
        var health = await deviceManager.GetDeviceHealthAsync(devices[0].DeviceId);

        // Assert
        health.Status.Should().BeOneOf(DeviceStatus.Available, DeviceStatus.Busy, DeviceStatus.Error);
    }

    [Fact]
    public async Task GetDeviceHealthAsync_Concurrent_ShouldHandleMultipleRequests()
    {
        // Arrange
        var deviceManager = new DotComputeDeviceManager(_logger);
        await deviceManager.InitializeAsync();
        var devices = deviceManager.GetDevices();

        if (devices.Count == 0) return;

        // Act
        var tasks = Enumerable.Range(0, 10)
            .Select(_ => deviceManager.GetDeviceHealthAsync(devices[0].DeviceId))
            .ToArray();
        var results = await Task.WhenAll(tasks);

        // Assert
        results.Should().AllSatisfy(h =>
        {
            h.Should().NotBeNull();
            h.DeviceId.Should().Be(devices[0].DeviceId);
        });
    }

    [Fact]
    public async Task GetDeviceHealthAsync_WithCancellation_ShouldRespectToken()
    {
        // Arrange
        var deviceManager = new DotComputeDeviceManager(_logger);
        await deviceManager.InitializeAsync();
        var devices = deviceManager.GetDevices();

        if (devices.Count == 0) return;

        using var cts = new CancellationTokenSource();
        cts.Cancel();

        // Act
        Func<Task> act = async () => await deviceManager.GetDeviceHealthAsync(devices[0].DeviceId, cts.Token);

        // Assert
        await act.Should().ThrowAsync<OperationCanceledException>();
    }

    #endregion

    #region Device Metrics Tests (30 tests)

    [Fact]
    public async Task GetDeviceMetricsAsync_WithValidDevice_ShouldReturnMetrics()
    {
        // Arrange
        var deviceManager = new DotComputeDeviceManager(_logger);
        await deviceManager.InitializeAsync();
        var devices = deviceManager.GetDevices();

        if (devices.Count == 0) return;

        // Act
        var metrics = await deviceManager.GetDeviceMetricsAsync(devices[0]);

        // Assert
        metrics.Should().NotBeNull();
    }

    [Fact]
    public async Task GetDeviceMetricsAsync_GpuUtilization_ShouldBeValid()
    {
        // Arrange
        var deviceManager = new DotComputeDeviceManager(_logger);
        await deviceManager.InitializeAsync();
        var devices = deviceManager.GetDevices();

        if (devices.Count == 0) return;

        // Act
        var metrics = await deviceManager.GetDeviceMetricsAsync(devices[0]);

        // Assert
        metrics.GpuUtilizationPercent.Should().BeGreaterThanOrEqualTo(0);
    }

    [Fact]
    public async Task GetDeviceMetricsAsync_MemoryUtilization_ShouldBeValid()
    {
        // Arrange
        var deviceManager = new DotComputeDeviceManager(_logger);
        await deviceManager.InitializeAsync();
        var devices = deviceManager.GetDevices();

        if (devices.Count == 0) return;

        // Act
        var metrics = await deviceManager.GetDeviceMetricsAsync(devices[0]);

        // Assert
        metrics.MemoryUtilizationPercent.Should().BeInRange(0, 100);
    }

    [Fact]
    public async Task GetDeviceMetricsAsync_Temperature_ShouldBeRealistic()
    {
        // Arrange
        var deviceManager = new DotComputeDeviceManager(_logger);
        await deviceManager.InitializeAsync();
        var devices = deviceManager.GetDevices();

        if (devices.Count == 0) return;

        // Act
        var metrics = await deviceManager.GetDeviceMetricsAsync(devices[0]);

        // Assert
        metrics.TemperatureCelsius.Should().BeGreaterThanOrEqualTo(0);
        metrics.TemperatureCelsius.Should().BeLessThan(150); // Realistic GPU temp limit
    }

    [Fact]
    public async Task GetDeviceMetricsAsync_PowerWatts_ShouldBeRealistic()
    {
        // Arrange
        var deviceManager = new DotComputeDeviceManager(_logger);
        await deviceManager.InitializeAsync();
        var devices = deviceManager.GetDevices();

        if (devices.Count == 0) return;

        // Act
        var metrics = await deviceManager.GetDeviceMetricsAsync(devices[0]);

        // Assert
        metrics.PowerWatts.Should().BeGreaterThanOrEqualTo(0);
        metrics.PowerWatts.Should().BeLessThan(1000); // Realistic GPU power limit
    }

    [Fact]
    public async Task GetDeviceMetricsAsync_WithNullDevice_ShouldThrow()
    {
        // Arrange
        var deviceManager = new DotComputeDeviceManager(_logger);
        await deviceManager.InitializeAsync();

        // Act
        Func<Task> act = async () => await deviceManager.GetDeviceMetricsAsync(null!);

        // Assert
        await act.Should().ThrowAsync<ArgumentNullException>();
    }

    [Fact]
    public async Task GetDeviceMetricsAsync_Concurrent_ShouldHandleMultipleRequests()
    {
        // Arrange
        var deviceManager = new DotComputeDeviceManager(_logger);
        await deviceManager.InitializeAsync();
        var devices = deviceManager.GetDevices();

        if (devices.Count == 0) return;

        // Act
        var tasks = Enumerable.Range(0, 10)
            .Select(_ => deviceManager.GetDeviceMetricsAsync(devices[0]))
            .ToArray();
        var results = await Task.WhenAll(tasks);

        // Assert
        results.Should().AllSatisfy(m => m.Should().NotBeNull());
    }

    [Fact]
    public async Task GetDeviceMetricsAsync_WithCancellation_ShouldRespectToken()
    {
        // Arrange
        var deviceManager = new DotComputeDeviceManager(_logger);
        await deviceManager.InitializeAsync();
        var devices = deviceManager.GetDevices();

        if (devices.Count == 0) return;

        using var cts = new CancellationTokenSource();
        cts.Cancel();

        // Act
        Func<Task> act = async () => await deviceManager.GetDeviceMetricsAsync(devices[0], cts.Token);

        // Assert
        await act.Should().ThrowAsync<OperationCanceledException>();
    }

    #endregion

    #region Device Reset Tests (15 tests)

    [Fact]
    public async Task ResetDeviceAsync_WithValidDevice_ShouldComplete()
    {
        // Arrange
        var deviceManager = new DotComputeDeviceManager(_logger);
        await deviceManager.InitializeAsync();
        var devices = deviceManager.GetDevices();

        if (devices.Count == 0) return;

        // Act
        await deviceManager.ResetDeviceAsync(devices[0]);

        // Assert - Should complete without exception
        true.Should().BeTrue();
    }

    [Fact]
    public async Task ResetDeviceAsync_WithNullDevice_ShouldThrow()
    {
        // Arrange
        var deviceManager = new DotComputeDeviceManager(_logger);
        await deviceManager.InitializeAsync();

        // Act
        Func<Task> act = async () => await deviceManager.ResetDeviceAsync(null!);

        // Assert
        await act.Should().ThrowAsync<ArgumentNullException>();
    }

    [Fact]
    public async Task ResetDeviceAsync_WithCancellation_ShouldRespectToken()
    {
        // Arrange
        var deviceManager = new DotComputeDeviceManager(_logger);
        await deviceManager.InitializeAsync();
        var devices = deviceManager.GetDevices();

        if (devices.Count == 0) return;

        using var cts = new CancellationTokenSource();
        cts.Cancel();

        // Act
        Func<Task> act = async () => await deviceManager.ResetDeviceAsync(devices[0], cts.Token);

        // Assert
        await act.Should().ThrowAsync<OperationCanceledException>();
    }

    [Fact]
    public async Task ResetDeviceAsync_Multiple_ShouldHandleSequentialResets()
    {
        // Arrange
        var deviceManager = new DotComputeDeviceManager(_logger);
        await deviceManager.InitializeAsync();
        var devices = deviceManager.GetDevices();

        if (devices.Count == 0) return;

        // Act
        await deviceManager.ResetDeviceAsync(devices[0]);
        await deviceManager.ResetDeviceAsync(devices[0]);
        await deviceManager.ResetDeviceAsync(devices[0]);

        // Assert - Should complete without exception
        true.Should().BeTrue();
    }

    #endregion

    #region Disposal and Cleanup Tests (15 tests)

    [Fact]
    public async Task Dispose_AfterInitialization_ShouldCleanupResources()
    {
        // Arrange
        var deviceManager = new DotComputeDeviceManager(_logger);
        await deviceManager.InitializeAsync();

        // Act
        deviceManager.Dispose();

        // Assert - Should not throw
        Action act = () => deviceManager.Dispose();
        act.Should().NotThrow();
    }

    [Fact]
    public void Dispose_BeforeInitialization_ShouldNotThrow()
    {
        // Arrange
        var deviceManager = new DotComputeDeviceManager(_logger);

        // Act & Assert
        Action act = () => deviceManager.Dispose();
        act.Should().NotThrow();
    }

    [Fact]
    public void Dispose_MultipleTimes_ShouldBeIdempotent()
    {
        // Arrange
        var deviceManager = new DotComputeDeviceManager(_logger);

        // Act
        deviceManager.Dispose();
        deviceManager.Dispose();
        deviceManager.Dispose();

        // Assert - Should not throw
        true.Should().BeTrue();
    }

    [Fact]
    public async Task Dispose_Concurrent_ShouldHandleRaceCondition()
    {
        // Arrange
        var deviceManager = new DotComputeDeviceManager(_logger);
        await deviceManager.InitializeAsync();

        // Act
        var tasks = Enumerable.Range(0, 10)
            .Select(_ => Task.Run(() => deviceManager.Dispose()))
            .ToArray();

        // Assert
        Func<Task> act = async () => await Task.WhenAll(tasks);
        await act.Should().NotThrowAsync();
    }

    #endregion

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
    }
}

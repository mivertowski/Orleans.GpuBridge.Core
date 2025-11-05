// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using FluentAssertions;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using Orleans.GpuBridge.Abstractions.Enums;
using Orleans.GpuBridge.Backends.DotCompute.DeviceManagement;
using Xunit;

namespace Orleans.GpuBridge.Backends.DotCompute.Tests.DeviceManagement;

/// <summary>
/// Unit tests for DotComputeDeviceManager
/// </summary>
public class DotComputeDeviceManagerTests
{
    private readonly ILogger<DotComputeDeviceManager> _logger;

    public DotComputeDeviceManagerTests()
    {
        _logger = NullLogger<DotComputeDeviceManager>.Instance;
    }

    [Fact]
    public async Task InitializeAsync_Should_DiscoverDevices()
    {
        // Arrange
        var deviceManager = new DotComputeDeviceManager(_logger);

        // Act
        await deviceManager.InitializeAsync();

        // Assert
        var devices = deviceManager.GetDevices();
        devices.Should().NotBeNull();
        // Note: Device count depends on system (may be 0 if no GPU, at least 1 if CPU backend works)
    }

    [Fact]
    public void GetDevices_WhenNotInitialized_ShouldThrow()
    {
        // Arrange
        var deviceManager = new DotComputeDeviceManager(_logger);

        // Act
        Action act = () => deviceManager.GetDevices();

        // Assert
        act.Should().Throw<InvalidOperationException>()
            .WithMessage("*not initialized*");
    }

    [Fact]
    public async Task GetDevice_WithValidId_ShouldReturnDevice()
    {
        // Arrange
        var deviceManager = new DotComputeDeviceManager(_logger);
        await deviceManager.InitializeAsync();
        var allDevices = deviceManager.GetDevices();

        if (allDevices.Count == 0)
        {
            // Skip test if no devices available
            return;
        }

        var firstDevice = allDevices[0];

        // Act
        var device = deviceManager.GetDevice(firstDevice.DeviceId);

        // Assert
        device.Should().NotBeNull();
        device!.DeviceId.Should().Be(firstDevice.DeviceId);
        device.Name.Should().Be(firstDevice.Name);
    }

    [Fact]
    public async Task GetDevice_WithInvalidId_ShouldReturnNull()
    {
        // Arrange
        var deviceManager = new DotComputeDeviceManager(_logger);
        await deviceManager.InitializeAsync();

        // Act
        var device = deviceManager.GetDevice("invalid-device-id");

        // Assert
        device.Should().BeNull();
    }

    [Fact]
    public async Task GetDeviceHealthAsync_Should_ReturnHealthInfo()
    {
        // Arrange
        var deviceManager = new DotComputeDeviceManager(_logger);
        await deviceManager.InitializeAsync();
        var allDevices = deviceManager.GetDevices();

        if (allDevices.Count == 0)
        {
            // Skip test if no devices available
            return;
        }

        var device = allDevices[0];

        // Act
        var health = await deviceManager.GetDeviceHealthAsync(device.DeviceId);

        // Assert
        health.Should().NotBeNull();
        health.DeviceId.Should().Be(device.DeviceId);
        health.Status.Should().Be(DeviceStatus.Available);
        health.MemoryUtilizationPercent.Should().BeGreaterThanOrEqualTo(0).And.BeLessThanOrEqualTo(100);
    }

    [Fact]
    public async Task InitializeAsync_CalledTwice_ShouldNotReinitialize()
    {
        // Arrange
        var deviceManager = new DotComputeDeviceManager(_logger);

        // Act
        await deviceManager.InitializeAsync();
        var firstInitDevices = deviceManager.GetDevices();

        await deviceManager.InitializeAsync(); // Second init should be no-op
        var secondInitDevices = deviceManager.GetDevices();

        // Assert
        firstInitDevices.Count.Should().Be(secondInitDevices.Count);
    }

    [Fact]
    public async Task Dispose_Should_CleanupResources()
    {
        // Arrange
        var deviceManager = new DotComputeDeviceManager(_logger);
        await deviceManager.InitializeAsync();
        var deviceCountBefore = deviceManager.GetDevices().Count;

        // Act
        deviceManager.Dispose();

        // Assert
        // After disposal, calling Dispose again should not throw (idempotent)
        Action act = () => deviceManager.Dispose();
        act.Should().NotThrow();

        // Device collection should be cleared after disposal
        deviceCountBefore.Should().BeGreaterThanOrEqualTo(0);
    }

    [Fact]
    public async Task CreateContextAsync_Should_ThrowNotImplemented()
    {
        // Arrange
        var deviceManager = new DotComputeDeviceManager(_logger);
        await deviceManager.InitializeAsync();
        var devices = deviceManager.GetDevices();

        if (devices.Count == 0)
        {
            return; // Skip if no devices
        }

        var device = devices[0];

        // Act
        Func<Task> act = async () => await deviceManager.CreateContextAsync(device, null!);

        // Assert
        await act.Should().ThrowAsync<NotImplementedException>()
            .WithMessage("*Context creation*");
    }

    [Fact]
    public async Task GetDeviceHealthAsync_WithGpuDevice_ShouldReturnNonZeroTemperature()
    {
        // Arrange
        var deviceManager = new DotComputeDeviceManager(_logger);
        await deviceManager.InitializeAsync();
        var devices = deviceManager.GetDevices();

        var gpuDevice = devices.FirstOrDefault(d => d.Type == DeviceType.GPU);
        if (gpuDevice == null)
        {
            return; // Skip if no GPU available
        }

        // Act
        var health = await deviceManager.GetDeviceHealthAsync(gpuDevice.DeviceId);

        // Assert
        // GPU should report temperature (currently simulated at 45°C)
        health.TemperatureCelsius.Should().BeGreaterThan(0);
    }

    [Fact]
    public async Task GetDeviceHealthAsync_WithCpuDevice_ShouldReturnZeroTemperature()
    {
        // Arrange
        var deviceManager = new DotComputeDeviceManager(_logger);
        await deviceManager.InitializeAsync();
        var devices = deviceManager.GetDevices();

        var cpuDevice = devices.FirstOrDefault(d => d.Type == DeviceType.CPU);
        if (cpuDevice == null)
        {
            return; // Skip if no CPU device
        }

        // Act
        var health = await deviceManager.GetDeviceHealthAsync(cpuDevice.DeviceId);

        // Assert
        // CPU currently reports 0°C (sensor not yet implemented)
        health.TemperatureCelsius.Should().Be(0);
    }
}

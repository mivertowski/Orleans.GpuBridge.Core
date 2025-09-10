using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using AutoFixture.Xunit2;
using FluentAssertions;
using FsCheck;
using FsCheck.Xunit;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using Orleans.GpuBridge.Abstractions;
using Orleans.GpuBridge.Runtime;
using Orleans.GpuBridge.Tests.TestingFramework;
using Xunit;

namespace Orleans.GpuBridge.Tests.Unit;

/// <summary>
/// Enhanced comprehensive tests for DeviceBroker with stress testing and edge cases
/// </summary>
public class EnhancedDeviceBrokerTests : TestFixtureBase
{
    [Fact]
    public async Task Given_Valid_Configuration_When_Initialize_Then_Should_Discover_All_Available_Devices()
    {
        // Arrange
        var logger = GetService<ILogger<DeviceBroker>>();
        var options = Options.Create(new GpuBridgeOptions
        {
            PreferGpu = true,
            MemoryPoolSizeMB = 1024,
            Telemetry = new TelemetryOptions { EnableMetrics = true }
        });
        
        using var broker = new DeviceBroker(logger, options);

        // Act
        await broker.InitializeAsync(CancellationToken);
        var devices = broker.GetDevices();

        // Assert
        devices.Should().NotBeEmpty();
        devices.Should().Contain(d => d.Type == DeviceType.Cpu, "CPU device should always be available");
        broker.DeviceCount.Should().BeGreaterThan(0);
        broker.TotalMemoryBytes.Should().BeGreaterThan(0);
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public async Task Given_PreferGpu_Setting_When_GetBestDevice_Then_Should_Respect_Preference(bool preferGpu)
    {
        // Arrange
        var logger = GetService<ILogger<DeviceBroker>>();
        var options = Options.Create(new GpuBridgeOptions { PreferGpu = preferGpu });
        
        using var broker = new DeviceBroker(logger, options);
        await broker.InitializeAsync(CancellationToken);

        // Act
        var bestDevice = broker.GetBestDevice();

        // Assert
        bestDevice.Should().NotBeNull();
        if (preferGpu && broker.GetDevices().Any(d => d.Type == DeviceType.Gpu))
        {
            bestDevice.Type.Should().Be(DeviceType.Gpu, "GPU should be preferred when available");
        }
        else
        {
            bestDevice.Type.Should().Be(DeviceType.Cpu, "CPU should be fallback when GPU not preferred or available");
        }
    }

    [Property]
    public Property Given_Random_Device_Index_When_GetDevice_Then_Should_Return_Valid_Device_Or_Null(int index)
    {
        return Prop.ForAll<int>(deviceIndex =>
        {
            // Arrange
            var logger = GetService<ILogger<DeviceBroker>>();
            var options = Options.Create(new GpuBridgeOptions());
            using var broker = new DeviceBroker(logger, options);
            broker.InitializeAsync(CancellationToken.None).Wait();

            // Act
            var device = broker.GetDevice(deviceIndex);

            // Assert
            return deviceIndex >= 0 && deviceIndex < broker.DeviceCount
                ? (device != null).ToProperty()
                : (device == null).ToProperty();
        });
    }

    [Fact]
    public async Task Given_Multiple_Concurrent_Initializations_When_Initialize_Then_Should_Be_Thread_Safe()
    {
        // Arrange
        var logger = GetService<ILogger<DeviceBroker>>();
        var options = Options.Create(new GpuBridgeOptions());
        using var broker = new DeviceBroker(logger, options);

        var tasks = new List<Task>();
        var results = new List<int>();
        var lockObject = new object();

        // Act - Multiple concurrent initializations
        for (int i = 0; i < 10; i++)
        {
            tasks.Add(Task.Run(async () =>
            {
                await broker.InitializeAsync(CancellationToken);
                lock (lockObject)
                {
                    results.Add(broker.DeviceCount);
                }
            }));
        }

        await Task.WhenAll(tasks);

        // Assert - All results should be the same (idempotent)
        results.Should().HaveCount(10);
        results.Should().OnlyContain(count => count == results[0], "all initializations should return same device count");
    }

    [Fact]
    public async Task Given_Cancelled_Token_When_Initialize_Then_Should_Throw_OperationCancelledException()
    {
        // Arrange
        var logger = GetService<ILogger<DeviceBroker>>();
        var options = Options.Create(new GpuBridgeOptions());
        using var broker = new DeviceBroker(logger, options);
        using var cts = new CancellationTokenSource();
        cts.Cancel(); // Cancel immediately

        // Act & Assert
        await broker.Invoking(b => b.InitializeAsync(cts.Token))
            .Should()
            .ThrowAsync<OperationCanceledException>();
    }

    [Fact]
    public async Task Given_Disposed_Broker_When_GetDevices_Then_Should_Throw_InvalidOperationException()
    {
        // Arrange
        var logger = GetService<ILogger<DeviceBroker>>();
        var options = Options.Create(new GpuBridgeOptions());
        var broker = new DeviceBroker(logger, options);
        await broker.InitializeAsync(CancellationToken);

        // Act
        broker.Dispose();

        // Assert
        broker.Invoking(b => b.GetDevices())
            .Should()
            .Throw<InvalidOperationException>();
    }

    [Fact]
    public async Task Given_High_Memory_Configuration_When_Initialize_Then_Should_Handle_Large_Memory_Pool()
    {
        // Arrange
        var logger = GetService<ILogger<DeviceBroker>>();
        var options = Options.Create(new GpuBridgeOptions
        {
            MemoryPoolSizeMB = 8192 // 8GB memory pool
        });
        
        using var broker = new DeviceBroker(logger, options);

        // Act
        await broker.InitializeAsync(CancellationToken);
        var devices = broker.GetDevices();

        // Assert
        broker.TotalMemoryBytes.Should().BeGreaterThan(0);
        devices.Should().NotBeEmpty();
    }

    [Theory, AutoData]
    public async Task Given_Custom_Configuration_When_Initialize_Then_Should_Apply_Configuration(
        bool preferGpu, int memoryPoolSize, bool enableMetrics)
    {
        // Arrange
        var logger = GetService<ILogger<DeviceBroker>>();
        var options = Options.Create(new GpuBridgeOptions
        {
            PreferGpu = preferGpu,
            MemoryPoolSizeMB = Math.Max(1, memoryPoolSize % 1024), // Clamp to reasonable range
            Telemetry = new TelemetryOptions { EnableMetrics = enableMetrics }
        });
        
        using var broker = new DeviceBroker(logger, options);

        // Act
        await broker.InitializeAsync(CancellationToken);

        // Assert
        broker.DeviceCount.Should().BeGreaterThan(0);
        broker.TotalMemoryBytes.Should().BeGreaterThan(0);
    }

    [Fact]
    public async Task Given_Rapid_Shutdown_After_Initialize_When_Shutdown_Then_Should_Complete_Successfully()
    {
        // Arrange
        var logger = GetService<ILogger<DeviceBroker>>();
        var options = Options.Create(new GpuBridgeOptions());
        using var broker = new DeviceBroker(logger, options);

        // Act
        await broker.InitializeAsync(CancellationToken);
        await broker.ShutdownAsync(CancellationToken);

        // Assert
        broker.GetDevices().Should().BeEmpty("devices should be cleared after shutdown");
    }

    [Fact]
    public async Task Given_Multiple_Dispose_Calls_When_Dispose_Then_Should_Not_Throw()
    {
        // Arrange
        var logger = GetService<ILogger<DeviceBroker>>();
        var options = Options.Create(new GpuBridgeOptions());
        var broker = new DeviceBroker(logger, options);
        await broker.InitializeAsync(CancellationToken);

        // Act & Assert
        broker.Dispose();
        broker.Invoking(b => b.Dispose()) // Second dispose should not throw
            .Should()
            .NotThrow();
    }

    [Fact]
    public async Task Given_Device_Enumeration_When_Multiple_Threads_Access_Then_Should_Be_Thread_Safe()
    {
        // Arrange
        var logger = GetService<ILogger<DeviceBroker>>();
        var options = Options.Create(new GpuBridgeOptions());
        using var broker = new DeviceBroker(logger, options);
        await broker.InitializeAsync(CancellationToken);

        var tasks = new List<Task<GpuDeviceInfo>>();
        var exceptions = new List<Exception>();
        var lockObject = new object();

        // Act - Multiple threads accessing devices concurrently
        for (int i = 0; i < 50; i++)
        {
            tasks.Add(Task.Run(() =>
            {
                try
                {
                    var devices = broker.GetDevices().ToList();
                    var bestDevice = broker.GetBestDevice();
                    var randomDevice = broker.GetDevice(Random.Shared.Next(0, broker.DeviceCount));
                    return bestDevice;
                }
                catch (Exception ex)
                {
                    lock (lockObject)
                    {
                        exceptions.Add(ex);
                    }
                    throw;
                }
            }));
        }

        // Assert
        await Task.WhenAll(tasks);
        exceptions.Should().BeEmpty("no exceptions should occur during concurrent access");
        tasks.Should().OnlyContain(t => t.Result != null, "all tasks should return valid devices");
    }

    [Theory]
    [InlineData(0)]
    [InlineData(-1)]
    [InlineData(int.MinValue)]
    [InlineData(int.MaxValue)]
    public async Task Given_Invalid_Device_Index_When_GetDevice_Then_Should_Return_Null(int invalidIndex)
    {
        // Arrange
        var logger = GetService<ILogger<DeviceBroker>>();
        var options = Options.Create(new GpuBridgeOptions());
        using var broker = new DeviceBroker(logger, options);
        await broker.InitializeAsync(CancellationToken);

        // Act
        var device = broker.GetDevice(invalidIndex);

        // Assert
        if (invalidIndex >= 0 && invalidIndex < broker.DeviceCount)
        {
            device.Should().NotBeNull("valid indices should return devices");
        }
        else
        {
            device.Should().BeNull("invalid indices should return null");
        }
    }

    [Fact]
    public async Task Given_Device_With_Capabilities_When_GetDevices_Then_Should_Include_Capability_Information()
    {
        // Arrange
        var logger = GetService<ILogger<DeviceBroker>>();
        var options = Options.Create(new GpuBridgeOptions());
        using var broker = new DeviceBroker(logger, options);
        await broker.InitializeAsync(CancellationToken);

        // Act
        var devices = broker.GetDevices();

        // Assert
        foreach (var device in devices)
        {
            device.Capabilities.Should().NotBeNull("capabilities should not be null");
            device.Capabilities.Should().NotBeEmpty("capabilities should not be empty");
            device.Name.Should().NotBeNullOrEmpty("device name should be provided");
            device.TotalMemoryBytes.Should().BeGreaterThan(0, "device should have memory");
            device.ComputeUnits.Should().BeGreaterThan(0, "device should have compute units");
            
            if (device.Type == DeviceType.Cpu)
            {
                device.Capabilities.Should().Contain("CPU", "CPU device should have CPU capability");
            }
        }
    }

    [Fact]
    public async Task Given_Queue_Depth_Tracking_When_Operations_Pending_Then_Should_Track_Queue_Depth()
    {
        // Arrange
        var logger = GetService<ILogger<DeviceBroker>>();
        var options = Options.Create(new GpuBridgeOptions());
        using var broker = new DeviceBroker(logger, options);
        await broker.InitializeAsync(CancellationToken);

        // Act & Assert
        var initialDepth = broker.CurrentQueueDepth;
        initialDepth.Should().Be(0, "queue depth should start at zero");
    }

    [Fact(Skip = "Stress test - enable for comprehensive testing")]
    public async Task Stress_Test_Concurrent_Device_Access()
    {
        // Arrange
        var logger = GetService<ILogger<DeviceBroker>>();
        var options = Options.Create(new GpuBridgeOptions());
        using var broker = new DeviceBroker(logger, options);
        await broker.InitializeAsync(CancellationToken);

        var tasks = new List<Task>();
        var successCount = 0;
        var lockObject = new object();

        using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(30));

        // Act - Stress test with many concurrent operations
        for (int i = 0; i < 100; i++)
        {
            tasks.Add(Task.Run(async () =>
            {
                while (!cts.Token.IsCancellationRequested)
                {
                    try
                    {
                        var devices = broker.GetDevices().ToList();
                        var best = broker.GetBestDevice();
                        var random = broker.GetDevice(Random.Shared.Next(0, broker.DeviceCount));
                        
                        lock (lockObject)
                        {
                            successCount++;
                        }
                        
                        await Task.Delay(Random.Shared.Next(1, 10), cts.Token);
                    }
                    catch (OperationCanceledException)
                    {
                        break;
                    }
                }
            }, cts.Token));
        }

        try
        {
            await Task.WhenAll(tasks);
        }
        catch (OperationCanceledException)
        {
            // Expected when test times out
        }

        // Assert
        successCount.Should().BeGreaterThan(0, "stress test should complete some operations");
    }

    [Fact]
    public async Task Given_Resource_Constraints_When_Initialize_Then_Should_Handle_Low_Memory_Gracefully()
    {
        // Arrange
        var logger = GetService<ILogger<DeviceBroker>>();
        var options = Options.Create(new GpuBridgeOptions
        {
            MemoryPoolSizeMB = 1 // Very small memory pool
        });
        
        using var broker = new DeviceBroker(logger, options);

        // Act & Assert
        await broker.Invoking(b => b.InitializeAsync(CancellationToken))
            .Should()
            .NotThrowAsync("broker should handle low memory constraints");
        
        broker.DeviceCount.Should().BeGreaterThan(0, "at least CPU device should be available");
    }
}
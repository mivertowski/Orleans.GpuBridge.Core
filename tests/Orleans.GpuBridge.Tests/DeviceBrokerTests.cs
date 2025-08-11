using System;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using Orleans.GpuBridge.Abstractions;
using Orleans.GpuBridge.Runtime;
using Xunit;

namespace Orleans.GpuBridge.Tests;

public class DeviceBrokerTests : IDisposable
{
    private readonly DeviceBroker _broker;
    private readonly ILogger<DeviceBroker> _logger;
    private readonly GpuBridgeOptions _options;

    public DeviceBrokerTests()
    {
        _logger = new TestLogger<DeviceBroker>();
        _options = new GpuBridgeOptions
        {
            PreferGpu = true,
            MemoryPoolSizeMB = 1024,
            Telemetry = new TelemetryOptions { EnableMetrics = true }
        };
        _broker = new DeviceBroker(_logger, Options.Create(_options));
    }

    [Fact]
    public async Task InitializeAsync_Should_Initialize_Broker()
    {
        // Act
        await _broker.InitializeAsync(CancellationToken.None);

        // Assert
        Assert.True(_broker.DeviceCount > 0);
        Assert.True(_broker.TotalMemoryBytes > 0);
    }

    [Fact]
    public async Task InitializeAsync_Should_Always_Include_CPU_Device()
    {
        // Act
        await _broker.InitializeAsync(CancellationToken.None);
        var devices = _broker.GetDevices();

        // Assert
        Assert.NotNull(devices);
        Assert.Contains(devices, d => d.Type == DeviceType.Cpu);
    }

    [Fact]
    public async Task GetBestDevice_Should_Return_Device_With_Highest_Score()
    {
        // Arrange
        await _broker.InitializeAsync(CancellationToken.None);

        // Act
        var device = _broker.GetBestDevice();

        // Assert
        Assert.NotNull(device);
        Assert.True(device.TotalMemoryBytes > 0);
        Assert.True(device.ComputeUnits > 0);
    }

    [Fact]
    public async Task GetDevice_Should_Return_Correct_Device_By_Index()
    {
        // Arrange
        await _broker.InitializeAsync(CancellationToken.None);
        var devices = _broker.GetDevices();

        // Act & Assert
        foreach (var device in devices)
        {
            var retrieved = _broker.GetDevice(device.Index);
            Assert.NotNull(retrieved);
            Assert.Equal(device.Index, retrieved.Index);
            Assert.Equal(device.Name, retrieved.Name);
        }
    }

    [Fact]
    public async Task GetDevice_Should_Return_Null_For_Invalid_Index()
    {
        // Arrange
        await _broker.InitializeAsync(CancellationToken.None);

        // Act
        var device = _broker.GetDevice(-1);
        var device2 = _broker.GetDevice(999);

        // Assert
        Assert.Null(device);
        Assert.Null(device2);
    }

    [Fact]
    public async Task CurrentQueueDepth_Should_Return_Zero_Initially()
    {
        // Arrange
        await _broker.InitializeAsync(CancellationToken.None);

        // Act
        var depth = _broker.CurrentQueueDepth;

        // Assert
        Assert.Equal(0, depth);
    }

    [Fact]
    public async Task ShutdownAsync_Should_Clear_All_Resources()
    {
        // Arrange
        await _broker.InitializeAsync(CancellationToken.None);
        var initialCount = _broker.DeviceCount;

        // Act
        await _broker.ShutdownAsync(CancellationToken.None);

        // Assert
        Assert.Empty(_broker.GetDevices());
    }

    [Fact]
    public async Task Multiple_Initialization_Should_Be_Idempotent()
    {
        // Act
        await _broker.InitializeAsync(CancellationToken.None);
        var count1 = _broker.DeviceCount;
        
        await _broker.InitializeAsync(CancellationToken.None);
        var count2 = _broker.DeviceCount;

        // Assert
        Assert.Equal(count1, count2);
    }

    [Fact]
    public void GetDevices_Without_Initialization_Should_Throw()
    {
        // Act & Assert
        Assert.Throws<InvalidOperationException>(() => _broker.GetDevices());
    }

    [Fact]
    public void GetBestDevice_Without_Initialization_Should_Throw()
    {
        // Act & Assert
        Assert.Throws<InvalidOperationException>(() => _broker.GetBestDevice());
    }

    [Fact]
    public async Task Initialization_With_Cancellation_Should_Respect_Token()
    {
        // Arrange
        using var cts = new CancellationTokenSource();
        cts.Cancel();

        // Act & Assert
        await Assert.ThrowsAsync<OperationCanceledException>(
            () => _broker.InitializeAsync(cts.Token));
    }

    [Fact]
    public async Task Device_Detection_Should_Include_Capabilities()
    {
        // Arrange
        await _broker.InitializeAsync(CancellationToken.None);

        // Act
        var devices = _broker.GetDevices();

        // Assert
        foreach (var device in devices)
        {
            Assert.NotNull(device.Capabilities);
            Assert.NotEmpty(device.Capabilities);
            
            if (device.Type == DeviceType.Cpu)
            {
                Assert.Contains("CPU", device.Capabilities);
            }
        }
    }

    [Fact]
    public async Task Device_Should_Have_Valid_Properties()
    {
        // Arrange
        await _broker.InitializeAsync(CancellationToken.None);

        // Act
        var devices = _broker.GetDevices();

        // Assert
        foreach (var device in devices)
        {
            Assert.True(device.Index >= 0);
            Assert.NotEmpty(device.Name);
            Assert.True(device.TotalMemoryBytes > 0);
            Assert.True(device.ComputeUnits > 0);
        }
    }

    [Fact]
    public async Task Dispose_Should_Cleanup_Resources()
    {
        // Arrange
        await _broker.InitializeAsync(CancellationToken.None);

        // Act
        _broker.Dispose();
        _broker.Dispose(); // Second dispose should not throw

        // Assert
        Assert.Throws<InvalidOperationException>(() => _broker.GetDevices());
    }

    public void Dispose()
    {
        _broker?.Dispose();
    }
}

internal class TestLogger<T> : ILogger<T>
{
    public IDisposable? BeginScope<TState>(TState state) where TState : notnull => new NoopDisposable();
    public bool IsEnabled(LogLevel logLevel) => true;
    public void Log<TState>(LogLevel logLevel, EventId eventId, TState state, 
        Exception? exception, Func<TState, Exception?, string> formatter)
    {
        // Capture logs for testing if needed
    }

    private class NoopDisposable : IDisposable
    {
        public void Dispose() { }
    }
}
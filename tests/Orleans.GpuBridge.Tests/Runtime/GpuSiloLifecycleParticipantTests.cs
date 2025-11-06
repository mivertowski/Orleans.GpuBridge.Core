using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using FluentAssertions;
using Microsoft.Extensions.Logging.Abstractions;
using Moq;
using Orleans;
using Orleans.GpuBridge.Abstractions;
using Orleans.GpuBridge.Abstractions.Capacity;
using Orleans.GpuBridge.Abstractions.Enums;
using Orleans.GpuBridge.Runtime;
using Orleans.GpuBridge.Runtime.Infrastructure;
using Orleans.Runtime;
using Orleans.TestingHost;

namespace Orleans.GpuBridge.Tests.Runtime;

/// <summary>
/// Unit tests for GPU silo lifecycle participant
/// </summary>
public class GpuSiloLifecycleParticipantTests
{
    private readonly Mock<IGrainFactory> _mockGrainFactory;
    private readonly Mock<IGpuCapacityGrain> _mockCapacityGrain;
    private readonly Mock<ILocalSiloDetails> _mockSiloDetails;
    private readonly Mock<DeviceBroker> _mockDeviceBroker;
    private readonly GpuSiloLifecycleParticipant _participant;
    private readonly SiloAddress _testSiloAddress;

    public GpuSiloLifecycleParticipantTests()
    {
        _testSiloAddress = SiloAddress.New(
            new System.Net.IPEndPoint(System.Net.IPAddress.Loopback, 11111), 0);

        _mockGrainFactory = new Mock<IGrainFactory>();
        _mockCapacityGrain = new Mock<IGpuCapacityGrain>();
        _mockSiloDetails = new Mock<ILocalSiloDetails>();

        // Create mock for DeviceBroker (requires ILogger and IOptions)
        var mockLogger = NullLogger<DeviceBroker>.Instance;
        var mockOptions = Microsoft.Extensions.Options.Options.Create(
            new GpuBridgeOptions());
        _mockDeviceBroker = new Mock<DeviceBroker>(mockLogger, mockOptions);

        _mockGrainFactory
            .Setup(f => f.GetGrain<IGpuCapacityGrain>(0, null))
            .Returns(_mockCapacityGrain.Object);

        _mockSiloDetails
            .Setup(s => s.SiloAddress)
            .Returns(_testSiloAddress);

        _participant = new GpuSiloLifecycleParticipant(
            NullLogger<GpuSiloLifecycleParticipant>.Instance,
            _mockGrainFactory.Object,
            _mockSiloDetails.Object,
            _mockDeviceBroker.Object);
    }

    private static List<GpuDevice> CreateTestGpuDevices(int count = 1)
    {
        var devices = new List<GpuDevice>();
        for (int i = 0; i < count; i++)
        {
            devices.Add(new GpuDevice(
                Index: i,
                Name: $"Test GPU {i}",
                Type: DeviceType.CUDA,
                TotalMemoryBytes: 8192L * 1024 * 1024, // 8GB
                AvailableMemoryBytes: 6000L * 1024 * 1024, // 6GB available
                ComputeUnits: 3584,
                Capabilities: new List<string> { "Compute 8.9", "Tensor Cores" }));
        }
        return devices;
    }

    [Fact]
    public void Participate_Should_RegisterWithSiloLifecycle()
    {
        // Arrange
        var mockLifecycle = new Mock<ISiloLifecycle>();

        // Act
        _participant.Participate(mockLifecycle.Object);

        // Assert
        mockLifecycle.Verify(
            l => l.Subscribe(
                nameof(GpuSiloLifecycleParticipant),
                ServiceLifecycleStage.ApplicationServices,
                It.IsAny<Func<CancellationToken, Task>>(),
                It.IsAny<Func<CancellationToken, Task>>()),
            Times.Once);
    }

    [Fact]
    public async Task OnStart_Should_RegisterGpuCapacity_WhenGpuDevicesAvailable()
    {
        // Arrange
        var gpuDevices = CreateTestGpuDevices(2);
        _mockDeviceBroker
            .Setup(b => b.GetDevices())
            .Returns(gpuDevices);

        _mockDeviceBroker
            .Setup(b => b.CurrentQueueDepth)
            .Returns(5);

        var mockLifecycle = new Mock<ISiloLifecycle>();
        Func<CancellationToken, Task>? onStartCallback = null;

        mockLifecycle
            .Setup(l => l.Subscribe(
                It.IsAny<string>(),
                It.IsAny<int>(),
                It.IsAny<Func<CancellationToken, Task>>(),
                It.IsAny<Func<CancellationToken, Task>>()))
            .Callback<string, int, Func<CancellationToken, Task>, Func<CancellationToken, Task>>(
                (name, stage, onStart, onStop) => onStartCallback = onStart);

        _participant.Participate(mockLifecycle.Object);

        // Act
        await onStartCallback!.Invoke(CancellationToken.None);

        // Assert
        _mockCapacityGrain.Verify(
            g => g.RegisterSiloAsync(
                _testSiloAddress,
                It.Is<GpuCapacity>(c =>
                    c.DeviceCount == 2 &&
                    c.TotalMemoryMB == 16384 && // 2 * 8GB
                    c.AvailableMemoryMB == 12000 && // 2 * 6GB
                    c.QueueDepth == 5 &&
                    c.Backend == "CUDA")),
            Times.Once);
    }

    [Fact]
    public async Task OnStart_Should_RegisterNoneCapacity_WhenNoGpuDevices()
    {
        // Arrange
        var cpuDevice = new GpuDevice(
            Index: 0,
            Name: "CPU",
            Type: DeviceType.CPU, // CPU device should be filtered
            TotalMemoryBytes: 1024L * 1024 * 1024,
            AvailableMemoryBytes: 512L * 1024 * 1024,
            ComputeUnits: 16,
            Capabilities: new List<string>());

        _mockDeviceBroker
            .Setup(b => b.GetDevices())
            .Returns(new List<GpuDevice> { cpuDevice });

        var mockLifecycle = new Mock<ISiloLifecycle>();
        Func<CancellationToken, Task>? onStartCallback = null;

        mockLifecycle
            .Setup(l => l.Subscribe(
                It.IsAny<string>(),
                It.IsAny<int>(),
                It.IsAny<Func<CancellationToken, Task>>(),
                It.IsAny<Func<CancellationToken, Task>>()))
            .Callback<string, int, Func<CancellationToken, Task>, Func<CancellationToken, Task>>(
                (name, stage, onStart, onStop) => onStartCallback = onStart);

        _participant.Participate(mockLifecycle.Object);

        // Act
        await onStartCallback!.Invoke(CancellationToken.None);

        // Assert
        _mockCapacityGrain.Verify(
            g => g.RegisterSiloAsync(
                _testSiloAddress,
                It.Is<GpuCapacity>(c => c.DeviceCount == 0)),
            Times.Once);
    }

    [Fact]
    public async Task OnStop_Should_UnregisterGpuCapacity()
    {
        // Arrange
        var gpuDevices = CreateTestGpuDevices();
        _mockDeviceBroker
            .Setup(b => b.GetDevices())
            .Returns(gpuDevices);

        var mockLifecycle = new Mock<ISiloLifecycle>();
        Func<CancellationToken, Task>? onStartCallback = null;
        Func<CancellationToken, Task>? onStopCallback = null;

        mockLifecycle
            .Setup(l => l.Subscribe(
                It.IsAny<string>(),
                It.IsAny<int>(),
                It.IsAny<Func<CancellationToken, Task>>(),
                It.IsAny<Func<CancellationToken, Task>>()))
            .Callback<string, int, Func<CancellationToken, Task>, Func<CancellationToken, Task>>(
                (name, stage, onStart, onStop) =>
                {
                    onStartCallback = onStart;
                    onStopCallback = onStop;
                });

        _participant.Participate(mockLifecycle.Object);

        // Act - Start then stop
        await onStartCallback!.Invoke(CancellationToken.None);
        await onStopCallback!.Invoke(CancellationToken.None);

        // Assert
        _mockCapacityGrain.Verify(
            g => g.UnregisterSiloAsync(_testSiloAddress),
            Times.Once);
    }

    [Fact]
    public async Task OnStop_Should_NotThrow_WhenUnregistrationFails()
    {
        // Arrange
        var gpuDevices = CreateTestGpuDevices();
        _mockDeviceBroker
            .Setup(b => b.GetDevices())
            .Returns(gpuDevices);

        _mockCapacityGrain
            .Setup(g => g.UnregisterSiloAsync(It.IsAny<SiloAddress>()))
            .ThrowsAsync(new InvalidOperationException("Capacity grain unavailable"));

        var mockLifecycle = new Mock<ISiloLifecycle>();
        Func<CancellationToken, Task>? onStartCallback = null;
        Func<CancellationToken, Task>? onStopCallback = null;

        mockLifecycle
            .Setup(l => l.Subscribe(
                It.IsAny<string>(),
                It.IsAny<int>(),
                It.IsAny<Func<CancellationToken, Task>>(),
                It.IsAny<Func<CancellationToken, Task>>()))
            .Callback<string, int, Func<CancellationToken, Task>, Func<CancellationToken, Task>>(
                (name, stage, onStart, onStop) =>
                {
                    onStartCallback = onStart;
                    onStopCallback = onStop;
                });

        _participant.Participate(mockLifecycle.Object);

        await onStartCallback!.Invoke(CancellationToken.None);

        // Act & Assert - Should not throw
        await onStopCallback!.Invoking(cb => cb(CancellationToken.None))
            .Should().NotThrowAsync();
    }

    [Fact]
    public async Task OnStart_Should_CalculateCorrectMemoryInMB()
    {
        // Arrange
        var device = new GpuDevice(
            Index: 0,
            Name: "Test GPU",
            Type: DeviceType.CUDA,
            TotalMemoryBytes: 8_589_934_592L, // 8192 MB exactly
            AvailableMemoryBytes: 4_294_967_296L, // 4096 MB exactly
            ComputeUnits: 3584,
            Capabilities: new List<string>());

        _mockDeviceBroker
            .Setup(b => b.GetDevices())
            .Returns(new List<GpuDevice> { device });

        var mockLifecycle = new Mock<ISiloLifecycle>();
        Func<CancellationToken, Task>? onStartCallback = null;

        mockLifecycle
            .Setup(l => l.Subscribe(
                It.IsAny<string>(),
                It.IsAny<int>(),
                It.IsAny<Func<CancellationToken, Task>>(),
                It.IsAny<Func<CancellationToken, Task>>()))
            .Callback<string, int, Func<CancellationToken, Task>, Func<CancellationToken, Task>>(
                (name, stage, onStart, onStop) => onStartCallback = onStart);

        _participant.Participate(mockLifecycle.Object);

        // Act
        await onStartCallback!.Invoke(CancellationToken.None);

        // Assert
        _mockCapacityGrain.Verify(
            g => g.RegisterSiloAsync(
                _testSiloAddress,
                It.Is<GpuCapacity>(c =>
                    c.TotalMemoryMB == 8192 &&
                    c.AvailableMemoryMB == 4096)),
            Times.Once);
    }

    [Fact]
    public async Task OnStart_Should_UseCorrectBackendFromDeviceType()
    {
        // Arrange
        var openclDevice = new GpuDevice(
            Index: 0,
            Name: "AMD GPU",
            Type: DeviceType.OpenCL, // OpenCL device
            TotalMemoryBytes: 4096L * 1024 * 1024,
            AvailableMemoryBytes: 2048L * 1024 * 1024,
            ComputeUnits: 2560,
            Capabilities: new List<string>());

        _mockDeviceBroker
            .Setup(b => b.GetDevices())
            .Returns(new List<GpuDevice> { openclDevice });

        var mockLifecycle = new Mock<ISiloLifecycle>();
        Func<CancellationToken, Task>? onStartCallback = null;

        mockLifecycle
            .Setup(l => l.Subscribe(
                It.IsAny<string>(),
                It.IsAny<int>(),
                It.IsAny<Func<CancellationToken, Task>>(),
                It.IsAny<Func<CancellationToken, Task>>()))
            .Callback<string, int, Func<CancellationToken, Task>, Func<CancellationToken, Task>>(
                (name, stage, onStart, onStop) => onStartCallback = onStart);

        _participant.Participate(mockLifecycle.Object);

        // Act
        await onStartCallback!.Invoke(CancellationToken.None);

        // Assert
        _mockCapacityGrain.Verify(
            g => g.RegisterSiloAsync(
                _testSiloAddress,
                It.Is<GpuCapacity>(c => c.Backend == "OpenCL")),
            Times.Once);
    }

    [Fact]
    public async Task OnStart_Should_AggregateMultipleGpuDevices()
    {
        // Arrange
        var devices = new List<GpuDevice>
        {
            new GpuDevice(
                Index: 0,
                Name: "GPU 0",
                Type: DeviceType.CUDA,
                TotalMemoryBytes: 8192L * 1024 * 1024,
                AvailableMemoryBytes: 6000L * 1024 * 1024,
                ComputeUnits: 3584,
                Capabilities: new List<string>()),
            new GpuDevice(
                Index: 1,
                Name: "GPU 1",
                Type: DeviceType.CUDA,
                TotalMemoryBytes: 16384L * 1024 * 1024,
                AvailableMemoryBytes: 10000L * 1024 * 1024,
                ComputeUnits: 5120,
                Capabilities: new List<string>())
        };

        _mockDeviceBroker
            .Setup(b => b.GetDevices())
            .Returns(devices);

        var mockLifecycle = new Mock<ISiloLifecycle>();
        Func<CancellationToken, Task>? onStartCallback = null;

        mockLifecycle
            .Setup(l => l.Subscribe(
                It.IsAny<string>(),
                It.IsAny<int>(),
                It.IsAny<Func<CancellationToken, Task>>(),
                It.IsAny<Func<CancellationToken, Task>>()))
            .Callback<string, int, Func<CancellationToken, Task>, Func<CancellationToken, Task>>(
                (name, stage, onStart, onStop) => onStartCallback = onStart);

        _participant.Participate(mockLifecycle.Object);

        // Act
        await onStartCallback!.Invoke(CancellationToken.None);

        // Assert
        _mockCapacityGrain.Verify(
            g => g.RegisterSiloAsync(
                _testSiloAddress,
                It.Is<GpuCapacity>(c =>
                    c.DeviceCount == 2 &&
                    c.TotalMemoryMB == 24576 && // 8192 + 16384
                    c.AvailableMemoryMB == 16000)), // 6000 + 10000
            Times.Once);
    }
}

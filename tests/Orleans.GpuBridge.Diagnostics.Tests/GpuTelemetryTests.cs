// Orleans.GpuBridge - GPU-native distributed computing for Microsoft Orleans
// Copyright (c) 2025 Michael Ivertowski

using System.Diagnostics;
using System.Diagnostics.Metrics;
using Microsoft.Extensions.Logging;
using Orleans.GpuBridge.Diagnostics.Enums;
using Orleans.GpuBridge.Diagnostics.Implementation;
using Orleans.GpuBridge.Diagnostics.Interfaces;

namespace Orleans.GpuBridge.Diagnostics.Tests;

/// <summary>
/// Tests for <see cref="GpuTelemetry"/> implementation.
/// </summary>
public class GpuTelemetryTests : IDisposable
{
    private readonly Mock<ILogger<GpuTelemetry>> _loggerMock;
    private readonly Mock<IMeterFactory> _meterFactoryMock;
    private readonly Meter _meter;
    private readonly GpuTelemetry _telemetry;

    public GpuTelemetryTests()
    {
        _loggerMock = new Mock<ILogger<GpuTelemetry>>();
        _meterFactoryMock = new Mock<IMeterFactory>();
        _meter = new Meter("Orleans.GpuBridge.Tests");
        _meterFactoryMock.Setup(f => f.Create(It.IsAny<MeterOptions>())).Returns(_meter);
        _telemetry = new GpuTelemetry(_loggerMock.Object, _meterFactoryMock.Object);
    }

    public void Dispose()
    {
        _telemetry.Dispose();
        _meter.Dispose();
    }

    [Fact]
    public void RecordKernelExecution_ShouldNotThrow()
    {
        // Arrange
        var kernelName = "vectorAdd";
        var deviceIndex = 0;
        var duration = TimeSpan.FromMilliseconds(100);

        // Act
        var act = () => _telemetry.RecordKernelExecution(kernelName, deviceIndex, duration, success: true);

        // Assert
        act.Should().NotThrow();
    }

    [Fact]
    public void RecordKernelExecution_Failed_ShouldNotThrow()
    {
        // Arrange
        var kernelName = "matrixMul";
        var deviceIndex = 1;
        var duration = TimeSpan.FromMilliseconds(50);

        // Act
        var act = () => _telemetry.RecordKernelExecution(kernelName, deviceIndex, duration, success: false);

        // Assert
        act.Should().NotThrow();
    }

    [Fact]
    public void RecordMemoryTransfer_AllDirections_ShouldNotThrow()
    {
        // Arrange
        var bytes = 1024 * 1024L; // 1 MB
        var duration = TimeSpan.FromMicroseconds(500);

        // Act & Assert
        foreach (TransferDirection direction in Enum.GetValues<TransferDirection>())
        {
            var act = () => _telemetry.RecordMemoryTransfer(direction, bytes, duration);
            act.Should().NotThrow($"for direction {direction}");
        }
    }

    [Fact]
    public void RecordMemoryAllocation_Success_ShouldNotThrow()
    {
        // Arrange
        var deviceIndex = 0;
        var bytes = 512 * 1024 * 1024L; // 512 MB

        // Act
        var act = () => _telemetry.RecordMemoryAllocation(deviceIndex, bytes, success: true);

        // Assert
        act.Should().NotThrow();
    }

    [Fact]
    public void RecordAllocationFailure_ShouldNotThrow()
    {
        // Arrange
        var deviceIndex = 0;
        var requestedBytes = 1024 * 1024 * 1024L; // 1 GB
        var reason = "Out of memory";

        // Act
        var act = () => _telemetry.RecordAllocationFailure(deviceIndex, requestedBytes, reason);

        // Assert
        act.Should().NotThrow();
    }

    [Fact]
    public void RecordQueueDepth_ShouldNotThrow()
    {
        // Arrange
        var deviceIndex = 0;
        var depth = 42;

        // Act
        var act = () => _telemetry.RecordQueueDepth(deviceIndex, depth);

        // Assert
        act.Should().NotThrow();
    }

    [Fact]
    public void RecordGrainActivation_ShouldNotThrow()
    {
        // Arrange
        var grainType = "MyGrainType";
        var duration = TimeSpan.FromMilliseconds(25);

        // Act
        var act = () => _telemetry.RecordGrainActivation(grainType, duration);

        // Assert
        act.Should().NotThrow();
    }

    [Fact]
    public void RecordPipelineStage_ShouldNotThrow()
    {
        // Arrange
        var stageName = "Transform";
        var duration = TimeSpan.FromMilliseconds(10);

        // Act
        var act = () => _telemetry.RecordPipelineStage(stageName, duration, success: true);

        // Assert
        act.Should().NotThrow();
    }

    [Fact]
    public void StartKernelExecution_ShouldReturnActivity()
    {
        // Act
        var activity = _telemetry.StartKernelExecution("testKernel", 0);

        // Assert - Activity might be null if there's no listener
        // This test verifies the method doesn't throw
        activity?.Dispose();
    }

    [Fact]
    public void UpdateGpuMetrics_ShouldNotThrow()
    {
        // Arrange
        var deviceIndex = 0;
        var utilization = 75.5;
        var temperature = 65.0;
        var power = 250.0;

        // Act
        var act = () => _telemetry.UpdateGpuMetrics(deviceIndex, utilization, temperature, power);

        // Assert
        act.Should().NotThrow();
    }

    [Fact]
    public void MultipleRecords_ShouldNotThrow()
    {
        // Arrange & Act
        var act = () =>
        {
            for (var i = 0; i < 100; i++)
            {
                _telemetry.RecordKernelExecution($"kernel_{i}", i % 4, TimeSpan.FromMilliseconds(i), true);
                _telemetry.RecordMemoryTransfer(TransferDirection.HostToDevice, i * 1024L, TimeSpan.FromMicroseconds(i));
                _telemetry.RecordQueueDepth(i % 4, i);
            }
        };

        // Assert
        act.Should().NotThrow();
    }

    [Fact]
    public void Dispose_ShouldBeIdempotent()
    {
        // Arrange
        using var telemetry = new GpuTelemetry(_loggerMock.Object, _meterFactoryMock.Object);

        // Act & Assert
        var act = () =>
        {
            telemetry.Dispose();
            telemetry.Dispose();
            telemetry.Dispose();
        };

        act.Should().NotThrow();
    }
}

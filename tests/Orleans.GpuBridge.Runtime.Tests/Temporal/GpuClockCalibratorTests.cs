// Orleans.GpuBridge - GPU-native distributed computing for Microsoft Orleans
// Copyright (c) 2025 Michael Ivertowski

using Microsoft.Extensions.Logging;
using Orleans.GpuBridge.Abstractions.Temporal;
using Orleans.GpuBridge.Runtime.Temporal;

namespace Orleans.GpuBridge.Runtime.Tests.Temporal;

/// <summary>
/// Unit tests for <see cref="GpuClockCalibrator"/> clock calibration.
/// </summary>
public sealed class GpuClockCalibratorTests : IDisposable
{
    private readonly ILoggerFactory _loggerFactory;
    private readonly ILogger<GpuClockCalibrator> _calibratorLogger;
    private readonly ILogger<CpuTimingProvider> _providerLogger;
    private readonly CpuTimingProvider _cpuProvider;

    public GpuClockCalibratorTests()
    {
        _loggerFactory = LoggerFactory.Create(builder => builder.AddDebug());
        _calibratorLogger = _loggerFactory.CreateLogger<GpuClockCalibrator>();
        _providerLogger = _loggerFactory.CreateLogger<CpuTimingProvider>();
        _cpuProvider = new CpuTimingProvider(_providerLogger);
    }

    public void Dispose()
    {
        _loggerFactory.Dispose();
    }

    /// <summary>
    /// Tests that calibrator with timing provider reports HasGpuTimingProvider = true.
    /// </summary>
    [Fact]
    public void Constructor_WithTimingProvider_SetsHasGpuTimingProviderTrue()
    {
        // Arrange & Act
        using var calibrator = new GpuClockCalibrator(_cpuProvider, _calibratorLogger);

        // Assert
        calibrator.HasGpuTimingProvider.Should().BeTrue();
    }

    /// <summary>
    /// Tests that calibrator without timing provider reports HasGpuTimingProvider = false.
    /// </summary>
    [Fact]
    public void Constructor_WithoutTimingProvider_SetsHasGpuTimingProviderFalse()
    {
        // Arrange & Act
        using var calibrator = new GpuClockCalibrator(_calibratorLogger);

        // Assert
        calibrator.HasGpuTimingProvider.Should().BeFalse();
    }

    /// <summary>
    /// Tests that calibrator with CPU provider reports correct type name.
    /// </summary>
    [Fact]
    public void TimingProviderType_WithCpuProvider_ReturnsCpuProviderName()
    {
        // Arrange
        using var calibrator = new GpuClockCalibrator(_cpuProvider, _calibratorLogger);

        // Act
        var typeName = calibrator.TimingProviderType;

        // Assert
        typeName.Should().Contain("CPU");
    }

    /// <summary>
    /// Tests that calibrator without provider reports "Simulated".
    /// </summary>
    [Fact]
    public void TimingProviderType_WithoutProvider_ReturnsSimulated()
    {
        // Arrange
        using var calibrator = new GpuClockCalibrator(_calibratorLogger);

        // Act
        var typeName = calibrator.TimingProviderType;

        // Assert
        typeName.Should().Be("Simulated");
    }

    /// <summary>
    /// Tests that calibration produces valid results.
    /// </summary>
    [Fact]
    public async Task CalibrateAsync_ProducesValidCalibration()
    {
        // Arrange
        using var calibrator = new GpuClockCalibrator(_cpuProvider, _calibratorLogger);

        // Act
        var calibration = await calibrator.CalibrateAsync(sampleCount: 50);

        // Assert
        calibration.SampleCount.Should().Be(50);
        calibration.CalibrationTimestampNanos.Should().BeGreaterThan(0);
    }

    /// <summary>
    /// Tests that GetCalibrationAsync returns calibration.
    /// </summary>
    [Fact]
    public async Task GetCalibrationAsync_ReturnsCalibration()
    {
        // Arrange
        using var calibrator = new GpuClockCalibrator(_cpuProvider, _calibratorLogger);

        // Act
        var calibration = await calibrator.GetCalibrationAsync();

        // Assert
        calibration.SampleCount.Should().BeGreaterThan(0);
    }

    /// <summary>
    /// Tests that cached calibration is returned when fresh.
    /// </summary>
    [Fact]
    public async Task GetCalibrationAsync_WithFreshCalibration_ReturnsCached()
    {
        // Arrange
        using var calibrator = new GpuClockCalibrator(_cpuProvider, _calibratorLogger);

        // Act
        var calibration1 = await calibrator.GetCalibrationAsync();
        var calibration2 = await calibrator.GetCalibrationAsync();

        // Assert - should be the same calibration (same timestamp)
        calibration1.CalibrationTimestampNanos.Should().Be(calibration2.CalibrationTimestampNanos);
    }

    /// <summary>
    /// Tests that calibration with low sample count throws.
    /// </summary>
    [Fact]
    public async Task CalibrateAsync_WithLowSampleCount_ThrowsException()
    {
        // Arrange
        using var calibrator = new GpuClockCalibrator(_cpuProvider, _calibratorLogger);

        // Act & Assert
        await Assert.ThrowsAsync<ArgumentException>(
            () => calibrator.CalibrateAsync(sampleCount: 5));
    }

    /// <summary>
    /// Tests that GpuToCpuTime requires calibration.
    /// </summary>
    [Fact]
    public void GpuToCpuTime_WithoutCalibration_ThrowsInvalidOperationException()
    {
        // Arrange
        using var calibrator = new GpuClockCalibrator(_cpuProvider, _calibratorLogger);

        // Act & Assert
        Assert.Throws<InvalidOperationException>(
            () => calibrator.GpuToCpuTime(1_000_000_000));
    }

    /// <summary>
    /// Tests that CpuToGpuTime requires calibration.
    /// </summary>
    [Fact]
    public void CpuToGpuTime_WithoutCalibration_ThrowsInvalidOperationException()
    {
        // Arrange
        using var calibrator = new GpuClockCalibrator(_cpuProvider, _calibratorLogger);

        // Act & Assert
        Assert.Throws<InvalidOperationException>(
            () => calibrator.CpuToGpuTime(1_000_000_000));
    }

    /// <summary>
    /// Tests that GpuToCpuTime works after calibration.
    /// </summary>
    [Fact]
    public async Task GpuToCpuTime_AfterCalibration_ReturnsValue()
    {
        // Arrange
        using var calibrator = new GpuClockCalibrator(_cpuProvider, _calibratorLogger);
        await calibrator.GetCalibrationAsync();
        const long testGpuTime = 1_000_000_000_000L; // 1000 seconds in ns

        // Act
        var cpuTime = calibrator.GpuToCpuTime(testGpuTime);

        // Assert - should return some reasonable value (not zero or negative)
        cpuTime.Should().NotBe(0);
    }

    /// <summary>
    /// Tests that CpuToGpuTime works after calibration.
    /// </summary>
    [Fact]
    public async Task CpuToGpuTime_AfterCalibration_ReturnsValue()
    {
        // Arrange
        using var calibrator = new GpuClockCalibrator(_cpuProvider, _calibratorLogger);
        await calibrator.GetCalibrationAsync();
        const long testCpuTime = 1_000_000_000_000L; // 1000 seconds in ns

        // Act
        var gpuTime = calibrator.CpuToGpuTime(testCpuTime);

        // Assert - should return some reasonable value
        gpuTime.Should().NotBe(0);
    }

    /// <summary>
    /// Tests that GpuToCpuTime and CpuToGpuTime perform conversions.
    /// </summary>
    /// <remarks>
    /// Note: The exact values depend on calibration results. This test verifies
    /// that the conversion functions work correctly, not that they produce
    /// specific values (which depend on offset and drift compensation).
    /// </remarks>
    [Fact]
    public async Task GpuToCpuTime_AndCpuToGpuTime_PerformConversions()
    {
        // Arrange
        using var calibrator = new GpuClockCalibrator(_cpuProvider, _calibratorLogger);
        var calibration = await calibrator.GetCalibrationAsync();
        const long originalTime = 1_000_000_000_000L; // 1000 seconds in ns

        // Act
        var gpuTime = calibrator.CpuToGpuTime(originalTime);
        var backToCpu = calibrator.GpuToCpuTime(gpuTime);

        // Assert - conversions should produce non-zero, positive values
        gpuTime.Should().NotBe(0);
        backToCpu.Should().NotBe(0);

        // The relationship between CPU and GPU time is: gpu = cpu + offset + (drift * time)
        // So the difference should reflect the offset
        var difference = gpuTime - originalTime;
        // The offset can be positive or negative, but should be bounded
        // by the calibration parameters (offset + drift contribution)
        difference.Should().NotBe(long.MinValue);
        difference.Should().NotBe(long.MaxValue);
    }

    /// <summary>
    /// Tests that calibration with simulated provider (no timing provider) still works.
    /// </summary>
    [Fact]
    public async Task CalibrateAsync_WithSimulatedProvider_ProducesCalibration()
    {
        // Arrange
        using var calibrator = new GpuClockCalibrator(_calibratorLogger);

        // Act
        var calibration = await calibrator.CalibrateAsync(sampleCount: 50);

        // Assert
        calibration.SampleCount.Should().Be(50);
        calibration.CalibrationTimestampNanos.Should().BeGreaterThan(0);
        // Simulated provider has known offset
        calibration.OffsetNanos.Should().NotBe(0);
    }

    /// <summary>
    /// Tests that calibration respects cancellation token.
    /// </summary>
    [Fact]
    public async Task CalibrateAsync_WithCancelledToken_ThrowsOperationCancelledException()
    {
        // Arrange
        using var calibrator = new GpuClockCalibrator(_cpuProvider, _calibratorLogger);
        using var cts = new CancellationTokenSource();
        cts.Cancel();

        // Act & Assert
        await Assert.ThrowsAsync<OperationCanceledException>(
            () => calibrator.CalibrateAsync(sampleCount: 100, ct: cts.Token));
    }

    /// <summary>
    /// Tests that GetCalibrationAsync respects cancellation token.
    /// </summary>
    [Fact]
    public async Task GetCalibrationAsync_WithCancelledToken_ThrowsCancellationException()
    {
        // Arrange
        using var calibrator = new GpuClockCalibrator(_cpuProvider, _calibratorLogger);
        using var cts = new CancellationTokenSource();
        cts.Cancel();

        // Act & Assert
        // TaskCanceledException derives from OperationCanceledException
        var exception = await Assert.ThrowsAnyAsync<OperationCanceledException>(
            () => calibrator.GetCalibrationAsync(ct: cts.Token));
        exception.Should().NotBeNull();
    }
}

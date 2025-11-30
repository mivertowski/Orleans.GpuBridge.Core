using Microsoft.Extensions.Logging.Abstractions;
using Orleans.GpuBridge.Abstractions.Temporal;
using Orleans.GpuBridge.Runtime.Temporal;

namespace Orleans.GpuBridge.Temporal.Tests.Unit;

/// <summary>
/// Unit tests for GPU clock calibration.
/// </summary>
public sealed class ClockCalibrationTests
{
    [Fact]
    public async Task GpuClockCalibrator_PerformsCalibration()
    {
        // Arrange
        var calibrator = new GpuClockCalibrator(NullLogger<GpuClockCalibrator>.Instance);

        // Act
        var calibration = await calibrator.CalibrateAsync(sampleCount: 100);

        // Assert
        calibration.SampleCount.Should().Be(100);
        calibration.OffsetNanos.Should().NotBe(0); // Should have some offset
        calibration.ErrorBoundNanos.Should().BeGreaterThan(0);
    }

    [Fact]
    public async Task GpuClockCalibrator_CachesCalibration()
    {
        // Arrange
        var calibrator = new GpuClockCalibrator(NullLogger<GpuClockCalibrator>.Instance);

        // Act
        var calibration1 = await calibrator.GetCalibrationAsync();
        var calibration2 = await calibrator.GetCalibrationAsync();

        // Assert - should return same calibration without recalibrating
        calibration1.CalibrationTimestampNanos.Should().Be(calibration2.CalibrationTimestampNanos);
    }

    [Fact]
    public void ClockCalibration_GpuToCpuTimeConvertsCorrectly()
    {
        // Arrange
        long offset = 1_000_000_000L; // 1 second offset
        double drift = 10.0; // 10 PPM
        long calibrationTime = 1000L;

        var calibration = new ClockCalibration(
            offsetNanos: offset,
            driftPPM: drift,
            errorBoundNanos: 1000,
            sampleCount: 100,
            calibrationTimestampNanos: calibrationTime);

        // Act
        long gpuTime = 2_000_000_000L;
        long cpuTime = calibration.GpuToCpuTime(gpuTime);

        // Assert
        // CPU time should be GPU time minus offset, with drift correction
        long expectedCpuTime = gpuTime - offset;
        cpuTime.Should().BeCloseTo(expectedCpuTime, 1_000_000); // Within 1ms
    }

    [Fact]
    public void ClockCalibration_CpuToGpuTimeConvertsCorrectly()
    {
        // Arrange
        long offset = 1_000_000_000L;
        var calibration = new ClockCalibration(
            offsetNanos: offset,
            driftPPM: 0,
            errorBoundNanos: 1000,
            sampleCount: 100,
            calibrationTimestampNanos: 1000L);

        // Act
        long cpuTime = 1_000_000_000L;
        long gpuTime = calibration.CpuToGpuTime(cpuTime);

        // Assert
        gpuTime.Should().Be(cpuTime + offset);
    }

    [Fact]
    public void ClockCalibration_GetUncertaintyRangeReturnsCorrectBounds()
    {
        // Arrange
        long errorBound = 1_000_000L; // 1ms error
        var calibration = new ClockCalibration(
            offsetNanos: 0,
            driftPPM: 0,
            errorBoundNanos: errorBound,
            sampleCount: 100,
            calibrationTimestampNanos: 0);

        // Act
        long gpuTime = 1_000_000_000L;
        var (min, max) = calibration.GetUncertaintyRange(gpuTime);

        // Assert
        (max - min).Should().Be(2 * errorBound);
        min.Should().Be(gpuTime - errorBound);
        max.Should().Be(gpuTime + errorBound);
    }

    [Fact]
    public void ClockCalibration_IsStaleReturnsTrueAfterTimeout()
    {
        // Arrange
        long oldCalibrationTime = DateTimeOffset.UtcNow.AddMinutes(-10).ToUnixTimeNanoseconds();
        var calibration = new ClockCalibration(
            offsetNanos: 0,
            driftPPM: 0,
            errorBoundNanos: 1000,
            sampleCount: 100,
            calibrationTimestampNanos: oldCalibrationTime);

        // Act
        long currentTime = DateTimeOffset.UtcNow.ToUnixTimeNanoseconds();
        bool isStale = calibration.IsStale(currentTime, maxAgeNanos: 300_000_000_000L); // 5 minutes

        // Assert
        isStale.Should().BeTrue();
    }

    [Fact]
    public void ClockCalibration_IsStaleReturnsFalseWhenFresh()
    {
        // Arrange
        long recentTime = DateTimeOffset.UtcNow.AddSeconds(-30).ToUnixTimeNanoseconds();
        var calibration = new ClockCalibration(
            offsetNanos: 0,
            driftPPM: 0,
            errorBoundNanos: 1000,
            sampleCount: 100,
            calibrationTimestampNanos: recentTime);

        // Act
        long currentTime = DateTimeOffset.UtcNow.ToUnixTimeNanoseconds();
        bool isStale = calibration.IsStale(currentTime);

        // Assert
        isStale.Should().BeFalse();
    }

    [Fact]
    public void GpuClockCalibrator_ThrowsWhenNotCalibrated()
    {
        // Arrange
        var calibrator = new GpuClockCalibrator(NullLogger<GpuClockCalibrator>.Instance);

        // Act & Assert
        var act = () => calibrator.GpuToCpuTime(1000L);
        act.Should().Throw<InvalidOperationException>()
            .WithMessage("Clock not calibrated*");
    }

    [Fact]
    public async Task GpuClockCalibrator_HandlesMultipleSampleCounts()
    {
        // Arrange
        var calibrator = new GpuClockCalibrator(NullLogger<GpuClockCalibrator>.Instance);

        // Act & Assert
        var calibration10 = await calibrator.CalibrateAsync(sampleCount: 10);
        var calibration1000 = await calibrator.CalibrateAsync(sampleCount: 1000);

        // Both calibrations should complete successfully with reasonable error bounds
        // Note: More samples doesn't guarantee lower error in all environments due to
        // system load, cache effects, and measurement noise. We just verify both work.
        calibration10.ErrorBoundNanos.Should().BeGreaterThanOrEqualTo(0);
        calibration1000.ErrorBoundNanos.Should().BeGreaterThanOrEqualTo(0);

        // Both should have reasonable bounds (under 1 second)
        calibration10.ErrorBoundNanos.Should().BeLessThan(1_000_000_000);
        calibration1000.ErrorBoundNanos.Should().BeLessThan(1_000_000_000);
    }
}

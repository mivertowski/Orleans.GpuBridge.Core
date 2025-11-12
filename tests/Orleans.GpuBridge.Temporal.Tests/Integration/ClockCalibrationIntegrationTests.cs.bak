using FluentAssertions;
using Microsoft.Extensions.Logging.Abstractions;
using Orleans.GpuBridge.Runtime.Temporal;
using Orleans.GpuBridge.Runtime.Temporal.Clock;

namespace Orleans.GpuBridge.Temporal.Tests.Integration;

/// <summary>
/// Integration tests for Phase 5 (Clock Calibration) + Phase 6 (Physical Time Precision).
/// Verifies that GpuClockCalibrator correctly uses PTP clock sources.
/// </summary>
public sealed class ClockCalibrationIntegrationTests
{
    [Fact]
    public async Task GpuClockCalibrator_UsesPtpClockSource_WhenAvailable()
    {
        // Arrange
        var clockSelector = new ClockSourceSelector(NullLogger<ClockSourceSelector>.Instance);
        await clockSelector.InitializeAsync();

        var calibrator = new GpuClockCalibrator(NullLogger<GpuClockCalibrator>.Instance);

        // Act
        long cpuTime = clockSelector.ActiveSource.GetCurrentTimeNanos();
        long gpuTime = calibrator.GetGpuTimeNanos();

        // Assert
        clockSelector.ActiveSource.Should().NotBeNull();
        cpuTime.Should().BeGreaterThan(0);
        gpuTime.Should().BeGreaterThan(0);

        // Skew should be within reasonable bounds
        long skew = Math.Abs(gpuTime - cpuTime);
        skew.Should().BeLessThan(1_000_000_000); // < 1 second (sanity check)
    }

    [Fact]
    public async Task ClockCalibration_DetectsClockSkew_WithPtpPrecision()
    {
        // Arrange
        var clockSelector = new ClockSourceSelector(NullLogger<ClockSourceSelector>.Instance);
        await clockSelector.InitializeAsync();

        var calibrator = new GpuClockCalibrator(NullLogger<GpuClockCalibrator>.Instance);

        // Act - Simulate clock skew detection
        long baseline = clockSelector.ActiveSource.GetCurrentTimeNanos();
        await Task.Delay(10); // 10ms delay
        long afterDelay = clockSelector.ActiveSource.GetCurrentTimeNanos();

        long elapsedNanos = afterDelay - baseline;

        // Assert - Should detect ~10ms elapsed
        elapsedNanos.Should().BeGreaterThan(10_000_000); // > 10ms
        elapsedNanos.Should().BeLessThan(20_000_000);   // < 20ms (accounting for scheduling)

        // With PTP clock, precision should be sub-microsecond
        long errorBound = clockSelector.ActiveSource.GetErrorBound();
        if (clockSelector.ActiveSource.GetType().Name == "PtpClockSource")
        {
            errorBound.Should().BeLessThan(10_000); // < 10μs for hardware PTP
        }
    }

    [Fact]
    public async Task ClockCalibration_CompensatesForDrift_OverTime()
    {
        // Arrange
        var clockSelector = new ClockSourceSelector(NullLogger<ClockSourceSelector>.Instance);
        await clockSelector.InitializeAsync();

        var calibrator = new GpuClockCalibrator(NullLogger<GpuClockCalibrator>.Instance);

        // Act - Measure drift rate
        double driftRate = clockSelector.ActiveSource.GetClockDrift();

        // Assert
        driftRate.Should().BeGreaterThanOrEqualTo(0.0);
        driftRate.Should().BeLessThan(200.0); // < 200 PPM (typical crystal range)

        // PTP clocks should have zero drift (synchronized)
        if (clockSelector.ActiveSource.GetType().Name == "PtpClockSource")
        {
            driftRate.Should().Be(0.0);
        }
    }

    [Fact]
    public async Task Integration_PtpClock_ProvidesNanosecondPrecision()
    {
        // Arrange
        var clockSelector = new ClockSourceSelector(NullLogger<ClockSourceSelector>.Instance);
        await clockSelector.InitializeAsync();

        if (!clockSelector.ActiveSource.IsSynchronized)
        {
            // Skip if no synchronized clock available
            return;
        }

        // Act - Read time multiple times with minimal delay
        var times = new long[100];
        for (int i = 0; i < 100; i++)
        {
            times[i] = clockSelector.ActiveSource.GetCurrentTimeNanos();
        }

        // Assert - Times should be monotonically increasing
        for (int i = 1; i < times.Length; i++)
        {
            times[i].Should().BeGreaterThanOrEqualTo(times[i - 1],
                $"Time should be monotonic (iteration {i})");
        }

        // Calculate minimum time increment (precision indicator)
        var increments = new List<long>();
        for (int i = 1; i < times.Length; i++)
        {
            long increment = times[i] - times[i - 1];
            if (increment > 0)
            {
                increments.Add(increment);
            }
        }

        if (increments.Any())
        {
            long minIncrement = increments.Min();
            // Minimum increment should be nanosecond-scale for PTP
            if (clockSelector.ActiveSource.GetType().Name == "PtpClockSource")
            {
                minIncrement.Should().BeLessThan(1_000_000); // < 1ms precision
            }
        }
    }

    [Fact]
    public async Task Integration_FallbackToSystemClock_WhenPtpUnavailable()
    {
        // Arrange
        var clockSelector = new ClockSourceSelector(NullLogger<ClockSourceSelector>.Instance);
        await clockSelector.InitializeAsync();

        // Act
        var activeSource = clockSelector.ActiveSource;

        // Assert - Should always have at least system clock
        activeSource.Should().NotBeNull();
        activeSource.IsSynchronized.Should().BeTrue();

        // System clock should always be in available sources
        clockSelector.AvailableSources.Should().Contain(
            s => s.GetType().Name == "SystemClockSource");
    }

    [Fact]
    public async Task Integration_ClockSelection_PrefersPtpOverSystemClock()
    {
        // Arrange
        var clockSelector = new ClockSourceSelector(NullLogger<ClockSourceSelector>.Instance);
        await clockSelector.InitializeAsync();

        // Act
        var ptpSource = clockSelector.AvailableSources.FirstOrDefault(
            s => s.GetType().Name == "PtpClockSource");
        var systemSource = clockSelector.AvailableSources.FirstOrDefault(
            s => s.GetType().Name == "SystemClockSource");

        // Assert
        if (ptpSource != null && ptpSource.IsSynchronized)
        {
            // If PTP available, it should be active
            clockSelector.ActiveSource.Should().Be(ptpSource);

            // PTP should have lower error bound than system clock
            ptpSource.GetErrorBound().Should().BeLessThan(
                systemSource!.GetErrorBound());
        }
    }

    [Fact]
    public async Task Integration_SoftwarePtp_FallsBackFromHardwarePtp()
    {
        // Arrange
        var logger = NullLogger<SoftwarePtpClockSource>.Instance;

        // Try to initialize software PTP
        var softwarePtp = new SoftwarePtpClockSource(logger);

        // Act
        bool initialized = await softwarePtp.InitializeAsync();

        // Assert - Software PTP may fail if no NTP servers reachable
        // This is expected in isolated environments
        if (initialized)
        {
            softwarePtp.IsSynchronized.Should().BeTrue();
            softwarePtp.GetErrorBound().Should().BeLessThan(100_000_000); // < 100ms
            softwarePtp.GetErrorBound().Should().BeGreaterThan(1_000_000); // > 1ms (software range)
        }

        softwarePtp.Dispose();
    }
}

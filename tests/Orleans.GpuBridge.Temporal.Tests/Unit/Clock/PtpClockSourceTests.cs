using FluentAssertions;
using Microsoft.Extensions.Logging.Abstractions;
using Orleans.GpuBridge.Runtime.Temporal.Clock;

namespace Orleans.GpuBridge.Temporal.Tests.Unit.Clock;

/// <summary>
/// Unit tests for PTP hardware clock source.
/// </summary>
public sealed class PtpClockSourceTests
{
    [Fact]
    public async Task PtpClockSource_InitializesSuccessfully_WhenHardwareAvailable()
    {
        // Arrange
        var ptpClock = new PtpClockSource(NullLogger<PtpClockSource>.Instance);

        // Act
        bool initialized = await ptpClock.InitializeAsync();

        // Assert
        if (initialized)
        {
            ptpClock.IsSynchronized.Should().BeTrue();
            ptpClock.ClockId.Should().BeGreaterThanOrEqualTo(0);
            // PTP hardware available - test passed
        }
        else
        {
            // PTP hardware not available - not a test failure
            // This is expected on systems without PTP-capable NICs
            ptpClock.IsSynchronized.Should().BeFalse();
        }
    }

    [Fact]
    public async Task PtpClockSource_ReturnsNanosecondPrecision()
    {
        // Arrange
        var ptpClock = new PtpClockSource(NullLogger<PtpClockSource>.Instance);
        await ptpClock.InitializeAsync();

        if (!ptpClock.IsSynchronized)
        {
            // Skip test if PTP unavailable
            return;
        }

        // Act
        long time1 = ptpClock.GetCurrentTimeNanos();
        await Task.Delay(1); // 1ms delay
        long time2 = ptpClock.GetCurrentTimeNanos();

        // Assert
        long deltaNanos = time2 - time1;
        deltaNanos.Should().BeGreaterThan(1_000_000); // > 1ms
        deltaNanos.Should().BeLessThan(10_000_000);   // < 10ms
    }

    [Fact]
    public async Task PtpClockSource_ErrorBoundIsReasonable()
    {
        // Arrange
        var ptpClock = new PtpClockSource(NullLogger<PtpClockSource>.Instance);
        await ptpClock.InitializeAsync();

        // Act
        long errorBound = ptpClock.GetErrorBound();

        // Assert
        if (ptpClock.IsSynchronized)
        {
            // PTP hardware: 50ns (Mellanox) to 10μs (Hyper-V)
            errorBound.Should().BeInRange(50, 10_000);
        }
        else
        {
            // Not synchronized - large error bound
            errorBound.Should().Be(100_000_000); // 100ms
        }
    }

    [Fact]
    public async Task PtpClockSource_ThrowsWhenReadingWithoutInitialization()
    {
        // Arrange
        var ptpClock = new PtpClockSource(NullLogger<PtpClockSource>.Instance);

        // Act
        var act = () => ptpClock.GetCurrentTimeNanos();

        // Assert
        act.Should().Throw<InvalidOperationException>()
            .WithMessage("*not initialized*");
    }

    [Fact]
    public async Task PtpClockSource_CanInitializeMultipleTimes()
    {
        // Arrange
        var ptpClock = new PtpClockSource(NullLogger<PtpClockSource>.Instance);

        // Act
        bool init1 = await ptpClock.InitializeAsync();
        bool init2 = await ptpClock.InitializeAsync();

        // Assert
        // Should not throw on second initialization
        init1.Should().Be(init2);
    }

    [Fact]
    public async Task PtpClockSource_DisposeCleansUpResources()
    {
        // Arrange
        var ptpClock = new PtpClockSource(NullLogger<PtpClockSource>.Instance);
        await ptpClock.InitializeAsync();
        bool wasInitialized = ptpClock.IsSynchronized;

        // Act
        ptpClock.Dispose();

        // Assert
        ptpClock.IsSynchronized.Should().BeFalse();

        if (wasInitialized)
        {
            // Should throw after disposal
            var act = () => ptpClock.GetCurrentTimeNanos();
            act.Should().Throw<InvalidOperationException>();
        }
    }

    [Fact]
    public async Task PtpClockSource_ReturnsConsistentTime()
    {
        // Arrange
        var ptpClock = new PtpClockSource(NullLogger<PtpClockSource>.Instance);
        await ptpClock.InitializeAsync();

        if (!ptpClock.IsSynchronized)
        {
            return; // Skip if PTP unavailable
        }

        // Act - Read time multiple times
        var times = new long[10];
        for (int i = 0; i < 10; i++)
        {
            times[i] = ptpClock.GetCurrentTimeNanos();
            await Task.Delay(1);
        }

        // Assert - Time should be monotonically increasing
        for (int i = 1; i < times.Length; i++)
        {
            times[i].Should().BeGreaterThan(times[i - 1],
                $"Time should increase monotonically (iteration {i})");
        }
    }

    [Fact]
    public async Task PtpClockSource_HandlesNonExistentDevice()
    {
        // Arrange
        var ptpClock = new PtpClockSource(
            NullLogger<PtpClockSource>.Instance,
            "/dev/ptp999"); // Non-existent device

        // Act
        bool initialized = await ptpClock.InitializeAsync();

        // Assert
        initialized.Should().BeFalse();
        ptpClock.IsSynchronized.Should().BeFalse();
    }
}

// Orleans.GpuBridge - GPU-native distributed computing for Microsoft Orleans
// Copyright (c) 2025 Michael Ivertowski

using Microsoft.Extensions.Logging;
using Orleans.GpuBridge.Abstractions.Temporal;
using Orleans.GpuBridge.Runtime.Temporal;

namespace Orleans.GpuBridge.Runtime.Tests.Temporal;

/// <summary>
/// Unit tests for <see cref="CpuTimingProvider"/> CPU fallback timing implementation.
/// </summary>
public sealed class CpuTimingProviderTests
{
    private readonly ILogger<CpuTimingProvider> _logger;

    public CpuTimingProviderTests()
    {
        var loggerFactory = LoggerFactory.Create(builder => builder.AddDebug());
        _logger = loggerFactory.CreateLogger<CpuTimingProvider>();
    }

    /// <summary>
    /// Tests that the CPU timing provider reports it is not GPU-backed.
    /// </summary>
    [Fact]
    public void IsGpuBacked_ReturnsFalse()
    {
        // Arrange
        var provider = new CpuTimingProvider(_logger);

        // Act & Assert
        provider.IsGpuBacked.Should().BeFalse();
    }

    /// <summary>
    /// Tests that the provider type name is correctly set.
    /// </summary>
    [Fact]
    public void ProviderTypeName_ReturnsCpuFallback()
    {
        // Arrange
        var provider = new CpuTimingProvider(_logger);

        // Act & Assert
        provider.ProviderTypeName.Should().Contain("CPU");
        provider.ProviderTypeName.Should().Contain("Stopwatch");
    }

    /// <summary>
    /// Tests that timestamps are monotonically increasing.
    /// </summary>
    [Fact]
    public async Task GetGpuTimestampAsync_ReturnsMonotonicallyIncreasingValues()
    {
        // Arrange
        var provider = new CpuTimingProvider(_logger);

        // Act
        var timestamp1 = await provider.GetGpuTimestampAsync();
        await Task.Delay(1);
        var timestamp2 = await provider.GetGpuTimestampAsync();
        await Task.Delay(1);
        var timestamp3 = await provider.GetGpuTimestampAsync();

        // Assert
        timestamp2.Should().BeGreaterThan(timestamp1);
        timestamp3.Should().BeGreaterThan(timestamp2);
    }

    /// <summary>
    /// Tests that timestamps are positive nanosecond values.
    /// </summary>
    [Fact]
    public async Task GetGpuTimestampAsync_ReturnsPositiveNanoseconds()
    {
        // Arrange
        var provider = new CpuTimingProvider(_logger);

        // Act
        var timestamp = await provider.GetGpuTimestampAsync();

        // Assert
        timestamp.Should().BeGreaterThan(0);
        // Should be at least milliseconds worth of nanoseconds (since system startup)
        timestamp.Should().BeGreaterThan(1_000_000);
    }

    /// <summary>
    /// Tests that batch timestamps return requested count.
    /// </summary>
    [Fact]
    public async Task GetGpuTimestampsBatchAsync_ReturnsRequestedCount()
    {
        // Arrange
        var provider = new CpuTimingProvider(_logger);
        const int count = 100;

        // Act
        var timestamps = await provider.GetGpuTimestampsBatchAsync(count);

        // Assert
        timestamps.Should().HaveCount(count);
    }

    /// <summary>
    /// Tests that batch timestamps are all positive.
    /// </summary>
    [Fact]
    public async Task GetGpuTimestampsBatchAsync_ReturnsPositiveValues()
    {
        // Arrange
        var provider = new CpuTimingProvider(_logger);

        // Act
        var timestamps = await provider.GetGpuTimestampsBatchAsync(10);

        // Assert
        timestamps.Should().OnlyContain(t => t > 0);
    }

    /// <summary>
    /// Tests that batch timestamps are non-decreasing.
    /// </summary>
    [Fact]
    public async Task GetGpuTimestampsBatchAsync_ReturnsNonDecreasingValues()
    {
        // Arrange
        var provider = new CpuTimingProvider(_logger);

        // Act
        var timestamps = await provider.GetGpuTimestampsBatchAsync(100);

        // Assert
        for (int i = 1; i < timestamps.Length; i++)
        {
            timestamps[i].Should().BeGreaterThanOrEqualTo(timestamps[i - 1]);
        }
    }

    /// <summary>
    /// Tests that batch with invalid count throws.
    /// </summary>
    [Fact]
    public async Task GetGpuTimestampsBatchAsync_WithZeroCount_ThrowsException()
    {
        // Arrange
        var provider = new CpuTimingProvider(_logger);

        // Act & Assert
        await Assert.ThrowsAsync<ArgumentOutOfRangeException>(
            () => provider.GetGpuTimestampsBatchAsync(0));
    }

    /// <summary>
    /// Tests that batch with negative count throws.
    /// </summary>
    [Fact]
    public async Task GetGpuTimestampsBatchAsync_WithNegativeCount_ThrowsException()
    {
        // Arrange
        var provider = new CpuTimingProvider(_logger);

        // Act & Assert
        await Assert.ThrowsAsync<ArgumentOutOfRangeException>(
            () => provider.GetGpuTimestampsBatchAsync(-1));
    }

    /// <summary>
    /// Tests that clock frequency is a reasonable value.
    /// </summary>
    [Fact]
    public void GetGpuClockFrequency_ReturnsReasonableValue()
    {
        // Arrange
        var provider = new CpuTimingProvider(_logger);

        // Act
        var frequency = provider.GetGpuClockFrequency();

        // Assert
        // Should be at least 1 MHz (1,000,000 Hz)
        frequency.Should().BeGreaterThan(1_000_000);
        // Should not exceed 10 GHz (10,000,000,000 Hz)
        frequency.Should().BeLessThan(10_000_000_000L);
    }

    /// <summary>
    /// Tests that timer resolution is a reasonable value.
    /// </summary>
    [Fact]
    public void GetTimerResolutionNanos_ReturnsReasonableValue()
    {
        // Arrange
        var provider = new CpuTimingProvider(_logger);

        // Act
        var resolution = provider.GetTimerResolutionNanos();

        // Assert
        // Should be at least 1 nanosecond
        resolution.Should().BeGreaterThanOrEqualTo(1);
        // Should not exceed 1 millisecond (1,000,000 nanoseconds)
        resolution.Should().BeLessThan(1_000_000);
    }

    /// <summary>
    /// Tests that timestamp injection can be enabled.
    /// </summary>
    [Fact]
    public void EnableTimestampInjection_CanBeEnabled()
    {
        // Arrange
        var provider = new CpuTimingProvider(_logger);

        // Act
        provider.EnableTimestampInjection(true);

        // Assert
        provider.IsTimestampInjectionEnabled.Should().BeTrue();
    }

    /// <summary>
    /// Tests that timestamp injection can be disabled.
    /// </summary>
    [Fact]
    public void EnableTimestampInjection_CanBeDisabled()
    {
        // Arrange
        var provider = new CpuTimingProvider(_logger);
        provider.EnableTimestampInjection(true);

        // Act
        provider.EnableTimestampInjection(false);

        // Assert
        provider.IsTimestampInjectionEnabled.Should().BeFalse();
    }

    /// <summary>
    /// Tests that calibration produces valid results.
    /// </summary>
    [Fact]
    public async Task CalibrateAsync_ProducesValidCalibration()
    {
        // Arrange
        var provider = new CpuTimingProvider(_logger);

        // Act
        var calibration = await provider.CalibrateAsync(sampleCount: 50);

        // Assert
        calibration.SampleCount.Should().Be(50);
        calibration.CalibrationTimestampNanos.Should().BeGreaterThan(0);
        // Error bound should be positive (uncertainty exists)
        calibration.ErrorBoundNanos.Should().BeGreaterThanOrEqualTo(0);
    }

    /// <summary>
    /// Tests that calibration with low sample count throws.
    /// </summary>
    [Fact]
    public async Task CalibrateAsync_WithLowSampleCount_ThrowsException()
    {
        // Arrange
        var provider = new CpuTimingProvider(_logger);

        // Act & Assert
        await Assert.ThrowsAsync<ArgumentOutOfRangeException>(
            () => provider.CalibrateAsync(sampleCount: 5));
    }

    /// <summary>
    /// Tests that calibration respects cancellation.
    /// </summary>
    [Fact]
    public async Task CalibrateAsync_WithCancelledToken_ThrowsOperationCancelledException()
    {
        // Arrange
        var provider = new CpuTimingProvider(_logger);
        using var cts = new CancellationTokenSource();
        cts.Cancel();

        // Act & Assert
        await Assert.ThrowsAsync<OperationCanceledException>(
            () => provider.CalibrateAsync(sampleCount: 100, ct: cts.Token));
    }

    /// <summary>
    /// Tests that GetGpuTimestampAsync respects cancellation.
    /// </summary>
    [Fact]
    public async Task GetGpuTimestampAsync_WithCancelledToken_ThrowsOperationCancelledException()
    {
        // Arrange
        var provider = new CpuTimingProvider(_logger);
        using var cts = new CancellationTokenSource();
        cts.Cancel();

        // Act & Assert
        await Assert.ThrowsAsync<OperationCanceledException>(
            () => provider.GetGpuTimestampAsync(ct: cts.Token));
    }
}

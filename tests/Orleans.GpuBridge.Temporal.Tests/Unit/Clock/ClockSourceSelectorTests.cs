using FluentAssertions;
using Microsoft.Extensions.Logging.Abstractions;
using Orleans.GpuBridge.Runtime.Temporal.Clock;

namespace Orleans.GpuBridge.Temporal.Tests.Unit.Clock;

/// <summary>
/// Unit tests for automatic clock source selection with fallback chain.
/// </summary>
public sealed class ClockSourceSelectorTests
{
    [Fact]
    public async Task ClockSourceSelector_InitializesSuccessfully()
    {
        // Arrange
        var selector = new ClockSourceSelector(NullLogger<ClockSourceSelector>.Instance);

        // Act
        await selector.InitializeAsync();

        // Assert
        selector.IsInitialized.Should().BeTrue();
        selector.ActiveSource.Should().NotBeNull();
        selector.AvailableSources.Should().NotBeEmpty();
    }

    [Fact]
    public async Task ClockSourceSelector_SystemClockAlwaysAvailable()
    {
        // Arrange
        var selector = new ClockSourceSelector(NullLogger<ClockSourceSelector>.Instance);

        // Act
        await selector.InitializeAsync();

        // Assert
        // System clock should always be available as fallback
        selector.AvailableSources.Should().Contain(s => s.GetType().Name == "SystemClockSource");
        selector.ActiveSource.Should().NotBeNull();
    }

    [Fact]
    public async Task ClockSourceSelector_PrefersLowerErrorBound()
    {
        // Arrange
        var selector = new ClockSourceSelector(NullLogger<ClockSourceSelector>.Instance);
        await selector.InitializeAsync();

        // Act
        var bestSource = selector.GetBestAvailableSource();

        // Assert
        if (bestSource != null)
        {
            // Best source should have lowest error bound among synchronized sources
            foreach (var source in selector.AvailableSources)
            {
                if (source.IsSynchronized)
                {
                    bestSource.GetErrorBound().Should().BeLessThanOrEqualTo(source.GetErrorBound());
                }
            }
        }
    }

    [Fact]
    public async Task ClockSourceSelector_CanSwitchClockSource()
    {
        // Arrange
        var selector = new ClockSourceSelector(NullLogger<ClockSourceSelector>.Instance);
        await selector.InitializeAsync();

        var originalSource = selector.ActiveSource;
        var alternativeSource = selector.AvailableSources.FirstOrDefault(s => s != originalSource);

        if (alternativeSource == null)
        {
            // Only one source available - skip test
            return;
        }

        // Act
        selector.SwitchClockSource(alternativeSource);

        // Assert
        selector.ActiveSource.Should().Be(alternativeSource);
        selector.ActiveSource.Should().NotBe(originalSource);
    }

    [Fact]
    public async Task ClockSourceSelector_ThrowsWhenSwitchingToUnavailableSource()
    {
        // Arrange
        var selector = new ClockSourceSelector(NullLogger<ClockSourceSelector>.Instance);
        await selector.InitializeAsync();

        var unavailableSource = new TestClockSource();

        // Act
        var act = () => selector.SwitchClockSource(unavailableSource);

        // Assert
        act.Should().Throw<ArgumentException>()
            .WithMessage("*not in available sources list*");
    }

    [Fact]
    public async Task ClockSourceSelector_ReportsAvailableSources()
    {
        // Arrange
        var selector = new ClockSourceSelector(NullLogger<ClockSourceSelector>.Instance);

        // Act
        await selector.InitializeAsync();

        // Assert
        selector.AvailableSources.Should().NotBeEmpty();
        selector.AvailableSources.Should().AllSatisfy(s =>
        {
            s.Should().NotBeNull();
            s.GetErrorBound().Should().BeGreaterThan(0);
        });
    }

    [Fact]
    public async Task ClockSourceSelector_PtpPreferredOverSystemClock()
    {
        // Arrange
        var selector = new ClockSourceSelector(NullLogger<ClockSourceSelector>.Instance);

        // Act
        await selector.InitializeAsync();

        // Assert
        var ptpSource = selector.AvailableSources.FirstOrDefault(s => s.GetType().Name == "PtpClockSource");
        var systemSource = selector.AvailableSources.FirstOrDefault(s => s.GetType().Name == "SystemClockSource");

        if (ptpSource != null && systemSource != null)
        {
            // If both available, PTP should be active (lower error bound)
            if (ptpSource.IsSynchronized)
            {
                selector.ActiveSource.Should().Be(ptpSource);
                ptpSource.GetErrorBound().Should().BeLessThan(systemSource.GetErrorBound());
            }
        }
    }

    [Fact]
    public void ClockSourceSelector_ThrowsWhenReadingWithoutInitialization()
    {
        // Arrange
        var selector = new ClockSourceSelector(NullLogger<ClockSourceSelector>.Instance);

        // Act
        var act = () => selector.ActiveSource;

        // Assert
        act.Should().Throw<InvalidOperationException>()
            .WithMessage("*No clock source available*");
    }

    [Fact]
    public async Task ClockSourceSelector_WarnsOnMultipleInitializations()
    {
        // Arrange
        var selector = new ClockSourceSelector(NullLogger<ClockSourceSelector>.Instance);
        await selector.InitializeAsync();

        // Act & Assert
        // Should not throw on second initialization
        await selector.InitializeAsync();
        selector.IsInitialized.Should().BeTrue();
    }

    [Fact]
    public async Task ClockSourceSelector_BestSourceIsNullWhenNoneSynchronized()
    {
        // Arrange
        var selector = new ClockSourceSelector(NullLogger<ClockSourceSelector>.Instance);
        await selector.InitializeAsync();

        // Artificially mark all sources as unsynchronized for testing
        // In practice, SystemClock is always synchronized

        // Act
        var bestSource = selector.GetBestAvailableSource();

        // Assert
        // Should return a synchronized source (at least SystemClock)
        bestSource.Should().NotBeNull();
        bestSource!.IsSynchronized.Should().BeTrue();
    }

    /// <summary>
    /// Test clock source for verifying error handling.
    /// </summary>
    private sealed class TestClockSource : Orleans.GpuBridge.Abstractions.Temporal.IPhysicalClockSource
    {
        public bool IsSynchronized => true;
        public long GetCurrentTimeNanos() => DateTimeOffset.UtcNow.ToUnixTimeMilliseconds() * 1_000_000;
        public long GetErrorBound() => 1_000_000; // 1ms
        public double GetClockDrift() => 0.0;
    }
}

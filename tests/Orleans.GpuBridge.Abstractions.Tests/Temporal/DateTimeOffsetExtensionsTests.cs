using Orleans.GpuBridge.Abstractions.Temporal;

namespace Orleans.GpuBridge.Abstractions.Tests.Temporal;

/// <summary>
/// Tests for DateTimeOffsetExtensions nanosecond conversion utilities.
/// </summary>
public sealed class DateTimeOffsetExtensionsTests
{
    /// <summary>
    /// Tests Unix epoch conversion to nanoseconds.
    /// </summary>
    [Fact]
    public void ToUnixTimeNanoseconds_UnixEpoch_ReturnsZero()
    {
        // Arrange
        var epoch = new DateTimeOffset(1970, 1, 1, 0, 0, 0, TimeSpan.Zero);

        // Act
        var nanos = epoch.ToUnixTimeNanoseconds();

        // Assert
        nanos.Should().Be(0);
    }

    /// <summary>
    /// Tests conversion of one second after epoch.
    /// </summary>
    [Fact]
    public void ToUnixTimeNanoseconds_OneSecondAfterEpoch_Returns1Billion()
    {
        // Arrange
        var oneSecond = new DateTimeOffset(1970, 1, 1, 0, 0, 1, TimeSpan.Zero);

        // Act
        var nanos = oneSecond.ToUnixTimeNanoseconds();

        // Assert - 1 second = 1,000,000,000 nanoseconds
        nanos.Should().Be(1_000_000_000L);
    }

    /// <summary>
    /// Tests round-trip conversion.
    /// </summary>
    [Fact]
    public void RoundTrip_PreservesTime()
    {
        // Arrange
        var original = new DateTimeOffset(2024, 6, 15, 12, 30, 45, 123, TimeSpan.Zero);

        // Act
        var nanos = original.ToUnixTimeNanoseconds();
        var restored = DateTimeOffsetExtensions.FromUnixTimeNanoseconds(nanos);

        // Assert - Should be equal within tick precision (100ns)
        var ticksOriginal = original.UtcDateTime.Ticks;
        var ticksRestored = restored.UtcDateTime.Ticks;
        Math.Abs(ticksOriginal - ticksRestored).Should().BeLessThanOrEqualTo(1);
    }

    /// <summary>
    /// Tests conversion from nanoseconds back to DateTimeOffset.
    /// </summary>
    [Fact]
    public void FromUnixTimeNanoseconds_ValidNanos_ReturnsCorrectTime()
    {
        // Arrange - 1 day = 86400 seconds = 86,400,000,000,000 nanoseconds
        var oneDayNanos = 86_400_000_000_000L;

        // Act
        var result = DateTimeOffsetExtensions.FromUnixTimeNanoseconds(oneDayNanos);

        // Assert
        result.Year.Should().Be(1970);
        result.Month.Should().Be(1);
        result.Day.Should().Be(2);
        result.Hour.Should().Be(0);
        result.Minute.Should().Be(0);
        result.Second.Should().Be(0);
    }

    /// <summary>
    /// Tests conversion near the maximum safe nanosecond range.
    /// Int64.MaxValue nanoseconds ≈ year 2262, so year 2200 is safely within range.
    /// </summary>
    [Fact]
    public void ToUnixTimeNanoseconds_WithinNanosecondRange_ReturnsPositive()
    {
        // Arrange - Use a date within nanosecond Int64 range
        // Int64.MaxValue / 1e9 / 86400 / 365.25 + 1970 ≈ 2262
        // Year 2200 is safely within this range
        var future = new DateTimeOffset(2200, 1, 1, 0, 0, 0, TimeSpan.Zero);

        // Act
        var nanos = future.ToUnixTimeNanoseconds();

        // Assert - Should be positive and reasonable (230 years * ~31.5M seconds/year * 1e9)
        nanos.Should().BePositive();
        nanos.Should().BeGreaterThan(0);
    }

    /// <summary>
    /// Tests that dates beyond year 2262 overflow Int64 nanosecond range.
    /// This documents the intentional limitation of nanosecond timestamps.
    /// </summary>
    [Fact]
    public void ToUnixTimeNanoseconds_BeyondYear2262_OverflowsToNegative()
    {
        // Arrange - Year 2500 exceeds Int64.MaxValue nanoseconds since epoch
        // ~530 years * 31,557,600 seconds/year * 1e9 ≈ 16.7e18 > Int64.MaxValue (~9.2e18)
        var farFuture = new DateTimeOffset(2500, 1, 1, 0, 0, 0, TimeSpan.Zero);

        // Act
        var nanos = farFuture.ToUnixTimeNanoseconds();

        // Assert - Demonstrates overflow behavior (returns negative due to unchecked overflow)
        // This is expected behavior - nanosecond timestamps have a ~292 year range from epoch
        nanos.Should().BeNegative("because year 2500 exceeds Int64 nanosecond range");
    }

    /// <summary>
    /// Tests conversion of negative nanoseconds (before Unix epoch).
    /// </summary>
    [Fact]
    public void FromUnixTimeNanoseconds_Negative_ReturnsBeforeEpoch()
    {
        // Arrange - 1 day before epoch
        var negativeNanos = -86_400_000_000_000L;

        // Act
        var result = DateTimeOffsetExtensions.FromUnixTimeNanoseconds(negativeNanos);

        // Assert
        result.Year.Should().Be(1969);
        result.Month.Should().Be(12);
        result.Day.Should().Be(31);
    }

    /// <summary>
    /// Tests millisecond precision is preserved.
    /// </summary>
    [Fact]
    public void ToUnixTimeNanoseconds_WithMilliseconds_PreservesMillisecondPrecision()
    {
        // Arrange
        var withMillis = new DateTimeOffset(1970, 1, 1, 0, 0, 0, 500, TimeSpan.Zero);

        // Act
        var nanos = withMillis.ToUnixTimeNanoseconds();

        // Assert - 500ms = 500,000,000 nanoseconds
        nanos.Should().Be(500_000_000L);
    }
}

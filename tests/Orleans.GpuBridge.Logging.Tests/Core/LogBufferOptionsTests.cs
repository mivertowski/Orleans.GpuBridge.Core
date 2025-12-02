// Orleans.GpuBridge - GPU-native distributed computing for Microsoft Orleans
// Copyright (c) 2025 Michael Ivertowski

using Orleans.GpuBridge.Logging.Core;

namespace Orleans.GpuBridge.Logging.Tests.Core;

/// <summary>
/// Tests for <see cref="LogBufferOptions"/> record class.
/// </summary>
public class LogBufferOptionsTests
{
    [Fact]
    public void DefaultValues_ShouldBeReasonable()
    {
        // Act
        var options = new LogBufferOptions();

        // Assert
        options.Capacity.Should().Be(10000);
        options.MaxBatchSize.Should().Be(100);
        options.FlushInterval.Should().Be(TimeSpan.FromSeconds(1));
        options.DropOnOverflow.Should().BeTrue();
        options.PrioritizeHighSeverity.Should().BeTrue();
    }

    [Fact]
    public void AllProperties_ShouldBeSettable()
    {
        // Arrange
        var options = new LogBufferOptions
        {
            Capacity = 5000,
            MaxBatchSize = 50,
            FlushInterval = TimeSpan.FromSeconds(5),
            DropOnOverflow = false,
            PrioritizeHighSeverity = false
        };

        // Assert
        options.Capacity.Should().Be(5000);
        options.MaxBatchSize.Should().Be(50);
        options.FlushInterval.Should().Be(TimeSpan.FromSeconds(5));
        options.DropOnOverflow.Should().BeFalse();
        options.PrioritizeHighSeverity.Should().BeFalse();
    }

    [Theory]
    [InlineData(100)]
    [InlineData(1000)]
    [InlineData(100000)]
    public void Capacity_ShouldAcceptVariousValues(int capacity)
    {
        // Arrange
        var options = new LogBufferOptions { Capacity = capacity };

        // Assert
        options.Capacity.Should().Be(capacity);
    }

    [Theory]
    [InlineData(10)]
    [InlineData(100)]
    [InlineData(1000)]
    public void MaxBatchSize_ShouldAcceptVariousValues(int batchSize)
    {
        // Arrange
        var options = new LogBufferOptions { MaxBatchSize = batchSize };

        // Assert
        options.MaxBatchSize.Should().Be(batchSize);
    }

    [Fact]
    public void LogBufferOptions_ShouldSupportEquality()
    {
        // Arrange
        var options1 = new LogBufferOptions { Capacity = 1000, MaxBatchSize = 50 };
        var options2 = new LogBufferOptions { Capacity = 1000, MaxBatchSize = 50 };

        // Assert
        options1.Should().Be(options2);
    }

    [Fact]
    public void LogBufferOptions_ShouldSupportWith()
    {
        // Arrange
        var original = new LogBufferOptions { Capacity = 1000 };

        // Act
        var modified = original with { Capacity = 2000 };

        // Assert
        original.Capacity.Should().Be(1000);
        modified.Capacity.Should().Be(2000);
    }
}

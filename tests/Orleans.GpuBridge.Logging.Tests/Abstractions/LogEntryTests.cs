// Orleans.GpuBridge - GPU-native distributed computing for Microsoft Orleans
// Copyright (c) 2025 Michael Ivertowski

using Microsoft.Extensions.Logging;
using Orleans.GpuBridge.Logging.Abstractions;

namespace Orleans.GpuBridge.Logging.Tests.Abstractions;

/// <summary>
/// Tests for <see cref="LogEntry"/> record class.
/// </summary>
public class LogEntryTests
{
    [Fact]
    public void Create_ShouldSetBasicProperties()
    {
        // Act
        var entry = LogEntry.Create(LogLevel.Information, "TestCategory", "Test message");

        // Assert
        entry.Level.Should().Be(LogLevel.Information);
        entry.Category.Should().Be("TestCategory");
        entry.Message.Should().Be("Test message");
        entry.Id.Should().NotBeNullOrEmpty();
        entry.Timestamp.Should().BeCloseTo(DateTimeOffset.UtcNow, TimeSpan.FromSeconds(1));
    }

    [Fact]
    public void CreateWithException_ShouldSetException()
    {
        // Arrange
        var exception = new InvalidOperationException("Test exception");

        // Act
        var entry = LogEntry.CreateWithException(LogLevel.Error, "TestCategory", "Error occurred", exception);

        // Assert
        entry.Level.Should().Be(LogLevel.Error);
        entry.Exception.Should().Be(exception);
        entry.Message.Should().Be("Error occurred");
    }

    [Fact]
    public void CreateWithProperties_ShouldSetProperties()
    {
        // Arrange
        var properties = new Dictionary<string, object?>
        {
            ["Key1"] = "Value1",
            ["Key2"] = 42
        };

        // Act
        var entry = LogEntry.CreateWithProperties(LogLevel.Debug, "TestCategory", "Test", properties);

        // Assert
        entry.Properties.Should().ContainKey("Key1").WhoseValue.Should().Be("Value1");
        entry.Properties.Should().ContainKey("Key2").WhoseValue.Should().Be(42);
    }

    [Fact]
    public void WithProperties_ShouldMergeProperties()
    {
        // Arrange
        var initialProperties = new Dictionary<string, object?> { ["Key1"] = "Value1" };
        var additionalProperties = new Dictionary<string, object?> { ["Key2"] = "Value2" };
        var entry = LogEntry.CreateWithProperties(LogLevel.Information, "Test", "Message", initialProperties);

        // Act
        var newEntry = entry.WithProperties(additionalProperties);

        // Assert
        newEntry.Properties.Should().ContainKey("Key1");
        newEntry.Properties.Should().ContainKey("Key2");
    }

    [Fact]
    public void WithCorrelation_ShouldSetCorrelationInfo()
    {
        // Arrange
        var entry = LogEntry.Create(LogLevel.Information, "Test", "Message");

        // Act
        var newEntry = entry.WithCorrelation("corr-123", "op-456");

        // Assert
        newEntry.CorrelationId.Should().Be("corr-123");
        newEntry.OperationId.Should().Be("op-456");
    }

    [Fact]
    public void WithMetrics_ShouldSetMetrics()
    {
        // Arrange
        var entry = LogEntry.Create(LogLevel.Information, "Test", "Message");
        var metrics = LogPerformanceMetrics.WithDuration(TimeSpan.FromSeconds(1));

        // Act
        var newEntry = entry.WithMetrics(metrics);

        // Assert
        newEntry.Metrics.Should().NotBeNull();
        newEntry.Metrics!.Duration.Should().Be(TimeSpan.FromSeconds(1));
    }

    [Fact]
    public void DefaultValues_ShouldBeSet()
    {
        // Act
        var entry = new LogEntry();

        // Assert
        entry.Id.Should().NotBeNullOrEmpty();
        entry.Category.Should().BeEmpty();
        entry.Message.Should().BeEmpty();
        entry.Level.Should().Be(LogLevel.Trace); // Default enum value is 0 = Trace
        entry.Exception.Should().BeNull();
        entry.Properties.Should().BeEmpty();
        entry.Scopes.Should().BeEmpty();
        entry.ThreadId.Should().Be(Environment.CurrentManagedThreadId);
    }

    [Fact]
    public void LogEntry_ShouldBeImmutable()
    {
        // Arrange
        var entry = LogEntry.Create(LogLevel.Information, "Test", "Message");

        // Act
        var newEntry = entry with { Message = "New message" };

        // Assert
        entry.Message.Should().Be("Message");
        newEntry.Message.Should().Be("New message");
    }

    [Theory]
    [InlineData(LogLevel.Trace)]
    [InlineData(LogLevel.Debug)]
    [InlineData(LogLevel.Information)]
    [InlineData(LogLevel.Warning)]
    [InlineData(LogLevel.Error)]
    [InlineData(LogLevel.Critical)]
    public void Create_ShouldAcceptAllLogLevels(LogLevel level)
    {
        // Act
        var entry = LogEntry.Create(level, "Test", "Message");

        // Assert
        entry.Level.Should().Be(level);
    }
}

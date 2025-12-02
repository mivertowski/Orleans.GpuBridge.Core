// Orleans.GpuBridge - GPU-native distributed computing for Microsoft Orleans
// Copyright (c) 2025 Michael Ivertowski

using Orleans.GpuBridge.Logging.Abstractions;

namespace Orleans.GpuBridge.Logging.Tests.Abstractions;

/// <summary>
/// Tests for <see cref="LogContext"/> class.
/// </summary>
public class LogContextTests
{
    [Fact]
    public void Constructor_ShouldGenerateCorrelationId()
    {
        // Act
        var context = new LogContext();

        // Assert
        context.CorrelationId.Should().NotBeNullOrEmpty();
        context.CorrelationId.Should().HaveLength(12);
    }

    [Fact]
    public void Constructor_WithCorrelationId_ShouldSetCorrelationId()
    {
        // Arrange
        var correlationId = "test-correlation-123";

        // Act
        var context = new LogContext(correlationId);

        // Assert
        context.CorrelationId.Should().Be(correlationId);
    }

    [Fact]
    public void SetProperty_ShouldAddProperty()
    {
        // Arrange
        var context = new LogContext();

        // Act
        context.SetProperty("Key1", "Value1");

        // Assert
        context.Properties.Should().ContainKey("Key1").WhoseValue.Should().Be("Value1");
    }

    [Fact]
    public void GetProperty_ShouldReturnValue()
    {
        // Arrange
        var context = new LogContext();
        context.SetProperty("IntKey", 42);

        // Act
        var value = context.GetProperty<int>("IntKey");

        // Assert
        value.Should().Be(42);
    }

    [Fact]
    public void GetProperty_ShouldReturnDefault_WhenKeyNotFound()
    {
        // Arrange
        var context = new LogContext();

        // Act
        var value = context.GetProperty<string>("NonExistent");

        // Assert
        value.Should().BeNull();
    }

    [Fact]
    public void CreateChild_ShouldInheritProperties()
    {
        // Arrange
        var parent = new LogContext("parent-id")
        {
            UserId = "user-123",
            TenantId = "tenant-456",
            Component = "TestComponent"
        };
        parent.SetProperty("CustomProp", "CustomValue");

        // Act
        var child = parent.CreateChild("child-op");

        // Assert
        child.CorrelationId.Should().Be("parent-id");
        child.OperationId.Should().Be("child-op");
        child.UserId.Should().Be("user-123");
        child.TenantId.Should().Be("tenant-456");
        child.Component.Should().Be("TestComponent");
        child.Properties.Should().ContainKey("CustomProp");
    }

    [Fact]
    public void Push_ShouldSetCurrentContext()
    {
        // Arrange
        var context = new LogContext("test-id");

        // Act
        using (context.Push())
        {
            // Assert
            LogContext.Current.Should().Be(context);
        }

        // After dispose, current should be null or previous
        LogContext.Current.Should().BeNull();
    }

    [Fact]
    public void Push_ShouldRestorePreviousContext()
    {
        // Arrange
        var outer = new LogContext("outer-id");
        var inner = new LogContext("inner-id");

        // Act & Assert
        using (outer.Push())
        {
            LogContext.Current.Should().Be(outer);

            using (inner.Push())
            {
                LogContext.Current.Should().Be(inner);
            }

            LogContext.Current.Should().Be(outer);
        }

        LogContext.Current.Should().BeNull();
    }

    [Fact]
    public void ToDictionary_ShouldContainAllProperties()
    {
        // Arrange
        var context = new LogContext("test-corr")
        {
            OperationId = "op-123",
            UserId = "user-456",
            SessionId = "sess-789",
            Component = "TestComp"
        };
        context.SetProperty("Custom", "Value");

        // Act
        var dict = context.ToDictionary();

        // Assert
        dict.Should().ContainKey("CorrelationId").WhoseValue.Should().Be("test-corr");
        dict.Should().ContainKey("OperationId").WhoseValue.Should().Be("op-123");
        dict.Should().ContainKey("UserId").WhoseValue.Should().Be("user-456");
        dict.Should().ContainKey("SessionId").WhoseValue.Should().Be("sess-789");
        dict.Should().ContainKey("Component").WhoseValue.Should().Be("TestComp");
        dict.Should().ContainKey("Custom").WhoseValue.Should().Be("Value");
        dict.Should().ContainKey("MachineName");
        dict.Should().ContainKey("ProcessId");
    }

    [Fact]
    public void MachineName_ShouldBeSet()
    {
        // Act
        var context = new LogContext();

        // Assert
        context.MachineName.Should().Be(Environment.MachineName);
    }

    [Fact]
    public void ProcessId_ShouldBeSet()
    {
        // Act
        var context = new LogContext();

        // Assert
        context.ProcessId.Should().Be(Environment.ProcessId);
    }

    [Fact]
    public void CreatedAt_ShouldBeRecent()
    {
        // Act
        var context = new LogContext();

        // Assert
        context.CreatedAt.Should().BeCloseTo(DateTimeOffset.UtcNow, TimeSpan.FromSeconds(1));
    }

    [Fact]
    public void Constructor_ShouldThrow_WhenCorrelationIdIsNull()
    {
        // Act
        var act = () => new LogContext(null!);

        // Assert
        act.Should().Throw<ArgumentNullException>();
    }
}

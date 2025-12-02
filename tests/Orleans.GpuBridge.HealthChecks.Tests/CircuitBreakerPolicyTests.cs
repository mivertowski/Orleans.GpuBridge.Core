// Orleans.GpuBridge - GPU-native distributed computing for Microsoft Orleans
// Copyright (c) 2025 Michael Ivertowski

using Microsoft.Extensions.Logging;
using Orleans.GpuBridge.HealthChecks.CircuitBreaker;
using Orleans.GpuBridge.HealthChecks.Exceptions;

namespace Orleans.GpuBridge.HealthChecks.Tests;

/// <summary>
/// Tests for <see cref="CircuitBreakerPolicy"/> implementation.
/// </summary>
public class CircuitBreakerPolicyTests
{
    private readonly Mock<ILogger<CircuitBreakerPolicy>> _loggerMock;
    private readonly CircuitBreakerPolicy _policy;

    public CircuitBreakerPolicyTests()
    {
        _loggerMock = new Mock<ILogger<CircuitBreakerPolicy>>();
        _policy = new CircuitBreakerPolicy(_loggerMock.Object);
    }

    [Fact]
    public async Task ExecuteAsync_WithResult_ShouldReturnResult_WhenOperationSucceeds()
    {
        // Arrange
        var expectedResult = 42;
        Func<Task<int>> operation = () => Task.FromResult(expectedResult);

        // Act
        var result = await _policy.ExecuteAsync(operation, "TestOperation");

        // Assert
        result.Should().Be(expectedResult);
    }

    [Fact]
    public async Task ExecuteAsync_WithoutResult_ShouldComplete_WhenOperationSucceeds()
    {
        // Arrange
        var executed = false;
        Func<Task> operation = () =>
        {
            executed = true;
            return Task.CompletedTask;
        };

        // Act
        await _policy.ExecuteAsync(operation, "TestOperation");

        // Assert
        executed.Should().BeTrue();
    }

    [Fact]
    public void GetCircuitState_ShouldReturnClosed_ForUnknownOperation()
    {
        // Act
        var state = _policy.GetCircuitState("UnknownOperation");

        // Assert
        state.Should().Be(CircuitState.Closed);
    }

    [Fact]
    public async Task GetCircuitState_ShouldReturnClosed_AfterSuccessfulExecution()
    {
        // Arrange
        await _policy.ExecuteAsync(() => Task.FromResult(1), "TestOperation");

        // Act
        var state = _policy.GetCircuitState("TestOperation");

        // Assert
        state.Should().Be(CircuitState.Closed);
    }

    [Fact]
    public async Task Reset_ShouldSetCircuitStateToClosed()
    {
        // Arrange - execute something first to create the policy
        await _policy.ExecuteAsync(() => Task.FromResult(1), "TestOperation");

        // Act
        _policy.Reset("TestOperation");
        var state = _policy.GetCircuitState("TestOperation");

        // Assert
        state.Should().Be(CircuitState.Closed);
    }

    [Fact]
    public async Task Isolate_ShouldSetCircuitStateToIsolated()
    {
        // Arrange - execute something first to create the policy
        await _policy.ExecuteAsync(() => Task.FromResult(1), "TestOperation");

        // Act
        _policy.Isolate("TestOperation");
        var state = _policy.GetCircuitState("TestOperation");

        // Assert
        state.Should().Be(CircuitState.Isolated);
    }

    [Fact]
    public void Reset_ShouldNotThrow_ForUnknownOperation()
    {
        // Act
        var act = () => _policy.Reset("UnknownOperation");

        // Assert
        act.Should().NotThrow();
    }

    [Fact]
    public void Isolate_ShouldNotThrow_ForUnknownOperation()
    {
        // Act
        var act = () => _policy.Isolate("UnknownOperation");

        // Assert
        act.Should().NotThrow();
    }

    [Fact]
    public async Task OnCircuitReset_ShouldBeRaised_WhenResetCalled()
    {
        // Arrange
        string? raisedOperation = null;
        _policy.OnCircuitReset += op => raisedOperation = op;
        await _policy.ExecuteAsync(() => Task.FromResult(1), "TestOperation");

        // Act
        _policy.Reset("TestOperation");

        // Assert
        raisedOperation.Should().Be("TestOperation");
    }

    [Fact]
    public async Task OnCircuitBreak_ShouldBeRaised_WhenIsolateCalled()
    {
        // Arrange
        string? raisedOperation = null;
        TimeSpan raisedDuration = default;
        _policy.OnCircuitBreak += (op, duration) =>
        {
            raisedOperation = op;
            raisedDuration = duration;
        };
        await _policy.ExecuteAsync(() => Task.FromResult(1), "TestOperation");

        // Act
        _policy.Isolate("TestOperation");

        // Assert
        raisedOperation.Should().Be("TestOperation");
        raisedDuration.Should().Be(TimeSpan.FromSeconds(30));
    }

    [Fact]
    public async Task ExecuteAsync_ShouldUseCustomOptions()
    {
        // Arrange
        var options = new CircuitBreakerOptions
        {
            RetryCount = 1,
            RetryDelayMs = 10,
            OperationTimeout = TimeSpan.FromSeconds(5)
        };
        var policy = new CircuitBreakerPolicy(_loggerMock.Object, options);

        // Act
        var result = await policy.ExecuteAsync(() => Task.FromResult(42), "TestOperation");

        // Assert
        result.Should().Be(42);
    }

    [Fact]
    public async Task ExecuteAsync_ShouldReusePolicy_ForSameOperationName()
    {
        // Arrange & Act
        await _policy.ExecuteAsync(() => Task.FromResult(1), "TestOperation");
        await _policy.ExecuteAsync(() => Task.FromResult(2), "TestOperation");
        var state1 = _policy.GetCircuitState("TestOperation");

        await _policy.ExecuteAsync(() => Task.FromResult(3), "OtherOperation");
        var state2 = _policy.GetCircuitState("OtherOperation");

        // Assert - both should be tracked separately
        state1.Should().Be(CircuitState.Closed);
        state2.Should().Be(CircuitState.Closed);
    }

    [Fact]
    public async Task ExecuteAsync_ShouldThrow_WhenOperationThrowsNonRetryableException()
    {
        // Arrange
        Func<Task<int>> operation = () => throw new OperationCanceledException();

        // Act
        var act = () => _policy.ExecuteAsync(operation, "TestOperation");

        // Assert
        await act.Should().ThrowAsync<OperationCanceledException>();
    }
}

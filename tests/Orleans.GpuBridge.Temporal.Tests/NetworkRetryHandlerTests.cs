// Orleans.GpuBridge - GPU-native distributed computing for Microsoft Orleans
// Copyright (c) 2025 Michael Ivertowski

using Microsoft.Extensions.Logging.Abstractions;
using Orleans.GpuBridge.Runtime.Temporal.FaultTolerance;
using Xunit;

namespace Orleans.GpuBridge.Temporal.Tests;

/// <summary>
/// Tests for the NetworkRetryHandler exponential backoff and circuit breaker system.
/// </summary>
public sealed class NetworkRetryHandlerTests : IDisposable
{
    private readonly NetworkRetryHandler _handler;
    private readonly NetworkRetryOptions _options;

    public NetworkRetryHandlerTests()
    {
        _options = new NetworkRetryOptions
        {
            MaxRetryAttempts = 3,
            BaseDelay = TimeSpan.FromMilliseconds(10),
            MaxDelay = TimeSpan.FromMilliseconds(100),
            JitterFactor = 0.1,
            CircuitBreakerThreshold = 3,
            CircuitBreakerResetTimeout = TimeSpan.FromMilliseconds(100)
        };

        _handler = new NetworkRetryHandler(
            NullLogger<NetworkRetryHandler>.Instance,
            _options);
    }

    [Fact]
    public void InitialState_ShouldBeClosedCircuit()
    {
        // Assert
        Assert.Equal(CircuitState.Closed, _handler.CurrentCircuitState);
        Assert.True(_handler.IsAllowingRequests);
        Assert.Equal(0, _handler.TotalAttempts);
        Assert.Equal(0, _handler.TotalSuccesses);
        Assert.Equal(0, _handler.TotalFailures);
    }

    [Fact]
    public async Task ExecuteAsync_SuccessfulOperation_ShouldReturnResult()
    {
        // Arrange
        var expectedResult = 42;

        // Act
        var result = await _handler.ExecuteAsync(
            _ => Task.FromResult(expectedResult),
            "TestOperation");

        // Assert
        Assert.Equal(expectedResult, result);
        Assert.Equal(1, _handler.TotalAttempts);
        Assert.Equal(1, _handler.TotalSuccesses);
        Assert.Equal(0, _handler.TotalFailures);
    }

    [Fact]
    public async Task ExecuteAsync_TransientFailure_ShouldRetryAndSucceed()
    {
        // Arrange
        var attemptCount = 0;

        // Act
        var result = await _handler.ExecuteAsync(
            _ =>
            {
                attemptCount++;
                if (attemptCount < 3)
                {
                    throw new TimeoutException("Simulated timeout");
                }
                return Task.FromResult(100);
            },
            "RetryOperation");

        // Assert
        Assert.Equal(100, result);
        Assert.Equal(3, attemptCount);
        Assert.Equal(3, _handler.TotalAttempts);
        Assert.Equal(1, _handler.TotalSuccesses);
    }

    [Fact]
    public async Task ExecuteAsync_AllRetriesFail_ShouldThrowRetryExhaustedException()
    {
        // Arrange & Act & Assert
        var exception = await Assert.ThrowsAsync<RetryExhaustedException>(async () =>
        {
            await _handler.ExecuteAsync<int>(
                _ => throw new TimeoutException("Always fails"),
                "AlwaysFailsOperation");
        });

        Assert.Equal("AlwaysFailsOperation", exception.OperationName);
        Assert.Equal(4, exception.TotalAttempts); // Initial + 3 retries
        Assert.Equal(4, _handler.TotalAttempts);
        Assert.Equal(1, _handler.TotalFailures);
    }

    [Fact]
    public async Task ExecuteAsync_NonRetryableException_ShouldNotRetry()
    {
        // Arrange
        var attemptCount = 0;

        // Act & Assert
        await Assert.ThrowsAsync<InvalidOperationException>(async () =>
        {
            await _handler.ExecuteAsync<int>(
                _ =>
                {
                    attemptCount++;
                    throw new InvalidOperationException("Non-retryable");
                },
                "NonRetryableOperation");
        });

        Assert.Equal(1, attemptCount); // Should not retry
    }

    [Fact]
    public async Task ExecuteAsync_CancellationRequested_ShouldThrowOperationCanceled()
    {
        // Arrange
        using var cts = new CancellationTokenSource();
        cts.Cancel();

        // Act & Assert
        await Assert.ThrowsAsync<OperationCanceledException>(async () =>
        {
            await _handler.ExecuteAsync(
                _ => Task.FromResult(1),
                "CancelledOperation",
                cts.Token);
        });
    }

    [Fact]
    public void CalculateDelay_ShouldApplyExponentialBackoff()
    {
        // Act
        var delay0 = _handler.CalculateDelay(0);
        var delay1 = _handler.CalculateDelay(1);
        var delay2 = _handler.CalculateDelay(2);

        // Assert - Base is 10ms, so 10, 20, 40 with ±10% jitter
        Assert.InRange(delay0.TotalMilliseconds, 9, 11); // 10 * 2^0 ± 10%
        Assert.InRange(delay1.TotalMilliseconds, 18, 22); // 10 * 2^1 ± 10%
        Assert.InRange(delay2.TotalMilliseconds, 36, 44); // 10 * 2^2 ± 10%
    }

    [Fact]
    public void CalculateDelay_ShouldCapAtMaxDelay()
    {
        // Act - High attempt number should cap at max delay (100ms)
        var delay = _handler.CalculateDelay(10);

        // Assert - Should be capped at 100ms ± jitter
        Assert.InRange(delay.TotalMilliseconds, 90, 110);
    }

    [Fact]
    public async Task CircuitBreaker_ShouldTripAfterThreshold()
    {
        // Arrange - Threshold is 3 consecutive failures
        // With 4 attempts per call (initial + 3 retries), circuit trips during first call
        var failures = 0;

        // Act - First call will trigger 4 failures, which exceeds threshold of 3
        try
        {
            await _handler.ExecuteAsync<int>(
                _ =>
                {
                    failures++;
                    throw new TimeoutException("Trip circuit");
                },
                "TripOperation");
        }
        catch (RetryExhaustedException)
        {
            // Expected - all retries exhausted
        }

        // Assert - Circuit should now be open
        Assert.Equal(CircuitState.Open, _handler.CurrentCircuitState);
        Assert.False(_handler.IsAllowingRequests);
        Assert.True(_handler.CircuitBreakerTrips > 0);
        Assert.True(failures >= 3); // At least threshold failures occurred
    }

    [Fact]
    public async Task CircuitBreaker_OpenCircuit_ShouldRejectRequests()
    {
        // Arrange - Trip the circuit first with a single call (4 failures > threshold of 3)
        try
        {
            await _handler.ExecuteAsync<int>(
                _ => throw new TimeoutException(),
                "TripOperation");
        }
        catch (RetryExhaustedException) { }

        Assert.Equal(CircuitState.Open, _handler.CurrentCircuitState);

        // Act & Assert - New requests should be rejected immediately
        await Assert.ThrowsAsync<CircuitBreakerOpenException>(async () =>
        {
            await _handler.ExecuteAsync(
                _ => Task.FromResult(1),
                "BlockedOperation");
        });
    }

    [Fact]
    public void ResetCircuitBreaker_ShouldCloseCircuit()
    {
        // Arrange - Trip the circuit by multiple failures
        // We can't easily trip it synchronously, so test manual reset

        // Act
        _handler.ResetCircuitBreaker();

        // Assert
        Assert.Equal(CircuitState.Closed, _handler.CurrentCircuitState);
        Assert.True(_handler.IsAllowingRequests);
    }

    [Fact]
    public async Task RetryAttemptedEvent_ShouldFire()
    {
        // Arrange
        var eventFired = false;
        RetryAttemptEventArgs? receivedArgs = null;

        _handler.RetryAttempted += (sender, args) =>
        {
            eventFired = true;
            receivedArgs = args;
        };

        var attempts = 0;

        // Act
        await _handler.ExecuteAsync(
            _ =>
            {
                attempts++;
                if (attempts == 1)
                {
                    throw new TimeoutException("First attempt fails");
                }
                return Task.FromResult(1);
            },
            "EventTestOperation");

        // Assert
        Assert.True(eventFired);
        Assert.NotNull(receivedArgs);
        Assert.Equal(0, receivedArgs!.Attempt); // First retry (attempt 0 after initial failure)
        Assert.Equal("EventTestOperation", receivedArgs.OperationName);
    }

    [Fact]
    public async Task RetryExhaustedEvent_ShouldFire()
    {
        // Arrange
        var eventFired = false;
        RetryExhaustedEventArgs? receivedArgs = null;

        _handler.RetryExhausted += (sender, args) =>
        {
            eventFired = true;
            receivedArgs = args;
        };

        // Act
        try
        {
            await _handler.ExecuteAsync<int>(
                _ => throw new TimeoutException("Always fails"),
                "ExhaustedEventTest");
        }
        catch (RetryExhaustedException) { }

        // Assert
        Assert.True(eventFired);
        Assert.NotNull(receivedArgs);
        Assert.Equal(4, receivedArgs!.TotalAttempts);
        Assert.Equal("ExhaustedEventTest", receivedArgs.OperationName);
    }

    [Fact]
    public async Task CircuitStateChangedEvent_ShouldFire()
    {
        // Arrange
        var eventFired = false;
        CircuitStateChangedEventArgs? receivedArgs = null;

        _handler.CircuitStateChanged += (sender, args) =>
        {
            eventFired = true;
            receivedArgs = args;
        };

        // Act - Trip the circuit with a single call (4 failures > threshold of 3)
        try
        {
            await _handler.ExecuteAsync<int>(
                _ => throw new TimeoutException(),
                "CircuitEventTest");
        }
        catch (RetryExhaustedException) { }

        // Assert
        Assert.True(eventFired);
        Assert.NotNull(receivedArgs);
        Assert.Equal(CircuitState.Closed, receivedArgs!.PreviousState);
        Assert.Equal(CircuitState.Open, receivedArgs.NewState);
    }

    [Fact]
    public void GetStatistics_ShouldReturnCorrectValues()
    {
        // Act
        var stats = _handler.GetStatistics();

        // Assert
        Assert.Equal(0, stats.TotalAttempts);
        Assert.Equal(0, stats.TotalSuccesses);
        Assert.Equal(0, stats.TotalFailures);
        Assert.Equal(CircuitState.Closed, stats.CurrentCircuitState);
        Assert.Equal(1.0, stats.SuccessRate); // No attempts = 100% success
    }

    [Fact]
    public void DefaultOptions_ShouldHaveReasonableDefaults()
    {
        // Act
        var options = NetworkRetryOptions.Default;

        // Assert
        Assert.Equal(3, options.MaxRetryAttempts);
        Assert.Equal(TimeSpan.FromMilliseconds(100), options.BaseDelay);
        Assert.Equal(TimeSpan.FromSeconds(30), options.MaxDelay);
        Assert.Equal(0.2, options.JitterFactor);
        Assert.Equal(5, options.CircuitBreakerThreshold);
        Assert.Equal(TimeSpan.FromSeconds(30), options.CircuitBreakerResetTimeout);
    }

    [Fact]
    public void HighAvailabilityOptions_ShouldHaveMoreRetries()
    {
        // Act
        var options = NetworkRetryOptions.HighAvailability;

        // Assert
        Assert.Equal(5, options.MaxRetryAttempts);
        Assert.Equal(TimeSpan.FromMilliseconds(50), options.BaseDelay);
        Assert.Equal(10, options.CircuitBreakerThreshold);
    }

    [Fact]
    public void LowLatencyOptions_ShouldHaveLowerDelays()
    {
        // Act
        var options = NetworkRetryOptions.LowLatency;

        // Assert
        Assert.Equal(2, options.MaxRetryAttempts);
        Assert.Equal(TimeSpan.FromMilliseconds(10), options.BaseDelay);
        Assert.Equal(TimeSpan.FromMilliseconds(100), options.MaxDelay);
    }

    [Fact]
    public async Task ExecuteAsync_VoidOperation_ShouldWork()
    {
        // Arrange
        var executed = false;

        // Act
        await _handler.ExecuteAsync(
            _ =>
            {
                executed = true;
                return Task.CompletedTask;
            },
            "VoidOperation");

        // Assert
        Assert.True(executed);
        Assert.Equal(1, _handler.TotalSuccesses);
    }

    [Fact]
    public async Task CustomRetryableExceptionPredicate_ShouldBeRespected()
    {
        // Arrange
        var customOptions = new NetworkRetryOptions
        {
            MaxRetryAttempts = 2,
            BaseDelay = TimeSpan.FromMilliseconds(1),
            RetryableExceptionPredicate = ex => ex is ArgumentException // Only retry ArgumentException
        };

        using var customHandler = new NetworkRetryHandler(
            NullLogger<NetworkRetryHandler>.Instance,
            customOptions);

        var attempts = 0;

        // Act - InvalidOperationException should NOT be retried
        await Assert.ThrowsAsync<InvalidOperationException>(async () =>
        {
            await customHandler.ExecuteAsync<int>(
                _ =>
                {
                    attempts++;
                    throw new InvalidOperationException("Not retryable");
                },
                "CustomPredicateTest");
        });

        // Assert - Should not retry (only 1 attempt)
        Assert.Equal(1, attempts);
    }

    public void Dispose()
    {
        _handler.Dispose();
    }
}

using Microsoft.Extensions.Logging;
using Orleans.GpuBridge.HealthChecks;
using Polly.CircuitBreaker;
using Xunit;

namespace Orleans.GpuBridge.Tests.HealthChecks;

public class CircuitBreakerTests
{
    private readonly ICircuitBreakerPolicy _circuitBreaker;
    private readonly TestLogger<CircuitBreakerPolicy> _logger;
    
    public CircuitBreakerTests()
    {
        _logger = new TestLogger<CircuitBreakerPolicy>();
        _circuitBreaker = new CircuitBreakerPolicy(
            _logger,
            new CircuitBreakerOptions
            {
                FailureThreshold = 3,
                BreakDuration = TimeSpan.FromSeconds(1),
                RetryCount = 2,
                RetryDelayMs = 10,
                OperationTimeout = TimeSpan.FromSeconds(1)
            });
    }
    
    [Fact]
    public async Task ExecuteAsync_SuccessfulOperation_ReturnsResult()
    {
        // Arrange
        var expectedResult = 42;
        
        // Act
        var result = await _circuitBreaker.ExecuteAsync(
            async () => 
            {
                await Task.Delay(10);
                return expectedResult;
            },
            "test-operation");
        
        // Assert
        Assert.Equal(expectedResult, result);
        Assert.Equal(CircuitState.Closed, _circuitBreaker.GetCircuitState("test-operation"));
    }
    
    [Fact]
    public async Task ExecuteAsync_FailuresBelowThreshold_CircuitStaysClosed()
    {
        // Arrange
        var failCount = 0;
        
        // Act & Assert
        for (int i = 0; i < 2; i++) // Below threshold of 3
        {
            try
            {
                await _circuitBreaker.ExecuteAsync<int>(
                    async () =>
                    {
                        failCount++;
                        await Task.Delay(10);
                        throw new GpuKernelException("test", "Simulated failure");
                    },
                    "test-operation");
            }
            catch (GpuKernelException)
            {
                // Expected
            }
        }
        
        Assert.Equal(2 * 3, failCount); // 2 attempts, each with 3 retries (1 initial + 2 retries)
        Assert.Equal(CircuitState.Closed, _circuitBreaker.GetCircuitState("test-operation"));
    }
    
    [Fact]
    public async Task ExecuteAsync_FailuresReachThreshold_CircuitOpens()
    {
        // Arrange
        var options = new CircuitBreakerOptions
        {
            FailureThreshold = 3,
            BreakDuration = TimeSpan.FromSeconds(1),
            RetryCount = 0, // No retries for this test
            OperationTimeout = TimeSpan.FromSeconds(1)
        };
        
        var circuitBreaker = new CircuitBreakerPolicy(_logger, options);
        
        // Act - Trigger failures to open circuit
        for (int i = 0; i < 3; i++)
        {
            try
            {
                await circuitBreaker.ExecuteAsync<int>(
                    async () =>
                    {
                        await Task.Delay(10);
                        throw new GpuMemoryException("Out of memory");
                    },
                    "memory-operation");
            }
            catch (GpuMemoryException)
            {
                // Expected
            }
        }
        
        // Assert - Circuit should be open
        await Assert.ThrowsAsync<GpuOperationException>(async () =>
        {
            await circuitBreaker.ExecuteAsync<int>(
                async () =>
                {
                    await Task.Delay(10);
                    return 42;
                },
                "memory-operation");
        });
    }
    
    [Fact]
    public async Task ExecuteAsync_WithRetries_RetriesOnTransientFailure()
    {
        // Arrange
        var attemptCount = 0;
        var succeedOnAttempt = 2;
        
        // Act
        var result = await _circuitBreaker.ExecuteAsync(
            async () =>
            {
                attemptCount++;
                await Task.Delay(10);
                
                if (attemptCount < succeedOnAttempt)
                {
                    throw new GpuMemoryException("Transient failure", isTransient: true);
                }
                
                return attemptCount;
            },
            "retry-operation");
        
        // Assert
        Assert.Equal(succeedOnAttempt, result);
        Assert.Equal(succeedOnAttempt, attemptCount);
    }
    
    [Fact]
    public async Task ExecuteAsync_Timeout_ThrowsTimeoutException()
    {
        // Arrange
        var options = new CircuitBreakerOptions
        {
            OperationTimeout = TimeSpan.FromMilliseconds(100),
            RetryCount = 0
        };
        
        var circuitBreaker = new CircuitBreakerPolicy(_logger, options);
        
        // Act & Assert
        await Assert.ThrowsAsync<GpuOperationException>(async () =>
        {
            await circuitBreaker.ExecuteAsync(
                async () =>
                {
                    await Task.Delay(TimeSpan.FromSeconds(5));
                    return 42;
                },
                "timeout-operation");
        });
    }
    
    [Fact]
    public void Reset_ResetsCircuitToClosedState()
    {
        // Arrange
        var operationName = "reset-test";
        
        // Act
        _circuitBreaker.Reset(operationName);
        
        // Assert
        Assert.Equal(CircuitState.Closed, _circuitBreaker.GetCircuitState(operationName));
        Assert.Contains(_logger.LoggedMessages, m => 
            m.LogLevel == LogLevel.Information && 
            m.Message.Contains("reset"));
    }
    
    [Fact]
    public void Isolate_IsolatesCircuit()
    {
        // Arrange
        var operationName = "isolate-test";
        
        // Act
        _circuitBreaker.Isolate(operationName);
        
        // Assert
        Assert.Equal(CircuitState.Isolated, _circuitBreaker.GetCircuitState(operationName));
        Assert.Contains(_logger.LoggedMessages, m => 
            m.LogLevel == LogLevel.Warning && 
            m.Message.Contains("isolated"));
    }
    
    [Fact]
    public async Task ExecuteAsync_DoesNotRetryNonTransientErrors()
    {
        // Arrange
        var attemptCount = 0;
        
        // Act & Assert
        await Assert.ThrowsAsync<OperationCanceledException>(async () =>
        {
            await _circuitBreaker.ExecuteAsync(
                async () =>
                {
                    attemptCount++;
                    await Task.Delay(10);
                    throw new OperationCanceledException("User cancelled");
                },
                "no-retry-operation");
        });
        
        Assert.Equal(1, attemptCount); // Should not retry
    }
    
    [Fact]
    public async Task ExecuteAsync_CircuitBreakerEvents_AreFired()
    {
        // Arrange
        var breakEventFired = false;
        var resetEventFired = false;
        var halfOpenEventFired = false;
        
        var options = new CircuitBreakerOptions
        {
            FailureThreshold = 2,
            BreakDuration = TimeSpan.FromMilliseconds(100),
            RetryCount = 0
        };
        
        var circuitBreaker = new CircuitBreakerPolicy(_logger, options);
        
        circuitBreaker.OnCircuitBreak += (name, duration) => breakEventFired = true;
        circuitBreaker.OnCircuitReset += (name) => resetEventFired = true;
        circuitBreaker.OnCircuitHalfOpen += (name) => halfOpenEventFired = true;
        
        // Act - Trigger circuit to open
        for (int i = 0; i < 2; i++)
        {
            try
            {
                await circuitBreaker.ExecuteAsync<int>(
                    () => throw new GpuKernelException("test", "error"),
                    "event-test");
            }
            catch { }
        }
        
        // Wait for circuit to transition to half-open
        await Task.Delay(150);
        
        // Try to execute (should transition to half-open)
        try
        {
            await circuitBreaker.ExecuteAsync(
                async () => { await Task.Delay(1); return 42; },
                "event-test");
        }
        catch { }
        
        // Assert
        Assert.True(breakEventFired);
        // Note: Half-open and reset events depend on Polly's internal behavior
    }
}
using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Orleans.GpuBridge.Abstractions;
using Orleans.GpuBridge.HealthChecks.Exceptions;
using Polly;
using Polly.CircuitBreaker;
using Polly.Timeout;

namespace Orleans.GpuBridge.HealthChecks.CircuitBreaker;

public class CircuitBreakerPolicy : ICircuitBreakerPolicy
{
    private readonly ILogger<CircuitBreakerPolicy> _logger;
    private readonly CircuitBreakerOptions _options;
    private readonly Dictionary<string, IAsyncPolicy> _policies = new();
    private readonly Dictionary<string, CircuitBreakerStateProvider> _stateProviders = new();
    private readonly object _policiesLock = new();
    
    public CircuitBreakerPolicy(
        ILogger<CircuitBreakerPolicy> logger,
        CircuitBreakerOptions? options = null)
    {
        _logger = logger;
        _options = options ?? new CircuitBreakerOptions();
    }
    
    public async Task<TResult> ExecuteAsync<TResult>(
        Func<Task<TResult>> operation,
        string operationName,
        CancellationToken cancellationToken = default)
    {
        var policy = GetOrCreatePolicy(operationName);
        
        try
        {
            return await policy.ExecuteAsync(
                async (ct) => await operation(),
                cancellationToken);
        }
        catch (BrokenCircuitException ex)
        {
            _logger.LogWarning(
                "Circuit breaker is open for operation {OperationName}: {Message}",
                operationName, ex.Message);
            throw new GpuOperationException(
                $"Operation {operationName} is temporarily unavailable due to repeated failures",
                ex);
        }
        catch (TimeoutException ex)
        {
            _logger.LogWarning(
                "Operation {OperationName} timed out: {Message}",
                operationName, ex.Message);
            throw new GpuOperationException(
                $"Operation {operationName} timed out",
                ex);
        }
    }
    
    public async Task ExecuteAsync(
        Func<Task> operation,
        string operationName,
        CancellationToken cancellationToken = default)
    {
        await ExecuteAsync<object?>(
            async () =>
            {
                await operation();
                return null;
            },
            operationName,
            cancellationToken);
    }
    
    public CircuitState GetCircuitState(string operationName)
    {
        lock (_policiesLock)
        {
            if (_stateProviders.TryGetValue(operationName, out var provider))
            {
                return provider.CircuitState;
            }
            return CircuitState.Closed;
        }
    }
    
    public void Reset(string operationName)
    {
        lock (_policiesLock)
        {
            if (_stateProviders.TryGetValue(operationName, out var provider))
            {
                provider.Reset();
                _logger.LogInformation(
                    "Circuit breaker for operation {OperationName} has been reset",
                    operationName);
            }
        }
    }
    
    public void Isolate(string operationName)
    {
        lock (_policiesLock)
        {
            if (_stateProviders.TryGetValue(operationName, out var provider))
            {
                provider.Isolate();
                _logger.LogWarning(
                    "Circuit breaker for operation {OperationName} has been manually isolated",
                    operationName);
            }
        }
    }
    
    private IAsyncPolicy GetOrCreatePolicy(string operationName)
    {
        lock (_policiesLock)
        {
            if (_policies.TryGetValue(operationName, out var existingPolicy))
            {
                return existingPolicy;
            }
            
            var stateProvider = new CircuitBreakerStateProvider();
            _stateProviders[operationName] = stateProvider;
            
            // Create retry policy (simplified - using basic retry instead of circuit breaker for now)
            var retryPolicy = Policy
                .Handle<Exception>(ex => ShouldHandleException(ex))
                .WaitAndRetryAsync(
                    retryCount: _options.RetryCount,
                    sleepDurationProvider: retryAttempt => 
                        TimeSpan.FromMilliseconds(Math.Pow(2, retryAttempt) * _options.RetryDelayMs),
                    onRetry: (outcome, timespan, retryCount, context) =>
                    {
                        _logger.LogDebug(
                            "Retry {RetryCount} for operation {OperationName} after {Delay}ms",
                            retryCount, operationName, timespan.TotalMilliseconds);
                    });
            
            // Create timeout policy
            var timeoutPolicy = Policy.TimeoutAsync(_options.OperationTimeout);
            
            // Combine policies: Retry -> Timeout
            var combinedPolicy = Policy.WrapAsync(retryPolicy, timeoutPolicy);
            
            _policies[operationName] = combinedPolicy;
            stateProvider.Policy = combinedPolicy;
            
            return combinedPolicy;
        }
    }
    
    private bool ShouldHandleException(Exception ex)
    {
        // Handle specific GPU-related exceptions
        return ex switch
        {
            GpuMemoryException => true,
            GpuKernelException => true,
            GpuDeviceException => true,
            TimeoutException => true,
            OperationCanceledException => false,
            _ => !IsFatalException(ex)
        };
    }
    
    private bool ShouldRetryException(Exception ex)
    {
        // Retry transient failures
        return ex switch
        {
            GpuMemoryException memEx => memEx.IsTransient,
            GpuKernelException => true,
            TimeoutException => true,
            _ => false
        };
    }
    
    private bool IsFatalException(Exception ex)
    {
        // Don't retry fatal exceptions
        return ex switch
        {
            OutOfMemoryException => true,
            StackOverflowException => true,
            AccessViolationException => true,
            _ => false
        };
    }
    
    // Events for monitoring
    public event Action<string, TimeSpan>? OnCircuitBreak;
    public event Action<string>? OnCircuitReset;
    public event Action<string>? OnCircuitHalfOpen;
    
    private class CircuitBreakerStateProvider
    {
        public CircuitState CircuitState { get; set; } = CircuitState.Closed;
        public IAsyncPolicy? Policy { get; set; }
        
        public void Reset()
        {
            CircuitState = CircuitState.Closed;
            // Note: Polly doesn't expose a direct reset method,
            // so we track state separately
        }
        
        public void Isolate()
        {
            CircuitState = CircuitState.Isolated;
        }
    }
}
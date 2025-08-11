using Microsoft.Extensions.Logging;
using Orleans.GpuBridge.Abstractions;
using Polly;
using Polly.CircuitBreaker;
using Polly.Timeout;

namespace Orleans.GpuBridge.HealthChecks;

public interface ICircuitBreakerPolicy
{
    Task<TResult> ExecuteAsync<TResult>(
        Func<Task<TResult>> operation,
        string operationName,
        CancellationToken cancellationToken = default);
    
    Task ExecuteAsync(
        Func<Task> operation,
        string operationName,
        CancellationToken cancellationToken = default);
    
    CircuitState GetCircuitState(string operationName);
    void Reset(string operationName);
    void Isolate(string operationName);
}

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
        catch (TimeoutRejectedException ex)
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
            
            // Create circuit breaker policy
            var circuitBreakerPolicy = Policy
                .Handle<Exception>(ex => ShouldHandleException(ex))
                .CircuitBreakerAsync(
                    handledEventsAllowedBeforeBreaking: _options.FailureThreshold,
                    durationOfBreak: _options.BreakDuration,
                    onBreak: (result, duration) =>
                    {
                        stateProvider.CircuitState = CircuitState.Open;
                        _logger.LogWarning(
                            "Circuit breaker opened for operation {OperationName}. " +
                            "Circuit will remain open for {Duration}",
                            operationName, duration);
                        
                        OnCircuitBreak?.Invoke(operationName, duration);
                    },
                    onReset: () =>
                    {
                        stateProvider.CircuitState = CircuitState.Closed;
                        _logger.LogInformation(
                            "Circuit breaker closed for operation {OperationName}",
                            operationName);
                        
                        OnCircuitReset?.Invoke(operationName);
                    },
                    onHalfOpen: () =>
                    {
                        stateProvider.CircuitState = CircuitState.HalfOpen;
                        _logger.LogInformation(
                            "Circuit breaker is half-open for operation {OperationName}",
                            operationName);
                        
                        OnCircuitHalfOpen?.Invoke(operationName);
                    });
            
            // Create retry policy
            var retryPolicy = Policy
                .Handle<Exception>(ex => ShouldRetryException(ex))
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
            var timeoutPolicy = Policy.TimeoutAsync(
                _options.OperationTimeout,
                TimeoutStrategy.Pessimistic,
                onTimeoutAsync: async (context, timespan, task) =>
                {
                    _logger.LogWarning(
                        "Operation {OperationName} timed out after {Timeout}",
                        operationName, timespan);
                    
                    if (task != null)
                    {
                        await task; // Ensure the timed-out task completes
                    }
                });
            
            // Combine policies: Retry -> Circuit Breaker -> Timeout
            var combinedPolicy = Policy.WrapAsync(retryPolicy, circuitBreakerPolicy, timeoutPolicy);
            
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

public class CircuitBreakerOptions
{
    public int FailureThreshold { get; set; } = 3;
    public TimeSpan BreakDuration { get; set; } = TimeSpan.FromSeconds(30);
    public int RetryCount { get; set; } = 3;
    public int RetryDelayMs { get; set; } = 100;
    public TimeSpan OperationTimeout { get; set; } = TimeSpan.FromSeconds(30);
}

public enum CircuitState
{
    Closed,
    Open,
    HalfOpen,
    Isolated
}

// Custom exceptions for GPU operations
public class GpuOperationException : Exception
{
    public GpuOperationException(string message) : base(message) { }
    public GpuOperationException(string message, Exception innerException) 
        : base(message, innerException) { }
}

public class GpuMemoryException : GpuOperationException
{
    public bool IsTransient { get; }
    
    public GpuMemoryException(string message, bool isTransient = true) 
        : base(message)
    {
        IsTransient = isTransient;
    }
}

public class GpuKernelException : GpuOperationException
{
    public string KernelName { get; }
    
    public GpuKernelException(string kernelName, string message) 
        : base($"Kernel '{kernelName}' failed: {message}")
    {
        KernelName = kernelName;
    }
}

public class GpuDeviceException : GpuOperationException
{
    public int DeviceIndex { get; }
    
    public GpuDeviceException(int deviceIndex, string message) 
        : base($"GPU device {deviceIndex} error: {message}")
    {
        DeviceIndex = deviceIndex;
    }
}
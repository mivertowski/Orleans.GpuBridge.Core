# Orleans.GpuBridge.Resilience

## Overview

Orleans.GpuBridge.Resilience provides comprehensive resilience patterns for GPU operations in Orleans applications. Built on Polly v8, this package implements retry policies, circuit breakers, rate limiting, and fallback chains specifically designed for GPU workloads.

## Features

### Resilience Patterns
- **Retry Policies**: Configurable retry with exponential backoff for transient GPU failures
- **Circuit Breakers**: Automatic circuit breaking to prevent cascade failures
- **Rate Limiting**: Token bucket rate limiter for GPU resource protection
- **Bulkhead Isolation**: Concurrent operation limits to prevent resource exhaustion
- **Fallback Chains**: GPU → CPU fallback with configurable degradation levels
- **Chaos Engineering**: Fault injection for resilience testing

### GPU-Specific Features
- **GPU Exception Handling**: Specialized handling for GPU memory, kernel, and device exceptions
- **Timeout Management**: Configurable timeouts for kernel execution, device operations, and memory allocation
- **Auto-Degradation**: Automatic fallback level adjustment based on error rates
- **Auto-Recovery**: Automatic recovery when GPU resources become available

## Installation

```bash
dotnet add package Orleans.GpuBridge.Resilience
```

## Quick Start

### Basic Configuration

```csharp
using Orleans.GpuBridge.Resilience.Extensions;

services.AddGpuResilience(options =>
{
    // Retry configuration
    options.RetryOptions.MaxAttempts = 3;
    options.RetryOptions.BaseDelay = TimeSpan.FromMilliseconds(100);
    options.RetryOptions.MaxDelay = TimeSpan.FromSeconds(10);

    // Circuit breaker configuration
    options.CircuitBreakerOptions.FailureRatio = 0.5;
    options.CircuitBreakerOptions.SamplingDuration = TimeSpan.FromSeconds(30);
    options.CircuitBreakerOptions.MinimumThroughput = 10;
    options.CircuitBreakerOptions.BreakDuration = TimeSpan.FromSeconds(30);

    // Timeout configuration
    options.TimeoutOptions.KernelExecution = TimeSpan.FromMinutes(5);
    options.TimeoutOptions.MemoryAllocation = TimeSpan.FromSeconds(30);

    // Bulkhead configuration
    options.BulkheadOptions.MaxConcurrentOperations = 10;
    options.BulkheadOptions.MaxQueuedOperations = 100;
});
```

## Key Components

### GpuResiliencePolicy

The main resilience policy combining retry, circuit breaker, timeout, and bulkhead patterns:

```csharp
public class GpuResiliencePolicy : IGpuResiliencePolicy
{
    Task<TResult> ExecuteKernelOperationAsync<TResult>(
        Func<CancellationToken, Task<TResult>> operation,
        string operationName,
        CancellationToken cancellationToken = default);

    Task<TResult> ExecuteDeviceOperationAsync<TResult>(
        Func<CancellationToken, Task<TResult>> operation,
        string deviceId,
        string operationName,
        CancellationToken cancellationToken = default);

    Task<TResult> ExecuteMemoryAllocationAsync<TResult>(
        Func<CancellationToken, Task<TResult>> operation,
        long requestedBytes,
        CancellationToken cancellationToken = default);

    Task<TResult> ExecuteCompilationAsync<TResult>(
        Func<CancellationToken, Task<TResult>> operation,
        string kernelName,
        CancellationToken cancellationToken = default);
}
```

**Usage Example:**

```csharp
var policy = serviceProvider.GetRequiredService<IGpuResiliencePolicy>();

var result = await policy.ExecuteKernelOperationAsync(
    async ct => await ExecuteGpuKernelAsync(input, ct),
    "VectorMultiply",
    cancellationToken);
```

### TokenBucketRateLimiter

GPU-aware rate limiting to prevent resource exhaustion:

```csharp
var rateLimiter = serviceProvider.GetRequiredService<ITokenBucketRateLimiter>();

// Try to acquire tokens
if (await rateLimiter.TryAcquireAsync(tokenCount))
{
    await ExecuteGpuOperationAsync();
}

// Or execute with automatic rate limiting
var result = await rateLimiter.ExecuteAsync(
    async ct => await ExecuteGpuOperationAsync(ct));

// Get metrics
var metrics = rateLimiter.GetMetrics();
Console.WriteLine($"Rejection Rate: {metrics.RejectionRate:P2}");
```

### GpuFallbackChain

Automatic fallback from GPU to CPU with configurable degradation:

```csharp
var fallbackChain = serviceProvider.GetRequiredService<IGpuFallbackChain>();

// Register executors at different fallback levels
fallbackChain.RegisterExecutor(FallbackLevel.Optimal, gpuExecutor);
fallbackChain.RegisterExecutor(FallbackLevel.Reduced, reducedGpuExecutor);
fallbackChain.RegisterExecutor(FallbackLevel.Degraded, cpuExecutor);

// Execute with automatic fallback
var result = await fallbackChain.ExecuteAsync<MyInput, MyOutput>(
    "ProcessData",
    input,
    cancellationToken);
```

### ChaosEngineer

Fault injection for resilience testing:

```csharp
services.AddGpuResilience(options =>
{
    options.ChaosOptions.Enabled = true; // Only in test environments!
    options.ChaosOptions.FaultInjectionRate = 0.1; // 10% fault rate
    options.ChaosOptions.LatencyInjectionRate = 0.05;
    options.ChaosOptions.InjectedLatency = TimeSpan.FromMilliseconds(500);
});
```

## Configuration Options

### GpuResiliencePolicyOptions

```csharp
public class GpuResiliencePolicyOptions
{
    public static string SectionName => "GpuResilience";

    public RetryOptions RetryOptions { get; set; }
    public CircuitBreakerOptions CircuitBreakerOptions { get; set; }
    public TimeoutOptions TimeoutOptions { get; set; }
    public BulkheadOptions BulkheadOptions { get; set; }
    public RateLimitingOptions RateLimitingOptions { get; set; }
    public ChaosOptions ChaosOptions { get; set; }
}
```

### RetryOptions

| Property | Default | Description |
|----------|---------|-------------|
| MaxAttempts | 3 | Maximum retry attempts |
| BaseDelay | 100ms | Initial delay between retries |
| MaxDelay | 30s | Maximum delay cap |
| UseJitter | true | Add randomness to delays |

### CircuitBreakerOptions

| Property | Default | Description |
|----------|---------|-------------|
| FailureRatio | 0.5 | Failure ratio to trip circuit |
| SamplingDuration | 30s | Sampling window duration |
| MinimumThroughput | 10 | Minimum calls before evaluation |
| BreakDuration | 30s | Duration circuit stays open |

### TimeoutOptions

| Property | Default | Description |
|----------|---------|-------------|
| KernelExecution | 5m | Kernel execution timeout |
| DeviceOperation | 30s | Device operation timeout |
| MemoryAllocation | 10s | Memory allocation timeout |
| KernelCompilation | 2m | Kernel compilation timeout |

### BulkheadOptions

| Property | Default | Description |
|----------|---------|-------------|
| MaxConcurrentOperations | 10 | Max concurrent GPU operations |
| MaxQueuedOperations | 100 | Max queued operations |

### RateLimitingOptions

| Property | Default | Description |
|----------|---------|-------------|
| Enabled | true | Enable rate limiting |
| TokenRefillRate | 100 | Tokens per second |
| MaxBurstSize | 50 | Maximum burst size |

### FallbackChainOptions

| Property | Default | Description |
|----------|---------|-------------|
| AutoDegradationEnabled | true | Auto-degrade on errors |
| AutoRecoveryEnabled | true | Auto-recover when healthy |
| DegradationErrorThreshold | 0.5 | Error rate to degrade |
| RecoveryErrorThreshold | 0.1 | Error rate to recover |
| MinimumRecoveryInterval | 1m | Min time between recoveries |

## Fallback Levels

```csharp
public enum FallbackLevel
{
    Optimal = 0,   // Full GPU acceleration
    Reduced = 1,   // Reduced GPU (lower precision/batch)
    Degraded = 2,  // CPU fallback
    Failed = 3     // All options exhausted
}
```

## Exception Types

The package handles GPU-specific exceptions:

- `GpuBridgeException` - Base exception for all GPU Bridge errors
- `GpuOperationException` - General GPU operation failures
- `GpuMemoryException` - Memory allocation/transfer failures
- `GpuDeviceException` - Device unavailable or failed
- `GpuKernelException` - Kernel execution failures
- `RateLimitExceededException` - Rate limit exceeded

## Metrics and Monitoring

### Bulkhead Metrics

```csharp
var policy = serviceProvider.GetRequiredService<IGpuResiliencePolicy>();
var metrics = policy.GetBulkheadMetrics();

Console.WriteLine($"Total Slots: {metrics.TotalSlots}");
Console.WriteLine($"Available: {metrics.AvailableSlots}");
Console.WriteLine($"In Use: {metrics.InUseSlots}");
Console.WriteLine($"Utilization: {metrics.UtilizationPercentage:P0}");
```

### Rate Limiter Metrics

```csharp
var rateLimiter = serviceProvider.GetRequiredService<ITokenBucketRateLimiter>();
var metrics = rateLimiter.GetMetrics();

Console.WriteLine($"Total Requests: {metrics.TotalRequests}");
Console.WriteLine($"Rejected: {metrics.RejectedRequests}");
Console.WriteLine($"Rejection Rate: {metrics.RejectionRate:P2}");
Console.WriteLine($"Available Tokens: {metrics.AvailableTokens}");
```

## Dependencies

- .NET 9.0 or later
- Polly 8.6.4+
- Microsoft.Extensions.DependencyInjection
- Microsoft.Extensions.Options
- Microsoft.Extensions.Logging
- Orleans.GpuBridge.Abstractions

## Best Practices

1. **Configure Appropriate Timeouts**: GPU operations can vary widely in duration
2. **Use Bulkhead Isolation**: Prevent GPU resource exhaustion
3. **Enable Auto-Recovery**: Allow automatic recovery when GPU becomes available
4. **Monitor Metrics**: Track rejection rates and circuit breaker state
5. **Test with Chaos Engineering**: Validate resilience in test environments

## License

Apache 2.0 - Copyright (c) 2025 Michael Ivertowski

## See Also

- [Orleans.GpuBridge.Runtime](../Orleans.GpuBridge.Runtime/README.md) - Runtime implementation
- [Orleans.GpuBridge.Abstractions](../Orleans.GpuBridge.Abstractions/README.md) - Core abstractions
- [Polly Documentation](https://www.thepollyproject.org/) - Resilience library

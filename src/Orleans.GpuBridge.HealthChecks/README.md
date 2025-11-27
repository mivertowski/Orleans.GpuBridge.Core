# Orleans.GpuBridge.HealthChecks

Health checks, circuit breakers, and resilience patterns for Orleans GPU Bridge.

## Overview

Orleans.GpuBridge.HealthChecks provides production-ready resilience patterns for GPU-accelerated applications. It includes ASP.NET Core health check integration, circuit breaker policies, and specialized GPU exception handling.

## Key Features

- **GPU Health Checks**: Monitor GPU device availability and health
- **Circuit Breaker Pattern**: Prevent cascade failures with configurable policies
- **Specialized Exceptions**: GPU-specific exception hierarchy for precise error handling
- **ASP.NET Core Integration**: Standard health check endpoint support
- **Automatic Recovery**: Self-healing with configurable recovery strategies

## Installation

```bash
dotnet add package Orleans.GpuBridge.HealthChecks
```

## Quick Start

### Health Check Configuration

```csharp
using Orleans.GpuBridge.HealthChecks;

var builder = WebApplication.CreateBuilder(args);

builder.Services.AddGpuBridge();
builder.Services.AddHealthChecks()
    .AddGpuHealthCheck(options =>
    {
        options.MinimumAvailableMemoryBytes = 1024 * 1024 * 512; // 512 MB
        options.MaxQueueDepth = 1000;
        options.HealthCheckTimeout = TimeSpan.FromSeconds(5);
    });

var app = builder.Build();
app.MapHealthChecks("/health");
```

### Circuit Breaker Usage

```csharp
using Orleans.GpuBridge.HealthChecks.CircuitBreaker;

public class GpuService
{
    private readonly ICircuitBreakerPolicy _circuitBreaker;

    public GpuService(ICircuitBreakerPolicy circuitBreaker)
    {
        _circuitBreaker = circuitBreaker;
    }

    public async Task<TResult> ExecuteWithResilienceAsync<TResult>(
        Func<Task<TResult>> operation)
    {
        return await _circuitBreaker.ExecuteAsync(operation);
    }
}
```

### Exception Handling

```csharp
try
{
    var result = await kernelCatalog.ExecuteAsync("my_kernel", input);
}
catch (GpuMemoryException ex)
{
    logger.LogWarning("GPU memory allocation failed: {Message}", ex.Message);
    // Retry with smaller batch or fallback to CPU
}
catch (GpuKernelException ex)
{
    logger.LogError(ex, "Kernel execution failed: {KernelId}", ex.KernelId);
    // Handle kernel-specific failure
}
catch (GpuDeviceException ex)
{
    logger.LogCritical(ex, "GPU device failure on device {DeviceIndex}", ex.DeviceIndex);
    // Trigger failover to another device
}
```

## Circuit Breaker Configuration

```csharp
services.AddCircuitBreakerPolicy(options =>
{
    // Failure thresholds
    options.FailureThreshold = 5;
    options.FailureTimeWindow = TimeSpan.FromMinutes(1);

    // Recovery settings
    options.OpenDuration = TimeSpan.FromSeconds(30);
    options.HalfOpenMaxAttempts = 3;

    // Exception filtering
    options.ShouldHandle = ex => ex is GpuOperationException;
});
```

### Circuit States

| State | Description | Behavior |
|-------|-------------|----------|
| `Closed` | Normal operation | Requests pass through, failures tracked |
| `Open` | Circuit tripped | Requests fail immediately |
| `HalfOpen` | Testing recovery | Limited requests allowed |

## Health Check Options

```csharp
public class GpuHealthCheckOptions
{
    // Memory requirements
    public long MinimumAvailableMemoryBytes { get; set; } = 256 * 1024 * 1024;

    // Queue depth limits
    public int MaxQueueDepth { get; set; } = 1000;

    // Timeout settings
    public TimeSpan HealthCheckTimeout { get; set; } = TimeSpan.FromSeconds(10);

    // Temperature thresholds (Celsius)
    public int WarningTemperature { get; set; } = 80;
    public int CriticalTemperature { get; set; } = 90;

    // Device requirements
    public int MinimumDeviceCount { get; set; } = 1;
}
```

## Exception Hierarchy

```
GpuOperationException (base)
├── GpuDeviceException      - Device-level failures
├── GpuMemoryException      - Memory allocation failures
└── GpuKernelException      - Kernel execution failures
```

### GpuDeviceException

```csharp
public class GpuDeviceException : GpuOperationException
{
    public int DeviceIndex { get; }
    public string DeviceName { get; }
    public DeviceErrorCode ErrorCode { get; }
}
```

### GpuMemoryException

```csharp
public class GpuMemoryException : GpuOperationException
{
    public long RequestedBytes { get; }
    public long AvailableBytes { get; }
    public MemoryType MemoryType { get; }
}
```

### GpuKernelException

```csharp
public class GpuKernelException : GpuOperationException
{
    public string KernelId { get; }
    public int ErrorCode { get; }
    public string ErrorMessage { get; }
}
```

## Health Check Response

```json
{
  "status": "Healthy",
  "totalDuration": "00:00:00.1234567",
  "entries": {
    "gpu": {
      "status": "Healthy",
      "data": {
        "deviceCount": 2,
        "totalMemoryBytes": 17179869184,
        "availableMemoryBytes": 12884901888,
        "averageQueueDepth": 45
      }
    }
  }
}
```

## Dependencies

- **Microsoft.Extensions.Diagnostics.HealthChecks** (>= 9.0.0)
- **Orleans.GpuBridge.Abstractions**

## License

MIT License - Copyright (c) 2025 Michael Ivertowski

---

For more information, see the [Orleans.GpuBridge.Core Documentation](https://github.com/mivertowski/Orleans.GpuBridge.Core).

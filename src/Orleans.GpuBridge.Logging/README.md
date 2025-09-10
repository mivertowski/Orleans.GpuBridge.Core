# Orleans.GpuBridge.Logging

A centralized, high-performance logging system for the Orleans GPU Bridge Core project. This system implements a delegate-based architecture for flexible logging targets with production-grade features including structured logging, performance metrics, correlation tracking, and telemetry integration.

## Key Features

- **Delegate-Based Architecture**: Centralized control over multiple logging targets
- **High-Performance Buffering**: Asynchronous batching with backpressure control
- **Structured Logging**: Rich metadata and context enrichment
- **Performance Metrics**: Built-in timing, memory, and custom counters
- **Correlation Tracking**: Distributed tracing with correlation IDs
- **OpenTelemetry Integration**: Native telemetry and observability support
- **Production Ready**: Thread-safe, async, configurable retention and rotation

## Architecture Overview

```
┌─────────────────┐    ┌──────────────┐    ┌─────────────────────┐
│   GpuBridge     │───▶│  LogBuffer   │───▶│ LoggerDelegateManager │
│     Logger      │    │              │    │                     │
└─────────────────┘    └──────────────┘    └─────────────────────┘
                                                       │
                       ┌────────────────────────────────┼────────────────────────┐
                       ▼                                ▼                        ▼
               ┌──────────────┐              ┌─────────────────┐    ┌──────────────────┐
               │   Console    │              │      File       │    │    Telemetry     │
               │   Delegate   │              │    Delegate     │    │    Delegate      │
               └──────────────┘              └─────────────────┘    └──────────────────┘
```

## Quick Start

### 1. Add to Dependency Injection

```csharp
// Using configuration
services.AddGpuBridgeLogging(configuration);

// Or using fluent configuration
services.AddGpuBridgeLogging(builder => builder
    .WithMinimumLevel(LogLevel.Information)
    .AddConsole(console => console.UseColors = true)
    .AddFile(file => {
        file.LogDirectory = "logs";
        file.MaxFileSizeBytes = 50 * 1024 * 1024; // 50MB
        file.RetentionDays = 7;
    })
    .AddTelemetry(telemetry => {
        telemetry.ServiceName = "GpuBridge";
        telemetry.OtlpEndpoint = "http://localhost:4317";
    }));
```

### 2. Configuration (appsettings.json)

```json
{
  "GpuBridgeLogging": {
    "MinimumLevel": "Information",
    "EnableStructuredLogging": true,
    "EnablePerformanceMetrics": true,
    "EnableCorrelationTracking": true,
    "Buffer": {
      "Capacity": 10000,
      "MaxBatchSize": 100,
      "FlushInterval": "00:00:01"
    },
    "Console": {
      "Enabled": true,
      "MinimumLevel": "Information",
      "UseColors": true,
      "IncludeMetrics": true
    },
    "File": {
      "Enabled": true,
      "LogDirectory": "logs",
      "BaseFileName": "gpu-bridge",
      "MaxFileSizeBytes": 104857600,
      "RetentionDays": 30
    },
    "Telemetry": {
      "Enabled": false,
      "ServiceName": "Orleans.GpuBridge",
      "FlushInterval": "00:00:30"
    },
    "CategoryLevels": {
      "Orleans.GpuBridge.Runtime": "Debug",
      "Orleans.GpuBridge.Grains": "Information"
    }
  }
}
```

### 3. Usage in Classes

```csharp
public class GpuKernelService
{
    private readonly ILogger<GpuKernelService> _logger;

    public GpuKernelService(ILogger<GpuKernelService> logger)
    {
        _logger = logger;
    }

    public async Task<TOut> ExecuteKernelAsync<TIn, TOut>(string kernelId, TIn input)
    {
        var stopwatch = Stopwatch.StartNew();
        
        try
        {
            _logger.LogInformation("Starting kernel execution for {KernelId}", kernelId);
            
            var result = await ProcessKernelAsync(kernelId, input);
            
            // Log with performance metrics
            _logger.LogKernelExecution(kernelId, stopwatch.Elapsed, 
                Marshal.SizeOf<TIn>(), Marshal.SizeOf<TOut>(), success: true);
                
            return result;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Kernel execution failed for {KernelId}", kernelId);
            _logger.LogKernelExecution(kernelId, stopwatch.Elapsed, 
                Marshal.SizeOf<TIn>(), 0, success: false);
            throw;
        }
    }
}
```

### 4. Advanced Usage with Context

```csharp
public class GpuBatchGrain : Grain
{
    private readonly ILogger<GpuBatchGrain> _logger;

    public async Task ProcessBatchAsync(BatchRequest request)
    {
        // Create correlation context
        using var context = new LogContext(request.CorrelationId)
        {
            UserId = request.UserId,
            Component = "GpuBatchGrain"
        }.Push();

        _logger.LogGrainOperation(LogLevel.Information, 
            nameof(GpuBatchGrain), nameof(ProcessBatchAsync), 
            this.GetPrimaryKeyString());

        // Processing with automatic context correlation
        await ProcessItems(request.Items);
    }
}
```

## Components

### Core Components

- **`ILoggerDelegate`**: Base interface for logging targets
- **`LoggerDelegateManager`**: Central coordinator for all delegates
- **`LogBuffer`**: High-performance batching with backpressure
- **`LoggerFactory`**: Factory with fluent configuration
- **`GpuBridgeLogger`**: Main logger implementation

### Delegates

- **`ConsoleLoggerDelegate`**: Color-coded console output with structured formatting
- **`FileLoggerDelegate`**: JSON file logging with automatic rotation and retention
- **`TelemetryLoggerDelegate`**: OpenTelemetry integration with distributed tracing

### Configuration

- **`GpuBridgeLoggingOptions`**: Comprehensive configuration options
- **`ServiceCollectionExtensions`**: Dependency injection integration
- **Validation**: Built-in configuration validation

## Performance Features

### Asynchronous Processing
- Non-blocking log entry queuing
- Batched processing for high throughput
- Backpressure control to prevent memory issues

### Memory Optimization
- Pooled objects where possible
- Efficient string handling
- Bounded channels for memory control

### Thread Safety
- Lock-free queuing where possible
- Thread-safe delegate operations
- Concurrent processing support

## Production Considerations

### Error Handling
- Graceful degradation on delegate failures
- Fallback logging to prevent data loss
- Comprehensive error reporting

### Resource Management
- Automatic cleanup of resources
- Configurable retention policies
- Memory usage monitoring

### Monitoring
- Built-in statistics and health checks
- Performance metrics tracking
- Buffer utilization monitoring

## Migration Guide

### From ILogger<T> to GpuBridgeLogger

**Before:**
```csharp
private readonly ILogger<MyClass> _logger;

_logger.LogInformation("Processing {Count} items", items.Count);
```

**After:**
```csharp
private readonly ILogger<MyClass> _logger;

// Same interface, enhanced functionality
_logger.LogInformation("Processing {Count} items", items.Count);

// Or use enhanced GPU-specific methods
_logger.LogGpuOperation(LogLevel.Information, "BatchProcess", 
    "vector-add", duration: elapsed, memoryUsage: bufferSize);
```

### Registering Custom Delegates

```csharp
services.AddGpuBridgeLogging(builder => builder
    .AddDelegate(factory => new CustomLoggerDelegate("custom-target")));
```

## Examples

See the `Examples/` directory for:
- Basic console logging setup
- File logging with rotation
- Telemetry integration
- Custom delegate implementation
- Orleans grain logging patterns
- Performance monitoring setup

## License

This project is licensed under the MIT License.
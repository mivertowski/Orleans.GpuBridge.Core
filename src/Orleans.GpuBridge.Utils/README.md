# Orleans.GpuBridge.Utils

A comprehensive collection of utility functions, extension methods, and helper classes for GPU operations in the Orleans.GpuBridge ecosystem. This library provides common functionality for memory management, performance optimization, diagnostics, and configuration management.

## Overview

Orleans.GpuBridge.Utils consolidates utility functions and helpers that are used across the GPU Bridge ecosystem. It provides high-performance, production-ready implementations for common operations including GPU memory management, performance benchmarking, telemetry collection, and device management.

## Features

### 🚀 Memory Management Utilities
- High-performance memory pools with NUMA awareness
- GPU-specific memory allocators and managers
- Pinned memory management for GPU transfers
- Memory usage tracking and diagnostics

### 📊 Performance Utilities
- GPU kernel performance benchmarking
- Vectorized operation optimizations
- Async pattern optimizations
- Performance metrics collection

### 🔧 Configuration Helpers
- Service collection extensions
- Options pattern implementations
- Configuration validation
- Environment-specific settings

### 📈 Diagnostic Tools
- GPU telemetry collection
- Performance profiling utilities
- Health check implementations
- Resource monitoring

### 🛠️ Extension Methods
- Service registration extensions
- Memory operation extensions
- Task and async utilities
- Collection manipulation helpers

## Installation

### Package Manager
```bash
dotnet add package Orleans.GpuBridge.Utils
```

### PackageReference
```xml
<PackageReference Include="Orleans.GpuBridge.Utils" Version="1.0.0" />
```

## Key Utility Categories

### Memory Management

#### High-Performance Memory Pool
```csharp
using Orleans.GpuBridge.Utils.Memory;

// Create NUMA-aware memory pool
var memoryPool = new HighPerformanceMemoryPool<float>(
    logger, 
    maxBuffersPerBucket: 100,
    useNumaOptimization: true);

// Rent memory with automatic sizing
using var buffer = memoryPool.Rent(1024);
var span = buffer.Memory.Span;

// Get pool statistics
var stats = memoryPool.GetStatistics();
Console.WriteLine($"Efficiency: {stats.EfficiencyPercent:F1}%");
```

#### GPU Memory Helpers
```csharp
using Orleans.GpuBridge.Utils.Memory;

// Allocate pinned memory for GPU transfers
using var pinnedMemory = GpuMemoryHelpers.AllocatePinned<float>(1024);

// Copy data with optimal transfer patterns
await GpuMemoryHelpers.CopyToDeviceAsync(hostData, deviceBuffer);

// Validate memory alignment
bool isAligned = GpuMemoryHelpers.IsAligned(buffer, 256);
```

### Performance Utilities

#### Kernel Performance Benchmarking
```csharp
using Orleans.GpuBridge.Utils.Performance;

var benchmarker = new KernelBenchmarker(logger);

// Benchmark kernel performance
var results = await benchmarker.BenchmarkAsync(
    kernelId: "vector_add",
    inputSizes: new[] { 1000, 10000, 100000 },
    iterations: 100);

// Analyze results
foreach (var result in results)
{
    Console.WriteLine($"Size: {result.InputSize}, " +
                     $"Avg: {result.AverageLatency.TotalMilliseconds:F2}ms");
}
```

#### Vectorized Operations
```csharp
using Orleans.GpuBridge.Utils.Performance;

// High-performance vectorized operations
var vectorOps = new VectorizedOperations();

// Parallel vector addition
Span<float> a = stackalloc float[1000];
Span<float> b = stackalloc float[1000];
Span<float> result = stackalloc float[1000];

vectorOps.Add(a, b, result); // Uses SIMD when available
```

### Configuration Extensions

#### Service Registration
```csharp
using Orleans.GpuBridge.Utils.Extensions;

services.AddGpuBridgeUtilities(options =>
{
    options.EnableHighPerformanceMemoryPool = true;
    options.EnablePerformanceProfiling = true;
    options.MemoryPoolMaxBuffersPerBucket = 100;
    options.ProfilingLevel = ProfilingLevel.Detailed;
});

// Add telemetry with utilities
services.AddGpuTelemetryWithUtils(telemetry =>
{
    telemetry.ServiceName = "MyGpuApp";
    telemetry.EnableDetailedMetrics = true;
    telemetry.CollectionInterval = TimeSpan.FromSeconds(5);
});
```

### Diagnostic Utilities

#### GPU Health Monitoring
```csharp
using Orleans.GpuBridge.Utils.Diagnostics;

var healthMonitor = new GpuHealthMonitor(logger);

// Check GPU health status
var healthStatus = await healthMonitor.CheckHealthAsync();
if (healthStatus.IsHealthy)
{
    Console.WriteLine($"GPU Temperature: {healthStatus.Temperature}°C");
    Console.WriteLine($"Memory Usage: {healthStatus.MemoryUsagePercent:F1}%");
}
```

#### Performance Profiler
```csharp
using Orleans.GpuBridge.Utils.Diagnostics;

var profiler = new GpuProfiler(logger);

// Profile kernel execution
using (profiler.StartProfiling("vector_add"))
{
    var result = await kernelExecutor.ExecuteAsync(input);
}

// Generate performance report
var report = profiler.GenerateReport();
await report.SaveToFileAsync("performance_report.json");
```

## Extension Methods

### Service Collection Extensions
```csharp
// Core utilities registration
services.AddGpuBridgeUtilities();

// Memory pool registration
services.AddHighPerformanceMemoryPool<float>();

// Performance monitoring
services.AddGpuPerformanceMonitoring();

// Diagnostic services
services.AddGpuDiagnostics();
```

### Memory Extensions
```csharp
using Orleans.GpuBridge.Utils.Extensions;

// Memory validation
bool isValid = memory.IsValidGpuBuffer();

// Safe copying with validation
await sourceBuffer.CopyToGpuAsync(targetBuffer, validateAlignment: true);

// Memory statistics
var stats = memoryPool.GetDetailedStats();
```

### Task Extensions
```csharp
using Orleans.GpuBridge.Utils.Extensions;

// GPU-aware task scheduling
var result = await task.WithGpuContext(gpuContext);

// Timeout with GPU cleanup
var result = await kernelTask.WithGpuTimeout(TimeSpan.FromSeconds(30));

// Retry with GPU error handling
var result = await operation.RetryOnGpuError(maxRetries: 3);
```

## Configuration

### UtilsOptions Configuration
```csharp
public class GpuBridgeUtilsOptions
{
    /// <summary>
    /// Enable high-performance memory pool (default: true)
    /// </summary>
    public bool EnableHighPerformanceMemoryPool { get; set; } = true;
    
    /// <summary>
    /// Maximum buffers per bucket in memory pool (default: 50)
    /// </summary>
    public int MemoryPoolMaxBuffersPerBucket { get; set; } = 50;
    
    /// <summary>
    /// Enable NUMA-aware memory allocation (default: true on Windows)
    /// </summary>
    public bool EnableNumaOptimization { get; set; } = true;
    
    /// <summary>
    /// Enable performance profiling (default: false)
    /// </summary>
    public bool EnablePerformanceProfiling { get; set; } = false;
    
    /// <summary>
    /// Performance profiling level
    /// </summary>
    public ProfilingLevel ProfilingLevel { get; set; } = ProfilingLevel.Basic;
    
    /// <summary>
    /// Diagnostic data collection interval
    /// </summary>
    public TimeSpan DiagnosticInterval { get; set; } = TimeSpan.FromSeconds(10);
    
    /// <summary>
    /// Enable detailed logging (default: false)
    /// </summary>
    public bool EnableDetailedLogging { get; set; } = false;
}
```

### Configuration in appsettings.json
```json
{
  "GpuBridgeUtils": {
    "EnableHighPerformanceMemoryPool": true,
    "MemoryPoolMaxBuffersPerBucket": 100,
    "EnableNumaOptimization": true,
    "EnablePerformanceProfiling": true,
    "ProfilingLevel": "Detailed",
    "DiagnosticInterval": "00:00:05",
    "EnableDetailedLogging": false
  }
}
```

## Best Practices

### Memory Management
1. **Use High-Performance Memory Pool**: Always prefer the provided memory pool for frequent allocations
2. **Proper Disposal**: Always dispose memory owners properly using `using` statements
3. **Alignment Validation**: Validate memory alignment for GPU operations
4. **Monitor Pool Efficiency**: Regularly check pool statistics for optimization opportunities

### Performance Optimization
1. **Benchmark Regularly**: Use built-in benchmarking tools to identify performance bottlenecks
2. **Profile Before Optimizing**: Use the profiler to understand actual performance characteristics
3. **Vectorization**: Leverage SIMD operations where possible
4. **Batch Operations**: Group small operations into larger batches for better GPU utilization

### Diagnostics and Monitoring
1. **Health Checks**: Implement regular GPU health monitoring
2. **Telemetry Integration**: Use the telemetry utilities for comprehensive monitoring
3. **Error Handling**: Implement proper GPU error detection and recovery
4. **Resource Monitoring**: Monitor GPU memory and utilization continuously

### Configuration
1. **Environment-Specific Settings**: Configure options based on deployment environment
2. **Resource Limits**: Set appropriate limits for memory pools and diagnostic collection
3. **Logging Levels**: Use detailed logging only in development environments
4. **Performance Profiling**: Enable profiling selectively to avoid overhead

## Performance Considerations

### Memory Pool Efficiency
- **Bucket Sizing**: Default bucket sizes are optimized for common GPU workloads
- **NUMA Awareness**: Automatic NUMA optimization on supported platforms
- **Thread Safety**: Lock-free design for high-concurrency scenarios
- **Monitoring**: Built-in statistics for efficiency tracking

### GPU Operations
- **Async Patterns**: All GPU operations are fully asynchronous
- **Error Handling**: Comprehensive GPU error detection and recovery
- **Resource Cleanup**: Automatic resource cleanup and disposal
- **Performance Metrics**: Built-in performance measurement and reporting

## Dependencies

```xml
<PackageReference Include="Microsoft.Extensions.DependencyInjection" Version="8.0.0" />
<PackageReference Include="Microsoft.Extensions.Logging" Version="8.0.0" />
<PackageReference Include="Microsoft.Extensions.Options" Version="8.0.0" />
<PackageReference Include="Microsoft.Extensions.Hosting" Version="8.0.0" />
<PackageReference Include="System.Memory" Version="4.5.5" />
<PackageReference Include="System.Buffers" Version="4.5.1" />
<PackageReference Include="System.Numerics.Vectors" Version="4.5.0" />
<PackageReference Include="OpenTelemetry" Version="1.6.0" />
<PackageReference Include="OpenTelemetry.Extensions.Hosting" Version="1.6.0" />
```

## Examples

### Complete Setup Example
```csharp
using Orleans.GpuBridge.Utils.Extensions;

var builder = WebApplication.CreateBuilder(args);

// Add GPU Bridge utilities
builder.Services.AddGpuBridgeUtilities(options =>
{
    options.EnableHighPerformanceMemoryPool = true;
    options.EnablePerformanceProfiling = true;
    options.ProfilingLevel = ProfilingLevel.Detailed;
});

// Add telemetry with utilities
builder.Services.AddGpuTelemetryWithUtils(telemetry =>
{
    telemetry.ServiceName = "GpuWebApi";
    telemetry.EnablePrometheusExporter = true;
    telemetry.EnableDetailedMetrics = true;
});

var app = builder.Build();

// Use built-in health checks
app.MapHealthChecks("/health");
```

### Memory Pool Usage Example
```csharp
public class GpuDataProcessor
{
    private readonly HighPerformanceMemoryPool<float> _memoryPool;
    private readonly ILogger<GpuDataProcessor> _logger;

    public GpuDataProcessor(
        HighPerformanceMemoryPool<float> memoryPool,
        ILogger<GpuDataProcessor> logger)
    {
        _memoryPool = memoryPool;
        _logger = logger;
    }

    public async Task<float[]> ProcessAsync(float[] input)
    {
        using var buffer = _memoryPool.Rent(input.Length);
        var span = buffer.Memory.Span;
        
        // Copy input data
        input.CopyTo(span);
        
        // Process data (GPU operations)
        await ProcessOnGpuAsync(buffer.Memory);
        
        // Return results
        return span.ToArray();
    }
}
```

## Contributing

We welcome contributions to Orleans.GpuBridge.Utils! Please ensure all contributions:

1. Include comprehensive unit tests
2. Follow the established coding standards
3. Include XML documentation for public APIs
4. Add performance benchmarks for new utilities
5. Update the README with new functionality

## License

Copyright (c) 2025 Michael Ivertowski

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

## Support

For issues, feature requests, or questions about Orleans.GpuBridge.Utils:

- 📋 [Issue Tracker](https://github.com/mivertowski/orleans-gpu-bridge/issues)
- 📖 [Documentation](https://orleans-gpu-bridge.readthedocs.io)
- 💬 [Discussions](https://github.com/mivertowski/orleans-gpu-bridge/discussions)
# Orleans.GpuBridge.Runtime

## Overview

Orleans.GpuBridge.Runtime provides the core runtime implementation for GPU acceleration in Orleans applications. This library enables seamless integration of GPU compute resources with Orleans grains through a comprehensive bridge abstraction layer, featuring automatic device discovery, intelligent load balancing, and robust fallback mechanisms.

## Key Features

### 🚀 Core Capabilities
- **Automatic GPU Device Discovery**: Detects CUDA, OpenCL, DirectCompute, Metal, and Vulkan devices
- **Intelligent Device Selection**: Smart device scoring based on memory, queue depth, and performance
- **CPU Fallback Support**: Seamless fallback to CPU when GPU resources are unavailable
- **Memory Pool Management**: Efficient memory pooling for both GPU and CPU operations
- **Kernel Catalog System**: Centralized kernel registration and execution management

### 🔧 Advanced Features
- **Load Balancing**: Automatic work distribution across multiple GPU devices
- **Health Monitoring**: Real-time device health monitoring and recovery
- **Persistent Kernel Hosting**: Long-running kernel hosts for improved performance
- **Queue Depth Awareness**: Queue-aware placement strategies for optimal resource utilization
- **Production Hardening**: Comprehensive error handling, logging, and diagnostics

### 🛡️ Reliability & Monitoring
- **Resilience Patterns**: Built-in retry policies and circuit breakers
- **Telemetry Collection**: Comprehensive metrics and performance tracking
- **Device Health Monitoring**: Continuous monitoring with automatic recovery
- **Resource Cleanup**: Proper disposal and resource management

## Installation

### NuGet Package
```bash
dotnet add package Orleans.GpuBridge.Runtime
```

### Manual Installation
```bash
git clone https://github.com/your-repo/Orleans.GpuBridge.Core.git
cd Orleans.GpuBridge.Core/src/Orleans.GpuBridge.Runtime
dotnet build
```

## Quick Start

### Basic Configuration

```csharp
using Orleans.GpuBridge.Runtime.Extensions;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;

var builder = Host.CreateDefaultBuilder()
    .ConfigureServices(services =>
    {
        services.AddGpuBridge(options =>
        {
            options.PreferGpu = true;
            options.EnableFallback = true;
            options.MaxQueueDepth = 1000;
        });
    });

var host = builder.Build();
await host.RunAsync();
```

### Kernel Registration

```csharp
services.AddGpuBridge()
    .AddKernel(kernel => kernel
        .Id("vector_add")
        .In<float[]>()
        .Out<float[]>()
        .FromFactory(sp => new VectorAddKernel()))
    .AddKernel<MatrixMultiplyKernel>()
    .AddKernel(kernel => kernel
        .Id("image_filter")
        .In<ImageData>()
        .Out<ImageData>()
        .FromType<ImageFilterKernel>());
```

## Configuration

### Basic Options

```csharp
services.AddGpuBridge(options =>
{
    // Device Selection
    options.PreferGpu = true;
    options.EnableFallback = true;
    options.DeviceSelectionStrategy = DeviceSelectionStrategy.BestFit;
    
    // Performance Tuning
    options.MaxQueueDepth = 1000;
    options.BatchSize = 64;
    options.MemoryPoolSize = 1024 * 1024 * 256; // 256MB
    
    // Monitoring
    options.EnableHealthMonitoring = true;
    options.HealthCheckInterval = TimeSpan.FromSeconds(30);
    options.LoadBalancingInterval = TimeSpan.FromSeconds(5);
    
    // Diagnostics
    options.EnableDiagnostics = true;
    options.LogLevel = LogLevel.Information;
});
```

### Advanced Configuration

```csharp
services.AddGpuBridge(options =>
{
    // Backend Providers
    options.EnabledBackends = new[]
    {
        GpuBackendType.CUDA,
        GpuBackendType.OpenCL,
        GpuBackendType.DirectCompute
    };
    
    // Memory Management
    options.MemoryStrategy = MemoryStrategy.Pooled;
    options.MaxConcurrentKernels = 16;
    options.KernelTimeout = TimeSpan.FromMinutes(5);
    
    // Resilience
    options.RetryPolicy = new ExponentialBackoffRetryPolicy
    {
        MaxRetries = 3,
        BaseDelay = TimeSpan.FromMilliseconds(100)
    };
    
    // Device Filtering
    options.DeviceFilter = device => 
        device.TotalMemoryBytes >= 1024 * 1024 * 1024 && // Min 1GB
        device.ComputeUnits >= 8; // Min 8 compute units
})
.AddBackendProvider<CudaBackendProvider>()
.AddBackendProvider<OpenClBackendProvider>();
```

## Key Components

### DeviceBroker

The `DeviceBroker` manages GPU device discovery and work distribution:

```csharp
public class DeviceBroker : IDisposable
{
    // Device Management
    public Task InitializeAsync(CancellationToken ct);
    public IReadOnlyList<GpuDevice> GetDevices();
    public GpuDevice? GetBestDevice();
    
    // Resource Monitoring
    public int DeviceCount { get; }
    public long TotalMemoryBytes { get; }
    public int CurrentQueueDepth { get; }
}
```

**Usage Example:**
```csharp
var deviceBroker = serviceProvider.GetRequiredService<DeviceBroker>();
await deviceBroker.InitializeAsync(cancellationToken);

var bestDevice = deviceBroker.GetBestDevice();
var devices = deviceBroker.GetDevices();

Console.WriteLine($"Found {deviceBroker.DeviceCount} devices");
Console.WriteLine($"Total memory: {deviceBroker.TotalMemoryBytes:N0} bytes");
```

### KernelCatalog

Manages kernel registration and execution:

```csharp
public class KernelCatalog
{
    // Kernel Management
    public void RegisterKernel<TIn, TOut>(string id, IGpuKernel<TIn, TOut> kernel);
    public Task<TOut> ExecuteAsync<TIn, TOut>(string kernelId, TIn input);
    
    // Batch Operations
    public Task<TOut[]> ExecuteBatchAsync<TIn, TOut>(
        string kernelId, 
        TIn[] inputs, 
        int batchSize = 32);
}
```

**Usage Example:**
```csharp
var catalog = serviceProvider.GetRequiredService<KernelCatalog>();

// Execute single operation
var result = await catalog.ExecuteAsync<float[], float[]>("vector_add", inputData);

// Execute batch operation
var results = await catalog.ExecuteBatchAsync<ImageData, ImageData>(
    "image_filter", 
    imagesBatch, 
    batchSize: 16);
```

### Memory Pool Management

Efficient GPU memory pooling:

```csharp
public interface IGpuMemoryPool<T> where T : unmanaged
{
    IGpuMemory<T> Rent(int minimumLength);
    void Return(IGpuMemory<T> memory);
    MemoryPoolStats GetStats();
}
```

**Usage Example:**
```csharp
var memoryPool = serviceProvider.GetRequiredService<IGpuMemoryPool<float>>();

// Rent memory
using var memory = memoryPool.Rent(1024);
var span = memory.AsMemory().Span;

// Use the memory
for (int i = 0; i < span.Length; i++)
{
    span[i] = i * 2.0f;
}

// Memory is automatically returned on disposal
```

### Backend Providers

Support for multiple GPU backends:

```csharp
// CUDA Provider
services.AddGpuBridge()
    .AddBackendProvider<CudaBackendProvider>();

// OpenCL Provider
services.AddGpuBridge()
    .AddBackendProvider<OpenClBackendProvider>();

// Custom Provider
services.AddGpuBridge()
    .AddBackendProvider(sp => new CustomBackendProvider(
        sp.GetRequiredService<ILogger<CustomBackendProvider>>()));
```

## Performance Optimization

### Device Selection Strategies

```csharp
public enum DeviceSelectionStrategy
{
    FirstAvailable,    // Use first available device
    BestFit,          // Score-based selection (recommended)
    RoundRobin,       // Distribute evenly across devices
    LoadBalanced,     // Dynamic load balancing
    Custom            // User-defined strategy
}
```

### Batch Processing

```csharp
// Optimal batch sizes vary by kernel and hardware
var results = await GpuPipeline<InputData, OutputData>
    .For(grainFactory, "my_kernel")
    .WithBatchSize(64)  // Tune based on your data and GPU
    .WithMemoryStrategy(MemoryStrategy.Pooled)
    .ExecuteAsync(largeDataSet);
```

### Memory Management Tips

1. **Use Memory Pools**: Always use the provided memory pools for better performance
2. **Batch Operations**: Group small operations into larger batches
3. **Minimize Transfers**: Keep data on GPU between operations when possible
4. **Monitor Memory Usage**: Use diagnostic tools to track memory consumption

```csharp
// Good: Use pooled memory
using var memory = memoryPool.Rent(dataSize);
await kernel.ExecuteAsync(memory.AsMemory());

// Better: Batch multiple operations
var batchResults = await catalog.ExecuteBatchAsync(kernelId, inputs, batchSize: 32);

// Best: Chain operations on GPU
var pipeline = GpuPipeline<float[], float[]>
    .For(grainFactory, "preprocess")
    .ThenExecute("main_compute")
    .ThenExecute("postprocess")
    .WithMemoryStrategy(MemoryStrategy.Persistent);
```

## Error Handling and Diagnostics

### Error Handling Patterns

```csharp
try
{
    var result = await catalog.ExecuteAsync<float[], float[]>("my_kernel", input);
    return result;
}
catch (GpuDeviceException ex)
{
    logger.LogWarning(ex, "GPU operation failed, falling back to CPU");
    // Automatic CPU fallback is handled by the runtime
    throw;
}
catch (KernelNotFoundException ex)
{
    logger.LogError(ex, "Kernel {KernelId} not found", "my_kernel");
    throw;
}
catch (GpuMemoryException ex)
{
    logger.LogError(ex, "GPU memory allocation failed");
    // Consider reducing batch size or clearing cache
    throw;
}
```

### Diagnostic Information

```csharp
// Get device diagnostics
var diagnostics = serviceProvider.GetRequiredService<GpuDiagnostics>();
var deviceStats = await diagnostics.GetDeviceStatisticsAsync();

foreach (var stat in deviceStats)
{
    Console.WriteLine($"Device {stat.Index}: {stat.Name}");
    Console.WriteLine($"  Memory: {stat.UsedMemory:N0}/{stat.TotalMemory:N0} bytes");
    Console.WriteLine($"  Queue Depth: {stat.QueueDepth}");
    Console.WriteLine($"  Error Rate: {stat.ErrorRate:P2}");
}

// Get memory pool statistics
var memoryStats = memoryPool.GetStats();
Console.WriteLine($"Pool: {memoryStats.InUse:N0}/{memoryStats.TotalAllocated:N0} bytes");
Console.WriteLine($"Efficiency: {memoryStats.PooledItems} items pooled");
```

### Logging Configuration

```csharp
services.AddLogging(builder =>
{
    builder.AddConsole();
    builder.SetMinimumLevel(LogLevel.Information);
    
    // Specific GPU Bridge logging
    builder.AddFilter("Orleans.GpuBridge", LogLevel.Debug);
    builder.AddFilter("Orleans.GpuBridge.Runtime.DeviceBroker", LogLevel.Trace);
});
```

## Dependencies

### Required Packages
- **Microsoft.Orleans.Core** (≥ 8.0.0) - Orleans framework
- **Microsoft.Extensions.DependencyInjection** (≥ 8.0.0) - Dependency injection
- **Microsoft.Extensions.Hosting** (≥ 8.0.0) - Application hosting
- **Microsoft.Extensions.Logging** (≥ 8.0.0) - Logging infrastructure
- **Microsoft.Extensions.Options** (≥ 8.0.0) - Configuration options

### Optional Packages
- **Orleans.GpuBridge.Abstractions** - Core abstractions (auto-included)
- **Orleans.GpuBridge.BridgeFX** - High-level pipeline API
- **Orleans.GpuBridge.Grains** - GPU-enabled Orleans grains

### GPU Backend Dependencies
- **CUDA Toolkit** (≥ 12.0) - For CUDA backend support
- **OpenCL Runtime** - For OpenCL backend support
- **DirectX** - For DirectCompute backend support (Windows only)

## System Requirements

### Minimum Requirements
- **.NET 9.0** or later
- **4GB RAM** (8GB+ recommended)
- **GPU with 1GB+ VRAM** (optional, CPU fallback available)

### Recommended Requirements
- **.NET 9.0** or later
- **16GB+ RAM**
- **Modern GPU** with 4GB+ VRAM
- **NVMe SSD** for optimal data transfer

### Supported Platforms
- **Windows 10/11** (x64)
- **Linux** (Ubuntu 20.04+, CentOS 8+)
- **macOS** (Intel and Apple Silicon)

### GPU Support
| Backend | Windows | Linux | macOS | Notes |
|---------|---------|-------|-------|-------|
| CUDA | ✅ | ✅ | ❌ | NVIDIA GPUs only |
| OpenCL | ✅ | ✅ | ✅ | Most GPU vendors |
| DirectCompute | ✅ | ❌ | ❌ | DirectX 11+ required |
| Metal | ❌ | ❌ | ✅ | Apple Silicon/Intel |
| Vulkan | 🚧 | 🚧 | 🚧 | Coming soon |

## Examples

### Basic Kernel Implementation

```csharp
public class VectorAddKernel : IGpuKernel<VectorAddInput, float[]>
{
    public async Task<float[]> ExecuteAsync(VectorAddInput input, CancellationToken ct = default)
    {
        // CPU fallback implementation
        var result = new float[input.A.Length];
        for (int i = 0; i < input.A.Length; i++)
        {
            result[i] = input.A[i] + input.B[i];
        }
        return result;
    }
    
    public void Dispose() { }
}

public record VectorAddInput(float[] A, float[] B);
```

### Orleans Grain Integration

```csharp
[GpuAccelerated]
public class ComputeGrain : Grain, IComputeGrain
{
    private readonly KernelCatalog _kernelCatalog;
    
    public ComputeGrain(KernelCatalog kernelCatalog)
    {
        _kernelCatalog = kernelCatalog;
    }
    
    public async Task<float[]> ProcessDataAsync(float[] input)
    {
        return await _kernelCatalog.ExecuteAsync<float[], float[]>(
            "vector_process", input);
    }
}
```

### Advanced Pipeline

```csharp
public class ImageProcessingService
{
    private readonly IGrainFactory _grainFactory;
    
    public async Task<ProcessedImage[]> ProcessBatchAsync(Image[] images)
    {
        var results = await GpuPipeline<Image, ProcessedImage>
            .For(_grainFactory, "image_pipeline")
            .WithBatchSize(8)
            .WithMemoryStrategy(MemoryStrategy.Pooled)
            .WithRetryPolicy(new ExponentialBackoffRetryPolicy())
            .ExecuteAsync(images);
            
        return results;
    }
}
```

## Troubleshooting

### Common Issues

**1. No GPU devices detected**
```
Solution: Ensure GPU drivers are installed and hardware is compatible
Check: nvidia-smi (NVIDIA), clinfo (OpenCL), dxdiag (DirectX)
```

**2. GPU memory allocation failures**
```
Solution: Reduce batch sizes, increase system memory, or enable memory pooling
Configuration: options.MemoryStrategy = MemoryStrategy.Pooled
```

**3. Kernel execution timeouts**
```
Solution: Increase timeout values or optimize kernel performance
Configuration: options.KernelTimeout = TimeSpan.FromMinutes(10)
```

**4. High error rates**
```
Solution: Check GPU stability, update drivers, or enable CPU fallback
Monitoring: Use GpuDiagnostics to track device health
```

### Debug Configuration

```csharp
services.AddGpuBridge(options =>
{
    options.EnableDiagnostics = true;
    options.LogLevel = LogLevel.Trace;
    options.EnableHealthMonitoring = true;
})
.AddLogging(builder => builder
    .AddConsole()
    .SetMinimumLevel(LogLevel.Debug));
```

## Contributing

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

## License

**Apache License 2.0**

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

---

For more information, visit the [Orleans.GpuBridge.Core Documentation](../../../docs/) or check out the [Examples](../../../examples/) directory.
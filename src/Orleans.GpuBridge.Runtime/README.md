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
        // Basic GPU Bridge setup
        services.AddGpuBridge(options =>
        {
            options.PreferGpu = true;
            options.FallbackToCpu = true;
        });
    });

var host = builder.Build();
await host.RunAsync();
```

### Full Configuration with Ring Kernels

```csharp
using Orleans.GpuBridge.Runtime.Extensions;
using Orleans.GpuBridge.Backends.DotCompute.Extensions;

services.AddGpuBridge(options =>
{
    options.PreferGpu = true;
    options.FallbackToCpu = true;
    options.MaxConcurrentKernels = 100;
})
.AddDotGpuBackend() // Add DotCompute backend
.Services
.AddRingKernelSupport(options =>
{
    options.DefaultGridSize = 1;
    options.DefaultBlockSize = 256;
    options.DefaultQueueCapacity = 256;
    options.EnableKernelCaching = true;
    options.DeviceIndex = 0;
})
.AddK2KSupport()                    // Enable kernel-to-kernel messaging
.AddDotComputeRingKernelBridge();   // GPU-accelerated ring kernel bridge
```

## Configuration

### GpuBridgeOptions

```csharp
services.AddGpuBridge(options =>
{
    // GPU preferences
    options.PreferGpu = true;
    options.FallbackToCpu = true;
    options.MaxRetries = 3;

    // Performance tuning
    options.DefaultMicroBatch = 8192;
    options.MaxConcurrentKernels = 100;
    options.MemoryPoolSizeMB = 1024;
    options.BatchSize = 1024;

    // Device management
    options.MaxDevices = 4;
    options.EnableGpuDirectStorage = false;

    // Backend configuration
    options.DefaultBackend = "DotCompute";
    options.EnableProviderDiscovery = true;

    // Telemetry
    options.EnableProfiling = false;
    options.Telemetry = new TelemetryOptions
    {
        EnableMetrics = true,
        EnableTracing = true,
        SamplingRate = 0.1
    };
});
```

### RingKernelOptions

```csharp
services.AddRingKernelSupport(options =>
{
    // Kernel launch configuration
    options.DefaultGridSize = 1;      // Single block for single-actor
    options.DefaultBlockSize = 256;   // Optimal for most GPUs

    // Message queue
    options.DefaultQueueCapacity = 256; // Must be power of 2

    // Compilation
    options.EnableKernelCaching = true; // Cache compiled kernels
    options.DeviceIndex = 0;            // First GPU
});
```

## Key Components

### IGpuBridge

The primary interface for GPU bridge operations:

```csharp
public interface IGpuBridge
{
    ValueTask<GpuBridgeInfo> GetInfoAsync(CancellationToken ct = default);
    ValueTask<IGpuKernel<TIn, TOut>> GetKernelAsync<TIn, TOut>(
        KernelId kernelId, CancellationToken ct = default);
    ValueTask<IReadOnlyList<GpuDevice>> GetDevicesAsync(CancellationToken ct = default);
    ValueTask<object> ExecuteKernelAsync(string kernelId, object input, CancellationToken ct = default);
}
```

**Usage Example:**
```csharp
var gpuBridge = serviceProvider.GetRequiredService<IGpuBridge>();

// Get bridge info
var info = await gpuBridge.GetInfoAsync();
Console.WriteLine($"Backend: {info.BackendName}, Devices: {info.DeviceCount}");

// Get kernel and execute
var kernel = await gpuBridge.GetKernelAsync<float[], float[]>(new KernelId("vector-add"));
var result = await kernel.ExecuteAsync(inputData);
```

### IRingKernelBridge

For GPU-native actors using persistent ring kernels:

```csharp
public interface IRingKernelBridge
{
    bool IsAvailable { get; }

    ValueTask<GpuStateHandle<TState>> AllocateStateAsync<TState>(
        long actorId, TState initialState, CancellationToken ct = default)
        where TState : unmanaged;

    ValueTask<TResponse> SendMessageAsync<TState, TRequest, TResponse>(
        GpuStateHandle<TState> stateHandle, TRequest request, CancellationToken ct = default)
        where TState : unmanaged where TRequest : unmanaged where TResponse : unmanaged;

    ValueTask<TState> GetStateAsync<TState>(GpuStateHandle<TState> handle, CancellationToken ct = default)
        where TState : unmanaged;

    ValueTask ReleaseAsync<TState>(GpuStateHandle<TState> handle, CancellationToken ct = default)
        where TState : unmanaged;
}
```

**Usage Example:**
```csharp
var bridge = serviceProvider.GetRequiredService<IRingKernelBridge>();

// Allocate GPU state for actor
var stateHandle = await bridge.AllocateStateAsync(actorId, new CounterState { Value = 0 });

// Send message through GPU ring kernel
var response = await bridge.SendMessageAsync<CounterState, IncrementMessage, int>(
    stateHandle, new IncrementMessage { Amount = 5 });

// Get current state
var state = await bridge.GetStateAsync(stateHandle);

// Release when done
await bridge.ReleaseAsync(stateHandle);
```

### DeviceBroker

Manages GPU device discovery and selection:

```csharp
var deviceBroker = serviceProvider.GetRequiredService<DeviceBroker>();
await deviceBroker.InitializeAsync(cancellationToken);

var devices = deviceBroker.GetDevices();
foreach (var device in devices)
{
    Console.WriteLine($"Device {device.Index}: {device.Name}");
    Console.WriteLine($"  Memory: {device.TotalMemoryBytes / (1024 * 1024)}MB");
    Console.WriteLine($"  Compute Units: {device.ComputeUnits}");
}
```

### Backend Providers

Register GPU backend providers:

```csharp
// DotCompute backend (recommended - cross-platform)
services.AddGpuBridge()
    .AddDotGpuBackend();

// With custom configuration
services.AddGpuBridge()
    .AddDotGpuBackend(config =>
    {
        config.OptimizationLevel = OptimizationLevel.O3;
        config.MemorySettings.InitialPoolSize = 1024 * 1024 * 1024; // 1 GB
    });

// With factory for runtime configuration
services.AddGpuBridge()
    .AddDotGpuBackend(sp =>
    {
        var env = sp.GetRequiredService<IHostEnvironment>();
        var config = new DotGpuBackendConfiguration();
        config.EnableDebugMode = env.IsDevelopment();
        return new DotComputeBackendProvider(config, sp.GetRequiredService<ILoggerFactory>());
    });
```

## Performance Optimization

### Deployment Models

| Model | Latency | Throughput | Use Case |
|-------|---------|------------|----------|
| GPU-Offload (GpuGrainBase) | 10-100μs | 15K msg/s | Batch processing, infrequent GPU |
| GPU-Native (RingKernelGrainBase) | 100-500ns | 2M msg/s | High-frequency messaging |

### K2K Routing Strategies

```csharp
// Kernel-to-Kernel messaging strategies
public enum K2KRoutingStrategy
{
    Direct,      // Point-to-point messaging (100-500ns)
    Broadcast,   // One-to-many messaging
    Ring,        // Circular topology for consensus
    HashRouted   // Consistent hashing for load distribution
}
```

### Grain Base Classes

```csharp
// For GPU-offload model (batch processing)
public class MyComputeGrain : GpuGrainBase<MyState>, IMyComputeGrain
{
    public async ValueTask<float[]> ProcessAsync(float[] input)
    {
        return await InvokeKernelAsync<float[], float[]>("my-kernel", input);
    }
}

// For GPU-native model (high-frequency messaging)
public class MyHighFreqActor : RingKernelGrainBase<MyState, MyMessage>
{
    protected override string KernelId => "my-actor-kernel";

    public async ValueTask<int> ProcessAsync(MyMessage msg)
    {
        return await InvokeKernelAsync<MyMessage, int>(msg);
    }
}
```

### Memory Management Tips

1. **Use Ring Kernels for High-Frequency**: For >1000 msg/s, use `RingKernelGrainBase`
2. **Batch Operations**: For batch processing, use `GpuGrainBase` with large batches
3. **Enable Memory Pooling**: Set `MemoryPoolSizeMB` appropriately
4. **Monitor Telemetry**: Enable profiling during development

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
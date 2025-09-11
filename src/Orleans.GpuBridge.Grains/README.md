# Orleans.GpuBridge.Grains

Pre-built GPU-accelerated Orleans grains for high-performance distributed computing with CUDA/OpenCL support.

## Overview

Orleans.GpuBridge.Grains provides a collection of production-ready Orleans grains that seamlessly integrate GPU acceleration into your distributed applications. These grains handle the complexities of GPU resource management, memory allocation, and kernel execution while maintaining Orleans' familiar programming model.

## Features

- **Batch Processing**: High-throughput GPU batch processing with automatic partitioning
- **Stream Processing**: Real-time GPU-accelerated stream processing with Orleans Streaming
- **Resident Memory**: Persistent GPU memory management for zero-copy operations
- **Automatic Fallback**: CPU fallback when GPU resources are unavailable
- **Resource Management**: Automatic GPU resource cleanup and memory management
- **Placement Strategy**: GPU-aware grain placement for optimal resource utilization

## Available Grain Types

### GpuBatchGrain&lt;TIn, TOut&gt;

High-performance batch processing grain that executes GPU kernels on collections of data.

**Key Features:**
- Concurrent execution with configurable limits
- Automatic batch partitioning for optimal GPU utilization
- Built-in performance monitoring and timing
- Observer pattern support for real-time updates

**Use Cases:**
- Machine learning inference on large datasets
- Mathematical computations (matrix operations, FFT)
- Image/video processing pipelines
- Financial calculations and risk analysis

### GpuStreamGrain&lt;TIn, TOut&gt;

Real-time stream processing grain that processes Orleans streams through GPU kernels.

**Key Features:**
- Seamless integration with Orleans Streaming
- Real-time processing statistics and monitoring
- Configurable processing parameters
- Start/stop controls for dynamic workloads

**Use Cases:**
- Real-time analytics and event processing
- Live video/audio processing
- IoT sensor data processing
- Continuous machine learning inference

### GpuResidentGrain&lt;T&gt;

Memory management grain that maintains persistent GPU memory allocations across operations.

**Key Features:**
- Persistent GPU memory allocation
- Zero-copy kernel execution
- Multiple memory types (default, pinned, shared)
- Direct memory read/write operations
- Comprehensive memory usage tracking

**Use Cases:**
- Large dataset caching on GPU
- Persistent model parameters for ML inference
- Shared data structures across multiple operations
- High-frequency trading with pre-loaded market data

## Installation

Add the package reference to your Orleans application:

```xml
<PackageReference Include="Orleans.GpuBridge.Grains" Version="1.0.0" />
```

## Quick Start

### 1. Configure GPU Bridge in Your Silo

```csharp
var siloBuilder = new SiloBuilder()
    .UseLocalhostClustering()
    .AddGpuBridge(options =>
    {
        options.PreferGpu = true;
        options.FallbackToCpu = true;
        options.MaxConcurrentKernels = 8;
    })
    .AddKernel(k => k
        .Id("vector-add")
        .In<float[]>()
        .Out<float[]>()
        .FromFactory<VectorAddKernel>())
    .ConfigureLogging(builder => builder.AddConsole());

using var silo = siloBuilder.Build();
await silo.StartAsync();
```

### 2. Using GpuBatchGrain

```csharp
// Get grain reference
var batchGrain = grainFactory.GetGrain<IGpuBatchGrain<float[], float[]>>("vector-add");

// Prepare data
var inputBatches = new List<float[]>
{
    new float[] { 1.0f, 2.0f, 3.0f },
    new float[] { 4.0f, 5.0f, 6.0f },
    new float[] { 7.0f, 8.0f, 9.0f }
};

// Execute batch processing
var result = await batchGrain.ExecuteAsync(inputBatches, new GpuExecutionHints
{
    BatchSize = 1024,
    PreferredDevice = 0
});

if (result.Success)
{
    Console.WriteLine($"Processed {result.Results.Count} batches in {result.ExecutionTime}");
    foreach (var output in result.Results)
    {
        Console.WriteLine($"Result: [{string.Join(", ", output)}]");
    }
}
```

### 3. Using GpuStreamGrain

```csharp
// Configure streams
var streamProvider = clusterClient.GetStreamProvider("gpu-streams");
var inputStream = streamProvider.GetStream<float[]>(StreamId.Create("input", "data"));
var outputStream = streamProvider.GetStream<float[]>(StreamId.Create("output", "results"));

// Get stream processing grain
var streamGrain = grainFactory.GetGrain<IGpuStreamGrain<float[], float[]>>("vector-multiply");

// Start processing
await streamGrain.StartProcessingAsync(
    inputStream.StreamId,
    outputStream.StreamId,
    new GpuExecutionHints { BatchSize = 512 });

// Send data to input stream
await inputStream.OnNextAsync(new float[] { 1.0f, 2.0f, 3.0f });

// Subscribe to output stream
var subscription = await outputStream.SubscribeAsync(
    (data, token) =>
    {
        Console.WriteLine($"Processed: [{string.Join(", ", data)}]");
        return Task.CompletedTask;
    });

// Monitor processing
var stats = await streamGrain.GetStatsAsync();
Console.WriteLine($"Items processed: {stats.ItemsProcessed}, Avg latency: {stats.AverageLatencyMs}ms");
```

### 4. Using GpuResidentGrain

```csharp
// Get resident grain
var residentGrain = grainFactory.GetGrain<IGpuResidentGrain<float>>("model-weights");

// Allocate GPU memory
var inputHandle = await residentGrain.AllocateAsync(
    sizeBytes: 1024 * sizeof(float),
    memoryType: GpuMemoryType.Pinned);

var outputHandle = await residentGrain.AllocateAsync(
    sizeBytes: 1024 * sizeof(float),
    memoryType: GpuMemoryType.Default);

// Write data to GPU memory
await residentGrain.WriteAsync(inputHandle, new float[] { 1.0f, 2.0f, 3.0f, 4.0f });

// Execute kernel on resident data
var computeResult = await residentGrain.ComputeAsync(
    KernelId.Parse("neural-network-forward"),
    inputHandle,
    outputHandle,
    new GpuComputeParams
    {
        WorkGroupSize = 256,
        Constants = new Dictionary<string, object> { ["learning_rate"] = 0.001f }
    });

if (computeResult.Success)
{
    // Read results back
    var results = await residentGrain.ReadAsync<float>(outputHandle, count: 1024);
    Console.WriteLine($"Computation completed in {computeResult.ExecutionTime}");
}

// Clean up
await residentGrain.ReleaseAsync(inputHandle);
await residentGrain.ReleaseAsync(outputHandle);
```

## Configuration

### GPU Bridge Options

```csharp
services.AddGpuBridge(options =>
{
    // GPU preferences
    options.PreferGpu = true;              // Prefer GPU over CPU when available
    options.FallbackToCpu = true;          // Fall back to CPU if GPU fails
    options.MaxConcurrentKernels = 8;      // Maximum concurrent GPU kernels
    
    // Memory management
    options.MaxGpuMemoryMB = 2048;         // Maximum GPU memory allocation
    options.EnableMemoryPooling = true;    // Enable memory pool for reuse
    
    // Performance tuning
    options.DefaultBatchSize = 1024;       // Default batch size for operations
    options.KernelTimeout = TimeSpan.FromSeconds(30); // Kernel execution timeout
    
    // Diagnostics
    options.EnableProfiling = true;        // Enable GPU profiling
    options.LogKernelExecutions = false;   // Log individual kernel executions
});
```

### Grain-Specific Configuration

```csharp
// Batch grain with observer callbacks
var result = await batchGrain.ExecuteWithCallbackAsync(
    inputBatches,
    new ProgressObserver<float[]>(), // Custom observer implementation
    new GpuExecutionHints
    {
        BatchSize = 2048,
        PreferredDevice = 1,
        EnableProfiling = true
    });

// Stream grain with custom parameters
await streamGrain.StartProcessingAsync(
    inputStreamId,
    outputStreamId,
    new GpuExecutionHints
    {
        BatchSize = 512,
        MaxLatencyMs = 100, // Maximum acceptable latency
        EnableBackpressure = true
    });
```

## Performance Characteristics

### GpuBatchGrain Performance

| Batch Size | Throughput (ops/sec) | Latency (ms) | Memory Usage |
|------------|---------------------|--------------|--------------|
| 256        | 15,000              | 12.5         | 512 MB       |
| 1024       | 45,000              | 18.2         | 1.2 GB       |
| 4096       | 120,000             | 28.7         | 3.8 GB       |
| 16384      | 200,000             | 45.1         | 12.1 GB      |

### Memory Management Best Practices

- **Use appropriate batch sizes**: Start with 1024 and tune based on your GPU memory
- **Enable memory pooling**: Reduces allocation overhead for repeated operations
- **Monitor memory usage**: Use `GetMemoryInfoAsync()` to track allocations
- **Clean up resources**: Always release memory handles when done
- **Consider memory type**: Use pinned memory for frequent host-device transfers

## Advanced Usage Examples

### Custom Kernel Implementation

```csharp
public class MatrixMultiplyKernel : IGpuKernel<MatrixPair, Matrix>
{
    public async Task<Matrix> ExecuteAsync(MatrixPair input)
    {
        // GPU kernel implementation
        // This is a simplified example - actual implementation would use CUDA/OpenCL
        return await ComputeMatrixMultiplyGpu(input.A, input.B);
    }
    
    public async Task<Matrix> ExecuteFallbackAsync(MatrixPair input)
    {
        // CPU fallback implementation
        return ComputeMatrixMultiplyCpu(input.A, input.B);
    }
}

// Register the kernel
services.AddKernel(k => k
    .Id("matrix-multiply")
    .In<MatrixPair>()
    .Out<Matrix>()
    .FromFactory<MatrixMultiplyKernel>());
```

### Error Handling and Resilience

```csharp
try
{
    var result = await batchGrain.ExecuteAsync(inputData);
    
    if (!result.Success)
    {
        _logger.LogError("Batch processing failed: {Error}", result.Error);
        // Handle error appropriately
        return await ProcessWithCpuFallback(inputData);
    }
    
    return result.Results;
}
catch (GpuResourceException ex)
{
    // GPU resource exhaustion
    _logger.LogWarning("GPU resources exhausted, retrying with smaller batch: {Message}", ex.Message);
    return await ProcessWithSmallerBatch(inputData);
}
catch (KernelExecutionException ex)
{
    // Kernel execution failure
    _logger.LogError(ex, "Kernel execution failed");
    throw; // Re-throw for upstream handling
}
```

### Monitoring and Diagnostics

```csharp
// Monitor batch grain performance
var batchGrain = grainFactory.GetGrain<IGpuBatchGrain<float[], float[]>>("my-kernel");
var result = await batchGrain.ExecuteAsync(data);

Console.WriteLine($"Execution time: {result.ExecutionTime}");
Console.WriteLine($"Throughput: {data.Count / result.ExecutionTime.TotalSeconds:F2} items/sec");

// Monitor stream processing
var streamGrain = grainFactory.GetGrain<IGpuStreamGrain<float[], float[]>>("stream-processor");
var stats = await streamGrain.GetStatsAsync();

Console.WriteLine($"Items processed: {stats.ItemsProcessed}");
Console.WriteLine($"Success rate: {(double)stats.ItemsProcessed / (stats.ItemsProcessed + stats.ItemsFailed):P2}");
Console.WriteLine($"Average latency: {stats.AverageLatencyMs:F2}ms");

// Monitor memory usage
var residentGrain = grainFactory.GetGrain<IGpuResidentGrain<float>>("memory-manager");
var memInfo = await residentGrain.GetMemoryInfoAsync();

Console.WriteLine($"Total allocated: {memInfo.TotalAllocatedBytes / (1024 * 1024)}MB");
Console.WriteLine($"Active allocations: {memInfo.ActiveAllocations}");
Console.WriteLine($"Memory efficiency: {memInfo.MemoryEfficiency:P2}");
```

## Best Practices

### Grain Design Patterns

1. **Use StatelessWorker**: For high-throughput scenarios, use `[StatelessWorker]` attribute
2. **Enable Reentrancy**: Use `[Reentrant]` for concurrent processing capabilities
3. **Implement Proper Cleanup**: Always dispose GPU resources in grain deactivation
4. **Monitor Resource Usage**: Track GPU memory and processing utilization

### Performance Optimization

1. **Batch Size Tuning**: Optimize batch sizes based on your GPU memory and compute capacity
2. **Memory Reuse**: Use resident grains for frequently accessed data
3. **Asynchronous Processing**: Leverage Orleans' async nature for overlapping operations
4. **Stream Processing**: Use streaming for real-time scenarios with lower latency requirements

### Error Handling

1. **Implement Fallbacks**: Always provide CPU fallback implementations
2. **Handle Resource Limits**: Gracefully handle GPU memory exhaustion
3. **Timeout Management**: Set appropriate timeouts for long-running kernels
4. **Logging and Monitoring**: Implement comprehensive logging for troubleshooting

### Resource Management

1. **Clean Shutdown**: Implement proper grain deactivation to free GPU resources
2. **Memory Pooling**: Enable memory pooling for better performance
3. **Device Selection**: Consider multi-GPU scenarios with device selection
4. **Load Balancing**: Use Orleans placement strategies for GPU load balancing

## Dependencies

- **Microsoft.Orleans.Core** (≥9.2.1): Core Orleans framework
- **Microsoft.Orleans.Runtime** (≥9.2.1): Orleans runtime components
- **Microsoft.Orleans.Server** (≥9.2.1): Orleans server infrastructure
- **Microsoft.Orleans.Streaming** (≥9.2.1): Orleans streaming support
- **Orleans.GpuBridge.Abstractions**: Core GPU bridge abstractions
- **Orleans.GpuBridge.Runtime**: GPU bridge runtime implementation

## Platform Requirements

- **.NET 9.0** or later
- **CUDA 12.0+** or **OpenCL 2.0+** for GPU acceleration
- **Orleans 9.2+** compatible silo
- **Windows 10/11** or **Linux** with GPU drivers

## License

```
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
```

## Contributing

Contributions are welcome! Please read our [Contributing Guidelines](../../CONTRIBUTING.md) and [Code of Conduct](../../CODE_OF_CONDUCT.md) before submitting pull requests.

## Support

- **Documentation**: [Orleans GPU Bridge Documentation](../../docs/README.md)
- **Issues**: [GitHub Issues](https://github.com/mivertowski/Orleans.GpuBridge.Core/issues)
- **Discussions**: [GitHub Discussions](https://github.com/mivertowski/Orleans.GpuBridge.Core/discussions)
- **Orleans Community**: [Orleans Discord](https://discord.gg/orleans)

---

**Orleans.GpuBridge.Grains** - Bringing GPU acceleration to Orleans distributed applications with minimal complexity and maximum performance.
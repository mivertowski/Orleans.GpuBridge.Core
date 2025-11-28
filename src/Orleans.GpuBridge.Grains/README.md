# Orleans.GpuBridge.Grains

GPU-native Orleans grains for high-performance distributed computing with DotCompute backend support.

## Overview

Orleans.GpuBridge.Grains provides production-ready Orleans grain base classes and implementations for GPU acceleration. This package supports two deployment models:

| Model | Base Class | Latency | Throughput | Use Case |
|-------|-----------|---------|------------|----------|
| GPU-Offload | `GpuGrainBase<TState>` | 10-100μs | 15K msg/s | Batch processing |
| GPU-Native | `RingKernelGrainBase<TState, TMessage>` | 100-500ns | 2M msg/s | High-frequency messaging |

## Features

- **GPU-Offload Model**: Traditional grain pattern with GPU kernel execution
- **GPU-Native Model**: Actors that live permanently in GPU memory with sub-microsecond latency
- **Ring Kernels**: Persistent GPU threads processing messages without kernel launch overhead
- **K2K Messaging**: Kernel-to-Kernel communication for GPU-resident actors
- **Temporal Alignment**: HLC and Vector Clocks for causal ordering on GPU
- **Hypergraph Support**: Multi-way relationships with GPU-accelerated pattern matching
- **Automatic Fallback**: CPU fallback when GPU resources are unavailable

## Grain Base Classes

### GpuGrainBase&lt;TState&gt;

Base class for Orleans grains with GPU-offload kernel execution.

**Key Features:**
- Familiar Orleans grain pattern with GPU kernel invocation
- Automatic CPU fallback when GPU unavailable
- Kernel execution via `InvokeKernelAsync<TIn, TOut>`
- State management via Orleans persistence

**Use Cases:**
- Batch processing with large datasets
- Machine learning inference
- Image/video processing pipelines
- Infrequent GPU operations (< 1000 msg/s)

### RingKernelGrainBase&lt;TState, TMessage&gt;

Base class for GPU-native actors using persistent ring kernels.

**Key Features:**
- Sub-microsecond message latency (100-500ns)
- State resides in GPU memory
- Zero kernel launch overhead
- Lock-free atomic queue operations
- Automatic CPU fallback

**Use Cases:**
- High-frequency messaging (> 1000 msg/s)
- Real-time trading systems
- Game server tick processing
- Digital twin simulations

### Additional Grain Types

- **GpuBatchGrain&lt;TIn, TOut&gt;**: Generic batch processing grain
- **GpuStreamGrain&lt;TIn, TOut&gt;**: Orleans Streaming with GPU processing
- **GpuResidentGrain&lt;T&gt;**: Persistent GPU memory management
- **HypergraphVertexGrain**: Vertex actor for hypergraph structures
- **HypergraphHyperedgeGrain**: Hyperedge actor for multi-way relationships
- **PatternDetectorGrain**: Temporal pattern detection grain
- **TemporalGraphGrain**: Graph with temporal ordering

## Installation

Add the package reference to your Orleans application:

```xml
<PackageReference Include="Orleans.GpuBridge.Grains" Version="0.1.0" />
```

## Quick Start

### 1. Configure GPU Bridge with Ring Kernel Support

```csharp
using Orleans.GpuBridge.Runtime.Extensions;
using Orleans.GpuBridge.Backends.DotCompute.Extensions;

var builder = Host.CreateApplicationBuilder(args);

builder.Services
    .AddOrleans(orleans =>
    {
        orleans.UseLocalhostClustering();
    });

builder.Services
    .AddGpuBridge(options =>
    {
        options.PreferGpu = true;
        options.FallbackToCpu = true;
        options.MaxConcurrentKernels = 100;
    })
    .AddDotGpuBackend()
    .Services
    .AddRingKernelSupport()
    .AddK2KSupport()
    .AddDotComputeRingKernelBridge();

var host = builder.Build();
await host.RunAsync();
```

### 2. Using GpuGrainBase (GPU-Offload Model)

```csharp
using Orleans.GpuBridge.Grains.Base;

public interface IComputeGrain : IGrainWithGuidKey
{
    ValueTask<float[]> ProcessDataAsync(float[] input);
    ValueTask<ComputeStats> GetStatsAsync();
}

public class ComputeGrain : GpuGrainBase<ComputeState>, IComputeGrain
{
    public async ValueTask<float[]> ProcessDataAsync(float[] input)
    {
        // Execute kernel on GPU (or CPU fallback)
        var result = await InvokeKernelAsync<float[], float[]>("vector-process", input);

        // Update local state
        State.ProcessedCount++;
        State.LastProcessedAt = DateTime.UtcNow;

        return result;
    }

    public ValueTask<ComputeStats> GetStatsAsync() =>
        ValueTask.FromResult(new ComputeStats(State.ProcessedCount, State.LastProcessedAt));
}

public struct ComputeState
{
    public int ProcessedCount;
    public DateTime LastProcessedAt;
}

public record ComputeStats(int ProcessedCount, DateTime LastProcessedAt);
```

### 3. Using RingKernelGrainBase (GPU-Native Model)

```csharp
using Orleans.GpuBridge.Grains.Base;
using System.Runtime.InteropServices;

public interface IHighFrequencyActor : IGrainWithIntegerKey
{
    ValueTask<int> IncrementAsync(int amount);
    ValueTask<int> GetValueAsync();
    ValueTask ResetAsync();
}

public class HighFrequencyActor : RingKernelGrainBase<CounterState, CounterMessage>, IHighFrequencyActor
{
    protected override string KernelId => "counters/high-frequency";

    public async ValueTask<int> IncrementAsync(int amount)
    {
        // Message processed at 100-500ns latency on GPU
        var request = new CounterMessage { Operation = CounterOp.Increment, Amount = amount };
        return await InvokeKernelAsync<CounterMessage, int>(request);
    }

    public async ValueTask<int> GetValueAsync()
    {
        // Read current state from GPU memory
        var state = await GetGpuStateAsync();
        return state.Value;
    }

    public async ValueTask ResetAsync()
    {
        var request = new CounterMessage { Operation = CounterOp.Reset, Amount = 0 };
        await InvokeKernelAsync<CounterMessage, int>(request);
    }
}

[StructLayout(LayoutKind.Sequential)]
public struct CounterState
{
    public int Value;
    public long LastUpdated;
    public long MessageCount;
}

[StructLayout(LayoutKind.Sequential)]
public struct CounterMessage
{
    public CounterOp Operation;
    public int Amount;
}

public enum CounterOp : int { Increment = 0, Decrement = 1, Reset = 2 }
```

### 4. Using Hypergraph Grains

```csharp
using Orleans.GpuBridge.Abstractions.Hypergraph;

// Create a vertex
var vertex = grainFactory.GetGrain<IHypergraphVertex>("user-123");
await vertex.InitializeAsync(new VertexInitRequest
{
    EntityType = "User",
    InitialProperties = new Dictionary<string, object>
    {
        ["name"] = "Alice",
        ["score"] = 100
    }
});

// Create a hyperedge connecting multiple vertices
var hyperedge = grainFactory.GetGrain<IHypergraphHyperedge>("friendship-group-1");
await hyperedge.InitializeAsync(new HyperedgeInitRequest
{
    RelationType = "FriendshipGroup",
    Members = new[]
    {
        new HyperedgeMemberInit { VertexId = "user-123", Role = "Admin" },
        new HyperedgeMemberInit { VertexId = "user-456", Role = "Member" },
        new HyperedgeMemberInit { VertexId = "user-789", Role = "Member" }
    }
});
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
    options.DefaultGridSize = 1;        // Single block for single-actor
    options.DefaultBlockSize = 256;     // Optimal for most GPUs
    options.DefaultQueueCapacity = 256; // Message queue size (power of 2)
    options.EnableKernelCaching = true; // Cache compiled kernels
    options.DeviceIndex = 0;            // GPU device to use
});
```

### GpuExecutionHints

```csharp
// For GpuBatchGrain operations
var result = await batchGrain.ExecuteAsync(inputBatches, new GpuExecutionHints(
    PreferGpu: true,
    BatchSize: 1024,
    DeviceId: 0,
    Timeout: TimeSpan.FromSeconds(30)));
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
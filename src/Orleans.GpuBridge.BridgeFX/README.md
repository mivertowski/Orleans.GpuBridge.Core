# Orleans.GpuBridge.BridgeFX

High-level fluent GPU pipeline API for Orleans GPU Bridge.

## Overview

Orleans.GpuBridge.BridgeFX provides a high-level, fluent API for building GPU-accelerated data processing pipelines in Orleans applications. It abstracts away the complexity of kernel management, batching, and result aggregation while providing powerful customization options.

## Key Features

- **Fluent Pipeline Builder**: Intuitive method chaining for pipeline construction
- **Automatic Batching**: Smart data partitioning for optimal GPU utilization
- **Result Aggregation**: Automatic collection and ordering of parallel results
- **Retry Policies**: Built-in resilience with configurable retry strategies
- **Memory Strategies**: Pooled, streaming, and persistent memory management
- **Type Safety**: Strongly-typed input/output with compile-time validation

## Installation

```bash
dotnet add package Orleans.GpuBridge.BridgeFX
```

## Quick Start

### Basic Pipeline

```csharp
using Orleans.GpuBridge.BridgeFX;

// Simple pipeline execution
var results = await GpuPipeline<float[], float[]>
    .For(grainFactory, "vector_add")
    .ExecuteAsync(inputData);
```

### Batch Processing

```csharp
// Process large datasets in batches
var results = await GpuPipeline<ImageData, ProcessedImage>
    .For(grainFactory, "image_filter")
    .WithBatchSize(64)
    .ExecuteAsync(images);
```

### Advanced Pipeline

```csharp
var results = await GpuPipeline<SensorData, AnalysisResult>
    .For(grainFactory, "sensor_analysis")
    .WithBatchSize(128)
    .WithMemoryStrategy(MemoryStrategy.Pooled)
    .WithRetryPolicy(new ExponentialBackoffPolicy
    {
        MaxRetries = 3,
        BaseDelay = TimeSpan.FromMilliseconds(100)
    })
    .WithTimeout(TimeSpan.FromMinutes(5))
    .WithProgressCallback(progress =>
        Console.WriteLine($"Progress: {progress:P0}"))
    .ExecuteAsync(sensorReadings);
```

## Pipeline Configuration

### Batch Size

Control how data is partitioned for GPU processing:

```csharp
.WithBatchSize(64)  // Process 64 items per GPU kernel invocation
```

**Recommendations:**
- Small data items: 256-1024
- Large data items: 16-64
- Memory-intensive: 8-32

### Memory Strategies

```csharp
public enum MemoryStrategy
{
    Default,     // System decides based on data size
    Pooled,      // Reuse GPU memory allocations
    Streaming,   // Stream data to/from GPU
    Persistent   // Keep data on GPU between operations
}
```

```csharp
// For repeated operations on similar data
.WithMemoryStrategy(MemoryStrategy.Pooled)

// For large datasets that don't fit in GPU memory
.WithMemoryStrategy(MemoryStrategy.Streaming)

// For pipeline chains where data stays on GPU
.WithMemoryStrategy(MemoryStrategy.Persistent)
```

### Retry Policies

```csharp
// Exponential backoff
.WithRetryPolicy(new ExponentialBackoffPolicy
{
    MaxRetries = 3,
    BaseDelay = TimeSpan.FromMilliseconds(100),
    MaxDelay = TimeSpan.FromSeconds(10)
})

// Fixed interval
.WithRetryPolicy(new FixedIntervalPolicy
{
    MaxRetries = 5,
    Interval = TimeSpan.FromMilliseconds(500)
})

// Custom policy
.WithRetryPolicy(new CustomRetryPolicy(
    shouldRetry: (ex, attempt) => attempt < 3 && ex is GpuMemoryException,
    getDelay: attempt => TimeSpan.FromMilliseconds(100 * Math.Pow(2, attempt))
))
```

## Pipeline Chaining

Chain multiple GPU operations:

```csharp
var results = await GpuPipeline<RawData, FinalResult>
    .For(grainFactory, "preprocess")
    .ThenExecute("transform")
    .ThenExecute("analyze")
    .ThenExecute("postprocess")
    .WithMemoryStrategy(MemoryStrategy.Persistent) // Keep on GPU between steps
    .ExecuteAsync(rawData);
```

## Progress Monitoring

Track pipeline execution progress:

```csharp
.WithProgressCallback(progress =>
{
    progressBar.Value = progress;
    Console.WriteLine($"Completed: {progress:P0}");
})
```

## Error Handling

```csharp
try
{
    var results = await GpuPipeline<Input, Output>
        .For(grainFactory, "my_kernel")
        .ExecuteAsync(data);
}
catch (GpuPipelineException ex)
{
    // Pipeline-level failure
    logger.LogError(ex, "Pipeline failed at batch {BatchIndex}", ex.FailedBatchIndex);
}
catch (KernelNotFoundException ex)
{
    // Kernel not registered
    logger.LogError(ex, "Kernel {KernelId} not found", ex.KernelId);
}
catch (GpuMemoryException ex)
{
    // Insufficient GPU memory - try smaller batch size
    logger.LogWarning("GPU OOM - reducing batch size");
}
```

## Parallel Pipeline Execution

Execute multiple independent pipelines concurrently:

```csharp
var pipeline1 = GpuPipeline<A, B>.For(grainFactory, "kernel1").ExecuteAsync(data1);
var pipeline2 = GpuPipeline<C, D>.For(grainFactory, "kernel2").ExecuteAsync(data2);
var pipeline3 = GpuPipeline<E, F>.For(grainFactory, "kernel3").ExecuteAsync(data3);

await Task.WhenAll(pipeline1, pipeline2, pipeline3);

var results1 = await pipeline1;
var results2 = await pipeline2;
var results3 = await pipeline3;
```

## Performance Tips

1. **Batch Size Tuning**: Profile different batch sizes for your data
2. **Memory Strategy**: Use `Pooled` for repeated similar operations
3. **Pipeline Chains**: Use `Persistent` memory for multi-step pipelines
4. **Parallel Pipelines**: Execute independent pipelines concurrently
5. **Progress Callbacks**: Use sparingly in production (adds overhead)

## API Reference

### GpuPipeline<TIn, TOut>

```csharp
public static class GpuPipeline<TIn, TOut>
{
    // Create pipeline for kernel
    public static IGpuPipelineBuilder<TIn, TOut> For(
        IGrainFactory grainFactory,
        string kernelId);
}
```

### IGpuPipelineBuilder<TIn, TOut>

```csharp
public interface IGpuPipelineBuilder<TIn, TOut>
{
    IGpuPipelineBuilder<TIn, TOut> WithBatchSize(int batchSize);
    IGpuPipelineBuilder<TIn, TOut> WithMemoryStrategy(MemoryStrategy strategy);
    IGpuPipelineBuilder<TIn, TOut> WithRetryPolicy(IRetryPolicy policy);
    IGpuPipelineBuilder<TIn, TOut> WithTimeout(TimeSpan timeout);
    IGpuPipelineBuilder<TIn, TOut> WithProgressCallback(Action<double> callback);
    IGpuPipelineBuilder<TIn, TNext> ThenExecute<TNext>(string kernelId);

    Task<TOut[]> ExecuteAsync(TIn[] inputs, CancellationToken ct = default);
    Task<TOut> ExecuteAsync(TIn input, CancellationToken ct = default);
}
```

## Dependencies

- **Orleans.GpuBridge.Abstractions**
- **Orleans.GpuBridge.Runtime**
- **Microsoft.Orleans.Core** (>= 9.0.0)

## License

MIT License - Copyright (c) 2025 Michael Ivertowski

---

For more information, see the [Orleans.GpuBridge.Core Documentation](https://github.com/mivertowski/Orleans.GpuBridge.Core).

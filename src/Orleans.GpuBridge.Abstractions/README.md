# Orleans.GpuBridge.Abstractions

## Overview

Orleans.GpuBridge.Abstractions provides the core interfaces and contracts for GPU acceleration within the Microsoft Orleans distributed computing framework. This package defines the essential abstractions that enable seamless integration of GPU compute resources with Orleans grains, allowing developers to leverage GPU acceleration in distributed applications.

## Features

- **Core Interfaces**: Foundational contracts for GPU bridge operations
- **Kernel Abstractions**: Generic kernel execution interfaces with type-safe input/output
- **Configuration Models**: Comprehensive options for GPU bridge behavior customization
- **Memory Management**: Abstractions for GPU memory allocation and transfer strategies
- **Error Handling**: Specialized exceptions and error models for GPU operations
- **Telemetry Support**: Built-in metrics and monitoring interfaces
- **Backend Provider Contracts**: Extensible backend system for different GPU frameworks
- **Placement Strategies**: GPU-aware grain placement for optimal resource utilization

## Installation

```bash
dotnet add package Orleans.GpuBridge.Abstractions
```

## Key Components

### Core Interfaces

#### IGpuBridge
The primary interface for GPU bridge operations, providing methods for kernel execution and resource management.

```csharp
public interface IGpuBridge
{
    ValueTask<TOut> ExecuteKernelAsync<TIn, TOut>(string kernelId, TIn input, CancellationToken cancellationToken = default);
    ValueTask<IComputeDevice[]> GetAvailableDevicesAsync(CancellationToken cancellationToken = default);
    ValueTask<GpuMemoryInfo> GetMemoryInfoAsync(string deviceId, CancellationToken cancellationToken = default);
}
```

#### IGpuKernel
Defines the contract for GPU kernel implementations with strongly-typed input and output.

```csharp
public interface IGpuKernel<TIn, TOut>
{
    string Id { get; }
    ValueTask<TOut> ExecuteAsync(TIn input, CancellationToken cancellationToken = default);
}
```

### Configuration

#### GpuBridgeOptions
Comprehensive configuration options for GPU bridge behavior:

```csharp
services.Configure<GpuBridgeOptions>(options =>
{
    options.PreferGpu = true;
    options.FallbackToCpu = true;
    options.MaxRetries = 3;
    options.Timeout = TimeSpan.FromSeconds(30);
    options.MemoryPooling = new MemoryPoolingOptions
    {
        Enabled = true,
        MaxPoolSize = 1024 * 1024 * 512 // 512MB
    };
});
```

### Attributes

#### [GpuAccelerated]
Mark Orleans grains for GPU acceleration:

```csharp
[GpuAccelerated]
public interface IComputeGrain : IGrainWithGuidKey
{
    ValueTask<float[]> ProcessDataAsync(float[] input);
}
```

### Memory Management

The package provides abstractions for different GPU memory strategies:

- **DeviceMemory**: Direct GPU device memory allocation
- **PinnedMemory**: CPU memory pinned for faster GPU transfers
- **UnifiedMemory**: Unified memory accessible by both CPU and GPU
- **ManagedMemory**: Automatically managed memory with pooling support

### Error Handling

Specialized exceptions for GPU operations:

- `GpuExecutionException`: Kernel execution failures
- `GpuMemoryException`: Memory allocation/transfer errors
- `GpuDeviceException`: Device-related issues
- `GpuTimeoutException`: Operation timeout errors

## Usage Example

```csharp
// Define a custom kernel
public class VectorAddKernel : IGpuKernel<VectorPair, float[]>
{
    public string Id => "vector-add";
    
    public async ValueTask<float[]> ExecuteAsync(VectorPair input, CancellationToken cancellationToken)
    {
        // Kernel implementation
        return result;
    }
}

// Use in an Orleans grain
[GpuAccelerated]
public class ComputeGrain : Grain, IComputeGrain
{
    private readonly IGpuBridge _gpuBridge;
    
    public ComputeGrain(IGpuBridge gpuBridge)
    {
        _gpuBridge = gpuBridge;
    }
    
    public async ValueTask<float[]> ProcessDataAsync(float[] input)
    {
        var result = await _gpuBridge.ExecuteKernelAsync<float[], float[]>(
            "vector-add", 
            input);
        return result;
    }
}
```

## Extensibility

The abstractions package is designed for extensibility:

1. **Custom Kernel Types**: Implement `IGpuKernel<TIn, TOut>` for specialized operations
2. **Memory Strategies**: Extend memory management interfaces for custom allocation patterns
3. **Device Selection**: Implement custom device selection strategies
4. **Error Recovery**: Define custom error handling and retry policies
5. **Backend Providers**: Create custom GPU backend implementations

## Dependencies

- .NET 9.0 or later
- Microsoft.Orleans.Core.Abstractions (>= 9.0.0)
- System.Memory (for Span/Memory support)

## Performance Considerations

- Use appropriate batch sizes for GPU operations (typically 1000+ elements)
- Consider memory transfer overhead when designing kernel interfaces
- Leverage memory pooling for frequently executed kernels
- Profile actual GPU vs CPU performance for your specific workloads

## Thread Safety

All interfaces in this package are designed to be thread-safe and suitable for use in concurrent Orleans grain activations.

## Contributing

Contributions are welcome! Please ensure:
- All new interfaces include XML documentation
- Breaking changes are avoided when possible
- Unit tests cover new abstractions
- Performance implications are documented

## License

Apache 2.0 - Copyright (c) 2025 Michael Ivertowski

## Support

For issues, feature requests, or questions:
- GitHub Issues: [Orleans.GpuBridge.Core/issues](https://github.com/yourusername/Orleans.GpuBridge.Core/issues)
- Documentation: [Full Documentation](https://github.com/yourusername/Orleans.GpuBridge.Core/docs)

## See Also

- [Orleans.GpuBridge.Runtime](../Orleans.GpuBridge.Runtime/README.md) - Runtime implementation
- [Orleans.GpuBridge.Grains](../Orleans.GpuBridge.Grains/README.md) - Pre-built GPU-accelerated grains
- [Orleans Documentation](https://dotnet.github.io/orleans/) - Microsoft Orleans framework
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
The primary interface for GPU bridge operations, providing methods for kernel resolution and device management.

```csharp
public interface IGpuBridge
{
    /// <summary>Gets GPU bridge information and capabilities.</summary>
    ValueTask<GpuBridgeInfo> GetInfoAsync(CancellationToken ct = default);

    /// <summary>Gets a typed kernel instance for execution.</summary>
    ValueTask<IGpuKernel<TIn, TOut>> GetKernelAsync<TIn, TOut>(
        KernelId kernelId, CancellationToken ct = default);

    /// <summary>Gets available GPU devices.</summary>
    ValueTask<IReadOnlyList<GpuDevice>> GetDevicesAsync(CancellationToken ct = default);

    /// <summary>Executes a kernel with untyped input/output (for dynamic scenarios).</summary>
    ValueTask<object> ExecuteKernelAsync(string kernelId, object input, CancellationToken ct = default);
}
```

#### IGpuKernel<TIn, TOut>
Defines the contract for GPU kernel implementations with strongly-typed input and output.

```csharp
public interface IGpuKernel<TIn, TOut> : IDisposable
{
    string KernelId { get; }
    string DisplayName { get; }
    string BackendProvider { get; }
    bool IsInitialized { get; }
    bool IsGpuAccelerated { get; }

    Task InitializeAsync(CancellationToken cancellationToken = default);
    Task<TOut> ExecuteAsync(TIn input, CancellationToken cancellationToken = default);
    Task<TOut[]> ExecuteBatchAsync(TIn[] inputs, CancellationToken cancellationToken = default);

    long GetEstimatedExecutionTimeMicroseconds(int inputSize);
    KernelMemoryRequirements GetMemoryRequirements();
    KernelValidationResult ValidateInput(TIn input);
    Task WarmupAsync(CancellationToken cancellationToken = default);
}

public sealed record KernelMemoryRequirements(
    long InputMemoryBytes,
    long OutputMemoryBytes,
    long WorkingMemoryBytes,
    long TotalMemoryBytes);

public sealed record KernelValidationResult(
    bool IsValid,
    string? ErrorMessage = null,
    string[]? ValidationErrors = null);
```

### Configuration

#### GpuBridgeOptions
Comprehensive configuration options for GPU bridge behavior:

```csharp
services.Configure<GpuBridgeOptions>(options =>
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

### Attributes

#### [GpuAccelerated]
Mark Orleans grains for GPU acceleration:

```csharp
[GpuAccelerated("my-kernel-id")]
public interface IComputeGrain : IGrainWithGuidKey
{
    ValueTask<float[]> ProcessDataAsync(float[] input);
}
```

#### Source Generation Attributes
Additional attributes for GPU-native actor generation:

```csharp
// Mark state types for GPU residency
[GpuState]
public partial struct CounterState
{
    public int Value;
    public long LastUpdated;
}

// Mark message handler methods
[GpuHandler]
public partial interface ICounterActor : IGrainWithIntegerKey
{
    ValueTask<int> IncrementAsync(int amount);
}

// Enable K2K (Kernel-to-Kernel) messaging targets
[K2KTarget(RoutingStrategy.HashRouted)]
public partial interface IRouterActor : IGrainWithIntegerKey
{
    ValueTask RouteMessageAsync(byte[] payload);
}

// Enable temporal ordering with HLC timestamps
[TemporalOrdered]
public partial interface IAuditActor : IGrainWithIntegerKey
{
    ValueTask<AuditEntry> RecordAsync(AuditEvent evt);
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

### Using GpuGrainBase for GPU-Offload Model

```csharp
using Orleans.GpuBridge.Grains.Base;

// GPU-accelerated grain using the offload model
public class ComputeGrain : GpuGrainBase<MyGrainState>, IComputeGrain
{
    public async ValueTask<float[]> ProcessDataAsync(float[] input)
    {
        // Execute kernel on GPU (or CPU fallback)
        return await InvokeKernelAsync<float[], float[]>("vector-add", input);
    }
}
```

### Using RingKernelGrainBase for GPU-Native Model

```csharp
using Orleans.GpuBridge.Grains.Base;

// GPU-native actor using persistent ring kernels
public class HighFrequencyActor : RingKernelGrainBase<CounterState, CounterMessage>
{
    protected override string KernelId => "counters/high-frequency";

    public async ValueTask<int> IncrementAsync(int amount)
    {
        // Message processed at sub-microsecond latency on GPU
        var request = new CounterMessage { Amount = amount };
        var response = await InvokeKernelAsync<CounterMessage, int>(request);
        return response;
    }
}

[StructLayout(LayoutKind.Sequential)]
public struct CounterState
{
    public int Value;
    public long LastUpdated;
}

[StructLayout(LayoutKind.Sequential)]
public struct CounterMessage
{
    public int Amount;
}
```

### Implementing IGpuKernel<TIn, TOut>

```csharp
using Orleans.GpuBridge.Abstractions.Kernels;

public class VectorAddKernel : IGpuKernel<float[], float[]>
{
    public string KernelId => "vector-add";
    public string DisplayName => "Vector Addition";
    public string BackendProvider => "DotCompute";
    public bool IsInitialized { get; private set; }
    public bool IsGpuAccelerated => true;

    public async Task InitializeAsync(CancellationToken ct = default)
    {
        // Initialize GPU resources
        IsInitialized = true;
        await Task.CompletedTask;
    }

    public async Task<float[]> ExecuteAsync(float[] input, CancellationToken ct = default)
    {
        // Execute on GPU or CPU fallback
        var result = new float[input.Length];
        for (int i = 0; i < input.Length; i++)
            result[i] = input[i] * 2.0f;
        return result;
    }

    public async Task<float[][]> ExecuteBatchAsync(float[][] inputs, CancellationToken ct = default)
    {
        var results = new float[inputs.Length][];
        for (int i = 0; i < inputs.Length; i++)
            results[i] = await ExecuteAsync(inputs[i], ct);
        return results;
    }

    public long GetEstimatedExecutionTimeMicroseconds(int inputSize) => inputSize / 1000;

    public KernelMemoryRequirements GetMemoryRequirements() =>
        new(InputMemoryBytes: 4096, OutputMemoryBytes: 4096, WorkingMemoryBytes: 0, TotalMemoryBytes: 8192);

    public KernelValidationResult ValidateInput(float[] input) =>
        input.Length > 0 ? new(true) : new(false, "Input array cannot be empty");

    public Task WarmupAsync(CancellationToken ct = default) => Task.CompletedTask;

    public void Dispose() { }
}
```

## Extensibility

The abstractions package is designed for extensibility:

1. **Custom Kernel Types**: Implement `IGpuKernel<TIn, TOut>` for specialized operations
2. **Memory Strategies**: Extend memory management interfaces for custom allocation patterns
3. **Device Selection**: Implement custom device selection strategies
4. **Error Recovery**: Define custom error handling and retry policies
5. **Backend Providers**: Create custom GPU backend implementations

## Ring Kernel Bridge Interface

For GPU-native actors using persistent ring kernels:

```csharp
public interface IRingKernelBridge
{
    /// <summary>Gets runtime state and availability.</summary>
    bool IsAvailable { get; }

    /// <summary>Allocates GPU memory for state.</summary>
    ValueTask<GpuStateHandle<TState>> AllocateStateAsync<TState>(
        long actorId, TState initialState, CancellationToken ct = default)
        where TState : unmanaged;

    /// <summary>Sends message through ring kernel for GPU-native processing.</summary>
    ValueTask<TResponse> SendMessageAsync<TState, TRequest, TResponse>(
        GpuStateHandle<TState> stateHandle, TRequest request, CancellationToken ct = default)
        where TState : unmanaged where TRequest : unmanaged where TResponse : unmanaged;

    /// <summary>Gets current state from GPU memory.</summary>
    ValueTask<TState> GetStateAsync<TState>(GpuStateHandle<TState> handle, CancellationToken ct = default)
        where TState : unmanaged;

    /// <summary>Releases GPU memory.</summary>
    ValueTask ReleaseAsync<TState>(GpuStateHandle<TState> handle, CancellationToken ct = default)
        where TState : unmanaged;

    /// <summary>Gets telemetry data.</summary>
    RingKernelTelemetry GetTelemetry();
}
```

## Dependencies

- .NET 9.0 or later
- Microsoft.Orleans.Core.Abstractions (>= 9.2.1)
- System.Memory (for Span/Memory support)
- System.Runtime.InteropServices (for unmanaged struct support)

## Performance Characteristics

| Model | Message Latency | Throughput | Best For |
|-------|----------------|------------|----------|
| GPU-Offload (GpuGrainBase) | 10-100μs | 15K msg/s | Batch processing, infrequent GPU |
| GPU-Native (RingKernelGrainBase) | 100-500ns | 2M msg/s | High-frequency messaging |

## Thread Safety

All interfaces are designed to be thread-safe and suitable for concurrent Orleans grain activations.

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
- GitHub Issues: [Orleans.GpuBridge.Core/issues](https://github.com/mivertowski/Orleans.GpuBridge.Core/issues)
- Documentation: [Full Documentation](https://github.com/mivertowski/Orleans.GpuBridge.Core/docs)

## See Also

- [Orleans.GpuBridge.Runtime](../Orleans.GpuBridge.Runtime/README.md) - Runtime implementation
- [Orleans.GpuBridge.Backends.DotCompute](../Orleans.GpuBridge.Backends.DotCompute/README.md) - DotCompute backend
- [Orleans.GpuBridge.Grains](../Orleans.GpuBridge.Grains/README.md) - Pre-built GPU-accelerated grains
- [Orleans Documentation](https://dotnet.github.io/orleans/) - Microsoft Orleans framework
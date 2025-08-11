# Orleans.GpuBridge Abstractions

## Overview

Orleans.GpuBridge provides a clean abstraction layer that decouples GPU operations from specific hardware implementations. This document details the core abstractions and their design principles.

## Core Abstractions

### IGpuBridge

The primary interface for GPU operations, providing a unified API regardless of the underlying GPU backend.

```csharp
public interface IGpuBridge
{
    // Device Discovery & Management
    Task<IGpuDevice> GetDeviceAsync(int? deviceIndex = null);
    Task<IReadOnlyList<IGpuDevice>> GetAvailableDevicesAsync();
    
    // Kernel Operations
    Task<IKernel> GetKernelAsync(string kernelId);
    Task<IKernel> CompileKernelAsync(string source, KernelOptions options = null);
    
    // Memory Management
    Task<IGpuMemory<T>> AllocateAsync<T>(int count) where T : unmanaged;
    
    // High-Level Execution
    Task<TOut> ExecuteAsync<TIn, TOut>(string kernelId, TIn input);
    Task<TOut[]> ExecuteBatchAsync<TIn, TOut>(string kernelId, TIn[] inputs);
}
```

**Design Principles:**
- **Async-First**: All operations are asynchronous to prevent blocking
- **Type Safety**: Generic methods ensure compile-time type checking
- **Resource Management**: Explicit allocation/deallocation for control

### IGpuDevice

Represents a physical or virtual GPU device.

```csharp
public interface IGpuDevice
{
    // Device Properties
    int Index { get; }
    string Name { get; }
    DeviceType Type { get; }
    ComputeCapability Capability { get; }
    
    // Memory Information
    long TotalMemoryBytes { get; }
    long AvailableMemoryBytes { get; }
    
    // Compute Capabilities
    int ComputeUnits { get; }
    int MaxWorkGroupSize { get; }
    int MaxWorkItemDimensions { get; }
    
    // Context Creation
    Task<IComputeContext> CreateContextAsync();
    
    // Device Metrics
    Task<DeviceMetrics> GetMetricsAsync();
}
```

**Key Concepts:**
- **Device Abstraction**: Unified interface for different GPU types
- **Capability Discovery**: Runtime detection of device features
- **Context Isolation**: Each context is independent

### IGpuMemory<T>

Type-safe GPU memory abstraction.

```csharp
public interface IGpuMemory<T> : IDisposable where T : unmanaged
{
    // Memory Properties
    int Length { get; }
    long SizeInBytes { get; }
    MemoryType Type { get; }
    bool IsDisposed { get; }
    
    // Data Transfer
    Task CopyToDeviceAsync(T[] source, int offset = 0, int? count = null);
    Task CopyFromDeviceAsync(T[] destination, int offset = 0, int? count = null);
    
    // Zero-Copy Access (when supported)
    Memory<T> AsMemory();
    Span<T> AsSpan();
    
    // Synchronization
    Task SynchronizeAsync();
}
```

**Memory Types:**
```csharp
public enum MemoryType
{
    Default,      // Device memory
    Pinned,       // Page-locked host memory
    Shared,       // Unified memory (CPU+GPU)
    Texture,      // Texture memory (GPU only)
    Constant      // Constant memory (GPU only)
}
```

### IKernel

Compiled GPU kernel ready for execution.

```csharp
public interface IKernel
{
    // Kernel Metadata
    string Id { get; }
    string Name { get; }
    KernelMetadata Metadata { get; }
    
    // Execution
    Task<TOut> ExecuteAsync<TIn, TOut>(TIn input);
    Task ExecuteAsync(KernelArguments args);
    
    // Argument Management
    void SetArgument(int index, object value);
    void SetArguments(params object[] args);
    
    // Performance
    Task<KernelProfile> ProfileAsync(KernelArguments args);
}
```

**Kernel Arguments:**
```csharp
public class KernelArguments
{
    public object[] Arguments { get; set; }
    public uint[] GlobalWorkSize { get; set; }
    public uint[] LocalWorkSize { get; set; }
    public uint[] GlobalWorkOffset { get; set; }
    public Dictionary<string, object> Parameters { get; set; }
}
```

## Pipeline Abstractions (BridgeFX)

### IPipelineStage

Base interface for all pipeline stages.

```csharp
public interface IPipelineStage<TIn, TOut>
{
    string Name { get; }
    Task<TOut> ProcessAsync(TIn input);
    Task<TOut[]> ProcessBatchAsync(TIn[] inputs);
}
```

### Pipeline Builder Pattern

```csharp
public interface IGpuPipelineBuilder<TInput, TOutput>
{
    // Add GPU kernel stage
    IGpuPipelineBuilder<TInput, TNext> AddKernel<TNext>(
        string kernelId, 
        Action<KernelConfiguration> configure = null);
    
    // Add transformation stage
    IGpuPipelineBuilder<TInput, TNext> Transform<TNext>(
        Func<TOutput, TNext> transform);
    
    // Add async transformation
    IGpuPipelineBuilder<TInput, TNext> TransformAsync<TNext>(
        Func<TOutput, Task<TNext>> transform);
    
    // Add parallel execution
    IGpuPipelineBuilder<TInput, TOutput> Parallel(
        int maxConcurrency);
    
    // Add batching
    IGpuPipelineBuilder<TInput, TOutput[]> Batch(
        int size, 
        TimeSpan? timeout = null);
    
    // Add filtering
    IGpuPipelineBuilder<TInput, TOutput> Filter(
        Func<TOutput, bool> predicate);
    
    // Add side effects
    IGpuPipelineBuilder<TInput, TOutput> Tap(
        Action<TOutput> action);
    
    // Build executable pipeline
    ExecutablePipeline<TInput, TOutput> Build();
}
```

### Executable Pipeline

```csharp
public class ExecutablePipeline<TIn, TOut>
{
    // Single item execution
    public Task<TOut> ExecuteAsync(TIn input);
    
    // Batch execution
    public Task<TOut[]> ExecuteAsync(TIn[] inputs);
    
    // Stream processing
    public IAsyncEnumerable<TOut> ExecuteAsync(
        IAsyncEnumerable<TIn> inputs);
    
    // Channel-based processing
    public Task ProcessChannelAsync(
        ChannelReader<TIn> input,
        ChannelWriter<TOut> output);
    
    // Metrics
    public PipelineMetrics GetMetrics();
    public void ResetMetrics();
}
```

## Grain Abstractions

### IGpuGrain

Base interface for GPU-accelerated grains.

```csharp
public interface IGpuGrain : IGrain
{
    Task<GpuCapabilities> GetCapabilitiesAsync();
    Task<bool> IsGpuAvailableAsync();
}
```

### Placement Attributes

```csharp
[AttributeUsage(AttributeTargets.Class)]
public class GpuResidentAttribute : PlacementAttribute
{
    public int? PreferredDeviceIndex { get; set; }
    public DeviceType PreferredDeviceType { get; set; }
    public int MinimumMemoryMB { get; set; }
    public ComputeCapability MinimumCapability { get; set; }
}

[AttributeUsage(AttributeTargets.Method)]
public class GpuAcceleratedAttribute : Attribute
{
    public string KernelId { get; }
    public bool AutoFallback { get; set; } = true;
    
    public GpuAcceleratedAttribute(string kernelId)
    {
        KernelId = kernelId;
    }
}
```

### Usage Example

```csharp
[GpuResident(MinimumMemoryMB = 2048)]
public class ImageProcessorGrain : Grain, IImageProcessorGrain
{
    private readonly IGpuBridge _gpu;
    
    [GpuAccelerated("resize_kernel")]
    public async Task<byte[]> ResizeImageAsync(byte[] image, int width, int height)
    {
        // Automatically uses GPU kernel if available
        return await _gpu.ExecuteAsync<byte[], byte[]>(
            "resize_kernel", 
            image, 
            new { width, height });
    }
}
```

## Backend Provider Abstraction

### IBackendProvider

Interface for pluggable GPU backends.

```csharp
public interface IBackendProvider
{
    string Name { get; }
    BackendType Type { get; }
    bool IsAvailable { get; }
    Version Version { get; }
    
    Task InitializeAsync();
    Task ShutdownAsync();
    
    IComputeContext CreateContext();
    IEnumerable<IDevice> EnumerateDevices();
    
    bool SupportsFeature(ComputeFeature feature);
    Task<BackendCapabilities> GetCapabilitiesAsync();
}
```

### IComputeContext

Execution context for a specific backend.

```csharp
public interface IComputeContext : IDisposable
{
    BackendType Backend { get; }
    IDevice Device { get; }
    
    // Kernel operations
    Task<IKernel> CompileKernelAsync(string source, CompilerOptions options);
    Task<IKernel> LoadKernelAsync(byte[] binary);
    
    // Buffer operations
    IBuffer<T> CreateBuffer<T>(int size, BufferUsage usage) where T : unmanaged;
    
    // Execution
    Task ExecuteAsync(IKernel kernel, KernelArguments args);
    Task<T> ExecuteWithResultAsync<T>(IKernel kernel, KernelArguments args);
    
    // Synchronization
    Task SynchronizeAsync();
    void Flush();
}
```

## Error Handling Abstractions

### Exception Hierarchy

```csharp
// Base exception
public abstract class GpuBridgeException : Exception
{
    public ErrorCategory Category { get; }
    public ErrorSeverity Severity { get; }
    public Dictionary<string, object> Context { get; }
}

// Specific exceptions
public class GpuOutOfMemoryException : GpuBridgeException
{
    public long RequestedBytes { get; }
    public long AvailableBytes { get; }
}

public class KernelCompilationException : GpuBridgeException
{
    public string Source { get; }
    public CompilationError[] Errors { get; }
}

public class KernelExecutionException : GpuBridgeException
{
    public string KernelId { get; }
    public KernelArguments Arguments { get; }
}

public class DeviceNotFoundException : GpuBridgeException
{
    public int? RequestedIndex { get; }
    public DeviceType? RequestedType { get; }
}
```

### Fallback Strategy

```csharp
public interface IFallbackStrategy
{
    Task<TResult> ExecuteWithFallbackAsync<TResult>(
        Func<Task<TResult>> primaryOperation,
        Func<Task<TResult>> fallbackOperation,
        FallbackContext context);
}

public class FallbackContext
{
    public string OperationName { get; set; }
    public int RetryCount { get; set; }
    public TimeSpan Timeout { get; set; }
    public bool LogWarnings { get; set; }
}
```

## Telemetry Abstractions

### Metrics

```csharp
public interface IGpuMetrics
{
    // Device metrics
    double GetDeviceUtilization();
    long GetMemoryUsed();
    long GetMemoryAvailable();
    double GetTemperature();
    double GetPowerUsage();
    
    // Kernel metrics
    TimeSpan GetAverageKernelTime();
    long GetKernelExecutionCount();
    long GetKernelErrorCount();
    
    // Memory pool metrics
    long GetPoolAllocations();
    long GetPoolDeallocations();
    long GetPoolSize();
    double GetPoolFragmentation();
    
    // Queue metrics
    int GetQueueDepth();
    TimeSpan GetAverageQueueTime();
}
```

### Tracing

```csharp
public interface IGpuTracer
{
    ISpan StartKernelExecution(string kernelId);
    ISpan StartMemoryOperation(string operation);
    ISpan StartPipelineStage(string stageName);
    
    void RecordEvent(string name, Dictionary<string, object> attributes);
    void RecordError(Exception exception);
}
```

## Extension Points

### Custom Kernel Loader

```csharp
public interface IKernelLoader
{
    Task<IKernel> LoadFromSourceAsync(string source);
    Task<IKernel> LoadFromFileAsync(string path);
    Task<IKernel> LoadFromAssemblyAsync(Assembly assembly, string resourceName);
    Task<IKernel> LoadFromBinaryAsync(byte[] binary);
}
```

### Custom Memory Allocator

```csharp
public interface IMemoryAllocator
{
    Task<IGpuMemory<T>> AllocateAsync<T>(int size, AllocationHints hints) 
        where T : unmanaged;
    
    void Free<T>(IGpuMemory<T> memory) where T : unmanaged;
    
    Task<AllocationStatistics> GetStatisticsAsync();
}

public class AllocationHints
{
    public MemoryType PreferredType { get; set; }
    public bool PinMemory { get; set; }
    public bool ZeroInitialize { get; set; }
    public int Alignment { get; set; }
}
```

## Design Patterns

### Builder Pattern for Complex Operations

```csharp
var result = await GpuOperation
    .Create()
    .WithKernel("matmul")
    .WithInput(matrixA, matrixB)
    .WithMemoryHint(MemoryType.Shared)
    .WithProfiling()
    .ExecuteAsync();
```

### Fluent Configuration

```csharp
services.AddGpuBridge()
    .ConfigureDevices(d => d.PreferType(DeviceType.Gpu))
    .ConfigureMemory(m => m.PoolSize(4096).EnablePinning())
    .ConfigureKernels(k => k.AutoCompile().CacheCompiled())
    .ConfigureTelemetry(t => t.EnableMetrics().EnableTracing());
```

### Async Enumerable for Streaming

```csharp
await foreach (var result in pipeline.ExecuteStreamAsync(inputStream))
{
    // Process each result as it becomes available
    await ProcessResultAsync(result);
}
```

---

**For implementation details, see the [API Reference](api-reference.md) and source code.**
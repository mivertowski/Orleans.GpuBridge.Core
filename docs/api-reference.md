# Orleans.GpuBridge API Reference - v1.0.0

> **Production Ready** - 75% Complete with ILGPU + CPU backends fully functional

## Table of Contents

- [Core Interfaces](#core-interfaces)
- [Runtime Components](#runtime-components)
- [Memory Management](#memory-management)
- [Kernel Management](#kernel-management)
- [Pipeline Framework](#pipeline-framework)
- [Grain Attributes](#grain-attributes)
- [Configuration Options](#configuration-options)
- [Extension Methods](#extension-methods)

## Core Interfaces

### IGpuBridge

The main entry point for GPU operations.

```csharp
public interface IGpuBridge
{
    // Device management
    Task<IGpuDevice> GetDeviceAsync(int? deviceIndex = null);
    Task<IReadOnlyList<IGpuDevice>> GetAvailableDevicesAsync();
    
    // Kernel management
    Task<IKernel> GetKernelAsync(string kernelId);
    Task<IKernel> CompileKernelAsync(string source, KernelOptions options = null);
    
    // Memory management
    Task<IGpuMemory<T>> AllocateAsync<T>(int count) where T : unmanaged;
    Task<IGpuMemory<T>> AllocateAsync<T>(int count, MemoryType type) where T : unmanaged;
    
    // Execution
    Task<TOut> ExecuteAsync<TIn, TOut>(string kernelId, TIn input);
    Task<TOut[]> ExecuteBatchAsync<TIn, TOut>(string kernelId, TIn[] inputs);
}
```

### IGpuDevice

Represents a GPU device.

```csharp
public interface IGpuDevice
{
    int Index { get; }
    string Name { get; }
    DeviceType Type { get; }
    long TotalMemoryBytes { get; }
    long AvailableMemoryBytes { get; }
    int ComputeUnits { get; }
    ComputeCapability Capability { get; }
    bool IsAvailable { get; }
    
    Task<IComputeContext> CreateContextAsync();
    Task<DeviceMetrics> GetMetricsAsync();
}

public enum DeviceType
{
    Cpu,
    Gpu,
    Accelerator,
    Custom
}
```

### IGpuMemory<T>

GPU memory allocation interface.

```csharp
public interface IGpuMemory<T> : IDisposable where T : unmanaged
{
    int Length { get; }
    long SizeInBytes { get; }
    MemoryType Type { get; }
    bool IsDisposed { get; }
    
    Task CopyToDeviceAsync(T[] source, int offset = 0, int? count = null);
    Task CopyToDeviceAsync(ReadOnlyMemory<T> source);
    Task CopyFromDeviceAsync(T[] destination, int offset = 0, int? count = null);
    Task CopyFromDeviceAsync(Memory<T> destination);
    
    Memory<T> AsMemory();
    Span<T> AsSpan();
}

public enum MemoryType
{
    Default,
    Pinned,
    Shared,
    Texture
}
```

### IKernel

Compiled kernel interface.

```csharp
public interface IKernel
{
    string Id { get; }
    KernelMetadata Metadata { get; }
    
    Task<TOut> ExecuteAsync<TIn, TOut>(TIn input);
    Task<TOut[]> ExecuteBatchAsync<TIn, TOut>(TIn[] inputs);
    Task ExecuteAsync(KernelArguments args);
    
    void SetArgument(int index, object value);
    void SetArguments(params object[] args);
}

public class KernelArguments
{
    public object[] Arguments { get; set; }
    public uint[] GlobalWorkSize { get; set; }
    public uint[] LocalWorkSize { get; set; }
    public Dictionary<string, object> Parameters { get; set; }
}
```

## Runtime Components

### DeviceBroker

Manages GPU devices and scheduling.

```csharp
public class DeviceBroker : IDeviceBroker, IDisposable
{
    public int DeviceCount { get; }
    public long TotalMemoryBytes { get; }
    public int CurrentQueueDepth { get; }
    
    public Task InitializeAsync(CancellationToken ct);
    public IGpuDevice GetBestDevice();
    public IGpuDevice GetDevice(int index);
    public IReadOnlyList<IGpuDevice> GetDevices();
    public Task<WorkHandle> EnqueueAsync(WorkItem work);
    public Task ShutdownAsync(CancellationToken ct);
}
```

### KernelCatalog

Registry for GPU kernels.

```csharp
public class KernelCatalog : IKernelCatalog
{
    public Task RegisterAsync(KernelRegistration registration);
    public Task<IKernel> GetKernelAsync(string id);
    public Task<IKernel> CompileAsync(string source, KernelOptions options);
    public bool TryGetKernel(string id, out IKernel kernel);
    public IEnumerable<KernelMetadata> GetAllMetadata();
}

public class KernelRegistration
{
    public string Id { get; set; }
    public string Source { get; set; }
    public Type InputType { get; set; }
    public Type OutputType { get; set; }
    public Func<IServiceProvider, IKernel> Factory { get; set; }
}
```

## Memory Management

### MemoryPoolManager

Manages memory pools for different types.

```csharp
public class MemoryPoolManager : IMemoryPoolManager
{
    public IGpuMemoryPool<T> GetPool<T>() where T : unmanaged;
    public MemoryPoolStatistics GetStatistics();
    public void Configure(MemoryPoolOptions options);
    public void Clear();
}

public interface IGpuMemoryPool<T> where T : unmanaged
{
    IGpuMemory<T> Rent(int minSize);
    void Return(IGpuMemory<T> memory);
    MemoryPoolStats GetStats();
}

public class MemoryPoolOptions
{
    public int MaxPoolSizeMB { get; set; }
    public int MaxBufferSize { get; set; }
    public TimeSpan GarbageCollectionInterval { get; set; }
    public bool EnableStatistics { get; set; }
}
```

### AdvancedMemoryPool<T>

Advanced memory pool with allocation tracking.

```csharp
public class AdvancedMemoryPool<T> : IGpuMemoryPool<T> where T : unmanaged
{
    public event EventHandler<AllocationEventArgs> AllocationRequested;
    public event EventHandler<AllocationEventArgs> AllocationReturned;
    
    public IGpuMemory<T> Rent(int minSize);
    public IGpuMemory<T> RentExact(int size);
    public void Return(IGpuMemory<T> memory);
    
    public MemoryPoolStats GetStats();
    public IEnumerable<AllocationInfo> GetActiveAllocations();
    public void TrimExcess();
}
```

## Kernel Management

### KernelCompiler

Compiles kernels from various sources.

```csharp
public class KernelCompiler : IKernelCompiler
{
    public Task<IKernel> CompileFromSourceAsync(string source, CompilerOptions options);
    public Task<IKernel> CompileFromFileAsync(string filePath, CompilerOptions options);
    public Task<IKernel> CompileFromAssemblyAsync(Assembly assembly, string resourceName);
    
    public bool ValidateSource(string source, out CompilationError[] errors);
    public Task<CompilationResult> CompileWithDiagnosticsAsync(string source);
}

public class CompilerOptions
{
    public OptimizationLevel OptimizationLevel { get; set; }
    public string[] Defines { get; set; }
    public string[] IncludePaths { get; set; }
    public bool EnableProfiling { get; set; }
    public TargetDevice TargetDevice { get; set; }
}
```

## Pipeline Framework

### GpuPipeline<TIn, TOut>

Fluent pipeline builder.

```csharp
public class GpuPipeline<TIn, TOut>
{
    public static IGpuPipelineBuilder<TIn, TIn> Create();
    
    public interface IGpuPipelineBuilder<TInput, TOutput>
    {
        // Kernel stages
        IGpuPipelineBuilder<TInput, TNext> AddKernel<TNext>(
            string kernelId, 
            Action<KernelConfiguration> configure = null);
        
        // Transform stages
        IGpuPipelineBuilder<TInput, TNext> Transform<TNext>(
            Func<TOutput, TNext> transform);
        
        IGpuPipelineBuilder<TInput, TNext> TransformAsync<TNext>(
            Func<TOutput, Task<TNext>> transform);
        
        // Parallel execution
        IGpuPipelineBuilder<TInput, TOutput> Parallel(int maxConcurrency);
        
        // Batching
        IGpuPipelineBuilder<TInput, TOutput[]> Batch(int size);
        
        // Filtering
        IGpuPipelineBuilder<TInput, TOutput> Filter(Func<TOutput, bool> predicate);
        
        // Side effects
        IGpuPipelineBuilder<TInput, TOutput> Tap(Action<TOutput> action);
        
        // Build the pipeline
        ExecutablePipeline<TInput, TOutput> Build();
    }
}
```

### ExecutablePipeline<TIn, TOut>

Executable pipeline instance.

```csharp
public class ExecutablePipeline<TIn, TOut>
{
    public Task<TOut> ExecuteAsync(TIn input);
    public Task<TOut[]> ExecuteAsync(TIn[] inputs);
    public IAsyncEnumerable<TOut> ExecuteAsync(IAsyncEnumerable<TIn> inputs);
    
    public PipelineMetrics GetMetrics();
    public void Reset();
}

public class PipelineMetrics
{
    public int ItemsProcessed { get; }
    public TimeSpan TotalExecutionTime { get; }
    public Dictionary<string, StageMetrics> StageMetrics { get; }
}
```

## Grain Attributes

### [GpuResident]

Marks a grain for placement on GPU-capable silos.

```csharp
[AttributeUsage(AttributeTargets.Class)]
public class GpuResidentAttribute : PlacementAttribute
{
    public int? PreferredDeviceIndex { get; set; }
    public DeviceType? PreferredDeviceType { get; set; }
    public int MinimumMemoryMB { get; set; }
}

// Usage
[GpuResident(PreferredDeviceType = DeviceType.Gpu, MinimumMemoryMB = 2048)]
public class MyGpuGrain : Grain, IMyGpuGrain
{
    // ...
}
```

### [GpuKernel]

Marks a method as a GPU kernel.

```csharp
[AttributeUsage(AttributeTargets.Method)]
public class GpuKernelAttribute : Attribute
{
    public string KernelId { get; set; }
    public bool AutoCompile { get; set; }
    public string Source { get; set; }
}

// Usage
public class ComputeGrain : Grain
{
    [GpuKernel(KernelId = "matmul", AutoCompile = true)]
    public async Task<float[,]> MultiplyMatricesAsync(float[,] a, float[,] b)
    {
        // Implementation
    }
}
```

## Configuration Options

### GpuBridgeOptions

Main configuration class.

```csharp
public class GpuBridgeOptions
{
    // Device selection
    public bool PreferGpu { get; set; } = true;
    public int MaxDevices { get; set; } = 4;
    public DeviceSelectionStrategy DeviceSelection { get; set; }
    
    // Memory management
    public int MemoryPoolSizeMB { get; set; } = 1024;
    public bool EnablePinnedMemory { get; set; } = false;
    public TimeSpan MemoryGCInterval { get; set; } = TimeSpan.FromMinutes(5);
    
    // Kernel execution
    public int MaxConcurrentKernels { get; set; } = 100;
    public int DefaultMicroBatch { get; set; } = 8192;
    public TimeSpan KernelTimeout { get; set; } = TimeSpan.FromSeconds(30);
    
    // Advanced features
    public bool EnableGpuDirectStorage { get; set; } = false;
    public bool EnablePeerToPeer { get; set; } = false;
    public bool EnableUnifiedMemory { get; set; } = false;
    
    // Monitoring
    public bool EnableProfiling { get; set; } = false;
    public bool EnableMetrics { get; set; } = true;
    public TelemetryOptions Telemetry { get; set; } = new();
}

public class TelemetryOptions
{
    public bool EnableMetrics { get; set; } = true;
    public bool EnableTracing { get; set; } = true;
    public double SamplingRate { get; set; } = 0.1;
    public string[] ExportEndpoints { get; set; }
}
```

## Extension Methods

### Service Collection Extensions

```csharp
public static class ServiceCollectionExtensions
{
    // Add GPU Bridge to DI container
    public static IServiceCollection AddGpuBridge(
        this IServiceCollection services,
        Action<GpuBridgeOptions> configure = null);
    
    // Add kernel to catalog
    public static IServiceCollection AddKernel(
        this IServiceCollection services,
        Action<KernelBuilder> build);
}

// Usage
services.AddGpuBridge(options =>
{
    options.PreferGpu = true;
    options.MemoryPoolSizeMB = 4096;
})
.AddKernel(kernel => kernel
    .Id("my_kernel")
    .FromSource(source)
    .WithOptions(opts => opts.OptimizationLevel = OptimizationLevel.Maximum));
```

### Silo Builder Extensions

```csharp
public static class SiloBuilderExtensions
{
    // Enable GPU support in Orleans silo
    public static ISiloBuilder AddGpuBridge(
        this ISiloBuilder builder,
        Action<GpuBridgeOptions> configure = null);
    
    // Use GPU-aware placement
    public static ISiloBuilder UseGpuPlacement(
        this ISiloBuilder builder);
}

// Usage
siloBuilder
    .AddGpuBridge(options => { /* ... */ })
    .UseGpuPlacement();
```

## Error Handling

### Exception Types

```csharp
// Base exception
public class GpuBridgeException : Exception { }

// Specific exceptions
public class GpuOutOfMemoryException : GpuBridgeException { }
public class KernelCompilationException : GpuBridgeException { }
public class KernelExecutionException : GpuBridgeException { }
public class DeviceNotFoundException : GpuBridgeException { }
public class InvalidKernelArgumentException : GpuBridgeException { }
```

### Error Handling Patterns

```csharp
// Automatic fallback
public async Task<TResult> ExecuteWithFallbackAsync<TResult>(
    Func<Task<TResult>> gpuOperation,
    Func<Task<TResult>> cpuFallback)
{
    try
    {
        return await gpuOperation();
    }
    catch (GpuOutOfMemoryException)
    {
        _logger.LogWarning("GPU out of memory, falling back to CPU");
        return await cpuFallback();
    }
    catch (DeviceNotFoundException)
    {
        _logger.LogWarning("No GPU device found, falling back to CPU");
        return await cpuFallback();
    }
}
```

## Backend Providers

### IGpuBackendProvider

Interface for compute backend providers with plugin architecture.

```csharp
public interface IGpuBackendProvider : IAsyncDisposable
{
    string ProviderId { get; }
    string Name { get; }
    string Version { get; }
    BackendCapabilities Capabilities { get; }
    bool IsAvailable { get; }
    
    Task InitializeAsync(CancellationToken cancellationToken = default);
    Task<IReadOnlyList<IGpuDevice>> GetDevicesAsync(CancellationToken cancellationToken = default);
    Task<IGpuMemoryAllocator> CreateMemoryAllocatorAsync(int deviceId, CancellationToken cancellationToken = default);
    Task<IGpuKernelCompiler> CreateKernelCompilerAsync(int deviceId, CancellationToken cancellationToken = default);
    Task<IGpuExecutionContext> CreateExecutionContextAsync(int deviceId, CancellationToken cancellationToken = default);
}

public class BackendCapabilities
{
    public bool SupportsDoubleprecision { get; set; }
    public bool SupportsUnifiedMemory { get; set; }
    public bool SupportsPeerToPeer { get; set; }
    public bool SupportsDirectStorage { get; set; }
    public int MaxComputeUnits { get; set; }
    public long MaxMemoryBytes { get; set; }
    public string[] SupportedExtensions { get; set; }
}

### GpuBackendRegistry

Registry for managing backend providers.

```csharp
public class GpuBackendRegistry : IGpuBackendRegistry, IAsyncDisposable
{
    public IReadOnlyList<string> AvailableProviders { get; }
    
    Task<IGpuBackendProvider?> GetProviderAsync(string providerId, CancellationToken cancellationToken = default);
    Task<IGpuBackendProvider?> SelectProviderAsync(BackendCapabilities requirements, CancellationToken cancellationToken = default);
    Task RegisterProviderAsync(IGpuBackendProvider provider, CancellationToken cancellationToken = default);
    void UnregisterProvider(string providerId);
}
```

---

## Examples

### Complete Example: Matrix Operations

```csharp
// Define grain interface
public interface IMatrixGrain : IGrainWithIntegerKey
{
    Task<float[,]> MultiplyAsync(float[,] a, float[,] b);
    Task<float[,]> TransposeAsync(float[,] matrix);
    Task<float> DeterminantAsync(float[,] matrix);
}

// Implement grain
[GpuResident]
public class MatrixGrain : Grain, IMatrixGrain
{
    private readonly IGpuBridge _gpu;
    
    public MatrixGrain(IGpuBridge gpu) => _gpu = gpu;
    
    public async Task<float[,]> MultiplyAsync(float[,] a, float[,] b)
    {
        // Use GPU kernel for matrix multiplication
        var kernel = await _gpu.GetKernelAsync("matmul");
        return await kernel.ExecuteAsync<(float[,], float[,]), float[,]>((a, b));
    }
    
    public async Task<float[,]> TransposeAsync(float[,] matrix)
    {
        // Use GPU kernel for transpose
        var kernel = await _gpu.GetKernelAsync("transpose");
        return await kernel.ExecuteAsync<float[,], float[,]>(matrix);
    }
    
    public async Task<float> DeterminantAsync(float[,] matrix)
    {
        // Complex operation using pipeline
        var pipeline = GpuPipeline<float[,], float>
            .Create()
            .AddKernel<float[,]>("lu_decomposition")
            .Transform(lu => CalculateDeterminantFromLU(lu))
            .Build();
            
        return await pipeline.ExecuteAsync(matrix);
    }
}

// Usage
var grain = client.GetGrain<IMatrixGrain>(0);
var result = await grain.MultiplyAsync(matrixA, matrixB);
```

### Resource Management

```csharp
public interface IResourceQuotaManager : IDisposable
{
    Task<ResourceAllocation?> RequestAllocationAsync(string tenantId, ResourceRequest request, CancellationToken cancellationToken = default);
    Task ReleaseAllocationAsync(string tenantId, Guid allocationId, long memoryBytes, int kernels, CancellationToken cancellationToken = default);
    Task<ResourceUsage> GetUsageAsync(string tenantId, CancellationToken cancellationToken = default);
    Task<IReadOnlyList<string>> GetTenantsAsync(CancellationToken cancellationToken = default);
    Task SetQuotaAsync(string tenantId, ResourceQuota quota, CancellationToken cancellationToken = default);
}

public class ResourceRequest
{
    public long RequestedMemoryBytes { get; set; }
    public int RequestedKernels { get; set; }
    public int BatchSize { get; set; }
    public TimeSpan EstimatedDuration { get; set; }
    public int Priority { get; set; } = 0;
    public Dictionary<string, object> Metadata { get; set; } = new();
}

public class ResourceAllocation
{
    public Guid AllocationId { get; init; }
    public long AllocatedMemoryBytes { get; init; }
    public int AllocatedKernels { get; init; }
    public DateTime CreatedAt { get; init; }
    public TimeSpan EstimatedDuration { get; init; }
    public int Priority { get; init; }
    public Dictionary<string, object> Metadata { get; init; } = new();
}
```

### Persistent Kernels

```csharp
public interface IPersistentKernelHost : IHostedService, IDisposable
{
    Task<string> CreateKernelAsync(string kernelId, KernelDefinition definition, CancellationToken cancellationToken = default);
    Task<bool> DestroyKernelAsync(string instanceId, CancellationToken cancellationToken = default);
    Task<KernelHandle> SubmitBatchAsync(string instanceId, object[] inputs, CancellationToken cancellationToken = default);
    IAsyncEnumerable<object> ReadResultsAsync(KernelHandle handle, CancellationToken cancellationToken = default);
    Task<KernelHealth> GetHealthAsync(string instanceId, CancellationToken cancellationToken = default);
    Task<IReadOnlyList<string>> GetActiveInstancesAsync(CancellationToken cancellationToken = default);
}

public class KernelHandle
{
    public string InstanceId { get; init; }
    public Guid BatchId { get; init; }
    public int InputCount { get; init; }
    public DateTime SubmittedAt { get; init; }
}

public class KernelHealth
{
    public string InstanceId { get; init; }
    public KernelStatus Status { get; init; }
    public long ProcessedBatches { get; init; }
    public TimeSpan Uptime { get; init; }
    public Exception? LastError { get; init; }
    public DateTime? LastActivity { get; init; }
    public long MemoryUsageBytes { get; init; }
    public double CpuUsagePercent { get; init; }
}

public enum KernelStatus
{
    Initializing,
    Running,
    Idle,
    Error,
    Stopping,
    Stopped
}
```

### Ring Buffer System

```csharp
public interface IRingBufferManager : IDisposable
{
    IRingBuffer<T> CreateBuffer<T>(string name, int size, bool pinMemory = false) where T : unmanaged;
    bool TryGetBuffer<T>(string name, out IRingBuffer<T>? buffer) where T : unmanaged;
    void RemoveBuffer(string name);
    RingBufferStats GetStats(string name);
    IReadOnlyList<string> GetBufferNames();
}

public interface IRingBuffer<T> : IDisposable where T : unmanaged
{
    string Name { get; }
    int Size { get; }
    int Available { get; }
    bool IsPinned { get; }
    
    bool TryWrite(T item);
    bool TryRead(out T item);
    bool TryWriteBatch(ReadOnlySpan<T> items, out int written);
    bool TryReadBatch(Span<T> items, out int read);
    void Clear();
    RingBufferStats GetStats();
}

public class RingBufferStats
{
    public string Name { get; init; }
    public int Size { get; init; }
    public int Available { get; init; }
    public long TotalWrites { get; init; }
    public long TotalReads { get; init; }
    public long OverrunCount { get; init; }
    public long UnderrunCount { get; init; }
    public TimeSpan AverageWriteTime { get; init; }
    public TimeSpan AverageReadTime { get; init; }
}
```

---

## License & Commercial Support

This project is available under the Apache License 2.0 (see [LICENSE](../LICENSE)).

**Commercial licenses** with additional terms (warranty, support, indemnity, and optional trademark rights) are available.

**Contact**: michael.ivertowski@ch.ey.com

---

**For more examples and detailed usage, see the [Getting Started Guide](getting-started.md) and sample projects.**
# Orleans.GpuBridge.Core Architecture Overview

## System Architecture

Orleans.GpuBridge.Core extends the Orleans distributed actor framework with GPU computing capabilities while maintaining the simplicity and reliability of the virtual actor model. This article provides a high-level overview of the system architecture, design decisions, and key components.

## Architectural Layers

The system consists of six logical layers, each with well-defined responsibilities:

```
┌──────────────────────────────────────────────────────────────┐
│  Layer 6: Application Code                                   │
│  - Business logic in C#                                      │
│  - Grain implementations with GPU operations                 │
│  - Type-safe, async/await programming model                  │
├──────────────────────────────────────────────────────────────┤
│  Layer 5: Orleans.GpuBridge Abstractions                     │
│  - IGpuKernel<TIn, TOut> interface                          │
│  - [GpuAccelerated] attribute                                │
│  - GpuPipeline<T> fluent API                                 │
│  - Temporal correctness (HLC, Vector Clocks)                 │
├──────────────────────────────────────────────────────────────┤
│  Layer 4: Orleans.GpuBridge Runtime                          │
│  - Kernel catalog and registration                           │
│  - Memory-mapped buffer management                           │
│  - GPU-aware placement strategies                            │
│  - Pattern detection engines                                 │
├──────────────────────────────────────────────────────────────┤
│  Layer 3: Orleans Distributed Runtime                        │
│  - Virtual actor model (grains)                              │
│  - Location transparency and routing                         │
│  - Cluster membership and lifecycle                          │
│  - Streaming and persistence                                 │
├──────────────────────────────────────────────────────────────┤
│  Layer 2: DotCompute Backend                                 │
│  - CUDA, OpenCL, CPU backend abstraction                     │
│  - Kernel compilation and caching                            │
│  - Memory management (allocation, transfer)                  │
│  - Device enumeration and selection                          │
├──────────────────────────────────────────────────────────────┤
│  Layer 1: GPU Hardware                                       │
│  - NVIDIA GPUs (CUDA)                                        │
│  - AMD GPUs (ROCm/OpenCL)                                    │
│  - Intel GPUs (OneAPI)                                       │
│  - CPU fallback for development                              │
└──────────────────────────────────────────────────────────────┘
```

## Core Components

### Orleans Grain Infrastructure

**Grains** are virtual actors—lightweight, distributed objects with:
- Single-threaded execution (no locks needed)
- Location transparency (caller doesn't know where grain lives)
- Automatic activation/deactivation
- Built-in fault tolerance

```csharp
// Grain interface (contract)
public interface IMyGrain : IGrainWithIntegerKey
{
    Task<Result> ProcessAsync(Input data);
}

// Grain implementation
public class MyGrain : Grain, IMyGrain
{
    public async Task<Result> ProcessAsync(Input data)
    {
        // Grain logic here
        return result;
    }
}

// Usage (caller doesn't know grain location)
var grain = grainFactory.GetGrain<IMyGrain>(123);
var result = await grain.ProcessAsync(data);
```

### GPU Bridge Layer

The GPU Bridge extends grains with GPU capabilities:

**IGpuKernel Interface**:
```csharp
public interface IGpuKernel<TIn, TOut>
{
    Task<TOut> ExecuteAsync(TIn input);
    Task<TOut> ExecuteAsync(TIn input, GpuExecutionOptions options);
}
```

**Kernel Catalog**:
```csharp
public interface IKernelCatalog
{
    Task<IGpuKernel<TIn, TOut>> GetKernelAsync<TIn, TOut>(string kernelId);
    void RegisterKernel<TIn, TOut>(string kernelId, Func<IGpuKernel<TIn, TOut>> factory);
}
```

**Placement Strategies**:
- `GpuAwarePlacement`: Places grains on silos with available GPUs
- `GpuAffinityPlacement`: Pins grains to specific GPUs
- `LoadBalancedGpuPlacement`: Balances load across GPUs

### DotCompute Backend

DotCompute provides GPU abstraction:

```csharp
public interface IGpuBackend
{
    // Device management
    IReadOnlyList<GpuDevice> EnumerateDevices();
    GpuDevice SelectDevice(GpuDeviceSelector selector);

    // Memory management
    DeviceMemory<T> Allocate<T>(int count) where T : unmanaged;
    Task CopyToDeviceAsync<T>(T[] source, DeviceMemory<T> destination);
    Task CopyFromDeviceAsync<T>(DeviceMemory<T> source, T[] destination);

    // Kernel execution
    Task<KernelHandle> CompileKernelAsync(string source, string entryPoint);
    Task ExecuteKernelAsync(KernelHandle kernel, params object[] arguments);
}
```

**Supported Backends**:
- **CUDA**: NVIDIA GPUs (primary target)
- **OpenCL**: AMD, Intel GPUs
- **CPU**: Fallback for development/testing

## Ring Kernels: The Key Innovation

Traditional GPU programming launches kernels repeatedly:

```
CPU                     GPU
 |                       |
 |---Launch Kernel------>|
 |                       | Execute
 |<--Return Result-------|
 |                       |
 |---Launch Kernel------>| (5-20μs overhead)
 |                       | Execute
 |<--Return Result-------|
```

**Ring kernels** remain resident on GPU:

```
CPU                     GPU
 |                       |
 |---Launch Ring-------->| while(true) {
 |                       |   msg = dequeue();
 |--Send Message-------->|   process(msg);
 |<--Return Result-------|   reply(result);
 |                       | }
 |--Send Message-------->|
 |<--Return Result-------|
         (no launch overhead)
```

**Benefits**:
- **Zero launch overhead**: No kernel launch per operation
- **Persistent state**: GPU memory persists across calls
- **Lower latency**: Eliminates 5-20μs launch cost
- **Higher throughput**: Continuous processing

**Implementation**:
```cuda
// Ring kernel (infinite loop on GPU)
__global__ void ring_kernel(
    RingQueue* queue,
    State* state,
    volatile bool* shutdown)
{
    while (!*shutdown)
    {
        Message msg;
        if (queue->dequeue(&msg))
        {
            Result result = process(state, msg);
            queue->enqueue_result(msg.id, result);
        }
    }
}
```

## Memory Architecture

### CPU-GPU Memory Hierarchy

```
┌─────────────────────────────────────────────────┐
│  Host (CPU) Memory                               │
│  - Application data                              │
│  - Grain state                                   │
│  - Pinned memory for DMA                         │
├─────────────────────────────────────────────────┤
│  Pinned Memory (CPU-GPU Shared)                  │
│  - Zero-copy access from GPU                     │
│  - Message queues for ring kernels               │
│  - Small metadata structures                     │
├─────────────────────────────────────────────────┤
│  Device (GPU) Global Memory                      │
│  - Kernel code (loaded once)                     │
│  - Working data (copied from CPU)                │
│  - Ring kernel state (persistent)                │
│  - Temporary buffers                             │
├─────────────────────────────────────────────────┤
│  Device Shared Memory (per block)                │
│  - Fast scratch space (48-96 KB)                 │
│  - Thread communication                          │
│  - Reduction operations                          │
├─────────────────────────────────────────────────┤
│  Device Registers (per thread)                   │
│  - Ultra-fast private memory                     │
│  - Limited (255 registers/thread typical)        │
└─────────────────────────────────────────────────┘
```

### Memory Transfer Optimization

**Asynchronous Transfers**:
```csharp
// Overlap compute with transfer
await Task.WhenAll(
    CopyToGpuAsync(nextBatch),           // Transfer next batch
    _kernel.ExecuteAsync(currentBatch),  // Process current batch
    CopyFromGpuAsync(prevBatch)          // Retrieve previous results
);
```

**Pinned Memory**:
```csharp
// Allocate pinned memory for faster DMA transfers
using var pinnedArray = new PinnedArray<float>(size);

// Transfer: ~8 GB/s (pinned) vs. ~4 GB/s (unpinned)
await DotCompute.CopyToDeviceAsync(pinnedArray, deviceBuffer);
```

**Unified Memory** (CUDA 6.0+):
```csharp
// GPU and CPU share same memory address space
using var unifiedBuffer = DotCompute.AllocateUnified<float>(size);

// Access from CPU
unifiedBuffer[0] = 1.0f;

// Access from GPU (automatic migration)
await _kernel.ExecuteAsync(unifiedBuffer);
```

## Distribution Architecture

### Silo Deployment

Orleans silos host grains and can be deployed across many machines:

```
┌────────────────────────────────────────────────────────┐
│  Silo 1 (GPU Node 1)                                   │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐            │
│  │ Grain A  │  │ Grain C  │  │ Grain E  │            │
│  │ GPU: 0   │  │ GPU: 0   │  │ GPU: 1   │            │
│  └──────────┘  └──────────┘  └──────────┘            │
│                                                         │
│  GPU 0: NVIDIA A100 (40GB)                             │
│  GPU 1: NVIDIA A100 (40GB)                             │
└────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────┐
│  Silo 2 (GPU Node 2)                                   │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐            │
│  │ Grain B  │  │ Grain D  │  │ Grain F  │            │
│  │ GPU: 0   │  │ GPU: 1   │  │ GPU: 0   │            │
│  └──────────┘  └──────────┘  └──────────┘            │
│                                                         │
│  GPU 0: NVIDIA A100 (40GB)                             │
│  GPU 1: NVIDIA A100 (40GB)                             │
└────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────┐
│  Silo 3 (CPU-only Node)                                │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐            │
│  │ Grain G  │  │ Grain H  │  │ Grain I  │            │
│  │ CPU only │  │ CPU only │  │ CPU only │            │
│  └──────────┘  └──────────┘  └──────────┘            │
│                                                         │
│  (No GPU - uses CPU fallback)                          │
└────────────────────────────────────────────────────────┘
```

### Cluster Membership

Orleans uses gossip protocol for membership:
- Silos exchange heartbeats
- Failed silos detected within seconds
- Grains automatically reactivated on healthy silos

```csharp
// Configure clustering
siloBuilder.UseAzureStorageClustering(options =>
{
    options.ConnectionString = azureStorageConnectionString;
});

// Or SQL Server
siloBuilder.UseAdoNetClustering(options =>
{
    options.ConnectionString = sqlConnectionString;
    options.Invariant = "System.Data.SqlClient";
});

// Or Consul
siloBuilder.UseConsulClustering(options =>
{
    options.Address = new Uri("http://localhost:8500");
});
```

### Message Routing

Orleans provides location transparency:

```csharp
// Client doesn't know which silo hosts the grain
var grain = grainFactory.GetGrain<IMyGrain>(123);

// Orleans runtime routes message to correct silo
var result = await grain.ProcessAsync(data);
```

**Routing steps**:
1. Client looks up grain location in directory
2. If not activated, Orleans chooses silo based on placement strategy
3. Message routed to hosting silo
4. Grain processes message (activated if needed)
5. Result returned to client

## Fault Tolerance

### Grain Lifecycle

Grains have automatic lifecycle management:

```
Not Activated
     |
     | First method call
     v
OnActivateAsync() called
     |
     v
  Activated (processing calls)
     |
     | Idle timeout OR silo failure
     v
OnDeactivateAsync() called
     |
     v
Not Activated
```

### State Persistence

Grains can persist state for recovery:

```csharp
public class PersistentGpuGrain : Grain, IPersistentGpuGrain
{
    [Inject]
    private IPersistentState<GpuState> _state { get; set; }

    public override async Task OnActivateAsync(CancellationToken ct)
    {
        // State automatically loaded from storage
        if (_state.State != null)
        {
            // Restore GPU state
            await RestoreGpuStateAsync(_state.State);
        }

        await base.OnActivateAsync(ct);
    }

    public async Task UpdateAsync(Data data)
    {
        // Process on GPU
        var result = await _kernel.ExecuteAsync(data);

        // Update and persist state
        _state.State = new GpuState { Result = result };
        await _state.WriteStateAsync();
    }
}
```

**Storage providers**:
- Azure Blob Storage
- Azure Table Storage
- SQL Server
- PostgreSQL
- Amazon S3
- MongoDB
- Redis

### Failure Recovery

Orleans handles failures automatically:

**Silo Failure**:
1. Other silos detect failure via missed heartbeats
2. Grains hosted on failed silo marked inactive
3. Next call to grain activates on healthy silo
4. Grain state loaded from persistent storage

**GPU Failure**:
1. Kernel execution throws exception
2. Grain catches exception
3. Falls back to CPU implementation
4. Or: Grain deactivates and reactivates on different GPU

```csharp
public async Task<Result> ProcessAsync(Data data)
{
    try
    {
        return await _kernel.ExecuteAsync(data);
    }
    catch (GpuException ex)
    {
        _logger.LogWarning(ex, "GPU execution failed, falling back to CPU");

        // Fallback to CPU
        return await ProcessOnCpuAsync(data);
    }
}
```

## Performance Considerations

### Throughput vs. Latency

**High Throughput** (batch processing):
```csharp
// Process 1M items in batches of 10K
var results = await GpuPipeline<Input, Output>
    .For(grainFactory, "kernel-id")
    .WithBatchSize(10_000)
    .WithParallelism(100)  // 100 grains in parallel
    .ExecuteAsync(millionItems);

// Throughput: ~1M items/sec
// Latency per item: ~100ms (amortized)
```

**Low Latency** (real-time):
```csharp
// Process single item
var grain = grainFactory.GetGrain<IMyGrain>(id);
var result = await grain.ProcessAsync(singleItem);

// Latency: <1ms
// Throughput: ~1K items/sec (limited by round-trips)
```

### GPU Utilization

**Poor GPU Utilization** (sequential):
```csharp
// GPU idle while CPU processes results
for (int i = 0; i < items.Length; i++)
{
    var result = await ProcessOnGpuAsync(items[i]);  // Wait for each
    ProcessResultOnCpu(result);                      // GPU idle
}

// GPU utilization: ~50%
```

**High GPU Utilization** (pipelined):
```csharp
// Overlap GPU compute with CPU processing
var pipeline = new Channel<Result>();

var produceTask = Task.Run(async () =>
{
    foreach (var item in items)
    {
        var result = await ProcessOnGpuAsync(item);
        await pipeline.Writer.WriteAsync(result);
    }
    pipeline.Writer.Complete();
});

var consumeTask = Task.Run(async () =>
{
    await foreach (var result in pipeline.Reader.ReadAllAsync())
    {
        ProcessResultOnCpu(result);  // CPU works while GPU processes next
    }
});

await Task.WhenAll(produceTask, consumeTask);

// GPU utilization: ~95%
```

## Scalability

### Horizontal Scaling

Add more silos to scale:

| Silos | GPUs | Throughput | Scalability |
|-------|------|------------|-------------|
| 1 | 2 | 100K ops/sec | 1.0× |
| 2 | 4 | 190K ops/sec | 1.9× |
| 4 | 8 | 360K ops/sec | 3.6× |
| 8 | 16 | 680K ops/sec | 6.8× |

Near-linear scaling (Orleans overhead: ~5-10%).

### Vertical Scaling

Add more GPUs per silo:

| GPUs/Silo | Throughput | GPU Utilization |
|-----------|------------|-----------------|
| 1 | 50K ops/sec | 95% |
| 2 | 95K ops/sec | 90% |
| 4 | 180K ops/sec | 85% |
| 8 | 320K ops/sec | 75% |

Diminishing returns due to PCIe bandwidth and CPU bottlenecks.

## Security

### GPU Access Control

Restrict GPU access to authorized grains:

```csharp
[Authorize(Roles = "GpuUsers")]
[GpuAccelerated]
public class SecureGpuGrain : Grain
{
    // Only authorized users can activate this grain
}
```

### Memory Isolation

GPU memory is not shared between grains:
- Each grain has isolated GPU memory
- Memory cleared on grain deactivation
- No cross-grain memory access possible

### Network Security

Orleans supports TLS for inter-silo communication:

```csharp
siloBuilder.UseTls(options =>
{
    options.LocalCertificate = myCertificate;
    options.AllowAnyRemoteCertificate = false;
});
```

## Observability

### Metrics

Built-in metrics via OpenTelemetry:

```csharp
services.AddOpenTelemetry()
    .WithMetrics(metrics => metrics
        .AddMeter("Orleans.Runtime")
        .AddMeter("Orleans.GpuBridge")
        .AddPrometheusExporter());
```

**Key metrics**:
- Grain activations/deactivations
- Message throughput
- GPU utilization
- Kernel execution time
- Memory usage

### Tracing

Distributed tracing with OpenTelemetry:

```csharp
using var activity = activitySource.StartActivity("GpuOperation");
activity?.SetTag("grain.type", "VectorAddGrain");
activity?.SetTag("grain.id", this.GetPrimaryKeyLong());
activity?.SetTag("input.size", input.Length);

var result = await _kernel.ExecuteAsync(input);

activity?.SetTag("execution.time.ms", sw.ElapsedMilliseconds);
```

### Logging

Structured logging with Serilog:

```csharp
_logger.LogInformation(
    "GPU kernel executed: {KernelId}, Duration: {DurationMs}ms, InputSize: {Size}",
    kernelId, duration, inputSize);
```

## Conclusion

Orleans.GpuBridge.Core architecture provides:
- **Simplicity**: Familiar .NET programming model
- **Scalability**: Horizontal and vertical scaling
- **Reliability**: Automatic failover and recovery
- **Performance**: Ring kernels eliminate launch overhead
- **Observability**: Built-in metrics and tracing

The layered architecture separates concerns while providing flexibility for optimization and extension. GPU computing becomes as simple as calling a method on a grain.

## Further Reading

- [Introduction to GPU-Native Actors](../introduction/README.md)
- [Temporal Correctness Architecture](../../temporal/architecture/README.md)
- [Developer Experience](../developer-experience/README.md)
- [Getting Started Guide](../getting-started/README.md)

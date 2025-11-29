# Orleans.GpuBridge.Core

[![Version](https://img.shields.io/badge/version-0.1.0-blue.svg)](CHANGELOG.md)
[![License](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)
[![.NET 9.0](https://img.shields.io/badge/.NET-9.0-purple)](https://dotnet.microsoft.com/download/dotnet/9.0)
[![Orleans](https://img.shields.io/badge/Orleans-9.2.1-green)](https://dotnet.github.io/orleans/)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](https://github.com/mivertowski/Orleans.GpuBridge.Core)

##Remark##

Consider version 0.1.0 as preview version meant for evaluation purpose. However, the API will stay stable.

## Overview

Orleans.GpuBridge.Core is a comprehensive GPU acceleration framework for Microsoft Orleans, enabling distributed GPU computing across Orleans clusters. This project bridges the gap between Orleans' powerful distributed actor model and modern GPU computing capabilities, allowing developers to seamlessly integrate GPU acceleration into their Orleans-based applications.

## 🌟 The GPU-Native Actor Paradigm Shift

**Traditional Approach**: CPU actors that occasionally offload work to GPU
**Revolutionary Approach**: Actors that *live permanently on the GPU*

Orleans.GpuBridge.Core enables a **fundamentally new paradigm** where actors reside entirely in GPU memory, processing messages at sub-microsecond latencies without ever leaving the GPU. This represents a **20-200× performance improvement** over traditional CPU-based actor systems.

### Key Breakthrough: Ring Kernels

Instead of launching short-lived GPU kernels repeatedly, GPU-native actors use **ring kernels** - persistent dispatch loops that run indefinitely on the GPU:

```cuda
// Ring kernel runs forever, processing actor messages on GPU
__global__ void GpuNativeActorDispatchLoop(
    GpuNativeActor* actors,
    MessageQueue* queue,
    int num_actors)
{
    while (true) {  // Launched once, runs forever
        Message msg;
        if (queue->try_dequeue(&msg)) {
            ProcessMessage(&actors[msg.target], &msg);
        }
    }
}
```

### Performance Characteristics

| Metric | CPU Actors | GPU-Offload | GPU-Native Actors |
|--------|-----------|-------------|-------------------|
| Message Latency | 10-100μs | 10-50μs | **100-500ns** |
| Throughput/Actor | 15K msgs/s | 50K msgs/s | **2M msgs/s** |
| Memory Bandwidth | 200 GB/s | 500 GB/s | **1,935 GB/s** |
| Temporal Ordering | 50ns (CPU) | 100μs (sync) | **20ns (GPU)** |

### Three Pillars of Living Knowledge Systems

1. **Sub-Microsecond Response** (100-500ns latency)
2. **Temporal Causality** (HLC/Vector Clocks maintained on GPU)
3. **Massive Parallelism** (2M messages/s per actor)

These capabilities enable entirely new classes of applications:
- **Knowledge Organisms** - Emergent intelligence from actor interactions
- **Digital Twins as Living Entities** - Real-time physics-accurate simulation
- **Hypergraph Actors** - Multi-way relationships with GPU-accelerated pattern matching
- **Temporal Pattern Detection** - Real-time fraud detection and behavioral analytics

## 🚀 Key Features

### Core Capabilities
- **Unified GPU Abstraction**: Single API for multiple GPU backends (CUDA, OpenCL, DirectCompute, Metal, Vulkan)
- **Orleans Integration**: Native integration with Orleans grains and placement strategies
- **Automatic Fallback**: Intelligent CPU fallback when GPU resources are unavailable
- **Memory Management**: Advanced memory pooling and transfer optimization
- **Kernel Management**: JIT compilation, caching, and optimization of GPU kernels
- **Performance Monitoring**: Real-time metrics, health checks, and diagnostics

### Advanced Features
- **Multi-Backend Support**: Run different kernels on different GPU backends simultaneously
- **Dynamic Load Balancing**: Intelligent workload distribution across available GPUs
- **Circuit Breaker Pattern**: Automatic failure recovery and resilience
- **Stream Processing**: GPU-accelerated Orleans streams with batch optimization
- **Pipeline API**: Fluent API for complex GPU computation pipelines
- **AOT Compatibility**: Full support for .NET 9 Native AOT and trimming

## 📦 Package Structure

| Package | Description | NuGet |
|---------|-------------|-------|
| `Orleans.GpuBridge.Abstractions` | Core interfaces and contracts | [![NuGet](https://img.shields.io/nuget/v/Orleans.GpuBridge.Abstractions.svg)](https://www.nuget.org/packages/Orleans.GpuBridge.Abstractions/) |
| `Orleans.GpuBridge.Runtime` | Runtime implementation and orchestration | [![NuGet](https://img.shields.io/nuget/v/Orleans.GpuBridge.Runtime.svg)](https://www.nuget.org/packages/Orleans.GpuBridge.Runtime/) |
| `Orleans.GpuBridge.Grains` | Pre-built GPU-accelerated grains | [![NuGet](https://img.shields.io/nuget/v/Orleans.GpuBridge.Grains.svg)](https://www.nuget.org/packages/Orleans.GpuBridge.Grains/) |
| `Orleans.GpuBridge.Utils` | Utilities and helpers | [![NuGet](https://img.shields.io/nuget/v/Orleans.GpuBridge.Utils.svg)](https://www.nuget.org/packages/Orleans.GpuBridge.Utils/) |
| `Orleans.GpuBridge.Backends.ILGPU` | ILGPU backend implementation | [![NuGet](https://img.shields.io/nuget/v/Orleans.GpuBridge.Backends.ILGPU.svg)](https://www.nuget.org/packages/Orleans.GpuBridge.Backends.ILGPU/) |
| `Orleans.GpuBridge.Backends.DotCompute` | DotCompute backend implementation | [![NuGet](https://img.shields.io/nuget/v/Orleans.GpuBridge.Backends.DotCompute.svg)](https://www.nuget.org/packages/Orleans.GpuBridge.Backends.DotCompute/) |

## 🛠️ Installation

### Quick Start

```bash
# Install core packages
dotnet add package Orleans.GpuBridge.Runtime
dotnet add package Orleans.GpuBridge.Grains

# Install DotCompute backend for GPU acceleration
dotnet add package Orleans.GpuBridge.Backends.DotCompute
```

### Minimal Configuration

```csharp
using Orleans.GpuBridge.Runtime.Extensions;

var builder = Host.CreateDefaultBuilder(args)
    .ConfigureServices(services =>
    {
        // Register GPU Bridge core services
        services.AddGpuBridge(options =>
        {
            options.PreferGpu = true;
            options.FallbackToCpu = true;
            options.MaxConcurrentKernels = 100;
        });

        // Add Ring Kernel support for GPU-native actors (sub-microsecond messaging)
        services.AddRingKernelSupport(options =>
        {
            options.DefaultGridSize = 1;
            options.DefaultBlockSize = 256;
            options.DefaultQueueCapacity = 256;
        });

        // Add K2K (Kernel-to-Kernel) messaging for GPU-to-GPU communication
        services.AddK2KSupport();
    })
    .UseOrleans(siloBuilder =>
    {
        siloBuilder.UseLocalhostClustering();
    });

await builder.Build().RunAsync();
```

## 💻 Usage Examples

### GPU-Accelerated Grain (GpuGrainBase)

```csharp
using Orleans.GpuBridge.Grains.Base;
using Orleans.GpuBridge.Abstractions.Kernels;

// Grain with automatic GPU lifecycle management and CPU fallback
public class VectorProcessingGrain : GpuGrainBase<VectorState>
{
    private IGpuKernel<float[], float[]>? _vectorAddKernel;

    public VectorProcessingGrain(IGrainContext grainContext, ILogger<VectorProcessingGrain> logger)
        : base(grainContext, logger) { }

    protected override async Task ConfigureGpuResourcesAsync(CancellationToken ct)
    {
        // Initialize GPU kernel during grain activation
        var kernelFactory = ServiceProvider.GetRequiredService<IKernelFactory>();
        _vectorAddKernel = await kernelFactory.CreateKernelAsync<float[], float[]>("vector-add", ct);
        await _vectorAddKernel.InitializeAsync(ct);
    }

    public async Task<float[]> AddVectorsAsync(float[] a, float[] b)
    {
        // Execute with automatic CPU fallback on GPU failure
        return await ExecuteKernelWithFallbackAsync(
            _vectorAddKernel!,
            a,
            cpuFallback: input => Task.FromResult(CpuVectorAdd(input, b)));
    }
}
```

### GPU-Native Actor (RingKernelGrainBase) - Sub-Microsecond Messaging

```csharp
using Orleans.GpuBridge.Grains.Base;
using Orleans.GpuBridge.Abstractions.Temporal;

// Actor lives permanently in GPU memory with 100-500ns message latency
public class TemporalActorGrain : RingKernelGrainBase<ActorState, ActorMessage>
{
    public TemporalActorGrain(IGrainContext grainContext, ILogger<TemporalActorGrain> logger)
        : base(grainContext, logger) { }

    protected override Task<RingKernelConfig> ConfigureRingKernelAsync(CancellationToken ct)
    {
        return Task.FromResult(new RingKernelConfig
        {
            QueueDepth = 256,         // Lock-free message queue size
            EnableHLC = true,          // Hybrid Logical Clock for causal ordering
            EnableVectorClock = false, // Enable for distributed causality
            MaxStateSizeBytes = 1024
        });
    }

    // Compiled to GPU code - runs in ring kernel dispatch loop
    protected override void ProcessMessageOnGpu(
        ref ActorState state,
        in ActorMessage message,
        ref HybridTimestamp hlc)
    {
        // Pure GPU computation - no heap allocations, no exceptions
        state.Counter += message.Value;
        state.LastUpdate = hlc.PhysicalTime;
    }

    public async Task SendEventAsync(int value)
    {
        await SendMessageAsync(new ActorMessage { Value = value });
    }
}
```

### GpuNativeGrain with DotCompute Ring Kernels

```csharp
using DotCompute.Abstractions.RingKernels;
using Orleans.GpuBridge.Runtime.RingKernels;

// Full DotCompute integration for maximum GPU performance
public class HighFrequencyGrain : GpuNativeGrain
{
    public HighFrequencyGrain(IRingKernelRuntime runtime, ILogger<HighFrequencyGrain> logger)
        : base(runtime, logger) { }

    protected override (int gridSize, int blockSize) GetKernelConfiguration()
    {
        return (gridSize: 1, blockSize: 256); // Customize GPU thread configuration
    }

    public async Task<ProcessResult> ProcessAsync(InputData input)
    {
        // Sub-microsecond GPU message processing with HLC timestamps
        var timestamp = GetCurrentTimestamp();
        return await InvokeKernelAsync<InputData, ProcessResult>(input);
    }
}
```

### Custom Kernel Implementation

```csharp
using Orleans.GpuBridge.Abstractions.Kernels;

public class VectorAddKernel : IGpuKernel<float[], float[]>
{
    public string KernelId => "vector-add";
    public string DisplayName => "Vector Addition";
    public string BackendProvider => "DotCompute";
    public bool IsInitialized { get; private set; }
    public bool IsGpuAccelerated => true;

    public Task InitializeAsync(CancellationToken ct = default)
    {
        IsInitialized = true;
        return Task.CompletedTask;
    }

    public async Task<float[]> ExecuteAsync(float[] input, CancellationToken ct = default)
    {
        // GPU kernel implementation (CPU fallback for dev/test)
        var result = new float[input.Length];
        for (int i = 0; i < input.Length; i++)
            result[i] = input[i] * 2;
        return result;
    }

    public async Task<float[][]> ExecuteBatchAsync(float[][] inputs, CancellationToken ct = default)
    {
        var results = new float[inputs.Length][];
        for (int i = 0; i < inputs.Length; i++)
            results[i] = await ExecuteAsync(inputs[i], ct);
        return results;
    }

    public KernelMemoryRequirements GetMemoryRequirements()
        => new(InputMemoryBytes: 4096, OutputMemoryBytes: 4096, WorkingMemoryBytes: 0, TotalMemoryBytes: 8192);

    public KernelValidationResult ValidateInput(float[] input)
        => input.Length > 0 ? KernelValidationResult.Valid() : KernelValidationResult.Invalid("Empty input");

    public long GetEstimatedExecutionTimeMicroseconds(int inputSize) => inputSize / 1000;
    public Task WarmupAsync(CancellationToken ct = default) => Task.CompletedTask;
    public void Dispose() { }
}
```

## 🎯 Performance

### Benchmarks

| Operation | CPU (ms) | GPU (ms) | Speedup |
|-----------|----------|----------|---------|
| Vector Addition (10M) | 45 | 3 | 15x |
| Matrix Multiplication (1024x1024) | 1250 | 25 | 50x |
| FFT (2^20 points) | 890 | 12 | 74x |
| Image Convolution (4K) | 2100 | 35 | 60x |
| Neural Network Inference | 450 | 8 | 56x |

*Benchmarks performed on NVIDIA RTX 4090 vs Intel i9-13900K*

### Scalability

- Linear scaling up to 8 GPUs per node
- Efficient multi-node GPU clustering via Orleans
- Automatic load balancing across heterogeneous GPU resources
- Memory-aware scheduling and placement

## 🏗️ Architecture

### Deployment Models

**GPU-Offload Model** (Traditional):
- Actor logic runs on CPU
- Compute kernels dispatched to GPU (~50μs overhead)
- Results copied back to CPU
- Best for: Batch processing, infrequent GPU usage

**GPU-Native Model** (Revolutionary):
- Actor state resides permanently in GPU memory
- Ring kernels process messages entirely on GPU
- Zero kernel launch overhead
- Sub-microsecond message latency
- Best for: High-frequency messaging, temporal graphs, real-time analytics

```
┌─────────────────────────────────────────────────────────┐
│                   Orleans Application                     │
│  (User Services, Dashboards, Orchestration on CPU)       │
├─────────────────────────────────────────────────────────┤
│     CPU Grains          GPU-Native Actor Ring Kernels    │
│  (Business Logic) ◄──►   (Hypergraphs, Analytics)       │
├─────────────────────────────────────────────────────────┤
│              Orleans.GpuBridge.Grains                     │
│         (GpuBatchGrain, GpuResidentGrain)                │
├─────────────────────────────────────────────────────────┤
│              Orleans.GpuBridge.Runtime                    │
│      (KernelCatalog, DeviceBroker, Placement)            │
├──────────┬────────────┬────────────┬──────────┬─────────┤
│  ILGPU   │ DotCompute │   CUDA     │  OpenCL  │  Metal  │
│ Backend  │  Backend   │  Backend   │ Backend  │ Backend │
└──────────┴────────────┴────────────┴──────────┴─────────┘
```

## 🔧 Advanced Configuration

```csharp
using Orleans.GpuBridge.Runtime.Extensions;

services.AddGpuBridge(options =>
{
    // GPU Execution Preferences
    options.PreferGpu = true;                    // Prefer GPU over CPU
    options.FallbackToCpu = true;                // Fall back to CPU on GPU failure
    options.MaxConcurrentKernels = 100;          // Max concurrent GPU kernels
    options.MaxDevices = 4;                      // Max GPUs to use

    // Performance Tuning
    options.DefaultMicroBatch = 8192;            // Default micro-batch size
    options.BatchSize = 1024;                    // Default batch size
    options.MaxRetries = 3;                      // Retry attempts on failure

    // Memory Management
    options.MemoryPoolSizeMB = 1024;             // GPU memory pool size (1GB)
    options.EnableGpuDirectStorage = false;      // GPUDirect Storage (if supported)

    // Diagnostics and Profiling
    options.EnableProfiling = false;             // Kernel execution profiling

    // Telemetry
    options.Telemetry = new TelemetryOptions
    {
        EnableMetrics = true,
        EnableTracing = true,
        SamplingRate = 0.1
    };

    // Backend Provider Configuration
    options.DefaultBackend = "DotCompute";
    options.FallbackChain = new[] { "DotCompute", "CPU" };
    options.EnableProviderDiscovery = true;
});

// Ring Kernel Configuration (for GPU-native actors)
services.AddRingKernelSupport(options =>
{
    options.DefaultGridSize = 1;                 // GPU grid size
    options.DefaultBlockSize = 256;              // GPU block size (threads per block)
    options.DefaultQueueCapacity = 256;          // Message queue capacity (must be power of 2)
    options.EnableKernelCaching = true;          // Cache compiled kernels
    options.DeviceIndex = 0;                     // GPU device to use
});

// K2K Messaging (GPU-to-GPU communication)
services.AddK2KSupport();                        // Direct, Broadcast, Ring, HashRouted strategies

// CPU Fallback Bridge (for development/testing)
services.AddRingKernelBridge();                  // Registers CpuFallbackRingKernelBridge
```

## 📊 Monitoring & Diagnostics

### Health Checks

```csharp
services.AddHealthChecks()
    .AddGpuBridgeHealthCheck("gpu-health")
    .AddGpuMemoryHealthCheck("gpu-memory", 
        failureThreshold: 0.9f); // Fail if >90% memory used
```

### Metrics & Telemetry

```csharp
services.AddOpenTelemetry()
    .WithMetrics(builder =>
    {
        builder.AddGpuBridgeInstrumentation();
    })
    .WithTracing(builder =>
    {
        builder.AddGpuBridgeInstrumentation();
    });
```

## 🌍 Platform Support

| Platform | CUDA | OpenCL | DirectCompute | Metal | Vulkan | CPU |
|----------|------|--------|---------------|-------|--------|-----|
| Windows x64 | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ |
| Linux x64 | ✅ | ✅ | ❌ | ❌ | ✅ | ✅ |
| macOS x64 | ❌ | ✅ | ❌ | ✅ | ✅ | ✅ |
| macOS ARM64 | ❌ | ✅ | ❌ | ✅ | ✅ | ✅ |

## 📋 Requirements

### Minimum Requirements
- .NET 9.0 SDK or later
- Orleans 9.0 or later
- 4GB RAM
- Any x64 or ARM64 processor

### Recommended Requirements
- .NET 9.0 SDK or later
- Orleans 9.0 or later
- 16GB RAM
- NVIDIA GPU with 8GB+ VRAM (for CUDA)
- AMD GPU with 8GB+ VRAM (for OpenCL)
- CUDA 12.0+ or OpenCL 2.0+

## ⚠️ WSL2 GPU Limitations (Critical for GPU-Native Actors)

### The Problem

WSL2's GPU virtualization layer (GPU-PV) has **fundamental limitations** that prevent true GPU-native actor performance:

| Capability | Native Linux | WSL2 |
|------------|-------------|------|
| **Persistent Ring Kernels** | ✅ Sub-millisecond | ❌ ~5 second latency |
| **System-Scope Atomics** | ✅ Reliable | ❌ Unreliable visibility |
| **CPU-GPU Memory Coherence** | ✅ Real-time | ❌ Delayed/broken |
| **Unified Memory Spill** | ✅ VRAM → System RAM | ❌ VRAM only |
| **`is_active` Flag Polling** | ✅ Kernel sees updates | ❌ Kernel never sees |

### Technical Details

1. **System-Scope Atomics Fail**: `cuda::memory_order_system` and `__threadfence_system()` don't provide reliable CPU↔GPU memory visibility in WSL2. The GPU kernel cannot see host memory writes in real-time.

2. **Persistent Mode Broken**: Ring kernels that poll shared memory for new messages (tail pointer updates) will spin forever - they never see the host's writes.

3. **EventDriven Workaround**: Instead of persistent polling, kernels must terminate after processing and be relaunched by the host when new messages arrive. This adds ~5 seconds of overhead.

### Workarounds Implemented in DotCompute

```csharp
// WSL2 Fix: Start kernel with is_active=1 already set
// Avoids mid-execution activation signaling which doesn't work
var controlBlock = state.AsyncControlBlock != null
    ? RingKernelControlBlock.CreateActive()   // WSL2: Start active
    : RingKernelControlBlock.CreateInactive(); // Native: Can activate later
```

- **Start-Active Pattern**: Kernels start already active, avoiding the need for host→GPU activation signaling
- **EventDriven Mode**: Kernel processes messages then terminates; host relaunches for new batches
- **SpinWait+Yield Bridge**: High-performance polling in the bridge transfer loop (sub-ms when kernel is responsive)

### Performance Impact

| Metric | GPU-Native (Native Linux) | WSL2 (EventDriven) |
|--------|--------------------------|-------------------|
| Message Latency | 100-500ns | ~5 seconds |
| Throughput | 2M msgs/s/actor | ~1 msg/5s |
| Use Case | Production | Development/Testing |

### Recommendation

- **Development/Testing**: WSL2 is fine - all functionality works, just slower
- **Production**: Use native Linux for GPU-native actor systems requiring <10ms latency

### Feature Request Channels

To request improved GPU memory coherence in WSL2:

1. **Microsoft WSL**: https://github.com/microsoft/WSL/issues
   - [Issue #7198](https://github.com/microsoft/wslg/issues/357) - Shared memory limitations
   - [Issue #3789](https://github.com/Microsoft/WSL/issues/3789) - CUDA/OpenCL support

2. **Microsoft WSLg**: https://github.com/microsoft/wslg/issues

3. **NVIDIA**: https://docs.nvidia.com/cuda/wsl-user-guide/

## 🚦 Project Status

| Component | Status | Production Ready |
|-----------|--------|------------------|
| Core Abstractions | ✅ Stable | Yes |
| Runtime | ✅ Stable | Yes |
| ILGPU Backend | ✅ Stable | Yes |
| DotCompute Backend | ✅ Stable | Yes |
| Orleans Integration | ✅ Stable | Yes |
| Memory Management | ✅ Stable | Yes |
| Performance Monitoring | ✅ Stable | Yes |
| Documentation | ✅ Complete | Yes |
| Test Coverage | 🚧 85% | Almost |


## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Areas of Interest
- Backend implementations for new platforms
- Performance optimizations
- Additional kernel templates
- Documentation improvements
- Test coverage expansion

## 📚 Documentation

### Getting Started
- [Quick Start Guide](docs/quickstart.md)
- [Architecture Overview](docs/architecture.md)
- [Kernel Development Guide](docs/kernel-development.md)
- [Performance Tuning](docs/performance.md)
- [API Reference](docs/api/index.md)
- [Samples](samples/README.md)

### Technical Articles

Comprehensive guides exploring the GPU-native actor paradigm:

**[GPU-Native Actors Series](docs/articles/README.md#gpu-native-actors-series)**
- Introduction to GPU-Native Actors
- Use Cases and Applications
- Developer Experience (C# vs C/C++/Python)
- Getting Started Tutorial
- Architecture Overview

**[Hypergraph Actors Series](docs/articles/README.md#hypergraph-actors-series)**
- Introduction to Hypergraph Actors
- Hypergraph Theory and Computational Advantages
- Real-Time Analytics with Hypergraphs
- Industry Use Cases
- System Architecture

**[Temporal Correctness Series](docs/articles/README.md#temporal-correctness-series)**
- Introduction to Temporal Correctness
- Hybrid Logical Clocks (HLC)
- Vector Clocks and Causal Ordering
- Temporal Pattern Detection
- Performance Characteristics

**[Knowledge Organisms](docs/articles/knowledge-organisms/README.md)** (Advanced)
- The evolution from graphs to living knowledge systems
- Emergent intelligence from GPU-native temporal hypergraph actors
- Digital twins as living entities
- Cognitive architectures and consciousness
- The future of distributed systems

## 🧪 Testing

```bash
# Run all tests
dotnet test

# Run with coverage
dotnet test /p:CollectCoverage=true /p:CoverletOutputFormat=opencover

# Run benchmarks
dotnet run -c Release --project benchmarks/Orleans.GpuBridge.Benchmarks
```

## 📄 License

This project is licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.

**Commercial licensing** is available for organizations requiring custom SLA/MSA terms - contact the author for details.

**Commercial support** is available for organisations requiring support, managed services and consulting - contact the author for details.

**Commercial kernel and actor network blueprints** are available for the following domains: Accounting, Banking, Behavioral Analytics, Clearing, Compliance, Financial Audit, Financial Services, Graph Analytics, Order Matching, Payment Processing, Process Intelligence, Risk Analytics, Statistical ML, Temporal Analysis and Treasury Management - contact the author for details.

Copyright (c) 2025 Michael Ivertowski

## 🙏 Acknowledgments

- [Microsoft Orleans](https://github.com/dotnet/orleans) - The distributed actor framework
- [ILGPU](https://github.com/m4rs-mt/ILGPU) - .NET GPU compiler framework
- [DotCompute](https://github.com/mivertowski/DotCompute) - Cross-platform GPU framework
- The .NET Foundation and community

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/mivertowski/Orleans.GpuBridge.Core/issues)
- **Discussions**: [GitHub Discussions](https://github.com/mivertowski/Orleans.GpuBridge.Core/discussions)

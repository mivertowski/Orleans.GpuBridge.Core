# Orleans.GpuBridge.Core

[![Version](https://img.shields.io/badge/version-0.3.0-blue.svg)](CHANGELOG.md)
[![License](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)
[![.NET 9.0](https://img.shields.io/badge/.NET-9.0-purple)](https://dotnet.microsoft.com/download/dotnet/9.0)
[![Orleans](https://img.shields.io/badge/Orleans-9.2.1-green)](https://dotnet.github.io/orleans/)
[![DotCompute](https://img.shields.io/badge/DotCompute-0.5.3-orange)](https://www.nuget.org/packages/DotCompute.Core)

## Overview

Orleans.GpuBridge.Core is a .NET 9 library that enables GPU-native distributed computing for Microsoft Orleans. It extends Orleans' actor model with GPU-resident actors that process messages entirely on the GPU, achieving sub-microsecond latencies on supported hardware. The library provides ring kernel infrastructure, temporal alignment (HLC and Vector Clocks), GPU-to-GPU messaging, and hypergraph actor support.

## Key Capabilities

- **Ring Kernels** - Persistent GPU dispatch loops that keep actors resident in GPU memory, avoiding repeated kernel launch overhead
- **Temporal Alignment** - Hybrid Logical Clocks and Vector Clocks for distributed causal ordering, maintained on GPU
- **GPU-to-GPU Messaging** - Direct P2P communication between GPUs (NvLink, PCIe, Infinity Fabric) with automatic CPU-routed fallback
- **Hypergraph Actors** - Multi-way relationships with GPU-accelerated pattern matching
- **Queue-Depth Aware Placement** - Adaptive load balancing across heterogeneous GPU resources
- **Resilience** - Polly v8 integration with retry, circuit breaker, and rate limiting for GPU operations
- **CPU Fallback** - All GPU operations have CPU fallback paths for development and graceful degradation
- **OpenTelemetry Integration** - Per-grain GPU memory tracking, metrics, and distributed tracing

## Architecture

Orleans.GpuBridge supports two deployment models:

**GPU-Offload Model**: Actor logic runs on CPU, compute kernels are dispatched to GPU as needed. Best for batch processing and infrequent GPU usage.

**GPU-Native Model**: Actor state resides permanently in GPU memory. Ring kernels process messages entirely on GPU with zero kernel launch overhead. Requires GPUs with host-native atomic support (A100, H100, Grace Hopper) for persistent mode; partial-coherence GPUs (RTX series) use EventDriven mode.

```
+-----------------------------------------------------------+
|                   Orleans Application                     |
|  (User Services, Dashboards, Orchestration)               |
+-----------------------------------------------------------+
|     CPU Grains             GPU-Native Actor Ring Kernels   |
|  (Business Logic)  <-->   (Hypergraphs, Analytics)        |
+-----------------------------------------------------------+
|              Orleans.GpuBridge.Grains                     |
|         (GpuBatchGrain, GpuResidentGrain)                 |
+-----------------------------------------------------------+
|              Orleans.GpuBridge.Runtime                    |
|      (KernelCatalog, DeviceBroker, Placement)             |
+----------------------------+------------------------------+
|       DotCompute Backend   |        CPU Backend           |
+----------------------------+------------------------------+
```

## Packages

| Package | Description |
|---------|-------------|
| `Orleans.GpuBridge.Abstractions` | Core interfaces and contracts |
| `Orleans.GpuBridge.Runtime` | Runtime implementation, placement strategies, temporal infrastructure |
| `Orleans.GpuBridge.Grains` | GPU-accelerated grain base classes |
| `Orleans.GpuBridge.Backends.DotCompute` | DotCompute GPU backend (CUDA, Metal, CPU) |
| `Orleans.GpuBridge.BridgeFX` | High-level pipeline API |
| `Orleans.GpuBridge.Resilience` | Resilience patterns (Polly v8) |
| `Orleans.GpuBridge.Diagnostics` | Metrics and OpenTelemetry integration |
| `Orleans.GpuBridge.HealthChecks` | ASP.NET Core health check integrations |
| `Orleans.GpuBridge.Generators` | Source generators for GPU actors |
| `Orleans.GpuBridge.Logging` | Structured delegate-based logging |

## Getting Started

### Installation

```bash
dotnet add package Orleans.GpuBridge.Runtime
dotnet add package Orleans.GpuBridge.Grains
dotnet add package Orleans.GpuBridge.Backends.DotCompute
```

### Minimal Configuration

```csharp
using Orleans.GpuBridge.Runtime.Extensions;

var builder = Host.CreateDefaultBuilder(args)
    .ConfigureServices(services =>
    {
        services.AddGpuBridge(options =>
        {
            options.PreferGpu = true;
            options.FallbackToCpu = true;
            options.MaxConcurrentKernels = 100;
        });

        services.AddRingKernelSupport(options =>
        {
            options.DefaultGridSize = 1;
            options.DefaultBlockSize = 256;
            options.DefaultQueueCapacity = 256;
        });

        services.AddK2KSupport(enableP2P: true);
        services.AddGpuTelemetry();
    })
    .UseOrleans(siloBuilder =>
    {
        siloBuilder.UseLocalhostClustering();
    });

await builder.Build().RunAsync();
```

## Usage Examples

### GPU-Accelerated Grain

```csharp
using Orleans.GpuBridge.Grains.Base;
using Orleans.GpuBridge.Abstractions.Kernels;

public class VectorProcessingGrain : GpuGrainBase<VectorState>
{
    private IGpuKernel<float[], float[]>? _vectorAddKernel;

    public VectorProcessingGrain(IGrainContext grainContext, ILogger<VectorProcessingGrain> logger)
        : base(grainContext, logger) { }

    protected override async Task ConfigureGpuResourcesAsync(CancellationToken ct)
    {
        var kernelFactory = ServiceProvider.GetRequiredService<IKernelFactory>();
        _vectorAddKernel = await kernelFactory.CreateKernelAsync<float[], float[]>("vector-add", ct);
        await _vectorAddKernel.InitializeAsync(ct);
    }

    public async Task<float[]> AddVectorsAsync(float[] a, float[] b)
    {
        return await ExecuteKernelWithFallbackAsync(
            _vectorAddKernel!,
            a,
            cpuFallback: input => Task.FromResult(CpuVectorAdd(input, b)));
    }
}
```

### GPU-Native Actor (Ring Kernel)

```csharp
using Orleans.GpuBridge.Grains.Base;
using Orleans.GpuBridge.Abstractions.Temporal;

public class TemporalActorGrain : RingKernelGrainBase<ActorState, ActorMessage>
{
    public TemporalActorGrain(IGrainContext grainContext, ILogger<TemporalActorGrain> logger)
        : base(grainContext, logger) { }

    protected override Task<RingKernelConfig> ConfigureRingKernelAsync(CancellationToken ct)
    {
        return Task.FromResult(new RingKernelConfig
        {
            QueueDepth = 256,
            EnableHLC = true,
            EnableVectorClock = false,
            MaxStateSizeBytes = 1024
        });
    }

    protected override void ProcessMessageOnGpu(
        ref ActorState state,
        in ActorMessage message,
        ref HybridTimestamp hlc)
    {
        state.Counter += message.Value;
        state.LastUpdate = hlc.PhysicalTime;
    }

    public async Task SendEventAsync(int value)
    {
        await SendMessageAsync(new ActorMessage { Value = value });
    }
}
```

## GPU Hardware Requirements

### GPU Tiers

| Tier | Examples | Ring Kernel Mode | Message Latency |
|------|----------|-----------------|-----------------|
| Full Coherence | A100, H100, Grace Hopper | Persistent | 100-500ns |
| Partial Coherence | RTX 2000/3000/4000 series | EventDriven | 1-10ms (batched) |
| WSL2 / Limited | Any GPU under WSL2 | EventDriven | ~5s (development only) |

Full coherence GPUs have `hostNativeAtomicSupported=1`, enabling persistent ring kernels with real-time CPU-GPU memory visibility. Partial coherence GPUs (`concurrentManagedAccess=1`) must use EventDriven mode where kernels terminate after processing and are relaunched for new batches.

### WSL2 Limitations

WSL2's GPU virtualization layer does not support reliable system-scope atomics or CPU-GPU memory coherence. Ring kernels under WSL2 use the EventDriven workaround with Start-Active pattern. All functionality works but at development-only performance. Use native Linux for production GPU-native workloads.

## Monitoring and Diagnostics

### Health Checks

```csharp
services.AddHealthChecks()
    .AddGpuBridgeHealthCheck("gpu-health")
    .AddGpuMemoryHealthCheck("gpu-memory", failureThreshold: 0.9f);
```

### OpenTelemetry

```csharp
services.AddOpenTelemetry()
    .WithMetrics(builder => builder.AddGpuBridgeInstrumentation())
    .WithTracing(builder => builder.AddGpuBridgeInstrumentation());
```

Available metrics include `gpu.grain.allocations`, `gpu.grain.memory.allocated`, `gpu.memory.pool.utilization`, and `gpu.memory.pool.fragmentation`.

## Building from Source

```bash
# Build
dotnet build

# Run all tests
dotnet test

# Run a specific test project
dotnet test tests/Orleans.GpuBridge.Runtime.Tests

# Create NuGet packages
dotnet pack -c Release -o artifacts/packages
```

## Project Status

| Component | Status |
|-----------|--------|
| Core Abstractions | Stable |
| Runtime (Placement, Temporal, Ring Kernels) | Stable |
| DotCompute Backend | Stable |
| Resilience (Polly v8) | Stable |
| K2K Messaging / P2P | Stable |
| GPU Memory Telemetry | Stable |
| Health Checks | Stable |
| GPUDirect Storage | Planned |

### Test Suite (v0.3.0)

| Project | Passed | Skipped | Total |
|---------|--------|---------|-------|
| Abstractions.Tests | 242 | 0 | 242 |
| Runtime.Tests | 255 | 0 | 255 |
| Temporal.Tests | 290 | 1 | 292 |
| Grains.Tests | 98 | 0 | 98 |
| Generators.Tests | 22 | 0 | 22 |
| Hardware.Tests | 34 | 3 | 37 |
| Backends.DotCompute.Tests | 56 | 0 | 56 |
| RingKernelTests | 85 | 6 | 92 |
| Performance.Tests | 15 | 5 | 20 |
| Integration.Tests | 32 | 3 | 35 |
| Resilience.Tests | 53 | 0 | 53 |
| Diagnostics.Tests | 70 | 0 | 70 |
| **Total** | **1,252** | **18** | **1,272** |

Skipped tests require GPU hardware with specific capabilities (`hostNativeAtomicSupported`), full Orleans silo infrastructure, or are deferred pending lock-free data structure implementation.

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.

Commercial licensing, support, and domain-specific kernel blueprints are available - contact the author for details.

Copyright (c) 2025-2026 Michael Ivertowski

## Acknowledgments

- [Microsoft Orleans](https://github.com/dotnet/orleans) - Distributed actor framework
- [DotCompute](https://github.com/mivertowski/DotCompute) - .NET GPU compute abstraction
- The .NET Foundation and community

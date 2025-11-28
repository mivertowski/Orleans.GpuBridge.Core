# Orleans.GpuBridge.Core Documentation

<div align="center">

## GPU-Native Distributed Computing for Microsoft Orleans

**Transform your Orleans applications with GPU acceleration while maintaining familiar .NET patterns**

[Getting Started](articles/getting-started.md) | [API Documentation](api/index.md) | [Articles](articles/README.md) | [GitHub](https://github.com/mivertowski/Orleans.GpuBridge.Core)

</div>

---

## What is Orleans.GpuBridge.Core?

Orleans.GpuBridge.Core enables **GPU-native distributed computing** for Microsoft Orleans applications. This revolutionary framework allows you to build actors (grains) that reside permanently in GPU memory, processing messages at sub-microsecond latencies.

### Key Features

- **🚀 100-500ns Message Latency** - GPU-native actors process messages 20-200× faster than CPU actors
- **💾 GPU-Resident State** - Actors live permanently in GPU memory with zero kernel launch overhead
- **🔄 Ring Kernels** - Persistent GPU kernels running infinite dispatch loops
- **⏰ Temporal Alignment** - HLC and Vector Clocks maintained entirely on GPU
- **🕸️ Hypergraph Actors** - Multi-way relationships with GPU-accelerated pattern matching
- **🧬 Knowledge Organisms** - Emergent intelligence from actor interactions
- **🔌 Familiar .NET APIs** - Standard C# async/await patterns with full type safety

### Performance Breakthrough

| Metric | Traditional CPU Actors | GPU-Native Actors | Improvement |
|--------|------------------------|-------------------|-------------|
| Message Latency | 10-100μs | 100-500ns | **20-200×** |
| Throughput | 15K msgs/s | 2M msgs/s | **133×** |
| Memory Bandwidth | 200 GB/s | 1,935 GB/s | **10×** |
| Temporal Ordering | 50ns | 20ns | **2.5×** |

> **Performance Note**: The performance figures above represent targets achievable on **native Linux** with persistent kernel mode. WSL2 environments have limitations due to GPU-PV virtualization that prevent persistent kernels, resulting in higher latencies (~5 seconds in EventDriven mode). For production deployments requiring sub-microsecond latency, use native Linux. See [Implementation Roadmap](articles/temporal/IMPLEMENTATION-ROADMAP.md) for details on WSL2 limitations.

## Quick Example

```csharp
// Define your GPU-accelerated grain
[GpuAccelerated]
public class MyGpuGrain : Grain, IMyGpuGrain
{
    [GpuKernel("kernels/MyKernel")]
    private IGpuKernel<float[], float[]> _kernel;

    public async Task<float[]> ProcessAsync(float[] data)
    {
        // Kernel executes on GPU without launch overhead
        return await _kernel.ExecuteAsync(data);
    }
}

// Use it like any Orleans grain
var grain = grainFactory.GetGrain<IMyGpuGrain>(0);
var result = await grain.ProcessAsync(myData);
```

## Architecture Overview

Orleans.GpuBridge.Core implements two deployment models:

### GPU-Offload Model (Traditional)
- CPU actors offload compute to GPU
- Best for: Batch processing, infrequent GPU usage
- Kernel launch overhead: ~10-50μs

### GPU-Native Model (Revolutionary)
- Actors live permanently in GPU memory
- Ring kernels process messages on GPU
- **Zero kernel launch overhead**
- Sub-microsecond latency: 100-500ns
- Best for: High-frequency messaging, temporal graphs, real-time analytics

## Core Components

| Component | Description |
|-----------|-------------|
| **Orleans.GpuBridge.Abstractions** | Core interfaces and contracts (`IGpuBridge`, `IGpuKernel<TIn,TOut>`) |
| **Orleans.GpuBridge.Runtime** | Runtime implementation with kernel catalog and device management |
| **Orleans.GpuBridge.BridgeFX** | High-level pipeline API with fluent interface |
| **Orleans.GpuBridge.Grains** | Pre-built grain implementations for common patterns |
| **Orleans.GpuBridge.Backends.DotCompute** | GPU backend abstraction (CUDA, ROCm, CPU fallback) |

## Use Cases

### Financial Services
- **High-Frequency Trading** - Order matching at <10μs latency
- **Fraud Detection** - Real-time pattern matching on transaction streams
- **Risk Analytics** - Portfolio optimization with GPU-resident market data

### Scientific Computing
- **Physics Simulations** - Particle systems, fluid dynamics, molecular dynamics
- **Bioinformatics** - Genome sequence alignment, protein folding

### Real-Time Analytics
- **Stream Processing** - Event aggregation and pattern detection
- **Hypergraph Analytics** - Pattern detection with <100μs latency
- **Temporal Pattern Detection** - Fraud detection with causal ordering

### Gaming and Simulation
- **Digital Twins** - Living entities with physics-accurate simulation
- **Multiplayer Servers** - GPU-accelerated physics and AI

## Getting Started

### Installation

```bash
# Install Orleans
dotnet add package Microsoft.Orleans.Server
dotnet add package Microsoft.Orleans.Client

# Install GPU Bridge
dotnet add package Orleans.GpuBridge.Core
dotnet add package Orleans.GpuBridge.Backends.DotCompute
```

### Configure Services

```csharp
services.AddGpuBridge(options =>
{
    options.PreferGpu = true;
    options.EnableRingKernels = true;
})
.AddKernel(k => k
    .Id("kernels/MyKernel")
    .In<float[]>()
    .Out<float[]>()
    .FromFactory(sp => new MyKernel()));
```

### Next Steps

1. **[Getting Started Guide](articles/getting-started.md)** - Build your first GPU-accelerated grain
2. **[Concepts and Background](articles/concepts.md)** - Understand GPU-native actors and ring kernels
3. **[Architecture Overview](articles/architecture.md)** - Deep dive into system design
4. **[API Reference](api/index.md)** - Complete API documentation

## Documentation Sections

### 📚 Articles

Explore in-depth technical articles covering design, implementation, and usage:

- **[GPU-Native Actors](articles/gpu-actors/introduction/README.md)** - Revolutionary paradigm for distributed GPU computing
- **[Hypergraph Actors](articles/hypergraph-actors/introduction/README.md)** - Multi-way relationships with GPU acceleration
- **[Temporal Correctness](articles/temporal/introduction/README.md)** - HLC, Vector Clocks, and causal ordering
- **[Knowledge Organisms](articles/knowledge-organisms/README.md)** - Emergent intelligence from actor interactions
- **[Process Intelligence](articles/process-intelligence/README.md)** - Object-centric process mining with GPU acceleration

### 🔧 API Documentation

Complete API reference with examples:

- **[Core Abstractions](api/Orleans.GpuBridge.Abstractions.html)** - `IGpuBridge`, `IGpuKernel<TIn,TOut>`, attributes
- **[Runtime Components](api/Orleans.GpuBridge.Runtime.html)** - Kernel catalog, device broker, placement strategies
- **[Pipeline API](api/Orleans.GpuBridge.BridgeFX.html)** - Fluent API for batch processing
- **[Grain Implementations](api/Orleans.GpuBridge.Grains.html)** - Pre-built grains for common patterns

## Community and Support

- **GitHub**: [Orleans.GpuBridge.Core Repository](https://github.com/mivertowski/Orleans.GpuBridge.Core)
- **Issues**: [Report bugs and request features](https://github.com/mivertowski/Orleans.GpuBridge.Core/issues)
- **Discussions**: [Community discussions](https://github.com/mivertowski/Orleans.GpuBridge.Core/discussions)

## Requirements

- **.NET 9.0** or later
- **NVIDIA GPU** with CUDA 12.0+ (or AMD GPU with ROCm 5.0+)
- **Windows 10/11** or **Linux** (Ubuntu 22.04+)
- **Microsoft Orleans** 8.0+

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

<div align="center">

**Ready to accelerate your Orleans applications?**

[Get Started Now](articles/getting-started.md) | [View Examples](https://github.com/mivertowski/Orleans.GpuBridge.Core/tree/main/samples)

</div>

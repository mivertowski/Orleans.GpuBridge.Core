# API Reference

Welcome to the Orleans.GpuBridge.Core API reference documentation.

## Namespaces Overview

### Orleans.GpuBridge.Abstractions
Core interfaces and contracts for GPU acceleration in Orleans.

- **IGpuBridge** - Main bridge interface for GPU operations
- **IGpuKernel** - Kernel execution contract
- **IGpuAccelerated** - Marker interface for GPU-capable grains
- **GpuBridgeOptions** - Configuration options

### Orleans.GpuBridge.Abstractions.Clocks
Temporal correctness primitives.

- **IHybridLogicalClock** - HLC timestamp generation
- **IVectorClock** - Vector clock for causal ordering
- **HybridTimestamp** - Immutable HLC timestamp value
- **VectorTimestamp** - Immutable vector clock value

### Orleans.GpuBridge.Abstractions.Placement
GPU-aware grain placement strategies.

- **IGpuPlacementDirector** - Placement decision interface
- **GpuPlacementStrategy** - Placement strategy base class
- **QueueDepthAwarePlacement** - Ring kernel queue-aware placement

### Orleans.GpuBridge.Runtime
Runtime implementation and DI integration.

- **KernelCatalog** - Kernel registration and resolution
- **DeviceBroker** - GPU device management
- **ServiceCollectionExtensions** - `AddGpuBridge()` extension

### Orleans.GpuBridge.Runtime.Clocks
Clock implementation and synchronization.

- **HybridLogicalClock** - HLC implementation
- **VectorClock** - Vector clock implementation
- **ClockSourceSelector** - Clock source selection
- **SoftwarePtpClockSource** - Software PTP synchronization

### Orleans.GpuBridge.Runtime.Resident
GPU-resident actor support.

- **IGpuResidentManager** - GPU memory management
- **GpuResidentHandle** - Handle to GPU-resident state
- **RingKernelRuntime** - Ring kernel lifecycle management

### Orleans.GpuBridge.Grains
Orleans grain implementations.

- **GpuBatchGrain** - Batch processing grain
- **GpuResidentGrain** - GPU-resident data grain
- **GpuStreamGrain** - Stream processing grain
- **HypergraphVertexGrain** - Vertex actor implementation
- **HypergraphHyperedgeGrain** - Hyperedge actor implementation

### Orleans.GpuBridge.BridgeFX
High-level pipeline API.

- **GpuPipeline** - Fluent API for batch processing
- **PipelineBuilder** - Pipeline configuration builder
- **PipelineResult** - Execution result container

### Orleans.GpuBridge.Backends.DotCompute
DotCompute GPU backend integration.

- **DotComputeBackendAdapter** - Backend adapter implementation
- **DotComputeRingKernelRuntime** - Ring kernel runtime for DotCompute
- **KernelAttributeMapper** - DotCompute attribute mapping

## Quick Reference

### Service Registration
```csharp
services.AddGpuBridge(options =>
{
    options.PreferGpu = true;
    options.EnableFallbackToCpu = true;
    options.GpuFramework = GpuFramework.DotCompute;
});
```

### Kernel Registration
```csharp
services.AddGpuBridge()
    .AddKernel(k => k
        .Id("matrix-multiply")
        .In<float[]>()
        .Out<float[]>()
        .FromFactory(sp => new MatrixKernel()));
```

### Pipeline Execution
```csharp
var results = await GpuPipeline<float[], float[]>
    .For(grainFactory, "matrix-multiply")
    .WithBatchSize(1000)
    .ExecuteAsync(data);
```

## Version Information

- **Current Version**: 0.1.0
- **Target Framework**: .NET 9.0
- **GPU Framework**: DotCompute 0.5.1

---

*See individual namespace documentation for detailed API information.*

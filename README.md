# Orleans.GpuBridge (Core)
_Last updated: 2025-08-08 22:18 UTC+02:00_

**Purpose:** Bring GPU acceleration into Orleans via **G‑Grains** and a clean abstraction (**BridgeFX**).  
**License:** Apache-2.0 (OSS).

## Highlights
- **AddGpuBridge()** DI extension
- **KernelCatalog** with **CPU fallback**
- **Custom placement** (skeleton)
- **BridgeFX** pipeline & attribute
- **VectorAdd sample** (CPU fallback today)

## Quick start
```csharp
// Startup
services.AddGpuBridge(o => o.PreferGpu = true)
        .AddKernel(k => k.Id("kernels/VectorAdd")
                        .In<float[]>().Out<float>()
                        .FromFactory(sp => new Orleans.GpuBridge.Runtime.CpuVectorAddKernel()));
// Use
var results = await Orleans.GpuBridge.BridgeFX.GpuPipeline<float[],float>
               .For(grainFactory, "kernels/VectorAdd")
               .WithBatchSize(8192)
               .ExecuteAsync(pairs);
```
See **docs/** for design, abstractions, and ops.

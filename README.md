# Orleans.GpuBridge (Core)

[![CI/CD Pipeline](https://github.com/orleans-gpubridge/Orleans.GpuBridge.Core/actions/workflows/ci.yml/badge.svg)](https://github.com/orleans-gpubridge/Orleans.GpuBridge.Core/actions/workflows/ci.yml)
[![CodeQL](https://github.com/orleans-gpubridge/Orleans.GpuBridge.Core/actions/workflows/codeql.yml/badge.svg)](https://github.com/orleans-gpubridge/Orleans.GpuBridge.Core/actions/workflows/codeql.yml)
[![Security Scan](https://github.com/orleans-gpubridge/Orleans.GpuBridge.Core/actions/workflows/security.yml/badge.svg)](https://github.com/orleans-gpubridge/Orleans.GpuBridge.Core/actions/workflows/security.yml)
[![codecov](https://codecov.io/gh/orleans-gpubridge/Orleans.GpuBridge.Core/branch/main/graph/badge.svg?token=YOUR_TOKEN)](https://codecov.io/gh/orleans-gpubridge/Orleans.GpuBridge.Core)
[![NuGet](https://img.shields.io/nuget/v/Orleans.GpuBridge.Runtime.svg)](https://www.nuget.org/packages/Orleans.GpuBridge.Runtime/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![.NET 9.0](https://img.shields.io/badge/.NET-9.0-purple)](https://dotnet.microsoft.com/download/dotnet/9.0)

_Last updated: 2025-08-11_

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

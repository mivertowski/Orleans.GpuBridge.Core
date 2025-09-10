# Orleans GPU Bridge - DotCompute Backend

This package provides a DotCompute backend implementation for Orleans GPU Bridge, enabling GPU acceleration through the DotCompute framework.

## Features

- GPU kernel compilation and execution
- Memory management with unified memory support
- Device discovery and management
- Production-ready error handling and logging
- CPU fallback for non-GPU environments

## Installation

```bash
dotnet add package Orleans.GpuBridge.Backends.DotCompute
```

## Usage

```csharp
services.AddGpuBridge()
        .AddDotComputeBackend(options =>
        {
            options.PreferGpu = true;
            options.EnableUnifiedMemory = true;
        });
```

## Requirements

- .NET 9.0 or higher
- CUDA-compatible GPU (for GPU execution)
- DotCompute runtime
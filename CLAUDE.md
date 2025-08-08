# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Orleans.GpuBridge.Core is a .NET library that provides GPU acceleration capabilities for the Orleans distributed computing framework. The project integrates GPU compute resources with Orleans grains through a bridge abstraction layer.

## Architecture Overview

### Core Components

1. **Orleans.GpuBridge.Abstractions** - Defines core interfaces and contracts:
   - `IGpuBridge` - Main bridge interface for GPU operations
   - `IGpuKernel<TIn,TOut>` - Kernel execution contract
   - `[GpuAccelerated]` attribute for grain marking
   - Configuration via `GpuBridgeOptions`

2. **Orleans.GpuBridge.Runtime** - Runtime implementation:
   - `KernelCatalog` - Manages kernel registration and execution
   - `DeviceBroker` - GPU device management (currently stub)
   - DI integration via `AddGpuBridge()` extension method
   - Placement strategies for GPU-aware grain placement

3. **Orleans.GpuBridge.BridgeFX** - High-level pipeline API:
   - `GpuPipeline<TIn,TOut>` - Fluent API for batch processing
   - Automatic partitioning and result aggregation

4. **Orleans.GpuBridge.Grains** - Orleans grain implementations:
   - `GpuBatchGrain` - Batch processing grain
   - `GpuResidentGrain` - GPU-resident data grain
   - `GpuStreamGrain` - Stream processing grain

### Key Design Patterns

**Service Registration Pattern:**
```csharp
services.AddGpuBridge(options => options.PreferGpu = true)
        .AddKernel(k => k.Id("kernels/VectorAdd")
                        .In<float[]>().Out<float>()
                        .FromFactory(sp => new CustomKernel()));
```

**Pipeline Execution Pattern:**
```csharp
var results = await GpuPipeline<TIn,TOut>
    .For(grainFactory, "kernel-id")
    .WithBatchSize(batchSize)
    .ExecuteAsync(data);
```

## Development Commands

Since the project currently lacks build configuration files, you'll need to create them first:

### Initial Setup (Required)
```bash
# Create solution file
dotnet new sln -n Orleans.GpuBridge.Core

# Create project files for each component
cd src/Orleans.GpuBridge.Abstractions
dotnet new classlib -n Orleans.GpuBridge.Abstractions -f net9.0
cd ../Orleans.GpuBridge.Runtime
dotnet new classlib -n Orleans.GpuBridge.Runtime -f net9.0
cd ../Orleans.GpuBridge.BridgeFX
dotnet new classlib -n Orleans.GpuBridge.BridgeFX -f net9.0
cd ../Orleans.GpuBridge.Grains
dotnet new classlib -n Orleans.GpuBridge.Grains -f net9.0

# Add projects to solution
cd ../..
dotnet sln add src/**/*.csproj
```

### Standard Commands (after setup)
```bash
# Build the solution
dotnet build

# Run tests (when added)
dotnet test

# Create NuGet packages
dotnet pack

# Clean build artifacts
dotnet clean
```

## Project Status and Implementation Notes

### Current State
- **Core abstractions**: Complete
- **Runtime infrastructure**: Basic implementation with CPU fallbacks
- **GPU execution**: Not yet implemented (all kernels use CPU fallback)
- **Testing**: No test projects exist yet
- **Build configuration**: Missing .csproj and .sln files

### CPU Fallback System
All GPU kernels currently fall back to CPU implementations. The `KernelCatalog` manages this through:
```csharp
public async Task<TOut> ExecuteAsync<TIn, TOut>(string kernelId, TIn input)
{
    // Currently always uses CPU fallback
    var kernel = ResolveKernel<TIn, TOut>(kernelId);
    return await kernel.ExecuteAsync(input);
}
```

### Planned GPU Implementation
According to ROADMAP.md, the project will integrate:
- DotCompute adapter for actual GPU execution
- Queue-depth aware placement strategies
- Persistent kernel hosts with mapped buffers
- GPUDirect Storage support

## Key Files to Understand

### Service Registration and DI
- `src/Orleans.GpuBridge.Runtime/ServiceCollectionExtensions.cs` - Entry point for service configuration
- `src/Orleans.GpuBridge.Runtime/KernelCatalog.cs` - Kernel registration and resolution

### Core Interfaces
- `src/Orleans.GpuBridge.Abstractions/IGpuBridge.cs` - Main bridge contract
- `src/Orleans.GpuBridge.Abstractions/IGpuKernel.cs` - Kernel execution interface

### High-Level API
- `src/Orleans.GpuBridge.BridgeFX/GpuPipeline.cs` - Fluent pipeline API implementation

### Orleans Integration
- `src/Orleans.GpuBridge.Grains/GpuBatchGrain.cs` - Primary grain for batch processing
- `src/Orleans.GpuBridge.Runtime/Placement/*.cs` - GPU-aware placement strategies

## Development Priorities

When implementing new features:

1. **Maintain CPU fallback**: Always provide CPU implementations for GPU kernels
2. **Follow Orleans patterns**: Use grain state, activation lifecycle properly
3. **Async throughout**: All GPU operations should be async
4. **Batch optimization**: Design for batch processing efficiency
5. **Resource management**: Proper GPU resource cleanup in `Dispose()` methods

## Testing Strategy

When adding tests:
```bash
# Create test projects
dotnet new xunit -n Orleans.GpuBridge.Abstractions.Tests
dotnet new xunit -n Orleans.GpuBridge.Runtime.Tests

# Add Orleans TestingHost for grain testing
dotnet add package Microsoft.Orleans.TestingHost
```

Test priorities:
1. Kernel registration and resolution
2. CPU fallback execution
3. Pipeline batch processing
4. Grain activation and placement
5. Resource cleanup

## Documentation Structure

- `docs/starter-kit/DESIGN.md` - Core architecture overview
- `docs/starter-kit/ABSTRACTION.md` - BridgeFX pipeline details
- `docs/starter-kit/KERNELS.md` - Kernel implementation guide
- `docs/starter-kit/OPERATIONS.md` - Operational considerations
- `docs/starter-kit/ROADMAP.md` - Future development plans

## Dependencies

Key NuGet packages to add when creating .csproj files:
- Microsoft.Orleans.Core
- Microsoft.Orleans.Runtime
- Microsoft.Extensions.DependencyInjection
- Microsoft.Extensions.Hosting
- Microsoft.Extensions.Logging
- Microsoft.Extensions.Options

## Important Implementation Considerations

1. **Thread Safety**: All kernel implementations must be thread-safe
2. **Memory Management**: GPU memory must be properly managed and released
3. **Error Handling**: GPU operations can fail - proper fallback to CPU required
4. **Performance**: Batch size optimization is critical for GPU efficiency
5. **Orleans Constraints**: Respect grain single-threaded execution model
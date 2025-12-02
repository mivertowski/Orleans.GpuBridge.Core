# Orleans.GpuBridge.Core - Claude Code Configuration

## Project Overview

Orleans.GpuBridge.Core is a .NET 9 library enabling **GPU-native distributed computing** for Microsoft Orleans. This project represents a paradigm shift from traditional CPU-based actor systems to actors that can live permanently on the GPU.

### Key Technologies
- **Ring Kernels**: Persistent GPU kernels running infinite dispatch loops
- **Temporal Alignment**: HLC and Vector Clocks for distributed ordering
- **GPU-to-GPU Messaging**: Actors communicate at 100-500ns latency (datacenter GPUs)
- **Hypergraph Actors**: Multi-way relationships with GPU-accelerated pattern matching
- **DotCompute Backend**: .NET-native GPU compute abstraction

## Build Commands

```bash
# Build the solution
dotnet build

# Run all tests
dotnet test

# Run specific test project
dotnet test tests/Orleans.GpuBridge.Runtime.Tests
dotnet test tests/Orleans.GpuBridge.Temporal.Tests

# Build in release mode
dotnet build -c Release

# Create NuGet packages
dotnet pack
```

## Test Suite Status (as of December 2024)

| Project | Passed | Skipped | Total |
|---------|--------|---------|-------|
| Abstractions.Tests | 242 | 0 | 242 |
| Runtime.Tests | 202 | 0 | 202 |
| Temporal.Tests | 286 | 6 | 292 |
| Grains.Tests | 98 | 0 | 98 |
| Generators.Tests | 22 | 0 | 22 |
| Hardware.Tests | 34 | 3 | 37 |
| Backends.DotCompute.Tests | 58 | 0 | 58 |
| Performance.Tests | 0 | 5 | 5 |
| Integration.Tests | 19 | 3 | 22 |
| **Total** | **961** | **17** | **978** |

### Skipped Tests (with valid reasons)
- **ConcurrentAccessTests**: IntervalTree thread-safety - requires implementation changes
- **RingKernelIntegrationTests**: Requires GPU with `hostNativeAtomicSupported`
- **Hardware CUDA tests**: GPU-dependent integration tests
- **Performance tests**: Require GPU and Orleans silo infrastructure
- **Integration tests**: Require full Orleans silo + GPU setup

## Architecture

### Core Components

1. **Orleans.GpuBridge.Abstractions** - Interfaces and contracts
2. **Orleans.GpuBridge.Runtime** - Runtime implementation, placement strategies
3. **Orleans.GpuBridge.Grains** - Orleans grain implementations
4. **Orleans.GpuBridge.Backends.DotCompute** - DotCompute GPU backend
5. **Orleans.GpuBridge.BridgeFX** - High-level pipeline API

### Implementation Status
- ✅ Ring kernel infrastructure
- ✅ GPU-resident message queues
- ✅ Temporal alignment (HLC, Vector Clocks)
- ✅ Hypergraph actors
- ✅ Queue-depth aware placement
- ✅ Adaptive load balancing
- 🚧 DotCompute backend (EventDriven mode working)
- 📋 GPUDirect Storage (planned)

## Development Guidelines

### Code Quality Requirements
- .NET 9 patterns only
- Production-grade code quality
- Fix all build errors and warnings
- No warning suppression unless absolutely necessary

### Orleans-Specific Rules
- **NO `.ConfigureAwait(false)`** in grain context
- Respect grain single-threaded execution model
- Use grain state and activation lifecycle properly
- All GPU operations must be async

### Testing Requirements
- Write tests before implementation (TDD)
- Maintain CPU fallbacks for all GPU operations
- Use FluentAssertions for readable assertions
- Use Moq for mocking

## GPU Hardware Considerations

### Current Development System
```
GPU: RTX 2000 Ada Laptop
Compute Capability: 8.9
hostNativeAtomicSupported: 0 (No CPU-GPU atomic coherence)
```

### GPU Categories for Ring Kernel Support

1. **Full Coherence GPUs** (A100, H100, Grace Hopper)
   - `hostNativeAtomicSupported=1`
   - Persistent kernels work
   - Target latency: 100-500ns

2. **Partial Coherence GPUs** (RTX 2000/3000/4000 series)
   - `concurrentManagedAccess=1`, `hostNativeAtomicSupported=0`
   - Must use EventDriven mode
   - Latency: 1-10ms (batched)

3. **WSL2/Limited GPUs**
   - Most limited, EventDriven only
   - Development use only

### CUDA Testing on WSL2
```bash
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:$LD_LIBRARY_PATH"
```

## DotCompute Integration

DotCompute source code location:
```
/home/mivertowski/DotCompute/DotCompute
```

For DotCompute feature requests, create a document in `docs/dotcompute-feature-requests.md`.

## File Organization

```
/src                    - Source code
/tests                  - Test projects
/docs                   - Documentation
```

**Never save working files to the root folder.**

## Key Files

### Service Registration
- `src/Orleans.GpuBridge.Runtime/ServiceCollectionExtensions.cs`

### Placement Strategies
- `src/Orleans.GpuBridge.Runtime/Placement/QueueDepthPlacementDirector.cs`
- `src/Orleans.GpuBridge.Runtime/Placement/AdaptiveLoadBalancer.cs`

### Temporal Infrastructure
- `src/Orleans.GpuBridge.Runtime/Temporal/Clock/HybridLogicalClock.cs`
- `src/Orleans.GpuBridge.Runtime/Temporal/Network/NetworkLatencyCompensator.cs`

### Ring Kernel Infrastructure
- `src/Orleans.GpuBridge.Runtime/RingKernel/RingKernelManager.cs`
- `src/Orleans.GpuBridge.Runtime/RingKernel/IRingKernelRuntime.cs`

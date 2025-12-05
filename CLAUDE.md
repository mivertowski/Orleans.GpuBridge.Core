# Orleans.GpuBridge.Core - Claude Code Configuration

## Project Overview

Orleans.GpuBridge.Core is a .NET 9 library enabling **GPU-native distributed computing** for Microsoft Orleans. This project represents a paradigm shift from traditional CPU-based actor systems to actors that can live permanently on the GPU.

### Key Technologies
- **Ring Kernels**: Persistent GPU kernels running infinite dispatch loops
- **Temporal Alignment**: HLC and Vector Clocks for distributed ordering
- **GPU-to-GPU Messaging**: Actors communicate at 100-500ns latency (datacenter GPUs)
- **Hypergraph Actors**: Multi-way relationships with GPU-accelerated pattern matching
- **DotCompute Backend**: .NET-native GPU compute abstraction (v0.5.1 from NuGet)

## Current Version

**Version 0.2.0** - Released December 2025

### Key Features in v0.2.0
- Resilience Module (Polly v8) with retry, circuit breaker, rate limiting
- GPU Direct Messaging with P2P support (NvLink, PCIe, Infinity Fabric)
- GPU Memory Telemetry with OpenTelemetry integration
- DotCompute NuGet packages (no local references)
- Comprehensive XML documentation

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
dotnet pack -c Release -o artifacts/packages
```

## Test Suite Status (December 2025)

| Project | Passed | Skipped | Total |
|---------|--------|---------|-------|
| Abstractions.Tests | 242 | 0 | 242 |
| Runtime.Tests | 249 | 0 | 249 |
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
| **Total** | **1,246** | **18** | **1,266** |

### Skipped Tests (with valid reasons)
- **ConcurrentAccessTests**: IntervalTree thread-safety - requires lock-free implementation
- **RingKernelIntegrationTests**: Requires GPU with `hostNativeAtomicSupported`
- **Hardware CUDA tests**: GPU-dependent integration tests
- **Performance tests**: Require GPU and Orleans silo infrastructure
- **Integration tests**: Require full Orleans silo + GPU setup

## Architecture

### Core Components

1. **Orleans.GpuBridge.Abstractions** - Interfaces and contracts
2. **Orleans.GpuBridge.Runtime** - Runtime implementation, placement strategies
3. **Orleans.GpuBridge.Grains** - Orleans grain implementations
4. **Orleans.GpuBridge.Backends.DotCompute** - DotCompute GPU backend (NuGet v0.5.1)
5. **Orleans.GpuBridge.BridgeFX** - High-level pipeline API
6. **Orleans.GpuBridge.Resilience** - Resilience patterns (retry, circuit breaker, rate limiting)
7. **Orleans.GpuBridge.Diagnostics** - Metrics and telemetry
8. **Orleans.GpuBridge.HealthChecks** - Health check integrations
9. **Orleans.GpuBridge.Generators** - Source generators for GPU actors
10. **Orleans.GpuBridge.Logging** - Structured logging support

### Implementation Status
- ✅ Ring kernel infrastructure
- ✅ GPU-resident message queues
- ✅ Temporal alignment (HLC, Vector Clocks)
- ✅ Hypergraph actors
- ✅ Queue-depth aware placement
- ✅ Adaptive load balancing
- ✅ Resilience patterns (Polly v8)
- ✅ Rate limiting (token bucket)
- ✅ Circuit breaker and retry policies
- ✅ GPU P2P messaging with fallback
- ✅ GPU memory telemetry
- ✅ DotCompute backend (NuGet v0.5.1)
- 📋 GPUDirect Storage (planned)
- 📋 OpenCL backend (planned)

## Development Guidelines

### Code Quality Requirements
- .NET 9 patterns only
- Production-grade code quality
- Fix all build errors and warnings
- No warning suppression unless absolutely necessary
- Comprehensive XML documentation for all public APIs

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

## Package Version Management

All package versions are centralized in `Directory.Build.props`:

```xml
<PropertyGroup>
  <Version>0.2.0</Version>
  <MicrosoftCodeAnalysisVersion>4.14.0</MicrosoftCodeAnalysisVersion>
  <MicrosoftExtensionsVersion>10.0.0</MicrosoftExtensionsVersion>
  <MicrosoftOrleansVersion>9.2.1</MicrosoftOrleansVersion>
  <DotComputeVersion>0.5.1</DotComputeVersion>
</PropertyGroup>
```

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

DotCompute packages from NuGet.org v0.5.1:
- `DotCompute.Abstractions`
- `DotCompute.Core`
- `DotCompute.Runtime`
- `DotCompute.Backends.CPU`
- `DotCompute.Backends.CUDA`
- `DotCompute.Generators`

For DotCompute feature requests, create a document in `docs/dotcompute-feature-requests.md`.

## File Organization

```
/src                    - Source code (10 packages)
/tests                  - Test projects
/docs                   - Documentation
/artifacts/packages     - NuGet packages output
```

**Never save working files to the root folder.**

## Key Files

### Version Management
- `Directory.Build.props` - Centralized version and package management

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

### Resilience Patterns
- `src/Orleans.GpuBridge.Resilience/Policies/GpuResiliencePolicy.cs`
- `src/Orleans.GpuBridge.Resilience/RateLimit/TokenBucketRateLimiter.cs`
- `src/Orleans.GpuBridge.Resilience/Fallback/GpuFallbackChain.cs`

## Publishing Packages

```bash
# Build release and create packages
dotnet pack -c Release -o artifacts/packages

# Push to NuGet (requires API key)
dotnet nuget push "artifacts/packages/*.nupkg" --api-key YOUR_API_KEY --source https://api.nuget.org/v3/index.json
```

## NuGet Package List

| Package | Description |
|---------|-------------|
| Orleans.GpuBridge.Abstractions | Core interfaces and contracts |
| Orleans.GpuBridge.Runtime | Runtime implementation |
| Orleans.GpuBridge.Grains | GPU-accelerated grain base classes |
| Orleans.GpuBridge.Backends.DotCompute | DotCompute GPU backend |
| Orleans.GpuBridge.BridgeFX | High-level pipeline API |
| Orleans.GpuBridge.Resilience | Resilience patterns (Polly v8) |
| Orleans.GpuBridge.Diagnostics | Metrics and telemetry |
| Orleans.GpuBridge.HealthChecks | ASP.NET Core health checks |
| Orleans.GpuBridge.Generators | Source generators |
| Orleans.GpuBridge.Logging | Structured logging |

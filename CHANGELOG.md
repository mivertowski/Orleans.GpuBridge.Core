# Changelog

All notable changes to Orleans.GpuBridge.Core will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2025-12-05

### Added

#### Resilience Module (Orleans.GpuBridge.Resilience)
- **GpuResiliencePolicy**: Unified resilience policy combining retry, circuit breaker, timeout, and bulkhead patterns
  - Built on Polly v8 ResiliencePipeline API
  - GPU-specific exception handling for memory, kernel, and device failures
  - Configurable timeouts for kernel execution, device operations, and memory allocation
- **TokenBucketRateLimiter**: GPU-aware rate limiting to prevent resource exhaustion
  - Configurable token refill rate and burst size
  - Metrics tracking for rejection rates
- **GpuFallbackChain**: Automatic GPU → CPU fallback with degradation levels
  - Four-level fallback: Optimal → Reduced → Degraded → Failed
  - Auto-degradation based on error thresholds
  - Auto-recovery when GPU resources become available
- **ChaosEngineer**: Fault injection for resilience testing
  - Configurable fault and latency injection rates
  - Test-environment only activation

#### GPU Direct Messaging with P2P Support (FR-002)
- **IGpuPeerToPeerMemory**: Interface for GPU peer-to-peer memory operations
  - `CanAccessPeer()`, `EnablePeerAccessAsync()`, `DisablePeerAccessAsync()`
  - `CopyPeerToPeerAsync()`, `MapPeerMemoryAsync()`, `UnmapPeerMemoryAsync()`
  - `GetP2PCapability()` for querying P2P capabilities between devices
- **P2PCapabilityInfo**: Record describing P2P capabilities between device pairs
  - Bandwidth and latency estimates
  - Access type (NvLink, PCIe P2P, Infinity Fabric, etc.)
  - Atomics support detection
- **GpuDirectMessagingMode**: Enum for routing configuration
  - `CpuRouted`: Default CPU-staged transfers (always works)
  - `PreferP2P`: Automatically selects best available path
  - `NvLink`, `PciExpressP2P`, `InfinityFabric`, `GpuDirectRdma`: Explicit modes
- **CpuFallbackPeerToPeerMemory**: CPU fallback implementation for development/testing
- **K2KDispatcher Enhancements**:
  - Device placement tracking via `RegisterActorDevice()`/`GetActorDevice()`
  - P2P capability caching for efficient routing decisions
  - Automatic P2P vs CPU-routed path selection
  - `K2KRoutingStats` for monitoring (P2P dispatches, latency, etc.)
- **RingKernelConfig P2P Options**:
  - `MessagingMode`: Select GPU direct messaging mode
  - `AutoEnableP2P`: Automatic P2P discovery and setup
  - `P2PMinBandwidthGBps`, `P2PMaxLatencyNs`: Thresholds for P2P selection
  - `EnableP2PAtomics`: Enable P2P atomic operations
- **DI Extensions**:
  - `AddK2KSupport(enableP2P: true)`: Register K2K with P2P support
  - `AddP2PMemoryProvider<T>()`: Register custom P2P providers

#### GPU Memory Telemetry (FR-005)
- **IGpuMemoryTelemetryProvider**: Interface for per-grain GPU memory tracking
  - `RecordGrainMemoryAllocation()`, `RecordGrainMemoryRelease()`
  - `GetGrainMemorySnapshot()`, `GetMemoryStatsByGrainType()`
  - `GetAllGrainTypeMemoryStats()`, `GetTotalAllocatedMemory()`
  - `RecordMemoryPoolStats()`, `GetMemoryPoolStats()`
  - `StreamEventsAsync()`: Real-time event streaming via IAsyncEnumerable
- **GpuMemoryTelemetryProvider**: Full implementation with OpenTelemetry integration
  - Thread-safe per-grain tracking using ConcurrentDictionary
  - Peak memory tracking per grain
  - Memory pool utilization and fragmentation monitoring
  - Event streaming via bounded Channel
- **OpenTelemetry Metrics**:
  - `gpu.grain.allocations` - Allocation count per grain type
  - `gpu.grain.deallocations` - Deallocation count
  - `gpu.grain.allocation.size` - Allocation size histogram
  - `gpu.grain.memory.allocated` - Current memory per grain type
  - `gpu.grain.active.count` - Active grain count
  - `gpu.memory.pool.utilization` - Pool utilization percentage
  - `gpu.memory.pool.fragmentation` - Fragmentation percentage
- **Data Types**:
  - `GrainMemorySnapshot`: Point-in-time grain memory state
  - `GrainTypeMemoryStats`: Aggregated statistics per grain type
  - `MemoryPoolStats`: GPU memory pool statistics
  - `GpuMemoryEvent`: Real-time memory events for streaming

#### Exception Types
- **GpuBridgeException**: Base exception class for all GPU Bridge errors
- **GpuOperationException**: General GPU operation failures
- **GpuMemoryException**: Memory allocation and transfer failures
- **GpuDeviceException**: Device unavailable or failed states
- **GpuKernelException**: Kernel execution failures
- **RateLimitExceededException**: Rate limit exceeded errors

### Changed

#### Build & Packaging
- **DotCompute NuGet Migration**: Replaced hardcoded local project references with NuGet packages v0.5.1
  - All DotCompute dependencies now sourced from nuget.org
  - Centralized package version management in Directory.Build.props
  - Fixed CI/CD pipeline compatibility
- **Package Version Centralization**: All Microsoft.Extensions, Microsoft.Orleans, and Microsoft.CodeAnalysis versions now managed centrally

### Improved

#### Test Coverage
- Added 43 new K2K tests (K2KDispatcherTests, CpuFallbackPeerToPeerMemoryTests)
- Added 23 new GPU memory telemetry tests (GpuMemoryTelemetryProviderTests)
- Added Orleans.GpuBridge.Resilience.Tests with 53 comprehensive tests
- Increased total test count from 978 to 1,231 tests

#### Documentation
- Added comprehensive XML documentation across all public APIs
- Updated README with P2P GPU messaging documentation
- Updated README with GPU memory telemetry usage examples
- Added P2P access types comparison table
- Added OpenTelemetry metrics reference
- Added package README for Orleans.GpuBridge.Resilience

### Fixed
- Fixed Polly v7 → v8 API migration issues
- Fixed exception parameter ordering in ChaosEngineer and GpuFallbackChain
- Fixed missing Priority property on IFallbackExecutor interface
- Fixed FluentAssertions method usage in tests
- Fixed Orleans `[GenerateSerializer]` attributes on AffinityGroupMetrics
- Fixed XML documentation warnings across Resilience module

### Performance
- All resilience operations designed for minimal allocation overhead
- Token bucket rate limiter operates at sub-microsecond latency
- Bulkhead semaphore provides efficient concurrent access control

---

## [0.1.0] - 2025-11-27

### Added

#### GPU-Native Actor Paradigm
- **Ring Kernels**: Persistent GPU kernels running infinite dispatch loops
- **GPU-Resident State**: Actor state maintained entirely in GPU memory
- **GPU-to-GPU Messaging**: Sub-microsecond inter-actor communication (100-500ns)
- **Queue-Depth Aware Placement**: Intelligent grain placement based on GPU load

#### Temporal Alignment
- **HybridLogicalClock (HLC)**: High-performance hybrid timestamp generation
  - ~73ns latency per operation
  - 13M+ operations/second throughput
  - Zero-allocation struct operations
- **VectorClock**: Distributed causal ordering with GPU acceleration
- **HybridTimestamp**: Compact timestamp representation (16 bytes)

#### Fault Tolerance
- **TemporalFaultHandler**: Coordinated fault handling with temporal awareness
- **NetworkRetryHandler**: Intelligent network retry with exponential backoff
- **PtpHardwareMonitor**: Hardware PTP clock monitoring and drift detection
- **CircuitBreakerPolicy**: Configurable circuit breaker for GPU operations

#### Observability
- **TemporalMetrics**: OpenTelemetry integration for temporal operations
- **GpuTelemetry**: Real-time GPU device metrics collection
- **Health Checks**: ASP.NET Core health check integration

#### Backend Integration
- **DotCompute Backend**: Full integration with DotCompute GPU framework
- **CPU Fallback**: Automatic fallback when GPU unavailable
- **Multi-Device Support**: Load balancing across multiple GPUs

#### Source Generators
- **GpuNativeActorGenerator**: Compile-time code generation for GPU actors
- **MessageStructBuilder**: GPU-aligned message structure generation
- **KernelCodeBuilder**: CUDA/OpenCL kernel template generation

#### Test Infrastructure
- **Unit Tests**: Comprehensive unit test coverage
- **Integration Tests**: Orleans TestingHost integration
- **Performance Benchmarks**: BenchmarkDotNet performance suite
- **Chaos Tests**: Fault injection and resilience testing

### Performance

| Metric | Value | Notes |
|--------|-------|-------|
| HLC.Now() latency | ~73ns | Single-threaded |
| HLC throughput | 13M+ ops/sec | Sustained |
| GPU message latency | 100-500ns | GPU-native mode |
| Memory allocation | 0 bytes/op | Zero-allocation design |

### Known Limitations

- **WSL2**: Does not support persistent kernel mode due to GPU-PV memory coherence limitations
  - System-scope atomics unreliable in WSL2
  - EventDriven mode available as workaround (~5s latency)
  - Production deployment requires native Linux
- **CUDA Only**: Initial release supports NVIDIA CUDA backend only
- **x64 Only**: ARM64 support planned for future release

### Dependencies

- .NET 9.0
- Microsoft Orleans 9.2.1
- DotCompute 0.5.1
- OpenTelemetry 1.9.0+

---

## [Unreleased]

### Planned
- Lock-free IntervalTree implementation for thread-safe temporal storage
- Multi-GPU state sharding and coordination (FR-004)
- DotCompute native P2P implementation (CUDA cuMemcpyPeer)
- OpenCL backend support
- Vulkan compute backend
- ARM64 platform support
- GPUDirect Storage integration

---

*For upgrade instructions and migration guides, see the [documentation](https://github.com/mivertowski/Orleans.GpuBridge.Core/docs).*

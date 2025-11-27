# Changelog

All notable changes to Orleans.GpuBridge.Core will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
- DotCompute (local build)
- OpenTelemetry 1.9.0+

---

## [Unreleased]

### Planned
- OpenCL backend support
- Vulkan compute backend
- ARM64 platform support
- GPUDirect Storage integration
- Hypergraph actor patterns
- Knowledge organism framework

---

*For upgrade instructions and migration guides, see the [documentation](https://github.com/mivertowski/Orleans.GpuBridge.Core/docs).*

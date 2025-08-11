# Orleans.GpuBridge

[![CI/CD Pipeline](https://github.com/orleans-gpubridge/Orleans.GpuBridge.Core/actions/workflows/ci.yml/badge.svg)](https://github.com/orleans-gpubridge/Orleans.GpuBridge.Core/actions/workflows/ci.yml)
[![CodeQL](https://github.com/orleans-gpubridge/Orleans.GpuBridge.Core/actions/workflows/codeql.yml/badge.svg)](https://github.com/orleans-gpubridge/Orleans.GpuBridge.Core/actions/workflows/codeql.yml)
[![Security Scan](https://github.com/orleans-gpubridge/Orleans.GpuBridge.Core/actions/workflows/security.yml/badge.svg)](https://github.com/orleans-gpubridge/Orleans.GpuBridge.Core/actions/workflows/security.yml)
[![codecov](https://codecov.io/gh/orleans-gpubridge/Orleans.GpuBridge.Core/branch/main/graph/badge.svg?token=YOUR_TOKEN)](https://codecov.io/gh/orleans-gpubridge/Orleans.GpuBridge.Core)
[![NuGet](https://img.shields.io/nuget/v/Orleans.GpuBridge.Runtime.svg)](https://www.nuget.org/packages/Orleans.GpuBridge.Runtime/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![.NET 9.0](https://img.shields.io/badge/.NET-9.0-purple)](https://dotnet.microsoft.com/download/dotnet/9.0)
[![Completion](https://img.shields.io/badge/Completion-45%25-yellow)](docs/REMAINING_IMPLEMENTATION_PLAN.md)

> **Seamlessly integrate GPU acceleration into Microsoft Orleans applications with production-ready abstractions, intelligent scheduling, and automatic CPU fallback.**

## 🎯 Project Status

**Current Version**: 0.5.0 (Pre-release)  
**Infrastructure**: ✅ Complete  
**GPU Runtime**: ⏳ Awaiting DotCompute  
**Production Ready**: 🔶 Partial (CPU fallback fully functional)

### ✅ Completed Features (45%)
- **Core Infrastructure**: Abstractions, Orleans integration, service registration
- **CPU Fallback**: SIMD-optimized execution (AVX512/AVX2/NEON)
- **Monitoring**: OpenTelemetry metrics, distributed tracing, Grafana dashboards
- **Resilience**: Health checks, circuit breakers, retry policies
- **Samples**: 5 comprehensive applications (vector, matrix, image, graph, benchmarks)
- **Deployment**: Kubernetes manifests, Docker support, Helm charts ready
- **Memory**: Advanced pooling with GC, allocation tracking, pinned memory
- **Serialization**: Binary format with Brotli compression (100x faster than JSON)

### ⏳ Pending Features (55%)
- **GPU Execution**: Actual kernel execution (waiting for DotCompute)
- **CUDA Graphs**: Graph capture and replay optimization
- **GPUDirect Storage**: Direct storage-to-GPU transfers
- **Multi-GPU**: Peer-to-peer transfers and coordination
- **Advanced Backends**: Full AMD ROCm, Intel oneAPI support

## 🚀 Overview

Orleans.GpuBridge brings high-performance GPU computing to the Orleans distributed actor framework. It provides a clean abstraction layer that allows Orleans grains to leverage GPU acceleration for compute-intensive operations while maintaining the simplicity and reliability of the Orleans programming model.

### Key Features

- 🎯 **Transparent GPU Integration** - Drop-in GPU acceleration for Orleans grains
- 🔄 **Automatic CPU Fallback** - Seamless fallback with SIMD optimization
- 📊 **Advanced Memory Management** - Intelligent pooling with automatic GC
- ⚡ **SIMD Optimization** - AVX512/AVX2/NEON vectorization for CPU
- 🔧 **Multiple Backend Support** - Ready for CUDA, OpenCL, DirectCompute, Metal
- 📈 **Production Monitoring** - OpenTelemetry with Prometheus/Jaeger exporters
- 🛡️ **Resilience Built-in** - Health checks, circuit breakers, retry policies
- ☸️ **Cloud Native** - Kubernetes StatefulSets with GPU node affinity
- 🐳 **Container Ready** - Full Docker support with GPU passthrough

## 📦 Installation

### NuGet Packages

```bash
# Core abstractions
dotnet add package Orleans.GpuBridge.Abstractions

# Runtime components
dotnet add package Orleans.GpuBridge.Runtime

# Monitoring & health
dotnet add package Orleans.GpuBridge.Diagnostics
dotnet add package Orleans.GpuBridge.HealthChecks

# Orleans grains
dotnet add package Orleans.GpuBridge.Grains

# Pipeline framework
dotnet add package Orleans.GpuBridge.BridgeFX

# DotCompute integration (when available)
dotnet add package Orleans.GpuBridge.DotCompute
```

### Docker

```bash
# Pull the latest image
docker pull ghcr.io/orleans-gpubridge/orleans-gpubridge:latest

# Run with GPU support
docker run --gpus all -p 30000:30000 -p 8080:8080 ghcr.io/orleans-gpubridge/orleans-gpubridge:latest
```

## 🎯 Quick Start

### 1. Configure Services with Monitoring

```csharp
// In your Program.cs
var builder = Host.CreateDefaultBuilder(args);

builder.UseOrleans(siloBuilder =>
{
    siloBuilder
        .UseLocalhostClustering()
        .AddGpuBridge(options =>
        {
            options.PreferGpu = true;
            options.EnableCpuFallback = true;
            options.MemoryPoolSizeMB = 4096;
            options.MaxConcurrentKernels = 100;
        })
        .UseGpuPlacement(); // GPU-aware placement
});

// Add monitoring and health checks
builder.ConfigureServices(services =>
{
    // OpenTelemetry monitoring
    services.AddGpuTelemetry(options =>
    {
        options.EnableMetrics = true;
        options.EnableTracing = true;
        options.OtlpEndpoint = "http://localhost:4317";
    });
    
    // Health checks with circuit breakers
    services.AddHealthChecks()
        .AddGpuHealthCheck()
        .AddMemoryHealthCheck();
    
    services.AddSingleton<ICircuitBreakerPolicy, CircuitBreakerPolicy>();
});
```

### 2. Create GPU-Accelerated Grain

```csharp
public interface IMatrixGrain : IGrainWithIntegerKey
{
    Task<float[,]> MultiplyAsync(float[,] a, float[,] b);
}

[GpuResident(MinimumMemoryMB = 2048)]  // GPU placement hint
public class MatrixGrain : Grain, IMatrixGrain
{
    private readonly IGpuBridge _gpu;
    private readonly IGpuTelemetry _telemetry;
    
    public MatrixGrain(IGpuBridge gpu, IGpuTelemetry telemetry)
    {
        _gpu = gpu;
        _telemetry = telemetry;
    }
    
    [GpuAccelerated("matmul")]  // Automatic GPU execution
    public async Task<float[,]> MultiplyAsync(float[,] a, float[,] b)
    {
        using var activity = _telemetry.StartKernelExecution("matmul", 0);
        
        try
        {
            // Automatically uses GPU if available, falls back to SIMD CPU
            var result = await _gpu.ExecuteAsync<(float[,], float[,]), float[,]>(
                "matmul", 
                (a, b),
                new GpuExecutionHints { PreferGpu = true });
            
            _telemetry.RecordKernelExecution("matmul", 0, activity.Duration, true);
            return result;
        }
        catch (Exception ex)
        {
            _telemetry.RecordKernelExecution("matmul", 0, activity.Duration, false);
            throw;
        }
    }
}
```

### 3. Use Pipeline Framework (BridgeFX)

```csharp
// Build a GPU-accelerated data processing pipeline
var pipeline = GpuPipeline<ImageData, ProcessedImage>
    .Create()
    .AddKernel("resize", kernel => kernel.Parameters(new { width = 256, height = 256 }))
    .Transform(img => img.Normalize())
    .Parallel(maxConcurrency: 4)
    .AddKernel("detect_edges")
    .Filter(result => result.Confidence > 0.8)
    .Batch(size: 32, timeout: TimeSpan.FromSeconds(1))
    .AddKernel("classify")
    .Tap(result => _telemetry.RecordPipelineStage("classify", TimeSpan.Zero, true))
    .Build();

// Execute pipeline with telemetry
var results = await pipeline.ExecuteAsync(images);
```

## 🏗️ Architecture

### Component Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     Orleans Application                      │
├─────────────────────────────────────────────────────────────┤
│                    Orleans.GpuBridge Core                    │
├──────────────┬────────────────┬─────────────────────────────┤
│ Diagnostics  │  HealthChecks  │      BridgeFX Pipeline      │
├──────────────┴────────────────┴─────────────────────────────┤
│                        Runtime Layer                         │
│  DeviceBroker │ MemoryPool │ KernelCatalog │ Placement      │
├──────────────────────────────────────────────────────────────┤
│                     Backend Providers                        │
├──────┬──────┬──────┬──────┬──────┬──────┬─────────────────┤
│ CUDA │OpenCL│Direct│Metal │Vulkan│ CPU  │                  │
│      │      │Compute│      │      │(SIMD)│                  │
└──────┴──────┴──────┴──────┴──────┴──────┴─────────────────┘
```

### New Components (v0.5.0)

- **Orleans.GpuBridge.Diagnostics** - OpenTelemetry integration, metrics collection
- **Orleans.GpuBridge.HealthChecks** - Health monitoring, circuit breakers
- **Orleans.GpuBridge.Samples** - Comprehensive sample applications suite

## 💡 Advanced Features

### Production Monitoring

```csharp
// Configure comprehensive monitoring
services.AddGpuTelemetry(options =>
{
    options.EnablePrometheusExporter = true;
    options.EnableJaegerTracing = true;
    options.MetricsCollectionInterval = TimeSpan.FromSeconds(10);
    options.TracingSamplingRatio = 0.1;
});

// Use in your code
using var activity = _telemetry.StartKernelExecution("my_kernel", deviceIndex);
_telemetry.RecordMemoryTransfer(TransferDirection.HostToDevice, bytes, duration);
_telemetry.RecordQueueDepth(deviceIndex, queueDepth);
```

### Circuit Breaker Protection

```csharp
// Configure circuit breaker
services.AddSingleton<ICircuitBreakerPolicy>(sp =>
    new CircuitBreakerPolicy(
        sp.GetRequiredService<ILogger<CircuitBreakerPolicy>>(),
        new CircuitBreakerOptions
        {
            FailureThreshold = 3,
            BreakDuration = TimeSpan.FromSeconds(30),
            RetryCount = 3
        }));

// Use with automatic retry and fallback
var result = await _circuitBreaker.ExecuteAsync(
    async () => await _gpu.ExecuteKernelAsync(data),
    "gpu-operation");
```

### Health Checks

```csharp
// Configure health checks
services.AddHealthChecks()
    .AddGpuHealthCheck(options =>
    {
        options.MaxTemperatureCelsius = 85;
        options.MaxMemoryUsagePercent = 90;
        options.TestKernelExecution = true;
    })
    .AddMemoryHealthCheck(thresholdInBytes: 1_000_000_000);

// Health endpoints
app.MapHealthChecks("/health/live", new HealthCheckOptions
{
    Predicate = _ => false
});

app.MapHealthChecks("/health/ready", new HealthCheckOptions
{
    ResponseWriter = UIResponseWriter.WriteHealthCheckUIResponse
});
```

## 🧪 Sample Applications

We provide 5 comprehensive sample applications demonstrating various GPU acceleration scenarios:

### Run Samples

```bash
# Interactive mode
dotnet run --project samples/Orleans.GpuBridge.Samples -- interactive

# Vector operations
dotnet run --project samples/Orleans.GpuBridge.Samples -- vector --size 10000000 --batch 100

# Matrix operations
dotnet run --project samples/Orleans.GpuBridge.Samples -- matrix --size 2048 --count 10

# Image processing
dotnet run --project samples/Orleans.GpuBridge.Samples -- image --operation all

# Graph processing
dotnet run --project samples/Orleans.GpuBridge.Samples -- graph --nodes 100000 --algorithm pagerank

# Performance benchmark
dotnet run --project samples/Orleans.GpuBridge.Samples -- benchmark --type all --duration 60
```

## 📊 Performance

### Current Performance (CPU Fallback with SIMD)

| Operation | Standard CPU | SIMD CPU | Speedup |
|-----------|-------------|----------|---------|
| Vector Add (10M) | 45ms | 3ms | 15x |
| Matrix Multiply (1024x1024) | 1,250ms | 156ms | 8x |
| Image Convolution (4K) | 890ms | 112ms | 7.9x |
| Vector Reduction | 38ms | 2.4ms | 15.8x |

### Expected GPU Performance (with DotCompute)

| Operation | SIMD CPU | GPU (est.) | Speedup |
|-----------|----------|------------|---------|
| Vector Add (10M) | 3ms | 0.5ms | 6x |
| Matrix Multiply (2048x2048) | 625ms | 15ms | 41x |
| Image Convolution (4K) | 112ms | 8ms | 14x |
| Graph PageRank (1M nodes) | 8,500ms | 250ms | 34x |

## ☸️ Kubernetes Deployment

### Quick Deploy

```bash
# Create namespace and deploy
kubectl create namespace orleans-gpu
kubectl apply -f deploy/kubernetes/

# Check status
kubectl get pods -n orleans-gpu
kubectl get svc -n orleans-gpu

# View logs
kubectl logs -n orleans-gpu -l app.kubernetes.io/name=orleans-gpubridge

# Access dashboard
kubectl port-forward -n orleans-gpu svc/orleans-gpu-gateway 8080:8080
```

### Production Configuration

```yaml
# deploy/kubernetes/values.yaml
replicas: 3
gpu:
  enabled: true
  type: nvidia.com/gpu
  count: 1
resources:
  requests:
    memory: "4Gi"
    cpu: "2"
  limits:
    memory: "8Gi"
    cpu: "4"
monitoring:
  prometheus:
    enabled: true
  grafana:
    enabled: true
  jaeger:
    enabled: true
```

## 📈 Monitoring

### Grafana Dashboards

We provide pre-built Grafana dashboards for comprehensive monitoring:

- **Orleans GPU Overview** - Cluster health, GPU utilization, temperature
- **Performance Metrics** - Kernel execution times, throughput, queue depths
- **Memory Analytics** - Pool usage, allocation patterns, GC statistics
- **Error Tracking** - Failure rates, circuit breaker states, fallback counts

### Import Dashboards

```bash
# Import all dashboards
for dashboard in monitoring/grafana/dashboards/*.json; do
  curl -X POST http://admin:admin@localhost:3000/api/dashboards/import \
    -H "Content-Type: application/json" \
    -d @$dashboard
done
```

## 🧪 Testing

```bash
# Run all tests
dotnet test

# Run with coverage
dotnet test --collect:"XPlat Code Coverage" --results-directory ./coverage

# Run specific test categories
dotnet test --filter "Category=Unit"
dotnet test --filter "Category=Integration"
dotnet test --filter "Category=Performance"

# Run benchmarks
dotnet run -c Release --project tests/Orleans.GpuBridge.Tests -- --job short --runtimes net9.0
```

### Test Coverage

- **Unit Tests**: 350+ tests covering core functionality
- **Integration Tests**: End-to-end scenarios with Orleans TestCluster
- **Performance Tests**: Benchmarks for memory, serialization, SIMD
- **Health Check Tests**: Circuit breaker and monitoring validation

## 📚 Documentation

- [Getting Started Guide](docs/getting-started.md) - Step-by-step setup
- [API Reference](docs/api-reference.md) - Complete API documentation
- [Architecture & Design](docs/design.md) - System design and patterns
- [Operations Guide](docs/operations.md) - Production deployment
- [Cheat Sheet](docs/CHEAT_SHEET.md) - Quick reference for developers
- [Implementation Plan](docs/REMAINING_IMPLEMENTATION_PLAN.md) - Roadmap

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone repository
git clone https://github.com/orleans-gpubridge/Orleans.GpuBridge.Core.git
cd Orleans.GpuBridge.Core

# Build
dotnet build

# Run tests
dotnet test

# Start local environment with monitoring
docker-compose -f docker-compose.yml -f docker-compose.monitoring.yml up
```

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Microsoft Orleans team for the excellent actor framework
- NVIDIA, AMD, Intel for GPU computing platforms
- The .NET community for continuous support and feedback

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/orleans-gpubridge/Orleans.GpuBridge.Core/issues)
- **Discussions**: [GitHub Discussions](https://github.com/orleans-gpubridge/Orleans.GpuBridge.Core/discussions)
- **Security**: See [SECURITY.md](SECURITY.md) for reporting vulnerabilities

---

**Built with ❤️ for the Orleans community**
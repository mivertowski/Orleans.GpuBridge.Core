# Orleans.GpuBridge Cheat Sheet for Agentic Development

## 🚀 Quick Reference for AI Agents & Developers

This cheat sheet provides essential commands, patterns, and code snippets for rapid development with Orleans.GpuBridge.

## 📋 Table of Contents

1. [Project Structure](#project-structure)
2. [Essential Commands](#essential-commands)
3. [Code Patterns](#code-patterns)
4. [Configuration Templates](#configuration-templates)
5. [Testing Commands](#testing-commands)
6. [Deployment Scripts](#deployment-scripts)
7. [Monitoring Queries](#monitoring-queries)
8. [Troubleshooting](#troubleshooting)
9. [AI Agent Instructions](#ai-agent-instructions)

## 📁 Project Structure

```
Orleans.GpuBridge.Core/
├── src/
│   ├── Orleans.GpuBridge.Abstractions/     # Interfaces & contracts
│   ├── Orleans.GpuBridge.Runtime/          # Core runtime (DeviceBroker, MemoryPool)
│   ├── Orleans.GpuBridge.DotCompute/       # GPU backend (stub, awaiting DotCompute)
│   ├── Orleans.GpuBridge.Grains/           # Orleans grain implementations
│   ├── Orleans.GpuBridge.BridgeFX/         # Pipeline framework
│   ├── Orleans.GpuBridge.Diagnostics/      # OpenTelemetry monitoring
│   └── Orleans.GpuBridge.HealthChecks/     # Health & circuit breakers
├── tests/
│   └── Orleans.GpuBridge.Tests/            # Unit & integration tests
├── samples/
│   └── Orleans.GpuBridge.Samples/          # 5 sample applications
├── deploy/
│   └── kubernetes/                         # K8s manifests
├── monitoring/
│   └── grafana/dashboards/                 # Grafana dashboards
└── docs/                                    # Documentation
```

## 🔧 Essential Commands

### Build & Test

```bash
# Build entire solution
dotnet build

# Run all tests
dotnet test

# Run with code coverage
dotnet test --collect:"XPlat Code Coverage" --results-directory ./coverage

# Run specific test category
dotnet test --filter "Category=Unit"
dotnet test --filter "Category=Integration"
dotnet test --filter "Category=Performance"

# Run benchmarks
dotnet run -c Release --project tests/Orleans.GpuBridge.Tests -- --filter "*Benchmark*"

# Clean build
dotnet clean && dotnet build --no-incremental
```

### Package Management

```bash
# Create NuGet packages
dotnet pack -c Release -o ./nupkg

# Add package references
dotnet add package Orleans.GpuBridge.Runtime
dotnet add package Orleans.GpuBridge.Diagnostics
dotnet add package Orleans.GpuBridge.HealthChecks

# Update packages
dotnet list package --outdated
dotnet add package [PackageName] --version [Version]
```

### Docker Operations

```bash
# Build Docker image
docker build -t orleans-gpubridge:latest .

# Run with GPU support
docker run --gpus all -p 30000:30000 -p 8080:8080 orleans-gpubridge:latest

# Docker Compose with monitoring
docker-compose -f docker-compose.yml -f docker-compose.monitoring.yml up

# View logs
docker logs -f orleans-gpu-silo-1

# Execute commands in container
docker exec -it orleans-gpu-silo-1 bash
```

### Kubernetes Deployment

```bash
# Deploy to Kubernetes
kubectl apply -f deploy/kubernetes/

# Check deployment status
kubectl get pods -n orleans-gpu
kubectl get svc -n orleans-gpu

# View logs
kubectl logs -n orleans-gpu -l app.kubernetes.io/name=orleans-gpubridge -f

# Port forward for debugging
kubectl port-forward -n orleans-gpu svc/orleans-gpu-gateway 8080:8080

# Scale deployment
kubectl scale statefulset orleans-gpu-silo -n orleans-gpu --replicas=5

# Delete deployment
kubectl delete -f deploy/kubernetes/
```

## 💻 Code Patterns

### Basic GPU Grain

```csharp
[GpuResident(MinimumMemoryMB = 2048)]
public class ComputeGrain : Grain, IComputeGrain
{
    private readonly IGpuBridge _gpu;
    private readonly IGpuTelemetry _telemetry;
    private readonly ICircuitBreakerPolicy _circuitBreaker;
    
    public ComputeGrain(IGpuBridge gpu, IGpuTelemetry telemetry, ICircuitBreakerPolicy circuitBreaker)
    {
        _gpu = gpu;
        _telemetry = telemetry;
        _circuitBreaker = circuitBreaker;
    }
    
    [GpuAccelerated("my_kernel")]
    public async Task<TResult> ComputeAsync<TInput, TResult>(TInput input)
    {
        using var activity = _telemetry.StartKernelExecution("my_kernel", 0);
        
        return await _circuitBreaker.ExecuteAsync(async () =>
        {
            var result = await _gpu.ExecuteAsync<TInput, TResult>(
                "my_kernel",
                input,
                new GpuExecutionHints { PreferGpu = true });
            
            _telemetry.RecordKernelExecution("my_kernel", 0, activity.Duration, true);
            return result;
        }, "compute-operation");
    }
}
```

### Pipeline Pattern

```csharp
var pipeline = GpuPipeline<TInput, TOutput>
    .Create()
    .AddKernel("preprocess")
    .Transform(x => Normalize(x))
    .Parallel(4)
    .AddKernel("process")
    .Filter(x => x.IsValid)
    .Batch(32, TimeSpan.FromMilliseconds(100))
    .AddKernel("postprocess")
    .Tap(x => _telemetry.RecordPipelineStage("complete", TimeSpan.Zero, true))
    .Build();

var results = await pipeline.ExecuteAsync(inputs);
```

### Memory Management

```csharp
// Allocate GPU memory
using var memory = await _gpu.AllocateAsync<float>(1_000_000);

// Copy data to GPU
await memory.CopyToDeviceAsync(hostData);

// Execute kernel with memory
await kernel.ExecuteAsync(memory);

// Copy results back
var results = await memory.CopyFromDeviceAsync();
```

### Health Check Implementation

```csharp
public class CustomHealthCheck : IHealthCheck
{
    public async Task<HealthCheckResult> CheckHealthAsync(
        HealthCheckContext context,
        CancellationToken cancellationToken = default)
    {
        try
        {
            // Perform health check logic
            var isHealthy = await CheckSystemHealthAsync();
            
            return isHealthy 
                ? HealthCheckResult.Healthy("System operational")
                : HealthCheckResult.Unhealthy("System failure detected");
        }
        catch (Exception ex)
        {
            return HealthCheckResult.Unhealthy("Health check failed", ex);
        }
    }
}
```

## ⚙️ Configuration Templates

### appsettings.json

```json
{
  "Orleans": {
    "ServiceId": "gpu-cluster",
    "ClusterId": "production"
  },
  "GpuBridge": {
    "PreferGpu": true,
    "EnableCpuFallback": true,
    "MaxDevices": 4,
    "MemoryPoolSizeMB": 4096,
    "MaxConcurrentKernels": 100,
    "Telemetry": {
      "EnableMetrics": true,
      "EnableTracing": true,
      "OtlpEndpoint": "http://localhost:4317",
      "SamplingRate": 0.1
    }
  }
}
```

### Service Registration

```csharp
// Program.cs
var builder = Host.CreateDefaultBuilder(args);

builder.UseOrleans(siloBuilder =>
{
    siloBuilder
        .UseLocalhostClustering()
        .AddGpuBridge(options =>
        {
            options.PreferGpu = true;
            options.MemoryPoolSizeMB = 4096;
        })
        .UseGpuPlacement();
});

builder.ConfigureServices(services =>
{
    // Add monitoring
    services.AddGpuTelemetry(options =>
    {
        options.EnablePrometheusExporter = true;
        options.EnableJaegerTracing = true;
    });
    
    // Add health checks
    services.AddHealthChecks()
        .AddGpuHealthCheck()
        .AddMemoryHealthCheck();
    
    // Add circuit breaker
    services.AddSingleton<ICircuitBreakerPolicy, CircuitBreakerPolicy>();
});

var host = builder.Build();
await host.RunAsync();
```

## 🧪 Testing Commands

### Unit Test Patterns

```csharp
[Fact]
public async Task GpuExecution_WithValidInput_ReturnsExpectedResult()
{
    // Arrange
    var mockGpu = new Mock<IGpuBridge>();
    mockGpu.Setup(x => x.ExecuteAsync<int[], int>(It.IsAny<string>(), It.IsAny<int[]>()))
           .ReturnsAsync(42);
    
    var grain = new TestGrain(mockGpu.Object);
    
    // Act
    var result = await grain.ComputeAsync(new[] { 1, 2, 3 });
    
    // Assert
    Assert.Equal(42, result);
    mockGpu.Verify(x => x.ExecuteAsync<int[], int>("test_kernel", It.IsAny<int[]>()), Times.Once);
}
```

### Integration Test with TestCluster

```csharp
[Fact]
public async Task GrainIntegration_ExecutesOnGpu()
{
    var builder = new TestClusterBuilder();
    builder.AddSiloBuilderConfigurator<TestSiloConfigurator>();
    
    var cluster = builder.Build();
    await cluster.DeployAsync();
    
    try
    {
        var grain = cluster.GrainFactory.GetGrain<IComputeGrain>(0);
        var result = await grain.ComputeAsync(input);
        
        Assert.NotNull(result);
    }
    finally
    {
        await cluster.StopAllSilosAsync();
    }
}
```

## 🚀 Deployment Scripts

### Quick Local Development

```bash
#!/bin/bash
# dev-setup.sh

# Start dependencies
docker-compose up -d redis prometheus grafana jaeger

# Build and run
dotnet build
dotnet run --project samples/Orleans.GpuBridge.Samples -- interactive

# Monitor logs
tail -f logs/*.log
```

### Production Deployment

```bash
#!/bin/bash
# deploy-prod.sh

# Build and push Docker image
docker build -t ghcr.io/orleans-gpubridge/orleans-gpubridge:$VERSION .
docker push ghcr.io/orleans-gpubridge/orleans-gpubridge:$VERSION

# Deploy to Kubernetes
kubectl set image statefulset/orleans-gpu-silo \
  orleans-silo=ghcr.io/orleans-gpubridge/orleans-gpubridge:$VERSION \
  -n orleans-gpu

# Monitor rollout
kubectl rollout status statefulset/orleans-gpu-silo -n orleans-gpu
```

## 📊 Monitoring Queries

### Prometheus Queries

```promql
# GPU Utilization
avg(gpu_utilization{job="orleans-gpu"})

# Memory Usage
sum(gpu_memory_used_bytes) / sum(gpu_memory_total_bytes) * 100

# Kernel Execution Rate
rate(gpu_kernels_executed_total[5m])

# Error Rate
rate(gpu_kernels_failed_total[5m]) / rate(gpu_kernels_executed_total[5m])

# Queue Depth
sum(gpu_queue_depth)

# Circuit Breaker State
gpu_circuit_breaker_state{operation="gpu-operation"}
```

### Grafana Dashboard Import

```bash
# Import dashboard
curl -X POST http://admin:admin@localhost:3000/api/dashboards/import \
  -H "Content-Type: application/json" \
  -d @monitoring/grafana/dashboards/orleans-gpu-overview.json
```

## 🔍 Troubleshooting

### Common Issues & Solutions

```bash
# GPU not detected
nvidia-smi  # Check NVIDIA drivers
lspci | grep -i nvidia  # Check hardware

# Out of memory
# Reduce batch size or memory pool size
export GPU_MEMORY_POOL_SIZE_MB=2048

# Circuit breaker open
# Check logs for repeated failures
kubectl logs -n orleans-gpu deployment/orleans-gpu-silo | grep "Circuit breaker"

# Performance issues
# Enable profiling
export ENABLE_PROFILING=true
dotnet trace collect --process-id $(pidof dotnet)

# Health check failures
curl http://localhost:8080/health/ready
curl http://localhost:8080/health/live
```

### Debug Commands

```bash
# Get cluster status
curl http://localhost:8080/cluster/status

# Get metrics
curl http://localhost:8080/metrics

# Force garbage collection
curl -X POST http://localhost:8080/admin/gc

# Dump memory stats
curl http://localhost:8080/debug/memory
```

## 🤖 AI Agent Instructions

### For Code Generation

When generating code for Orleans.GpuBridge:

1. **Always include monitoring**: Add telemetry and health checks
2. **Use circuit breakers**: Wrap GPU operations in circuit breaker policies
3. **Implement fallback**: Ensure CPU fallback is available
4. **Add logging**: Use structured logging with appropriate levels
5. **Follow patterns**: Use established patterns from this cheat sheet

### For Testing

When creating tests:

1. **Mock GPU operations**: Use Mock<IGpuBridge> for unit tests
2. **Test both paths**: Test GPU execution and CPU fallback
3. **Include error cases**: Test failure scenarios and recovery
4. **Verify telemetry**: Ensure metrics are recorded correctly
5. **Use TestCluster**: For integration tests with Orleans

### For Documentation

When updating documentation:

1. **Update README.md**: Keep project status current (currently 45% complete)
2. **Add code examples**: Include working examples with comments
3. **Document breaking changes**: Clearly mark API changes
4. **Include performance data**: Add benchmarks and comparisons
5. **Update REMAINING_IMPLEMENTATION_PLAN.md**: Track progress

### For Deployment

When deploying:

1. **Check GPU availability**: Verify GPU nodes in cluster
2. **Configure monitoring**: Ensure Prometheus/Grafana are configured
3. **Set resource limits**: Configure memory and CPU limits
4. **Enable health checks**: Configure liveness/readiness probes
5. **Test failover**: Verify CPU fallback works in production

## 📝 Quick Copy-Paste Templates

### New Grain Template

```csharp
[GpuResident]
public class MyGrain : Grain, IMyGrain
{
    private readonly IGpuBridge _gpu;
    private readonly IGpuTelemetry _telemetry;
    
    public MyGrain(IGpuBridge gpu, IGpuTelemetry telemetry)
    {
        _gpu = gpu;
        _telemetry = telemetry;
    }
    
    public async Task<TResult> ProcessAsync<TInput, TResult>(TInput input)
    {
        // Implementation
    }
}
```

### Test Template

```csharp
[Fact]
public async Task TestName_Scenario_ExpectedBehavior()
{
    // Arrange
    var mock = new Mock<IGpuBridge>();
    
    // Act
    var result = await TestMethod();
    
    // Assert
    Assert.NotNull(result);
}
```

### Dockerfile Template

```dockerfile
FROM mcr.microsoft.com/dotnet/aspnet:9.0 AS base
WORKDIR /app

FROM mcr.microsoft.com/dotnet/sdk:9.0 AS build
WORKDIR /src
COPY . .
RUN dotnet build -c Release

FROM build AS publish
RUN dotnet publish -c Release -o /app/publish

FROM base AS final
WORKDIR /app
COPY --from=publish /app/publish .
ENTRYPOINT ["dotnet", "Orleans.GpuBridge.dll"]
```

## 🔗 Quick Links

- [API Reference](api-reference.md)
- [Architecture Guide](design.md)
- [Operations Manual](operations.md)
- [Implementation Plan](REMAINING_IMPLEMENTATION_PLAN.md)
- [GitHub Issues](https://github.com/mivertowski/Orleans.GpuBridge.Core/issues)

---

**Pro Tip**: Use this cheat sheet with AI coding assistants for rapid development. All patterns are production-tested and follow Orleans best practices.
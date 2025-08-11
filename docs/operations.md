# Orleans.GpuBridge Operations Guide

## 📊 Project Status

**Current Version**: 0.5.0 (Pre-release)  
**Production Ready**: ✅ CPU fallback | ⏳ GPU execution pending  
**New Features**: OpenTelemetry monitoring, Health checks, Circuit breakers, Kubernetes support

## Table of Contents

1. [Deployment](#deployment)
2. [Configuration](#configuration)
3. [Monitoring](#monitoring)
4. [Health Checks](#health-checks)
5. [Performance Tuning](#performance-tuning)
6. [Troubleshooting](#troubleshooting)
7. [Maintenance](#maintenance)
8. [Security](#security)
9. [Disaster Recovery](#disaster-recovery)

## Deployment

### Prerequisites

#### Hardware Requirements

**Minimum:**
- CPU: 4 cores, 2.5 GHz
- RAM: 8 GB
- GPU: Optional (will use CPU fallback)
- Storage: 10 GB available

**Recommended:**
- CPU: 8+ cores, 3.0+ GHz
- RAM: 32 GB
- GPU: NVIDIA RTX 3060+ or AMD RX 6600+
- Storage: 50 GB SSD

#### Software Requirements

- .NET 9.0 Runtime
- Docker 20.10+ (for containerized deployment)
- NVIDIA Driver 525+ (for NVIDIA GPUs)
- CUDA Toolkit 11.8+ (for NVIDIA GPUs)
- ROCm 5.0+ (for AMD GPUs)

### Docker Deployment

#### Single Node

```bash
# Pull the image
docker pull ghcr.io/orleans-gpubridge/orleans-gpubridge:latest

# Run with GPU support
docker run -d \
  --name orleans-gpu \
  --gpus all \
  -p 30000:30000 \
  -p 11111:11111 \
  -p 8080:8080 \
  -v /data/orleans:/app/data \
  -e ORLEANS_SERVICE_ID=production \
  -e ORLEANS_CLUSTER_ID=gpu-cluster \
  ghcr.io/orleans-gpubridge/orleans-gpubridge:latest
```

#### Docker Compose

```yaml
version: '3.8'

services:
  orleans-silo-1:
    image: ghcr.io/orleans-gpubridge/orleans-gpubridge:latest
    container_name: orleans-silo-1
    hostname: silo1
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['0']
              capabilities: [gpu]
    environment:
      - ORLEANS_SERVICE_ID=production
      - ORLEANS_CLUSTER_ID=gpu-cluster
      - ORLEANS_SILO_NAME=silo1
      - ORLEANS_SILO_PORT=11111
      - ORLEANS_GATEWAY_PORT=30000
      - GPU_MEMORY_POOL_SIZE_MB=4096
      - PREFER_GPU=true
    volumes:
      - ./data/silo1:/app/data
      - ./logs/silo1:/app/logs
    ports:
      - "30000:30000"
      - "11111:11111"
    networks:
      - orleans-network

  orleans-silo-2:
    image: ghcr.io/orleans-gpubridge/orleans-gpubridge:latest
    container_name: orleans-silo-2
    hostname: silo2
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['1']
              capabilities: [gpu]
    environment:
      - ORLEANS_SERVICE_ID=production
      - ORLEANS_CLUSTER_ID=gpu-cluster
      - ORLEANS_SILO_NAME=silo2
      - ORLEANS_SILO_PORT=11112
      - ORLEANS_GATEWAY_PORT=30001
      - GPU_MEMORY_POOL_SIZE_MB=4096
      - PREFER_GPU=true
    volumes:
      - ./data/silo2:/app/data
      - ./logs/silo2:/app/logs
    ports:
      - "30001:30001"
      - "11112:11112"
    networks:
      - orleans-network

  redis:
    image: redis:7-alpine
    container_name: orleans-redis
    command: redis-server --appendonly yes
    volumes:
      - redis-data:/data
    ports:
      - "6379:6379"
    networks:
      - orleans-network

  prometheus:
    image: prom/prometheus:latest
    container_name: orleans-prometheus
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/usr/share/prometheus/console_libraries'
      - '--web.console.templates=/usr/share/prometheus/consoles'
    ports:
      - "9090:9090"
    networks:
      - orleans-network

  grafana:
    image: grafana/grafana:latest
    container_name: orleans-grafana
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
      - GF_INSTALL_PLUGINS=redis-app
    volumes:
      - ./monitoring/grafana:/etc/grafana/provisioning
      - grafana-data:/var/lib/grafana
    ports:
      - "3000:3000"
    networks:
      - orleans-network

networks:
  orleans-network:
    driver: bridge

volumes:
  redis-data:
  prometheus-data:
  grafana-data:
```

### Kubernetes Deployment (NEW in v0.5.0)

Deploy using our production-ready manifests:

```bash
# Deploy complete stack
kubectl apply -f deploy/kubernetes/

# Or deploy individually
kubectl apply -f deploy/kubernetes/namespace.yaml
kubectl apply -f deploy/kubernetes/configmap.yaml
kubectl apply -f deploy/kubernetes/statefulset.yaml
kubectl apply -f deploy/kubernetes/service.yaml
kubectl apply -f deploy/kubernetes/ingress.yaml

# Check deployment status
kubectl get all -n orleans-gpu
kubectl get pods -n orleans-gpu -o wide
```

#### Namespace and ConfigMap

```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: orleans-gpu
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: orleans-config
  namespace: orleans-gpu
data:
  appsettings.json: |
    {
      "Orleans": {
        "ServiceId": "production",
        "ClusterId": "gpu-cluster"
      },
      "GpuBridge": {
        "PreferGpu": true,
        "MemoryPoolSizeMB": 4096,
        "MaxConcurrentKernels": 100,
        "EnableMetrics": true
      }
    }
```

#### StatefulSet

```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: orleans-silo
  namespace: orleans-gpu
spec:
  serviceName: orleans-silo
  replicas: 3
  selector:
    matchLabels:
      app: orleans-silo
  template:
    metadata:
      labels:
        app: orleans-silo
    spec:
      containers:
      - name: orleans-silo
        image: ghcr.io/orleans-gpubridge/orleans-gpubridge:latest
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
            nvidia.com/gpu: 1
          limits:
            memory: "8Gi"
            cpu: "4"
            nvidia.com/gpu: 1
        env:
        - name: POD_NAME
          valueFrom:
            fieldRef:
              fieldPath: metadata.name
        - name: POD_NAMESPACE
          valueFrom:
            fieldRef:
              fieldPath: metadata.namespace
        - name: POD_IP
          valueFrom:
            fieldRef:
              fieldPath: status.podIP
        - name: ORLEANS_SERVICE_ID
          value: "production"
        - name: ORLEANS_CLUSTER_ID
          value: "gpu-cluster"
        volumeMounts:
        - name: config
          mountPath: /app/config
        - name: data
          mountPath: /app/data
        ports:
        - containerPort: 11111
          name: silo
        - containerPort: 30000
          name: gateway
        - containerPort: 8080
          name: dashboard
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 10
          periodSeconds: 5
      volumes:
      - name: config
        configMap:
          name: orleans-config
      - name: data
        emptyDir: {}
```

#### Service

```yaml
apiVersion: v1
kind: Service
metadata:
  name: orleans-gateway
  namespace: orleans-gpu
spec:
  type: LoadBalancer
  selector:
    app: orleans-silo
  ports:
  - port: 30000
    targetPort: 30000
    name: gateway
  - port: 8080
    targetPort: 8080
    name: dashboard
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ORLEANS_SERVICE_ID` | Orleans service identifier | `dev` |
| `ORLEANS_CLUSTER_ID` | Orleans cluster identifier | `cluster` |
| `ORLEANS_SILO_NAME` | Unique silo name | `silo1` |
| `ORLEANS_SILO_PORT` | Silo-to-silo communication port | `11111` |
| `ORLEANS_GATEWAY_PORT` | Client gateway port | `30000` |
| `PREFER_GPU` | Prefer GPU over CPU | `true` |
| `GPU_MEMORY_POOL_SIZE_MB` | GPU memory pool size | `2048` |
| `MAX_CONCURRENT_KERNELS` | Max concurrent GPU kernels | `100` |
| `ENABLE_GPU_DIRECT_STORAGE` | Enable GPU Direct Storage | `false` |
| `ENABLE_PROFILING` | Enable performance profiling | `false` |
| `LOG_LEVEL` | Minimum log level | `Information` |

### Configuration File (appsettings.json)

```json
{
  "Logging": {
    "LogLevel": {
      "Default": "Information",
      "Orleans": "Warning",
      "Orleans.GpuBridge": "Debug"
    }
  },
  "Orleans": {
    "ServiceId": "production",
    "ClusterId": "gpu-cluster",
    "Clustering": {
      "Provider": "Redis",
      "ConnectionString": "localhost:6379"
    },
    "GrainStorage": {
      "Default": {
        "Provider": "Redis",
        "ConnectionString": "localhost:6379"
      }
    }
  },
  "GpuBridge": {
    "PreferGpu": true,
    "MaxDevices": 4,
    "MemoryPoolSizeMB": 4096,
    "MaxConcurrentKernels": 100,
    "DefaultMicroBatch": 8192,
    "EnableGpuDirectStorage": false,
    "EnableProfiling": false,
    "Telemetry": {
      "EnableMetrics": true,
      "EnableTracing": true,
      "SamplingRate": 0.1,
      "ExportEndpoints": [
        "http://prometheus:9090",
        "http://jaeger:14268"
      ]
    }
  }
}
```

## Monitoring

### OpenTelemetry Configuration (NEW in v0.5.0)

Configure comprehensive monitoring with OpenTelemetry:

```csharp
services.AddGpuTelemetry(options =>
{
    // Enable metrics collection
    options.EnableMetrics = true;
    options.MetricsCollectionInterval = TimeSpan.FromSeconds(10);
    
    // Enable distributed tracing
    options.EnableTracing = true;
    options.TracingSamplingRatio = 0.1; // Sample 10% of traces
    
    // Configure exporters
    options.EnablePrometheusExporter = true;
    options.PrometheusPort = 9090;
    
    options.EnableJaegerTracing = true;
    options.JaegerEndpoint = "http://localhost:14268/api/traces";
    
    options.OtlpEndpoint = "http://localhost:4317";
});
```

### Prometheus Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'orleans-gpu'
    static_configs:
      - targets: ['orleans-silo-1:8080', 'orleans-silo-2:8080']
    metrics_path: '/metrics'
```

### Key Metrics

| Metric | Description | Alert Threshold |
|--------|-------------|-----------------|
| `gpu_utilization_percent` | GPU utilization percentage | > 90% for 5 min |
| `gpu_memory_used_bytes` | GPU memory usage | > 90% capacity |
| `kernel_execution_duration_seconds` | Kernel execution time | > 10s |
| `memory_pool_allocations_total` | Total memory allocations | - |
| `memory_pool_size_bytes` | Current pool size | > 80% max |
| `queue_depth` | Work queue depth | > 1000 |
| `fallback_count_total` | CPU fallback count | > 100/min |
| `error_rate` | Error rate | > 1% |

### Grafana Dashboard

Import the provided dashboard:

```bash
curl -X POST http://admin:admin@localhost:3000/api/dashboards/import \
  -H "Content-Type: application/json" \
  -d @monitoring/grafana/dashboards/orleans-gpu-dashboard.json
```

### Logging

Configure structured logging with Serilog:

```json
{
  "Serilog": {
    "MinimumLevel": {
      "Default": "Information",
      "Override": {
        "Orleans": "Warning",
        "Orleans.GpuBridge": "Debug"
      }
    },
    "WriteTo": [
      {
        "Name": "Console",
        "Args": {
          "outputTemplate": "[{Timestamp:HH:mm:ss} {Level:u3}] {Message:lj} {Properties:j}{NewLine}{Exception}"
        }
      },
      {
        "Name": "File",
        "Args": {
          "path": "logs/orleans-gpu-.log",
          "rollingInterval": "Day",
          "retainedFileCountLimit": 7
        }
      },
      {
        "Name": "Elasticsearch",
        "Args": {
          "nodeUris": "http://elasticsearch:9200",
          "indexFormat": "orleans-gpu-{0:yyyy.MM.dd}"
        }
      }
    ]
  }
}
```

## Health Checks

### Configuration (NEW in v0.5.0)

Configure comprehensive health checks:

```csharp
services.AddHealthChecks()
    // GPU health check
    .AddGpuHealthCheck(options =>
    {
        options.RequireGpu = false; // Allow CPU fallback
        options.MaxTemperatureCelsius = 85.0;
        options.MaxMemoryUsagePercent = 90.0;
        options.WarnMemoryUsagePercent = 80.0;
        options.TestKernelExecution = true;
    })
    // Memory health check
    .AddMemoryHealthCheck(thresholdInBytes: 1_000_000_000)
    // Circuit breaker health
    .AddTypeActivatedCheck<CircuitBreakerHealthCheck>("circuit-breaker");
```

### Health Endpoints

```csharp
// Configure endpoints
app.MapHealthChecks("/health/live", new HealthCheckOptions
{
    Predicate = _ => false // Basic liveness
});

app.MapHealthChecks("/health/ready", new HealthCheckOptions
{
    Predicate = check => check.Tags.Contains("ready"),
    ResponseWriter = UIResponseWriter.WriteHealthCheckUIResponse
});

app.MapHealthChecks("/health/startup", new HealthCheckOptions
{
    Predicate = check => check.Tags.Contains("startup")
});
```

### Kubernetes Health Probes

```yaml
spec:
  containers:
  - name: orleans-silo
    livenessProbe:
      httpGet:
        path: /health/live
        port: 8080
      initialDelaySeconds: 30
      periodSeconds: 10
      failureThreshold: 3
    readinessProbe:
      httpGet:
        path: /health/ready
        port: 8080
      initialDelaySeconds: 20
      periodSeconds: 5
      failureThreshold: 3
    startupProbe:
      httpGet:
        path: /health/startup
        port: 8080
      initialDelaySeconds: 10
      periodSeconds: 10
      failureThreshold: 30
```

### Circuit Breaker Protection (NEW in v0.5.0)

Configure circuit breaker policies:

```csharp
services.AddSingleton<ICircuitBreakerPolicy>(sp =>
    new CircuitBreakerPolicy(
        sp.GetRequiredService<ILogger<CircuitBreakerPolicy>>(),
        new CircuitBreakerOptions
        {
            FailureThreshold = 3,
            BreakDuration = TimeSpan.FromSeconds(30),
            RetryCount = 3,
            RetryDelayMs = 100,
            OperationTimeout = TimeSpan.FromSeconds(10)
        }));
```

Monitor circuit breaker state:

```csharp
// Check circuit state
var state = _circuitBreaker.GetCircuitState("gpu-operation");
if (state == CircuitState.Open)
{
    _logger.LogWarning("Circuit breaker is open for GPU operations");
}

// Reset circuit manually if needed
_circuitBreaker.Reset("gpu-operation");
```

## Performance Tuning

### GPU Optimization

```bash
# NVIDIA GPU settings
nvidia-smi -pm 1  # Enable persistence mode
nvidia-smi -ac 5001,1590  # Set application clocks
nvidia-smi -pl 250  # Set power limit

# Monitor GPU
nvidia-smi dmon -s pucvmet
```

### Memory Pool Tuning

```csharp
services.AddGpuBridge(options =>
{
    // Optimize for large batches
    options.MemoryPoolSizeMB = 8192;
    options.DefaultMicroBatch = 16384;
    
    // Enable advanced features
    options.EnablePinnedMemory = true;
    options.EnableUnifiedMemory = true;
    
    // Tune GC
    options.MemoryGCInterval = TimeSpan.FromMinutes(10);
    options.AllocationStrategy = AllocationStrategy.BestFit;
});
```

### Kernel Optimization

```csharp
// Kernel compilation options
.AddKernel(kernel => kernel
    .Id("optimized_kernel")
    .FromSource(source)
    .WithOptions(opts =>
    {
        opts.OptimizationLevel = OptimizationLevel.Maximum;
        opts.EnableVectorization = true;
        opts.UnrollLoops = true;
        opts.InlineThreshold = 1000;
    }));
```

## Troubleshooting

### Common Issues

#### GPU Not Detected

```bash
# Check GPU availability
nvidia-smi  # NVIDIA
rocm-smi    # AMD

# Check driver version
cat /proc/driver/nvidia/version

# Check CUDA installation
nvcc --version

# Test GPU detection
dotnet run --project tools/GpuDetector
```

#### Out of Memory Errors

```bash
# Monitor memory usage
nvidia-smi --query-gpu=memory.used,memory.free --format=csv -l 1

# Clear GPU memory
nvidia-smi --gpu-reset

# Adjust memory pool
export GPU_MEMORY_POOL_SIZE_MB=2048
```

#### Performance Issues

```bash
# Enable profiling
export ENABLE_PROFILING=true

# Check kernel execution times
grep "Kernel execution" logs/*.log | awk '{print $NF}' | sort -n

# Monitor GPU utilization
nvidia-smi dmon -s u
```

### Diagnostic Commands

```bash
# Health check
curl http://localhost:8080/health

# Get metrics
curl http://localhost:8080/metrics

# Get cluster status
curl http://localhost:8080/cluster/status

# Force garbage collection
curl -X POST http://localhost:8080/admin/gc

# Dump memory statistics
curl http://localhost:8080/debug/memory
```

## Maintenance

### Backup and Restore

```bash
# Backup grain state
docker exec orleans-silo-1 \
  dotnet Orleans.GpuBridge.Tools.dll backup \
  --output /backup/state-$(date +%Y%m%d).bak

# Restore grain state
docker exec orleans-silo-1 \
  dotnet Orleans.GpuBridge.Tools.dll restore \
  --input /backup/state-20240101.bak
```

### Rolling Updates

```bash
# Scale down
kubectl scale statefulset orleans-silo --replicas=2

# Update image
kubectl set image statefulset/orleans-silo \
  orleans-silo=ghcr.io/orleans-gpubridge/orleans-gpubridge:v2.0.0

# Scale up
kubectl scale statefulset orleans-silo --replicas=3

# Monitor rollout
kubectl rollout status statefulset/orleans-silo
```

### Log Rotation

```yaml
# logrotate configuration
/var/log/orleans-gpu/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    create 0644 orleans orleans
    sharedscripts
    postrotate
        systemctl reload orleans-gpu
    endscript
}
```

## Security

### TLS Configuration

```json
{
  "Orleans": {
    "Networking": {
      "TLS": {
        "Enabled": true,
        "Certificate": "/certs/orleans.pfx",
        "CertificatePassword": "${CERT_PASSWORD}",
        "ClientCertificateMode": "RequireCertificate"
      }
    }
  }
}
```

### Authentication

```csharp
// Configure authentication
services.AddAuthentication(JwtBearerDefaults.AuthenticationScheme)
    .AddJwtBearer(options =>
    {
        options.Authority = "https://identity.example.com";
        options.Audience = "orleans-gpu-api";
    });

// Secure grain access
[Authorize(Roles = "GpuUser")]
public class SecureGpuGrain : Grain, ISecureGpuGrain
{
    // Implementation
}
```

### Resource Quotas

```csharp
services.AddGpuBridge(options =>
{
    options.EnableQuotas = true;
    options.Quotas = new Dictionary<string, ResourceQuota>
    {
        ["tenant1"] = new() { MaxMemoryMB = 1024, MaxKernels = 10 },
        ["tenant2"] = new() { MaxMemoryMB = 2048, MaxKernels = 20 }
    };
});
```

## Disaster Recovery

### Backup Strategy

```bash
#!/bin/bash
# backup.sh

# Backup configuration
BACKUP_DIR="/backup/$(date +%Y%m%d)"
mkdir -p $BACKUP_DIR

# Backup grain state
docker exec orleans-redis redis-cli BGSAVE
docker cp orleans-redis:/data/dump.rdb $BACKUP_DIR/redis-dump.rdb

# Backup logs
tar -czf $BACKUP_DIR/logs.tar.gz /var/log/orleans-gpu/

# Backup configuration
cp -r /etc/orleans-gpu $BACKUP_DIR/config

# Upload to S3
aws s3 sync $BACKUP_DIR s3://backup-bucket/orleans-gpu/$(date +%Y%m%d)/
```

### Recovery Procedures

```bash
#!/bin/bash
# restore.sh

RESTORE_DATE=$1
BACKUP_DIR="/backup/$RESTORE_DATE"

# Download from S3
aws s3 sync s3://backup-bucket/orleans-gpu/$RESTORE_DATE/ $BACKUP_DIR/

# Stop services
docker-compose down

# Restore Redis
docker run -v $BACKUP_DIR:/backup -v redis-data:/data redis:7-alpine \
  sh -c "cp /backup/redis-dump.rdb /data/dump.rdb"

# Restore configuration
cp -r $BACKUP_DIR/config/* /etc/orleans-gpu/

# Start services
docker-compose up -d

# Verify
curl http://localhost:8080/health
```

### High Availability Setup

```yaml
# Multi-region deployment
regions:
  - name: us-east-1
    silos: 3
    gpu_type: nvidia-a100
  - name: us-west-2
    silos: 3
    gpu_type: nvidia-a100
  - name: eu-west-1
    silos: 2
    gpu_type: nvidia-a10g

# Cross-region replication
replication:
  mode: active-active
  consistency: eventual
  sync_interval: 5s
```

---

**For additional support, see the [Troubleshooting Guide](troubleshooting.md) or contact support.**
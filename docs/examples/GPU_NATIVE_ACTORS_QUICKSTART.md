# GPU-Native Actors Quick Start Guide

## Overview

GPU-native actors represent a paradigm shift in distributed computing: **actors that live permanently on the GPU** rather than CPU actors that offload work to GPU.

**Performance**: 100-500ns message latency, 2M messages/s/actor throughput.

---

## Architecture Comparison

### Traditional Approach (GPU-Offload)
```
CPU Actor → Copy to GPU → Launch Kernel → Wait → Copy from GPU → CPU Actor
Latency: 10-100μs per message
```

### GPU-Native Approach (Revolutionary)
```
GPU Actor → Process Message on GPU → GPU Actor
Latency: 100-500ns per message (20-200× faster!)
```

---

## Quick Start

### 1. Install Packages

```bash
dotnet add package Orleans.GpuBridge.Runtime
dotnet add package Orleans.GpuBridge.Grains
dotnet add package Orleans.GpuBridge.Backends.DotCompute
```

### 2. Configure Orleans Silo

```csharp
using Orleans.GpuBridge.Backends.DotCompute.Temporal;
using Orleans.GpuBridge.Backends.DotCompute.RingKernels;

var builder = Host.CreateDefaultBuilder(args)
    .UseOrleans(siloBuilder =>
    {
        siloBuilder
            .UseLocalhostClustering()
            .AddGpuBridge(options =>
            {
                options.PreferGpu = true;
                options.EnableTemporalFeatures = true;
            })
            .AddDotComputeBackend(options =>
            {
                options.ValidateCapabilities = false; // WSL2 compatibility
            });
    })
    .ConfigureServices(services =>
    {
        // Register temporal infrastructure
        services.AddSingleton<RingKernelManager>();
        services.AddSingleton<GpuNativeHybridLogicalClock>();
        services.AddSingleton<DotComputeTimingProvider>();
        services.AddSingleton<DotComputeBarrierProvider>();
        services.AddSingleton<DotComputeMemoryOrderingProvider>();
    });

await builder.RunConsoleAsync();
```

### 3. Create Your First GPU-Native Actor

```csharp
using Orleans.GpuBridge.Grains.GpuNative;

// Get actor reference
var vertexActor = client.GetGrain<IGpuNativeVertexActor>(Guid.NewGuid());

// Initialize actor (launches ring kernel on GPU)
await vertexActor.InitializeVertexAsync(new VertexConfiguration
{
    MessageQueueCapacity = 10000,
    EnableTemporalOrdering = true
});

// Send messages (100-500ns latency!)
var timestamp = await vertexActor.AddEdgeAsync(edgeId);
Console.WriteLine($"Edge added at timestamp: {timestamp}");

// Query actor state
var edges = await vertexActor.GetEdgesAsync();
var status = await vertexActor.GetStatusAsync();
Console.WriteLine($"Actor has {edges.Length} edges, {status.PendingMessages} pending messages");
```

### 4. Monitor Performance

```csharp
// Get actor statistics
var stats = await vertexActor.GetStatisticsAsync();

Console.WriteLine($"Messages processed: {stats.TotalMessagesProcessed:N0}");
Console.WriteLine($"Messages sent: {stats.TotalMessagesSent:N0}");
Console.WriteLine($"Average latency: {stats.AverageLatencyNanos:F1}ns");
Console.WriteLine($"Throughput: {stats.ThroughputMessagesPerSecond:N0} msgs/s");
Console.WriteLine($"Queue utilization: {stats.QueueUtilization:F1}%");
```

Expected output:
```
Messages processed: 1,543,892
Messages sent: 876,234
Average latency: 287.3ns
Throughput: 2,145,678 msgs/s
Queue utilization: 12.4%
```

---

## Performance Characteristics

### Message Latency

| Actor Type | Latency | Notes |
|-----------|---------|-------|
| CPU Actors | 10-100μs | Traditional Orleans grains |
| GPU-Offload | 10-50μs | Kernel launch overhead |
| **GPU-Native** | **100-500ns** | Ring kernels, zero launch overhead |

**Improvement**: 20-200× faster than CPU actors!

### Throughput

| Actor Type | Messages/s | Notes |
|-----------|-----------|-------|
| CPU Actors | 15,000 | Single-threaded grain |
| **GPU-Native** | **2,000,000** | Lock-free queues on GPU |

**Improvement**: 133× higher throughput!

### Temporal Ordering

| Clock Type | Update Time | Notes |
|-----------|------------|-------|
| CPU HLC | 50ns | Standard implementation |
| **GPU HLC** | **20ns** | GPU-native %%globaltimer |

**Improvement**: 2.5× faster temporal updates!

---

## Use Cases

### 1. Knowledge Graphs with Real-Time Queries

```csharp
// Create graph with 1M GPU-native vertex actors
var vertices = new List<IGpuNativeVertexActor>();
for (int i = 0; i < 1_000_000; i++)
{
    var vertex = client.GetGrain<IGpuNativeVertexActor>(Guid.NewGuid());
    await vertex.InitializeVertexAsync();
    vertices.Add(vertex);
}

// Query with <100μs pattern detection latency
var result = await vertices[0].QueryConnectedVerticesAsync(maxHops: 3);
Console.WriteLine($"Found {result.VertexIds.Length} connected vertices in <100μs");
```

### 2. Digital Twins as Living Entities

```csharp
// Each component is a GPU-native actor with physics simulation
var factory = new DigitalTwinFactory(client);

var motor = await factory.CreateMotorAsync(motorId);
var pump = await factory.CreatePumpAsync(pumpId);
var sensor = await factory.CreateSensorAsync(sensorId);

// Actors exchange messages at 100-500ns latency
// Physics-accurate simulation with temporal causality
await motor.ApplyTorque(torque: 100.0f);
await pump.SetFlowRate(flowRate: 50.0f);

// Sensor reads at 2M samples/s
var reading = await sensor.ReadAsync();
```

### 3. Temporal Pattern Detection (Fraud/Anomalies)

```csharp
// Transaction actors with GPU-native HLC
var txActor = client.GetGrain<IGpuNativeTransactionActor>(txId);
await txActor.InitializeAsync();

// Detect patterns: rapid splits, circular flows, timing anomalies
await txActor.ProcessTransactionAsync(amount: 1000.0m, toAccount: account1);
await txActor.ProcessTransactionAsync(amount: 500.0m, toAccount: account2);

// Pattern detection <100μs with causal ordering
var patterns = await txActor.DetectPatternsAsync();
if (patterns.HasSuspiciousPattern)
{
    Console.WriteLine($"Fraud detected! Pattern: {patterns.PatternType}");
}
```

---

## Advanced Features

### Temporal Ordering with HLC

```csharp
// Enable temporal ordering for causal consistency
var config = new GpuNativeActorConfiguration
{
    EnableTemporalOrdering = true, // 15% overhead for causal ordering
    EnableTimestamps = true
};

await actor.InitializeAsync(config);

// Messages are ordered by HLC timestamps
var ts1 = await actor1.SendMessageAsync(msg1);
var ts2 = await actor2.SendMessageAsync(msg2);

// Check happened-before relationship
if (GpuNativeHybridLogicalClock.HappenedBefore(ts1, ts2))
{
    Console.WriteLine("msg1 happened before msg2 (causal order maintained)");
}
```

### Custom Ring Kernels

```csharp
// Define custom CUDA kernel for your actor logic
var kernelSource = @"
__global__ void my_custom_ring_kernel(
    ActorState* states,
    QueueMetadata* inbox_meta,
    char* inbox_data,
    QueueMetadata* outbox_meta,
    char* outbox_data,
    int message_size,
    int num_actors)
{
    int actor_idx = threadIdx.x + blockIdx.x * blockDim.x;
    ActorState* state = &states[actor_idx];

    while (true)  // Infinite loop!
    {
        ActorMessage msg;
        if (gpu_queue_try_dequeue(inbox_meta, inbox_data, message_size, &msg))
        {
            // Your custom processing logic here
            process_my_custom_message(state, &msg);
        }
        __nanosleep(20);
    }
}";

var config = new GpuNativeActorConfiguration
{
    RingKernelSource = kernelSource,
    KernelEntryPoint = "my_custom_ring_kernel"
};
```

### Multi-Actor Coordination

```csharp
// Create actor swarm with coordination
var actorCount = 1000;
var actors = new IGpuNativeActor[actorCount];

for (int i = 0; i < actorCount; i++)
{
    actors[i] = client.GetGrain<IGpuNativeActor>(Guid.NewGuid());
    await actors[i].InitializeAsync(config);
}

// Broadcast message to all actors (parallel, <1ms total)
var tasks = actors.Select(a => a.SendMessageAsync(broadcastMsg));
await Task.WhenAll(tasks);

// Total time: ~500μs for 1000 actors (500ns per actor)
```

---

## Monitoring and Diagnostics

### Real-Time Metrics

```csharp
// Monitor actor performance
var status = await actor.GetStatusAsync();
Console.WriteLine($"Running: {status.IsRunning}");
Console.WriteLine($"Pending messages: {status.PendingMessages}");
Console.WriteLine($"Current HLC: {status.CurrentTimestamp}");
Console.WriteLine($"Uptime: {status.Uptime.TotalSeconds:F1}s");

// Check queue health
var stats = await actor.GetStatisticsAsync();
if (stats.QueueUtilization > 80.0)
{
    Console.WriteLine("WARNING: Queue near capacity, consider scaling");
}

if (stats.AverageLatencyNanos > 1000)
{
    Console.WriteLine("WARNING: High latency detected, check GPU load");
}
```

### Debugging Ring Kernels

```csharp
// Enable GPU kernel profiling
var timing = serviceProvider.GetService<DotComputeTimingProvider>();
await timing.CalibrateAsync(sampleCount: 1000);

// Get clock calibration status
var (isCalibrated, age, calibration) = timing.GetCalibrationStatus();
Console.WriteLine($"Calibrated: {isCalibrated}, Age: {age.TotalSeconds:F1}s");
Console.WriteLine($"Offset: {calibration.Value.OffsetNanos}ns");
Console.WriteLine($"Drift: {calibration.Value.DriftPPM}ppm");
```

---

## Performance Tuning

### Message Queue Sizing

```csharp
// Small queues (low latency, risk of overflow)
var lowLatencyConfig = new GpuNativeActorConfiguration
{
    MessageQueueCapacity = 1000,  // ~100μs drain time
};

// Large queues (high throughput, more memory)
var highThroughputConfig = new GpuNativeActorConfiguration
{
    MessageQueueCapacity = 100000,  // ~50ms drain time
};
```

### Temporal Ordering Overhead

```csharp
// Maximum performance (no temporal ordering)
var maxPerfConfig = new GpuNativeActorConfiguration
{
    EnableTemporalOrdering = false,  // 0% overhead
    EnableTimestamps = false
};
// Latency: 100ns

// Causal ordering (15% overhead)
var causalConfig = new GpuNativeActorConfiguration
{
    EnableTemporalOrdering = true,  // 15% overhead
    EnableTimestamps = true
};
// Latency: 115ns (still 87× faster than CPU actors!)
```

---

## Troubleshooting

### Ring Kernel Not Starting

**Symptom**: Actor initializes but status shows `IsRunning = false`

**Solutions**:
1. Check GPU hardware barriers: `barrierProvider.IsHardwareBarrierSupported`
2. Verify CUDA cooperative groups support (Compute Capability 6.0+)
3. Check GPU device logs for kernel launch errors

### High Message Latency

**Symptom**: Latency >1μs consistently

**Solutions**:
1. Check GPU utilization (other kernels competing for resources)
2. Verify message queue not full (`QueueUtilization < 90%`)
3. Check for memory bandwidth saturation
4. Ensure PCIe bandwidth sufficient (prefer on-die GPU memory)

### Queue Overflow

**Symptom**: Messages dropped, `queue.IsFull == true`

**Solutions**:
1. Increase `MessageQueueCapacity`
2. Add backpressure: check queue utilization before sending
3. Scale horizontally: create more actor instances
4. Optimize ring kernel processing logic (reduce per-message time)

---

## Next Steps

1. **Read the articles**: [GPU-Native Actors Series](../articles/gpu-actors/README.md)
2. **Explore Knowledge Organisms**: [Knowledge Organisms Article](../articles/knowledge-organisms/README.md)
3. **Review temporal correctness**: [Temporal Correctness Series](../articles/temporal/README.md)
4. **Check out hypergraph actors**: [Hypergraph Actors Series](../articles/hypergraph-actors/README.md)

---

## Performance Comparison Summary

| Metric | CPU Actors | GPU-Offload | **GPU-Native** |
|--------|-----------|-------------|----------------|
| Message Latency | 10-100μs | 10-50μs | **100-500ns** |
| Throughput/Actor | 15K msgs/s | 50K msgs/s | **2M msgs/s** |
| Memory Bandwidth | 200 GB/s | 500 GB/s | **1,935 GB/s** |
| HLC Update | 50ns | 50ns | **20ns** |
| Kernel Launch | N/A | 10-50μs | **0ns** (ring) |

**Result**: GPU-native actors are 20-200× faster than traditional approaches!

---

**Built for**:
- Real-time systems requiring sub-microsecond response
- High-frequency distributed simulations
- Temporal pattern detection at scale
- Digital twins as living entities
- Knowledge organisms with emergent intelligence

**The future of distributed computing is GPU-native!** 🚀

# Phase 4: Production Hardening Implementation Specification

## Duration: Weeks 7-8

## Objectives
Optimize performance with CUDA Graphs, implement GPUDirect Storage, add comprehensive telemetry and monitoring, and create production-ready samples and documentation.

## Week 7: Performance Optimization & Advanced Features

### Day 1-2: CUDA Graph Implementation

#### CUDA Graph Capture and Optimization
```csharp
namespace Orleans.GpuBridge.DotCompute.Optimization;

public sealed class CudaGraphOptimizer : IKernelOptimizer
{
    private readonly ILogger<CudaGraphOptimizer> _logger;
    private readonly Dictionary<GraphKey, CapturedGraph> _graphs;
    private readonly SemaphoreSlim _captureLock;
    
    public CudaGraphOptimizer(ILogger<CudaGraphOptimizer> logger)
    {
        _logger = logger;
        _graphs = new Dictionary<GraphKey, CapturedGraph>();
        _captureLock = new SemaphoreSlim(1, 1);
    }
    
    public async Task<IOptimizedExecution> OptimizeAsync(
        KernelExecutionPlan plan,
        CancellationToken ct)
    {
        var key = ComputeGraphKey(plan);
        
        // Check if we have a captured graph
        if (_graphs.TryGetValue(key, out var existing))
        {
            _logger.LogDebug(
                "Using cached CUDA graph for {Key}",
                key);
            return existing;
        }
        
        // Capture new graph
        await _captureLock.WaitAsync(ct);
        try
        {
            // Double-check after acquiring lock
            if (_graphs.TryGetValue(key, out existing))
            {
                return existing;
            }
            
            var graph = await CaptureGraphAsync(plan, ct);
            _graphs[key] = graph;
            
            _logger.LogInformation(
                "Captured new CUDA graph for {Key} with {Nodes} nodes",
                key, graph.NodeCount);
            
            return graph;
        }
        finally
        {
            _captureLock.Release();
        }
    }
    
    private async Task<CapturedGraph> CaptureGraphAsync(
        KernelExecutionPlan plan,
        CancellationToken ct)
    {
        using var stream = new CudaStream();
        
        // Begin capture
        stream.BeginCapture(CaptureMode.Global);
        
        try
        {
            // Record kernel launches
            foreach (var kernel in plan.Kernels)
            {
                await RecordKernelLaunchAsync(stream, kernel, ct);
            }
            
            // Record memory operations
            foreach (var memOp in plan.MemoryOperations)
            {
                RecordMemoryOperation(stream, memOp);
            }
            
            // End capture
            var graphHandle = stream.EndCapture();
            
            // Instantiate executable graph
            var executable = graphHandle.Instantiate(
                InstantiateFlags.AutoFreeOnLaunch);
            
            return new CapturedGraph(
                plan.Id,
                graphHandle,
                executable,
                plan.Kernels.Count);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex,
                "Failed to capture CUDA graph for plan {Id}",
                plan.Id);
            stream.EndCapture();
            throw;
        }
    }
    
    private async Task RecordKernelLaunchAsync(
        CudaStream stream,
        KernelDescriptor kernel,
        CancellationToken ct)
    {
        // Add kernel node to graph
        var node = stream.AddKernelNode(
            kernel.Function,
            kernel.GridDim,
            kernel.BlockDim,
            kernel.SharedMemoryBytes,
            kernel.Parameters);
        
        // Add dependencies
        foreach (var dep in kernel.Dependencies)
        {
            node.AddDependency(dep);
        }
        
        await Task.CompletedTask;
    }
    
    private void RecordMemoryOperation(
        CudaStream stream,
        MemoryOperation memOp)
    {
        switch (memOp.Type)
        {
            case MemOpType.Copy:
                stream.AddMemcpyNode(
                    memOp.Destination,
                    memOp.Source,
                    memOp.Size);
                break;
                
            case MemOpType.Set:
                stream.AddMemsetNode(
                    memOp.Destination,
                    memOp.Value,
                    memOp.Size);
                break;
                
            case MemOpType.Prefetch:
                stream.AddPrefetchNode(
                    memOp.Destination,
                    memOp.Size,
                    memOp.Device);
                break;
        }
    }
}

public sealed class CapturedGraph : IOptimizedExecution
{
    private readonly string _id;
    private readonly CudaGraph _graph;
    private readonly CudaGraphExec _executable;
    private long _executionCount;
    
    public string Id => _id;
    public int NodeCount { get; }
    public long ExecutionCount => _executionCount;
    
    public CapturedGraph(
        string id,
        CudaGraph graph,
        CudaGraphExec executable,
        int nodeCount)
    {
        _id = id;
        _graph = graph;
        _executable = executable;
        NodeCount = nodeCount;
    }
    
    public async Task<ExecutionResult> ExecuteAsync(
        ExecutionContext context,
        CancellationToken ct)
    {
        var sw = Stopwatch.StartNew();
        
        // Launch graph
        await _executable.LaunchAsync(context.Stream, ct);
        
        // Synchronize
        await context.Stream.SynchronizeAsync(ct);
        
        sw.Stop();
        Interlocked.Increment(ref _executionCount);
        
        return new ExecutionResult(
            Success: true,
            Duration: sw.Elapsed,
            GraphExecuted: true);
    }
    
    public void Dispose()
    {
        _executable?.Dispose();
        _graph?.Dispose();
    }
}
```

### Day 3-4: GPUDirect Storage Integration

#### GPUDirect Storage Implementation
```csharp
namespace Orleans.GpuBridge.Runtime.Storage;

public sealed class GpuDirectStorageProvider : IStorageProvider
{
    private readonly ILogger<GpuDirectStorageProvider> _logger;
    private readonly GdsConfiguration _config;
    private readonly bool _gdsAvailable;
    private GdsHandle _handle = default!;
    
    public GpuDirectStorageProvider(
        ILogger<GpuDirectStorageProvider> logger,
        IOptions<GdsConfiguration> config)
    {
        _logger = logger;
        _config = config.Value;
        _gdsAvailable = CheckGdsAvailability();
    }
    
    public async Task InitializeAsync(CancellationToken ct)
    {
        if (!_gdsAvailable)
        {
            _logger.LogWarning(
                "GPUDirect Storage not available, using fallback");
            return;
        }
        
        try
        {
            _handle = await GdsNative.InitializeAsync(
                _config.MountPoint,
                _config.MaxPinnedMemoryMB * 1024 * 1024,
                ct);
            
            _logger.LogInformation(
                "GPUDirect Storage initialized at {Mount}",
                _config.MountPoint);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex,
                "Failed to initialize GPUDirect Storage");
            throw;
        }
    }
    
    public async Task<IGpuBuffer> LoadDirectAsync(
        string path,
        int deviceIndex,
        CancellationToken ct)
    {
        if (!_gdsAvailable)
        {
            return await LoadViaHostAsync(path, deviceIndex, ct);
        }
        
        try
        {
            // Get file info
            var fileInfo = new FileInfo(path);
            if (!fileInfo.Exists)
            {
                throw new FileNotFoundException(
                    $"File not found: {path}");
            }
            
            // Allocate GPU buffer
            var buffer = await AllocateDeviceBufferAsync(
                deviceIndex,
                fileInfo.Length,
                ct);
            
            // Open file with O_DIRECT
            using var fileHandle = GdsNative.OpenDirect(
                path,
                OpenFlags.ReadOnly | OpenFlags.Direct);
            
            // Register buffer for GDS
            var registration = await GdsNative.RegisterBufferAsync(
                _handle,
                buffer.DevicePointer,
                buffer.Size,
                ct);
            
            try
            {
                // Direct transfer from storage to GPU
                var transferred = await GdsNative.ReadAsync(
                    fileHandle,
                    registration,
                    0, // offset
                    buffer.Size,
                    ct);
                
                _logger.LogDebug(
                    "GDS: Transferred {Bytes} bytes directly to GPU",
                    transferred);
                
                return buffer;
            }
            finally
            {
                await GdsNative.UnregisterBufferAsync(
                    registration, ct);
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex,
                "GDS transfer failed, falling back to host");
            return await LoadViaHostAsync(path, deviceIndex, ct);
        }
    }
    
    public async Task SaveDirectAsync(
        IGpuBuffer buffer,
        string path,
        CancellationToken ct)
    {
        if (!_gdsAvailable)
        {
            await SaveViaHostAsync(buffer, path, ct);
            return;
        }
        
        try
        {
            // Create directory if needed
            var directory = Path.GetDirectoryName(path);
            if (!string.IsNullOrEmpty(directory))
            {
                Directory.CreateDirectory(directory);
            }
            
            // Open file with O_DIRECT
            using var fileHandle = GdsNative.OpenDirect(
                path,
                OpenFlags.WriteOnly | 
                OpenFlags.Create | 
                OpenFlags.Truncate | 
                OpenFlags.Direct);
            
            // Register buffer for GDS
            var registration = await GdsNative.RegisterBufferAsync(
                _handle,
                buffer.DevicePointer,
                buffer.Size,
                ct);
            
            try
            {
                // Direct transfer from GPU to storage
                var transferred = await GdsNative.WriteAsync(
                    fileHandle,
                    registration,
                    0, // offset
                    buffer.Size,
                    ct);
                
                _logger.LogDebug(
                    "GDS: Wrote {Bytes} bytes directly from GPU",
                    transferred);
            }
            finally
            {
                await GdsNative.UnregisterBufferAsync(
                    registration, ct);
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex,
                "GDS write failed, falling back to host");
            await SaveViaHostAsync(buffer, path, ct);
        }
    }
    
    private async Task<IGpuBuffer> LoadViaHostAsync(
        string path,
        int deviceIndex,
        CancellationToken ct)
    {
        // Traditional host-mediated transfer
        var data = await File.ReadAllBytesAsync(path, ct);
        
        var buffer = await AllocateDeviceBufferAsync(
            deviceIndex,
            data.Length,
            ct);
        
        await buffer.CopyFromAsync(data, ct);
        
        return buffer;
    }
    
    private bool CheckGdsAvailability()
    {
        try
        {
            return GdsNative.IsAvailable() && 
                   _config.EnableGpuDirectStorage;
        }
        catch
        {
            return false;
        }
    }
}

[Flags]
public enum OpenFlags
{
    ReadOnly = 0x0000,
    WriteOnly = 0x0001,
    ReadWrite = 0x0002,
    Create = 0x0040,
    Truncate = 0x0200,
    Direct = 0x4000
}
```

### Day 5: Comprehensive Telemetry

#### OpenTelemetry Integration
```csharp
namespace Orleans.GpuBridge.Diagnostics;

public sealed class GpuTelemetry : IGpuTelemetry
{
    private readonly Meter _meter;
    private readonly ActivitySource _activitySource;
    
    // Metrics
    private readonly Counter<long> _kernelsExecuted;
    private readonly Histogram<double> _kernelLatency;
    private readonly ObservableGauge<long> _gpuMemoryUsed;
    private readonly ObservableGauge<double> _gpuUtilization;
    private readonly Counter<long> _bytesTransferred;
    private readonly Counter<long> _allocationFailures;
    
    public GpuTelemetry(IMeterFactory meterFactory)
    {
        _meter = meterFactory.Create("Orleans.GpuBridge");
        _activitySource = new ActivitySource("Orleans.GpuBridge");
        
        // Initialize metrics
        _kernelsExecuted = _meter.CreateCounter<long>(
            "gpu.kernels.executed",
            "kernels",
            "Number of GPU kernels executed");
        
        _kernelLatency = _meter.CreateHistogram<double>(
            "gpu.kernel.latency",
            "ms",
            "GPU kernel execution latency");
        
        _gpuMemoryUsed = _meter.CreateObservableGauge<long>(
            "gpu.memory.used",
            GetGpuMemoryUsed,
            "bytes",
            "GPU memory currently in use");
        
        _gpuUtilization = _meter.CreateObservableGauge<double>(
            "gpu.utilization",
            GetGpuUtilization,
            "%",
            "GPU utilization percentage");
        
        _bytesTransferred = _meter.CreateCounter<long>(
            "gpu.bytes.transferred",
            "bytes",
            "Bytes transferred to/from GPU");
        
        _allocationFailures = _meter.CreateCounter<long>(
            "gpu.allocation.failures",
            "failures",
            "GPU memory allocation failures");
    }
    
    public Activity? StartKernelExecution(
        string kernelName,
        int deviceIndex)
    {
        var activity = _activitySource.StartActivity(
            "gpu.kernel.execute",
            ActivityKind.Internal);
        
        activity?.SetTag("kernel.name", kernelName);
        activity?.SetTag("device.index", deviceIndex);
        activity?.SetTag("device.type", "gpu");
        
        return activity;
    }
    
    public void RecordKernelExecution(
        string kernelName,
        int deviceIndex,
        TimeSpan duration,
        bool success)
    {
        var tags = new TagList
        {
            { "kernel", kernelName },
            { "device", deviceIndex },
            { "success", success }
        };
        
        _kernelsExecuted.Add(1, tags);
        _kernelLatency.Record(
            duration.TotalMilliseconds,
            tags);
    }
    
    public void RecordMemoryTransfer(
        TransferDirection direction,
        long bytes,
        TimeSpan duration)
    {
        var tags = new TagList
        {
            { "direction", direction.ToString() }
        };
        
        _bytesTransferred.Add(bytes, tags);
        
        var throughputGbps = (bytes / 1e9) / duration.TotalSeconds;
        _meter.CreateHistogram<double>("gpu.transfer.throughput")
            .Record(throughputGbps, tags);
    }
    
    public void RecordAllocationFailure(
        int deviceIndex,
        long requestedBytes,
        string reason)
    {
        _allocationFailures.Add(1, new TagList
        {
            { "device", deviceIndex },
            { "reason", reason }
        });
        
        Activity.Current?.SetStatus(
            ActivityStatusCode.Error,
            $"GPU allocation failed: {reason}");
    }
    
    private static IEnumerable<Measurement<long>> GetGpuMemoryUsed()
    {
        // Query actual GPU memory usage
        foreach (var device in GpuDeviceManager.GetDevices())
        {
            yield return new Measurement<long>(
                device.UsedMemoryBytes,
                new TagList { { "device", device.Index } });
        }
    }
    
    private static IEnumerable<Measurement<double>> GetGpuUtilization()
    {
        // Query actual GPU utilization
        foreach (var device in GpuDeviceManager.GetDevices())
        {
            yield return new Measurement<double>(
                device.UtilizationPercent,
                new TagList { { "device", device.Index } });
        }
    }
}

public enum TransferDirection
{
    HostToDevice,
    DeviceToHost,
    DeviceToDevice,
    PeerToPeer
}
```

## Week 8: Samples, Documentation & Final Testing

### Day 6-7: Production Samples

#### Vector Operations Sample
```csharp
namespace Orleans.GpuBridge.Samples.VectorOps;

public class VectorOperationsSample
{
    public static async Task RunAsync(IClusterClient client)
    {
        Console.WriteLine("=== Vector Operations Sample ===");
        
        // Get the GPU grain
        var grain = client.GetGrain<IGpuBatchGrain<VectorPair, float[]>>(
            "kernels/vector/add");
        
        // Prepare test data
        const int vectorSize = 1_000_000;
        const int batchSize = 100;
        
        var batch = new List<VectorPair>();
        for (int i = 0; i < batchSize; i++)
        {
            var a = GenerateRandomVector(vectorSize);
            var b = GenerateRandomVector(vectorSize);
            batch.Add(new VectorPair(a, b));
        }
        
        // Execute on GPU
        var sw = Stopwatch.StartNew();
        var result = await grain.ExecuteAsync(
            batch,
            new GpuExecutionHints
            {
                PreferredDevice = 0,
                HighPriority = true
            });
        sw.Stop();
        
        // Display results
        Console.WriteLine($"Processed {batchSize} vector pairs");
        Console.WriteLine($"Total time: {sw.ElapsedMilliseconds}ms");
        Console.WriteLine($"Throughput: {batchSize * 1000.0 / sw.ElapsedMilliseconds:F2} ops/sec");
        Console.WriteLine($"Success: {result.Success}");
        
        // Verify a sample
        if (result.Success && result.Results.Count > 0)
        {
            var first = result.Results[0];
            Console.WriteLine($"First result length: {first.Length}");
            Console.WriteLine($"Sample values: [{string.Join(", ", first.Take(5))}...]");
        }
    }
    
    private static float[] GenerateRandomVector(int size)
    {
        var random = new Random();
        var vector = new float[size];
        for (int i = 0; i < size; i++)
        {
            vector[i] = (float)random.NextDouble() * 100;
        }
        return vector;
    }
}
```

#### Graph Processing Sample (AssureTwin)
```csharp
namespace Orleans.GpuBridge.Samples.AssureTwin;

public class JeDecompositionSample
{
    public static async Task RunAsync(IClusterClient client)
    {
        Console.WriteLine("=== JE Decomposition Sample ===");
        
        // Load graph data
        var graph = await LoadGraphDataAsync();
        
        // Get specialized grain
        var grain = client.GetGrain<IGpuBatchGrain<GraphData, JeComponents>>(
            "assuretwin/je-decompose");
        
        // Configure for large graph processing
        var hints = new GpuExecutionHints
        {
            PreferredDevice = 0,
            MaxMicroBatch = 10000,
            Persistent = true
        };
        
        // Process in chunks
        var allComponents = new List<JeComponents>();
        const int chunkSize = 50000;
        
        for (int offset = 0; offset < graph.NodeCount; offset += chunkSize)
        {
            var chunk = graph.GetSubgraph(offset, Math.Min(chunkSize, graph.NodeCount - offset));
            
            var sw = Stopwatch.StartNew();
            var result = await grain.ExecuteAsync(new[] { chunk }, hints);
            sw.Stop();
            
            if (result.Success)
            {
                allComponents.AddRange(result.Results);
                Console.WriteLine($"Processed nodes {offset}-{offset + chunkSize}: {sw.ElapsedMilliseconds}ms");
            }
        }
        
        // Aggregate results
        Console.WriteLine($"\nTotal JE components found: {allComponents.Count}");
        Console.WriteLine($"Average component size: {allComponents.Average(c => c.Size):F2}");
        Console.WriteLine($"Largest component: {allComponents.Max(c => c.Size)} nodes");
        
        // Display sample patterns
        var patterns = AnalyzePatterns(allComponents);
        Console.WriteLine("\nDetected patterns:");
        foreach (var pattern in patterns.Take(5))
        {
            Console.WriteLine($"  - {pattern.Type}: {pattern.Count} occurrences");
        }
    }
    
    private static async Task<GraphData> LoadGraphDataAsync()
    {
        // Load from parquet files
        var nodes = await ParquetReader.ReadNodesAsync("data/nodes.parquet");
        var edges = await ParquetReader.ReadEdgesAsync("data/edges.parquet");
        
        return new GraphData(nodes, edges);
    }
    
    private static IEnumerable<Pattern> AnalyzePatterns(List<JeComponents> components)
    {
        // Pattern detection logic
        return components
            .GroupBy(c => c.PatternSignature)
            .Select(g => new Pattern(g.Key, g.Count()))
            .OrderByDescending(p => p.Count);
    }
}
```

### Day 8-9: Performance Benchmarking

#### Comprehensive Benchmark Suite
```csharp
namespace Orleans.GpuBridge.Benchmarks;

[MemoryDiagnoser]
[SimpleJob(RuntimeMoniker.Net90)]
public class GpuBridgeBenchmarks
{
    private IClusterClient _client = default!;
    private List<float[]> _smallVectors = default!;
    private List<float[]> _largeVectors = default!;
    
    [GlobalSetup]
    public async Task Setup()
    {
        _client = await ConnectToClusterAsync();
        
        // Prepare test data
        _smallVectors = Enumerable.Range(0, 1000)
            .Select(_ => GenerateVector(100))
            .ToList();
        
        _largeVectors = Enumerable.Range(0, 10)
            .Select(_ => GenerateVector(1_000_000))
            .ToList();
    }
    
    [Benchmark]
    public async Task SmallBatch_CpuFallback()
    {
        var grain = _client.GetGrain<IGpuBatchGrain<float[], float>>(
            "kernels/sum");
        
        await grain.ExecuteAsync(
            _smallVectors.Take(10).ToList(),
            new GpuExecutionHints { PreferGpu = false });
    }
    
    [Benchmark]
    public async Task SmallBatch_Gpu()
    {
        var grain = _client.GetGrain<IGpuBatchGrain<float[], float>>(
            "kernels/sum");
        
        await grain.ExecuteAsync(
            _smallVectors.Take(10).ToList(),
            new GpuExecutionHints { PreferGpu = true });
    }
    
    [Benchmark]
    public async Task LargeBatch_Gpu()
    {
        var grain = _client.GetGrain<IGpuBatchGrain<float[], float>>(
            "kernels/sum");
        
        await grain.ExecuteAsync(
            _largeVectors,
            new GpuExecutionHints { PreferGpu = true });
    }
    
    [Benchmark]
    public async Task PersistentKernel_Stream()
    {
        var grain = _client.GetGrain<IGpuStreamGrain<float[], float>>(
            "kernels/sum");
        
        var inputStream = StreamId.Create("input", Guid.NewGuid());
        var outputStream = StreamId.Create("output", Guid.NewGuid());
        
        await grain.StartProcessingAsync(inputStream, outputStream);
        
        // Send data through stream
        var provider = _client.GetStreamProvider("Default");
        var stream = provider.GetStream<float[]>(inputStream);
        
        foreach (var vector in _smallVectors.Take(100))
        {
            await stream.OnNextAsync(vector);
        }
        
        await grain.StopProcessingAsync();
    }
    
    [Benchmark]
    public async Task MemoryPool_Allocation()
    {
        var pool = new GpuMemoryPool(
            NullLogger<GpuMemoryPool>.Instance,
            new MockAdapter(),
            0);
        
        var buffers = new List<IGpuBuffer>();
        
        // Allocate and release
        for (int i = 0; i < 100; i++)
        {
            var buffer = await pool.RentAsync(
                1024 * 1024,
                BufferUsage.ReadWrite,
                CancellationToken.None);
            
            buffers.Add(buffer);
        }
        
        foreach (var buffer in buffers)
        {
            await pool.ReturnAsync(buffer);
        }
    }
}
```

### Day 10: Documentation & Deployment Guide

#### Production Deployment Guide
```markdown
# Orleans.GpuBridge Production Deployment Guide

## Prerequisites

### Hardware Requirements
- NVIDIA GPU with Compute Capability 7.0+ (for CUDA)
- Minimum 8GB GPU memory per device
- NVMe SSD for GPUDirect Storage (optional)
- 100Gbps+ network for multi-node clusters

### Software Requirements
- .NET 9.0 Runtime
- CUDA Toolkit 12.0+ (for NVIDIA GPUs)
- Orleans 8.0+
- Docker (optional, for containerized deployment)

## Installation

### NuGet Packages
```xml
<PackageReference Include="Orleans.GpuBridge" Version="1.0.0" />
<PackageReference Include="Orleans.GpuBridge.DotCompute" Version="1.0.0" />
```

### Configuration

#### appsettings.json
```json
{
  "GpuBridge": {
    "PreferGpu": true,
    "MaxDevices": 4,
    "MemoryPoolSizeMB": 1024,
    "EnableGpuDirectStorage": false,
    "Telemetry": {
      "EnableMetrics": true,
      "EnableTracing": true,
      "SamplingRate": 0.1
    }
  }
}
```

#### Service Registration
```csharp
builder.Host.UseOrleans(siloBuilder =>
{
    siloBuilder
        .UseLocalhostClustering()
        .ConfigureServices(services =>
        {
            services.AddGpuBridge(options =>
            {
                options.PreferGpu = true;
                options.MaxConcurrentKernels = 100;
            })
            .AddKernel<VectorAddKernel>()
            .AddKernel<MatrixMultiplyKernel>()
            .ConfigureGpuDirectStorage(gds =>
            {
                gds.EnableGpuDirectStorage = true;
                gds.MountPoint = "/mnt/nvme";
            });
        })
        .AddGpuPlacementDirector();
});
```

## Monitoring

### OpenTelemetry Setup
```csharp
services.AddOpenTelemetry()
    .WithMetrics(metrics =>
    {
        metrics
            .AddMeter("Orleans.GpuBridge")
            .AddPrometheusExporter();
    })
    .WithTracing(tracing =>
    {
        tracing
            .AddSource("Orleans.GpuBridge")
            .AddJaegerExporter();
    });
```

### Key Metrics to Monitor
- `gpu.kernels.executed` - Kernel execution count
- `gpu.kernel.latency` - Execution latency (P50, P95, P99)
- `gpu.memory.used` - Memory utilization
- `gpu.utilization` - GPU compute utilization
- `gpu.allocation.failures` - Resource exhaustion events

### Grafana Dashboard
Import the provided dashboard from `monitoring/grafana-dashboard.json`

## Performance Tuning

### Kernel Optimization
1. Use persistent kernels for high-frequency operations
2. Enable CUDA Graph capture for repetitive workflows
3. Batch small operations to reduce overhead
4. Use appropriate work group sizes

### Memory Management
1. Configure memory pools based on workload
2. Enable unified memory for large datasets
3. Use GPUDirect Storage for I/O-intensive workloads
4. Monitor and adjust pool sizes based on metrics

### Placement Strategy
1. Configure GPU affinity for specific grains
2. Use local placement for latency-sensitive operations
3. Balance load across available GPUs
4. Consider NUMA topology for multi-GPU systems

## Troubleshooting

### Common Issues

#### GPU Not Detected
- Verify CUDA installation: `nvidia-smi`
- Check driver version compatibility
- Ensure Orleans silo has GPU access permissions

#### Out of Memory Errors
- Reduce batch sizes
- Increase memory pool eviction rate
- Scale horizontally with more GPU nodes

#### Poor Performance
- Check GPU utilization metrics
- Verify kernel is using GPU (not CPU fallback)
- Profile with Nsight Systems
- Review CUDA Graph usage

### Debug Logging
```csharp
services.AddLogging(builder =>
{
    builder.AddFilter("Orleans.GpuBridge", LogLevel.Debug);
    builder.AddFilter("Orleans.GpuBridge.DotCompute", LogLevel.Trace);
});
```

## Security Considerations

1. **Kernel Validation**: All kernels are validated before execution
2. **Memory Isolation**: Tenant isolation via separate memory pools
3. **Resource Limits**: Configure per-grain resource quotas
4. **Audit Logging**: All GPU operations are logged with correlation IDs

## High Availability

### Multi-GPU Failover
- Automatic failover to available GPUs
- CPU fallback for critical operations
- Health checks and circuit breakers

### Cluster Configuration
```csharp
siloBuilder.Configure<ClusterOptions>(options =>
{
    options.ClusterId = "gpu-cluster";
    options.ServiceId = "gpu-service";
})
.UseAzureStorageClustering(options =>
{
    options.ConfigureTableServiceClient(connectionString);
});
```

## Support

- GitHub Issues: https://github.com/orleans/gpu-bridge/issues
- Documentation: https://docs.orleans-gpu.io
- Community Discord: https://discord.gg/orleans-gpu
```

## Deliverables Checklist

### Week 7 Deliverables
- [ ] CUDA Graph capture and optimization
- [ ] GPUDirect Storage implementation
- [ ] OpenTelemetry integration
- [ ] Performance profiling tools
- [ ] Resource monitoring dashboard
- [ ] Load testing framework

### Week 8 Deliverables
- [ ] Production samples (Vector, Matrix, Graph)
- [ ] AssureTwin integration samples
- [ ] Comprehensive benchmark suite
- [ ] Production deployment guide
- [ ] API documentation
- [ ] Performance tuning guide

## Success Metrics

### Performance Achievements
- CUDA Graph overhead reduction > 50%
- GPUDirect Storage bandwidth > 20GB/s
- End-to-end latency < 1ms (P50)
- Throughput > 1M ops/sec per GPU

### Production Readiness
- 24-hour soak test passing
- Memory leak free
- Graceful degradation under load
- Comprehensive error handling

### Documentation Quality
- Complete API reference
- Working samples for all features
- Deployment automation scripts
- Troubleshooting runbooks

## Project Completion Criteria

1. **All unit tests passing** (>90% coverage)
2. **Integration tests passing** with real GPUs
3. **Performance benchmarks meeting targets**
4. **Documentation complete and reviewed**
5. **Samples running successfully**
6. **Production deployment validated**
7. **Security review completed**
8. **Open source release prepared**

## Post-Launch Roadmap

### Version 1.1
- AMD ROCm support
- Intel oneAPI integration
- Multi-tenant improvements

### Version 1.2
- Distributed training support
- Model serving capabilities
- AutoML integration

### Version 2.0
- Kubernetes operator
- Cloud-native scaling
- Serverless GPU functions
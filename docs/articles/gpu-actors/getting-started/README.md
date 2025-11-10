# Getting Started with GPU-Native Actors

## Introduction

This guide walks you through building your first GPU-accelerated distributed application using Orleans.GpuBridge.Core. By the end, you'll have a working application that distributes vector operations across multiple GPUs using the actor model.

**Time to complete**: 30-45 minutes

**Prerequisites**:
- .NET 9.0 SDK or later
- NVIDIA GPU with CUDA 11.8+ (or CPU fallback mode)
- Basic C# knowledge
- Familiarity with async/await

## Installation

### 1. Install .NET SDK

Download and install from [dot.net](https://dot.net):

```bash
# Verify installation
dotnet --version
# Should output: 9.0.0 or later
```

### 2. Install CUDA (Optional)

For GPU acceleration, install NVIDIA CUDA Toolkit:

**Windows**:
- Download from [developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)
- Install CUDA 11.8 or later
- Verify: `nvcc --version`

**Linux (Ubuntu)**:
```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get install cuda-11-8

# Add to PATH
export PATH=/usr/local/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH
```

**Note**: CPU fallback mode works without CUDA for development and testing.

### 3. Create Project

```bash
# Create solution and projects
dotnet new sln -n MyGpuApp
dotnet new console -n MyGpuApp.Client
dotnet new classlib -n MyGpuApp.Grains
dotnet new classlib -n MyGpuApp.Interfaces

# Add projects to solution
dotnet sln add MyGpuApp.Client/MyGpuApp.Client.csproj
dotnet sln add MyGpuApp.Grains/MyGpuApp.Grains.csproj
dotnet sln add MyGpuApp.Interfaces/MyGpuApp.Interfaces.csproj

# Add project references
cd MyGpuApp.Client
dotnet add reference ../MyGpuApp.Interfaces/MyGpuApp.Interfaces.csproj
cd ../MyGpuApp.Grains
dotnet add reference ../MyGpuApp.Interfaces/MyGpuApp.Interfaces.csproj
cd ..
```

### 4. Install NuGet Packages

```bash
# Client project
cd MyGpuApp.Client
dotnet add package Microsoft.Orleans.Client
dotnet add package Orleans.GpuBridge.Core

# Grains project
cd ../MyGpuApp.Grains
dotnet add package Microsoft.Orleans.Server
dotnet add package Microsoft.Orleans.Runtime
dotnet add package Orleans.GpuBridge.Core
dotnet add package Orleans.GpuBridge.DotCompute

cd ..
```

## Your First GPU Grain

### Step 1: Define Grain Interface

Edit `MyGpuApp.Interfaces/IVectorAddGrain.cs`:

```csharp
using Orleans;

namespace MyGpuApp.Interfaces;

/// <summary>
/// Grain that performs GPU-accelerated vector addition.
/// </summary>
public interface IVectorAddGrain : IGrainWithIntegerKey
{
    /// <summary>
    /// Adds two vectors on GPU.
    /// </summary>
    /// <param name="a">First vector</param>
    /// <param name="b">Second vector</param>
    /// <returns>Sum of vectors</returns>
    Task<float[]> AddVectorsAsync(float[] a, float[] b);
}
```

### Step 2: Write GPU Kernel

Create `MyGpuApp.Grains/Kernels/VectorAdd.cu`:

```cuda
// CUDA kernel for vector addition
__global__ void vector_add_kernel(
    const float* a,
    const float* b,
    float* c,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n)
    {
        c[idx] = a[idx] + b[idx];
    }
}

// Host function (called from .NET)
extern "C" __declspec(dllexport)
void VectorAdd(const float* a, const float* b, float* c, int n)
{
    // Calculate grid dimensions
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernel
    vector_add_kernel<<<blocksPerGrid, threadsPerBlock>>>(a, b, c, n);

    // Synchronize
    cudaDeviceSynchronize();
}
```

**Note**: For CPU fallback, create `VectorAdd_CPU.cs`:

```csharp
namespace MyGpuApp.Grains.Kernels;

public static class VectorAdd_CPU
{
    public static void Execute(float[] a, float[] b, float[] c)
    {
        for (int i = 0; i < a.Length; i++)
        {
            c[i] = a[i] + b[i];
        }
    }
}
```

### Step 3: Implement Grain

Edit `MyGpuApp.Grains/VectorAddGrain.cs`:

```csharp
using Orleans;
using Orleans.GpuBridge.Abstractions;
using Orleans.Runtime;
using MyGpuApp.Interfaces;

namespace MyGpuApp.Grains;

/// <summary>
/// GPU-accelerated vector addition grain.
/// </summary>
[GpuAccelerated]
public class VectorAddGrain : Grain, IVectorAddGrain
{
    private readonly IGpuBridge _gpuBridge;
    private IGpuKernel<VectorInput, float[]>? _kernel;

    public VectorAddGrain(IGpuBridge gpuBridge)
    {
        _gpuBridge = gpuBridge;
    }

    public override async Task OnActivateAsync(CancellationToken cancellationToken)
    {
        // Load GPU kernel
        _kernel = await _gpuBridge.GetKernelAsync<VectorInput, float[]>(
            "Kernels/VectorAdd");

        await base.OnActivateAsync(cancellationToken);
    }

    public async Task<float[]> AddVectorsAsync(float[] a, float[] b)
    {
        if (a.Length != b.Length)
            throw new ArgumentException("Vectors must have same length");

        if (_kernel == null)
            throw new InvalidOperationException("Kernel not initialized");

        var input = new VectorInput { A = a, B = b };
        return await _kernel.ExecuteAsync(input);
    }
}

/// <summary>
/// Input for vector addition kernel.
/// </summary>
public record VectorInput
{
    public required float[] A { get; init; }
    public required float[] B { get; init; }
}
```

### Step 4: Configure Orleans Silo

Edit `MyGpuApp.Client/Program.cs`:

```csharp
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Orleans;
using Orleans.Hosting;
using Orleans.GpuBridge.Runtime;
using MyGpuApp.Interfaces;

// Build host with Orleans silo
var host = Host.CreateDefaultBuilder(args)
    .UseOrleans((context, siloBuilder) =>
    {
        siloBuilder
            // Use localhost clustering for development
            .UseLocalhostClustering()

            // Configure GPU bridge
            .AddGpuBridge(options =>
            {
                options.PreferGpu = true;  // Use GPU if available
                options.EnableCpuFallback = true;  // Fallback to CPU if no GPU
            })

            // Register kernels
            .ConfigureApplicationParts(parts =>
            {
                parts.AddApplicationPart(typeof(VectorAddGrain).Assembly)
                     .WithReferences();
            });
    })
    .ConfigureLogging(logging =>
    {
        logging.AddConsole();
        logging.SetMinimumLevel(LogLevel.Information);
    })
    .Build();

// Start silo
await host.StartAsync();

Console.WriteLine("Silo started. Press Enter to run example...");
Console.ReadLine();

// Get grain factory
var grainFactory = host.Services.GetRequiredService<IGrainFactory>();

// Example: Add two vectors
await RunVectorAddExample(grainFactory);

// Shutdown
await host.StopAsync();

static async Task RunVectorAddExample(IGrainFactory grainFactory)
{
    Console.WriteLine("\n=== Vector Addition Example ===\n");

    // Create test vectors
    var size = 1_000_000;
    var a = Enumerable.Range(0, size).Select(i => (float)i).ToArray();
    var b = Enumerable.Range(0, size).Select(i => (float)(i * 2)).ToArray();

    Console.WriteLine($"Adding two vectors of size {size:N0}...");

    // Get grain and execute
    var grain = grainFactory.GetGrain<IVectorAddGrain>(0);

    var sw = System.Diagnostics.Stopwatch.StartNew();
    var result = await grain.AddVectorsAsync(a, b);
    sw.Stop();

    Console.WriteLine($"Result: {result[0]}, {result[1]}, ..., {result[^1]}");
    Console.WriteLine($"Completed in {sw.ElapsedMilliseconds}ms");
    Console.WriteLine($"Throughput: {size / sw.Elapsed.TotalSeconds:N0} operations/sec");

    // Verify correctness
    var correct = result[100] == a[100] + b[100];
    Console.WriteLine($"Verification: {(correct ? "PASS" : "FAIL")}");
}
```

### Step 5: Run

```bash
# Build
dotnet build

# Run
dotnet run --project MyGpuApp.Client/MyGpuApp.Client.csproj
```

**Expected output**:
```
Silo started. Press Enter to run example...

=== Vector Addition Example ===

Adding two vectors of size 1,000,000...
Result: 0, 3, ..., 2999997
Completed in 15ms
Throughput: 66,666,667 operations/sec
Verification: PASS
```

## Next Steps: Distribution

### Distribute Across Multiple Grains

Modify the example to distribute work:

```csharp
static async Task RunDistributedVectorAdd(IGrainFactory grainFactory)
{
    Console.WriteLine("\n=== Distributed Vector Addition ===\n");

    // Split work across 10 grains
    var numGrains = 10;
    var sizePerGrain = 100_000;

    var tasks = new List<Task<float[]>>();

    for (int i = 0; i < numGrains; i++)
    {
        var grain = grainFactory.GetGrain<IVectorAddGrain>(i);

        // Create vectors for this grain
        var offset = i * sizePerGrain;
        var a = Enumerable.Range(offset, sizePerGrain)
            .Select(x => (float)x).ToArray();
        var b = Enumerable.Range(offset, sizePerGrain)
            .Select(x => (float)(x * 2)).ToArray();

        // Execute asynchronously
        tasks.Add(grain.AddVectorsAsync(a, b));
    }

    var sw = System.Diagnostics.Stopwatch.StartNew();
    var results = await Task.WhenAll(tasks);
    sw.Stop();

    Console.WriteLine($"Processed {numGrains} grains in parallel");
    Console.WriteLine($"Total size: {numGrains * sizePerGrain:N0}");
    Console.WriteLine($"Completed in {sw.ElapsedMilliseconds}ms");
    Console.WriteLine($"Throughput: {(numGrains * sizePerGrain) / sw.Elapsed.TotalSeconds:N0} ops/sec");
}
```

### Scale to Multiple Silos

For production, deploy across multiple machines:

**Silo configuration**:
```csharp
siloBuilder
    // Use Azure Table for clustering
    .UseAzureStorageClustering(options =>
    {
        options.ConnectionString = "YOUR_STORAGE_CONNECTION_STRING";
    })

    // Or use SQL Server
    .UseAdoNetClustering(options =>
    {
        options.ConnectionString = "YOUR_SQL_CONNECTION_STRING";
        options.Invariant = "System.Data.SqlClient";
    })

    // Configure endpoints
    .ConfigureEndpoints(
        siloPort: 11111,
        gatewayPort: 30000);
```

**Client configuration**:
```csharp
var client = new ClientBuilder()
    .UseAzureStorageClustering(options =>
    {
        options.ConnectionString = "YOUR_STORAGE_CONNECTION_STRING";
    })
    .ConfigureApplicationParts(parts =>
    {
        parts.AddApplicationPart(typeof(IVectorAddGrain).Assembly);
    })
    .Build();

await client.Connect();
```

## Common Patterns

### Pattern 1: Pipeline Processing

Chain multiple GPU operations:

```csharp
public interface IDataPipelineGrain : IGrainWithIntegerKey
{
    Task<Result> ProcessAsync(InputData data);
}

[GpuAccelerated]
public class DataPipelineGrain : Grain, IDataPipelineGrain
{
    [GpuKernel("Kernels/Preprocess")]
    private IGpuKernel<InputData, PreprocessedData> _preprocess;

    [GpuKernel("Kernels/Transform")]
    private IGpuKernel<PreprocessedData, TransformedData> _transform;

    [GpuKernel("Kernels/Aggregate")]
    private IGpuKernel<TransformedData, Result> _aggregate;

    public async Task<Result> ProcessAsync(InputData data)
    {
        var preprocessed = await _preprocess.ExecuteAsync(data);
        var transformed = await _transform.ExecuteAsync(preprocessed);
        var result = await _aggregate.ExecuteAsync(transformed);

        return result;
    }
}
```

### Pattern 2: Batch Processing

Process large datasets efficiently:

```csharp
var results = await GpuPipeline<InputData, Result>
    .For(grainFactory, "data-processor")
    .WithBatchSize(1000)  // Process 1000 items per grain
    .WithParallelism(10)  // Use 10 grains in parallel
    .ExecuteAsync(largeDataset);
```

### Pattern 3: Stateful Computation

Maintain GPU-resident state:

```csharp
[GpuAccelerated]
public class StatefulGpuGrain : Grain, IStatefulGpuGrain
{
    [GpuKernel("Kernels/UpdateState", persistent: true)]
    private IGpuKernel<Event, Statistics> _stateKernel;

    // Ring kernel maintains state across calls
    public async Task ProcessEventAsync(Event evt)
    {
        // State remains on GPU between calls
        await _stateKernel.ExecuteAsync(evt);
    }

    public async Task<Statistics> GetStatisticsAsync()
    {
        // Query GPU-resident state
        return await _stateKernel.ExecuteAsync(new QueryMessage());
    }
}
```

## Debugging

### Enable Logging

```csharp
.ConfigureLogging(logging =>
{
    logging.AddConsole();
    logging.AddDebug();
    logging.SetMinimumLevel(LogLevel.Debug);

    // GPU-specific logging
    logging.AddFilter("Orleans.GpuBridge", LogLevel.Trace);
})
```

### CPU Fallback for Debugging

Set `PreferGpu = false` to debug without GPU:

```csharp
.AddGpuBridge(options =>
{
    options.PreferGpu = false;  // Force CPU fallback
    options.EnableCpuFallback = true;
})
```

### Breakpoints

Standard Visual Studio debugging works for grain code:

```csharp
public async Task<float[]> AddVectorsAsync(float[] a, float[] b)
{
    // Set breakpoint here
    var input = new VectorInput { A = a, B = b };

    // Step into kernel execution
    var result = await _kernel.ExecuteAsync(input);

    return result;
}
```

## Testing

### Unit Tests

```csharp
using Xunit;
using Orleans.TestingHost;

public class VectorAddGrainTests : IClassFixture<ClusterFixture>
{
    private readonly TestCluster _cluster;

    public VectorAddGrainTests(ClusterFixture fixture)
    {
        _cluster = fixture.Cluster;
    }

    [Fact]
    public async Task AddVectors_ReturnsCorrectSum()
    {
        // Arrange
        var grain = _cluster.GrainFactory.GetGrain<IVectorAddGrain>(0);
        var a = new[] { 1.0f, 2.0f, 3.0f };
        var b = new[] { 4.0f, 5.0f, 6.0f };

        // Act
        var result = await grain.AddVectorsAsync(a, b);

        // Assert
        Assert.Equal(new[] { 5.0f, 7.0f, 9.0f }, result);
    }

    [Theory]
    [InlineData(10)]
    [InlineData(100)]
    [InlineData(1000)]
    public async Task AddVectors_WithVariousSizes_ReturnsCorrectResults(int size)
    {
        var grain = _cluster.GrainFactory.GetGrain<IVectorAddGrain>(0);
        var a = Enumerable.Range(0, size).Select(i => (float)i).ToArray();
        var b = Enumerable.Range(0, size).Select(i => (float)i).ToArray();

        var result = await grain.AddVectorsAsync(a, b);

        for (int i = 0; i < size; i++)
        {
            Assert.Equal(a[i] + b[i], result[i]);
        }
    }
}

public class ClusterFixture : IDisposable
{
    public TestCluster Cluster { get; }

    public ClusterFixture()
    {
        var builder = new TestClusterBuilder();
        builder.AddSiloBuilderConfigurator<SiloConfigurator>();
        Cluster = builder.Build();
        Cluster.Deploy();
    }

    public void Dispose()
    {
        Cluster.StopAllSilos();
    }

    private class SiloConfigurator : ISiloConfigurator
    {
        public void Configure(ISiloBuilder siloBuilder)
        {
            siloBuilder.AddGpuBridge(options =>
            {
                options.PreferGpu = false;  // Use CPU for tests
                options.EnableCpuFallback = true;
            });
        }
    }
}
```

## Performance Tuning

### 1. Batch Size

Adjust batch size based on GPU memory:

```csharp
// Small batches: Lower latency, higher overhead
.WithBatchSize(100)

// Large batches: Higher latency, better throughput
.WithBatchSize(10_000)

// Optimal: Balance based on GPU memory
.WithBatchSize(OptimalBatchSize())

static int OptimalBatchSize()
{
    var gpuMemory = GetAvailableGpuMemory();
    var itemSize = Marshal.SizeOf<MyDataType>();

    // Use 80% of available memory
    return (int)(gpuMemory * 0.8 / itemSize);
}
```

### 2. Grain Placement

Configure GPU-aware placement:

```csharp
.AddGpuBridge(options =>
{
    options.PlacementStrategy = GpuPlacementStrategy.LeastLoaded;
    // Or: RoundRobin, ConsistentHash, Custom
})
```

### 3. Memory Management

Reuse GPU buffers:

```csharp
[GpuAccelerated]
public class OptimizedGrain : Grain
{
    private GpuBuffer<float> _bufferPool;

    public override Task OnActivateAsync(CancellationToken cancellationToken)
    {
        // Pre-allocate GPU buffer
        _bufferPool = new GpuBuffer<float>(capacity: 1_000_000);
        return base.OnActivateAsync(cancellationToken);
    }

    public async Task<float[]> ProcessAsync(float[] data)
    {
        // Reuse pre-allocated buffer
        await _bufferPool.CopyFromAsync(data);
        var result = await _kernel.ExecuteAsync(_bufferPool);
        return result;
    }
}
```

## Troubleshooting

### Issue: "CUDA not found"

**Solution**: Install CUDA Toolkit or enable CPU fallback:

```csharp
.AddGpuBridge(options =>
{
    options.PreferGpu = true;
    options.EnableCpuFallback = true;  // Falls back to CPU if CUDA not found
})
```

### Issue: "Out of GPU memory"

**Solution**: Reduce batch size or use streaming:

```csharp
// Process in smaller chunks
var chunkSize = 10_000;
var results = new List<Result>();

for (int i = 0; i < data.Length; i += chunkSize)
{
    var chunk = data.Skip(i).Take(chunkSize).ToArray();
    var result = await grain.ProcessAsync(chunk);
    results.Add(result);
}
```

### Issue: "Grain activation timeout"

**Solution**: Increase timeout or optimize kernel loading:

```csharp
.Configure<GrainCollectionOptions>(options =>
{
    options.CollectionAge = TimeSpan.FromMinutes(10);
})
```

## Next Steps

Now that you have a working GPU-accelerated application:

1. **Add More Kernels**: Implement domain-specific GPU operations
2. **Distribute Further**: Deploy across multiple machines/GPUs
3. **Add Persistence**: Store grain state in databases
4. **Enable Monitoring**: Integrate OpenTelemetry metrics
5. **Optimize Performance**: Profile and tune GPU operations

## Additional Resources

- [Architecture Overview](../architecture/README.md)
- [Use Cases and Applications](../use-cases/README.md)
- [Developer Experience](../developer-experience/README.md)
- [Orleans Documentation](https://learn.microsoft.com/en-us/dotnet/orleans/)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)

## Getting Help

- **GitHub Issues**: Report bugs and request features
- **Stack Overflow**: Tag questions with `orleans` and `gpu`
- **Orleans Discord**: Join the Orleans community
- **Documentation**: Check the comprehensive docs in `/docs`

## Conclusion

You've built your first distributed GPU application using Orleans.GpuBridge.Core. The framework handles distribution, fault tolerance, and GPU management while you focus on business logic.

Key takeaways:
- GPU grains are just Orleans grains with `[GpuAccelerated]` attribute
- Kernels are loaded via `IGpuKernel<TIn, TOut>` interface
- Distribution is automatic through Orleans grain factory
- CPU fallback enables development without GPU hardware

Continue exploring to build production-grade distributed GPU applications.

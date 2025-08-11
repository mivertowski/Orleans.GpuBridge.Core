# Getting Started with Orleans.GpuBridge

This guide will help you get started with Orleans.GpuBridge, from installation to running your first GPU-accelerated Orleans application.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Basic Setup](#basic-setup)
4. [Your First GPU Grain](#your-first-gpu-grain)
5. [Using the Pipeline Framework](#using-the-pipeline-framework)
6. [Testing Your Application](#testing-your-application)
7. [Next Steps](#next-steps)

## Prerequisites

### Required Software

- **.NET 9.0 SDK** or later
- **Orleans 9.0** or later
- **Docker** (optional, for containerized deployment)
- **Visual Studio 2022** / **VS Code** / **Rider** (recommended IDEs)

### GPU Requirements (Optional)

Orleans.GpuBridge will automatically fall back to CPU if no GPU is available, but for GPU acceleration you'll need:

- **NVIDIA GPU**: CUDA 11.0+ and drivers
- **AMD GPU**: ROCm 5.0+ 
- **Intel GPU**: Level Zero runtime
- **Apple Silicon**: Metal support (macOS)

## Installation

### 1. Create a New Orleans Project

```bash
# Create a new Orleans project
dotnet new console -n MyGpuApp
cd MyGpuApp

# Add Orleans packages
dotnet add package Microsoft.Orleans.Server
dotnet add package Microsoft.Orleans.Client
dotnet add package Microsoft.Orleans.Sdk
```

### 2. Add Orleans.GpuBridge Packages

```bash
# Add core GPU bridge packages
dotnet add package Orleans.GpuBridge.Abstractions
dotnet add package Orleans.GpuBridge.Runtime
dotnet add package Orleans.GpuBridge.Grains
dotnet add package Orleans.GpuBridge.BridgeFX
```

## Basic Setup

### 1. Configure the Host

Create a `Program.cs` file:

```csharp
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.DependencyInjection;
using Orleans;
using Orleans.Hosting;
using Orleans.GpuBridge;

var host = new HostBuilder()
    .UseOrleans((context, siloBuilder) =>
    {
        siloBuilder
            .UseLocalhostClustering()
            .ConfigureApplicationParts(parts =>
            {
                parts.AddApplicationPart(typeof(Program).Assembly).WithReferences();
            })
            .AddGpuBridge(options =>
            {
                // Configure GPU bridge
                options.PreferGpu = true;
                options.MemoryPoolSizeMB = 2048;
                options.MaxConcurrentKernels = 50;
                options.EnableProfiling = true;
            })
            .UseGpuPlacement(); // Enable GPU-aware grain placement
    })
    .ConfigureServices(services =>
    {
        services.AddLogging();
    })
    .Build();

await host.RunAsync();
```

### 2. Register Kernels

Add GPU kernels to your configuration:

```csharp
siloBuilder.AddGpuBridge(options => { /* ... */ })
    .AddKernel(kernel => kernel
        .Id("vector_add")
        .In<float[]>()
        .Out<float[]>()
        .FromSource(@"
            __kernel void vector_add(__global float* a, __global float* b, __global float* c, int n) {
                int idx = get_global_id(0);
                if (idx < n) {
                    c[idx] = a[idx] + b[idx];
                }
            }"))
    .AddKernel(kernel => kernel
        .Id("matrix_multiply")
        .In<float[,]>()
        .Out<float[,]>()
        .FromAssembly(typeof(MyKernels).Assembly)); // Load from embedded resources
```

## Your First GPU Grain

### 1. Define the Grain Interface

```csharp
using Orleans;
using System.Threading.Tasks;

public interface IVectorProcessorGrain : IGrainWithIntegerKey
{
    Task<float[]> AddVectorsAsync(float[] a, float[] b);
    Task<float> DotProductAsync(float[] a, float[] b);
    Task<float[]> ScaleVectorAsync(float[] vector, float scalar);
}
```

### 2. Implement the Grain

```csharp
using Orleans;
using Orleans.GpuBridge.Abstractions;
using System;
using System.Linq;
using System.Threading.Tasks;

[GpuResident] // Ensures placement on GPU-capable silo
public class VectorProcessorGrain : Grain, IVectorProcessorGrain
{
    private readonly IGpuBridge _gpu;
    private readonly ILogger<VectorProcessorGrain> _logger;

    public VectorProcessorGrain(IGpuBridge gpu, ILogger<VectorProcessorGrain> logger)
    {
        _gpu = gpu;
        _logger = logger;
    }

    public async Task<float[]> AddVectorsAsync(float[] a, float[] b)
    {
        if (a.Length != b.Length)
            throw new ArgumentException("Vectors must have the same length");

        // Get the vector_add kernel
        var kernel = await _gpu.GetKernelAsync("vector_add");
        
        // Allocate GPU memory
        using var gpuA = await _gpu.AllocateAsync<float>(a.Length);
        using var gpuB = await _gpu.AllocateAsync<float>(b.Length);
        using var gpuC = await _gpu.AllocateAsync<float>(a.Length);
        
        // Copy data to GPU
        await gpuA.CopyToDeviceAsync(a);
        await gpuB.CopyToDeviceAsync(b);
        
        // Execute kernel
        await kernel.ExecuteAsync(new KernelArguments
        {
            Arguments = new object[] { gpuA, gpuB, gpuC, a.Length },
            GlobalWorkSize = new[] { (uint)a.Length },
            LocalWorkSize = new[] { 256u }
        });
        
        // Copy result back
        var result = new float[a.Length];
        await gpuC.CopyFromDeviceAsync(result);
        
        _logger.LogInformation("Processed vector addition of {Length} elements on GPU", a.Length);
        return result;
    }

    public async Task<float> DotProductAsync(float[] a, float[] b)
    {
        // For small operations, CPU might be faster
        if (a.Length < 1000)
        {
            return a.Zip(b, (x, y) => x * y).Sum();
        }

        // Use GPU for larger operations
        var multiplied = await MultiplyVectorsAsync(a, b);
        return multiplied.Sum(); // Final reduction on CPU
    }

    public async Task<float[]> ScaleVectorAsync(float[] vector, float scalar)
    {
        var kernel = await _gpu.GetKernelAsync("scale_vector");
        return await kernel.ExecuteAsync<float[], float[]>(vector, scalar);
    }

    private async Task<float[]> MultiplyVectorsAsync(float[] a, float[] b)
    {
        var kernel = await _gpu.GetKernelAsync("vector_multiply");
        return await kernel.ExecuteAsync<float[], float[]>((a, b));
    }
}
```

### 3. Use the Grain

```csharp
// Create Orleans client
var client = new ClientBuilder()
    .UseLocalhostClustering()
    .Build();

await client.Connect();

// Get grain reference
var grain = client.GetGrain<IVectorProcessorGrain>(0);

// Create test data
var vectorA = Enumerable.Range(1, 10000).Select(i => (float)i).ToArray();
var vectorB = Enumerable.Range(1, 10000).Select(i => (float)(i * 2)).ToArray();

// Execute on GPU
var result = await grain.AddVectorsAsync(vectorA, vectorB);
Console.WriteLine($"First element: {result[0]}, Last element: {result[^1]}");

// Calculate dot product
var dotProduct = await grain.DotProductAsync(vectorA, vectorB);
Console.WriteLine($"Dot product: {dotProduct}");
```

## Using the Pipeline Framework

The BridgeFX pipeline framework allows you to chain multiple GPU operations efficiently:

### 1. Define a Pipeline

```csharp
using Orleans.GpuBridge.BridgeFX;

// Create an image processing pipeline
var pipeline = GpuPipeline<byte[], ProcessedImage>
    .Create()
    .Transform(bytes => Image.FromBytes(bytes))           // Convert bytes to image
    .AddKernel("resize", k => k.Parameters(256, 256))     // Resize on GPU
    .AddKernel("gaussian_blur", k => k.Parameters(5.0f))  // Apply blur on GPU
    .Transform(img => img.ConvertToGrayscale())           // CPU transform
    .Parallel(maxConcurrency: 4)                          // Process 4 images in parallel
    .AddKernel("edge_detection")                          // Detect edges on GPU
    .Filter(result => result.EdgeCount > 100)             // Filter results
    .Batch(32)                                             // Batch for efficiency
    .AddKernel("feature_extraction")                      // Extract features on GPU
    .Transform(features => new ProcessedImage(features))   // Final transform
    .Build();

// Execute the pipeline
var images = LoadImages();
var processedImages = await pipeline.ExecuteAsync(images);
```

### 2. Create a Reusable Pipeline Grain

```csharp
public interface IImageProcessorGrain : IGrainWithStringKey
{
    Task<ProcessedImage[]> ProcessBatchAsync(byte[][] images);
}

[GpuResident]
public class ImageProcessorGrain : Grain, IImageProcessorGrain
{
    private ExecutablePipeline<byte[], ProcessedImage> _pipeline;

    public override Task OnActivateAsync(CancellationToken ct)
    {
        _pipeline = GpuPipeline<byte[], ProcessedImage>
            .Create()
            .AddKernel("resize", k => k.Parameters(256, 256))
            .AddKernel("normalize")
            .AddKernel("classify")
            .Build();
            
        return base.OnActivateAsync(ct);
    }

    public async Task<ProcessedImage[]> ProcessBatchAsync(byte[][] images)
    {
        return await _pipeline.ExecuteAsync(images);
    }
}
```

## Testing Your Application

### 1. Unit Testing GPU Grains

```csharp
using Xunit;
using Moq;
using Orleans.TestingHost;

public class VectorProcessorGrainTests
{
    [Fact]
    public async Task AddVectors_Should_Return_Correct_Sum()
    {
        // Arrange
        var builder = new TestClusterBuilder();
        builder.AddSiloBuilderConfigurator<TestSiloConfigurator>();
        var cluster = builder.Build();
        await cluster.DeployAsync();

        var grain = cluster.GrainFactory.GetGrain<IVectorProcessorGrain>(0);
        var a = new float[] { 1, 2, 3 };
        var b = new float[] { 4, 5, 6 };

        // Act
        var result = await grain.AddVectorsAsync(a, b);

        // Assert
        Assert.Equal(new float[] { 5, 7, 9 }, result);
        
        await cluster.StopAllSilosAsync();
    }
}

public class TestSiloConfigurator : ISiloConfigurator
{
    public void Configure(ISiloBuilder siloBuilder)
    {
        siloBuilder.AddGpuBridge(options =>
        {
            options.PreferGpu = false; // Use CPU fallback for testing
        });
    }
}
```

### 2. Performance Testing

```csharp
[Fact]
public async Task Benchmark_Vector_Addition_Performance()
{
    var grain = _cluster.GrainFactory.GetGrain<IVectorProcessorGrain>(0);
    var size = 1_000_000;
    var a = new float[size];
    var b = new float[size];
    
    var sw = Stopwatch.StartNew();
    var result = await grain.AddVectorsAsync(a, b);
    sw.Stop();
    
    _output.WriteLine($"Vector addition of {size} elements took {sw.ElapsedMilliseconds}ms");
    Assert.True(sw.ElapsedMilliseconds < 100, "Operation too slow");
}
```

## Next Steps

### Learn More

1. **[API Reference](api-reference.md)** - Detailed API documentation
2. **[Architecture Guide](design.md)** - Understanding the system architecture
3. **[Operations Guide](operations.md)** - Deployment and monitoring
4. **[Advanced Features](advanced.md)** - Multi-GPU, custom kernels, optimization

### Example Projects

Check out our example projects in the `samples/` directory:

- **VectorMath** - Basic vector operations
- **ImageProcessing** - Image processing pipeline
- **MachineLearning** - Neural network inference
- **Scientific** - Scientific computing examples

### Common Patterns

#### 1. Caching Compiled Kernels

```csharp
public class KernelCache
{
    private readonly ConcurrentDictionary<string, IKernel> _cache = new();
    
    public async Task<IKernel> GetOrCompileAsync(string id, string source)
    {
        return await _cache.GetOrAddAsync(id, async _ =>
        {
            return await _gpu.CompileKernelAsync(source);
        });
    }
}
```

#### 2. Memory Pooling

```csharp
// Reuse memory allocations for better performance
public class GpuMemoryPool<T> where T : unmanaged
{
    private readonly ConcurrentBag<IGpuMemory<T>> _pool = new();
    
    public async Task<PooledMemory<T>> RentAsync(int size)
    {
        if (_pool.TryTake(out var memory) && memory.Length >= size)
        {
            return new PooledMemory<T>(memory, this);
        }
        
        return new PooledMemory<T>(
            await _gpu.AllocateAsync<T>(size), 
            this);
    }
}
```

#### 3. Error Handling

```csharp
try
{
    var result = await kernel.ExecuteAsync(data);
}
catch (GpuOutOfMemoryException ex)
{
    _logger.LogWarning("GPU out of memory, falling back to CPU");
    result = await ExecuteOnCpu(data);
}
catch (KernelCompilationException ex)
{
    _logger.LogError(ex, "Kernel compilation failed");
    throw;
}
```

### Troubleshooting

#### GPU Not Detected

```bash
# Check GPU availability
nvidia-smi  # For NVIDIA
rocm-smi    # For AMD

# Check Orleans.GpuBridge detection
dotnet run --project tools/GpuDetector
```

#### Performance Issues

1. Enable profiling: `options.EnableProfiling = true`
2. Check memory pool statistics
3. Monitor kernel execution times
4. Use batching for small operations

#### Memory Leaks

1. Always dispose GPU memory allocations
2. Use `using` statements or `try-finally` blocks
3. Monitor memory pool statistics
4. Enable memory leak detection in debug builds

### Community and Support

- **GitHub Issues**: Report bugs and request features
- **Discussions**: Ask questions and share experiences
- **Discord**: Join our community chat
- **Stack Overflow**: Tag questions with `orleans-gpubridge`

---

**Ready to accelerate your Orleans applications? Let's get started!** 🚀
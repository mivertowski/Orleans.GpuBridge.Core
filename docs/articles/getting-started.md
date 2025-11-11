# Getting Started with Orleans.GpuBridge.Core

This guide will walk you through creating your first GPU-accelerated Orleans application using Orleans.GpuBridge.Core.

## Prerequisites

Before you begin, ensure you have:

- **.NET 9.0 SDK** or later
- **NVIDIA GPU** with CUDA 12.0+ or **AMD GPU** with ROCm 5.0+
- **Windows 10/11** or **Linux** (Ubuntu 22.04+)
- Basic familiarity with **Microsoft Orleans** and **C#**

### Verify GPU Setup

```bash
# For NVIDIA GPUs
nvidia-smi

# For AMD GPUs
rocm-smi
```

## Step 1: Create a New Orleans Project

```bash
# Create a new console application
dotnet new console -n MyGpuApp
cd MyGpuApp

# Add Orleans packages
dotnet add package Microsoft.Orleans.Server
dotnet add package Microsoft.Orleans.Client

# Add GPU Bridge packages
dotnet add package Orleans.GpuBridge.Core
dotnet add package Orleans.GpuBridge.Backends.DotCompute
```

## Step 2: Define Your Grain Interface

Create a file `IVectorAddGrain.cs`:

```csharp
using Orleans;

namespace MyGpuApp;

/// <summary>
/// Grain interface for GPU-accelerated vector addition
/// </summary>
public interface IVectorAddGrain : IGrainWithIntegerKey
{
    /// <summary>
    /// Adds two vectors using GPU acceleration
    /// </summary>
    /// <param name="a">First vector</param>
    /// <param name="b">Second vector</param>
    /// <returns>Sum of the two vectors</returns>
    Task<float[]> AddVectorsAsync(float[] a, float[] b);
}
```

## Step 3: Implement Your GPU Grain

Create a file `VectorAddGrain.cs`:

```csharp
using Orleans;
using Orleans.GpuBridge.Abstractions;
using Orleans.Runtime;

namespace MyGpuApp;

/// <summary>
/// GPU-accelerated grain for vector addition
/// </summary>
[GpuAccelerated]
public class VectorAddGrain : Grain, IVectorAddGrain
{
    private readonly ILogger<VectorAddGrain> _logger;

    [GpuKernel("kernels/VectorAdd")]
    private IGpuKernel<VectorAddInput, float[]>? _kernel;

    public VectorAddGrain(ILogger<VectorAddGrain> logger)
    {
        _logger = logger;
    }

    public override Task OnActivateAsync(CancellationToken cancellationToken)
    {
        _logger.LogInformation("VectorAddGrain {GrainId} activated", this.GetPrimaryKeyLong());
        return base.OnActivateAsync(cancellationToken);
    }

    public async Task<float[]> AddVectorsAsync(float[] a, float[] b)
    {
        if (a.Length != b.Length)
        {
            throw new ArgumentException("Vectors must have the same length");
        }

        _logger.LogInformation("Processing vector addition of length {Length}", a.Length);

        if (_kernel == null)
        {
            throw new InvalidOperationException("GPU kernel not initialized");
        }

        var input = new VectorAddInput { A = a, B = b };
        return await _kernel.ExecuteAsync(input);
    }
}

/// <summary>
/// Input data for vector addition kernel
/// </summary>
public record VectorAddInput
{
    public required float[] A { get; init; }
    public required float[] B { get; init; }
}
```

## Step 4: Implement the GPU Kernel

Create a file `VectorAddKernel.cs`:

```csharp
using Orleans.GpuBridge.Abstractions;

namespace MyGpuApp;

/// <summary>
/// CPU fallback implementation of vector addition
/// (GPU implementation would use CUDA/ROCm)
/// </summary>
public class VectorAddKernel : IGpuKernel<VectorAddInput, float[]>
{
    public Task<float[]> ExecuteAsync(VectorAddInput input)
    {
        // CPU fallback implementation
        var result = new float[input.A.Length];
        for (int i = 0; i < input.A.Length; i++)
        {
            result[i] = input.A[i] + input.B[i];
        }
        return Task.FromResult(result);
    }

    public void Dispose()
    {
        // Cleanup GPU resources if needed
    }
}
```

For production GPU implementation, you would write a CUDA kernel:

```cuda
// VectorAdd.cu - CUDA kernel implementation
__global__ void vectorAdd(const float* a, const float* b, float* c, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        c[idx] = a[idx] + b[idx];
    }
}
```

## Step 5: Configure the Silo (Server)

Update `Program.cs` to configure the Orleans silo:

```csharp
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Orleans;
using Orleans.GpuBridge.Runtime;
using Orleans.Hosting;
using MyGpuApp;

var builder = Host.CreateDefaultBuilder(args)
    .UseOrleans((context, siloBuilder) =>
    {
        siloBuilder
            .UseLocalhostClustering()
            .ConfigureApplicationParts(parts =>
            {
                parts.AddApplicationPart(typeof(VectorAddGrain).Assembly).WithReferences();
            })
            .UseDashboard(options => options.Port = 8080);

        // Configure GPU Bridge
        siloBuilder.Services.AddGpuBridge(options =>
        {
            options.PreferGpu = true;
            options.EnableRingKernels = false; // Start with simple offload model
        })
        .AddKernel(k => k
            .Id("kernels/VectorAdd")
            .In<VectorAddInput>()
            .Out<float[]>()
            .FromFactory(sp => new VectorAddKernel()));
    })
    .ConfigureLogging(logging =>
    {
        logging.AddConsole();
        logging.SetMinimumLevel(LogLevel.Information);
    });

var host = builder.Build();
await host.RunAsync();
```

## Step 6: Create a Client

Create a file `Client.cs` for testing:

```csharp
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Orleans;
using Orleans.Hosting;

namespace MyGpuApp;

public static class ClientApp
{
    public static async Task RunClientAsync()
    {
        var client = new HostBuilder()
            .UseOrleansClient(clientBuilder =>
            {
                clientBuilder.UseLocalhostClustering();
            })
            .ConfigureLogging(logging =>
            {
                logging.AddConsole();
                logging.SetMinimumLevel(LogLevel.Warning);
            })
            .Build();

        await client.StartAsync();
        var grainFactory = client.Services.GetRequiredService<IGrainFactory>();

        Console.WriteLine("Orleans GPU Bridge - Vector Addition Demo");
        Console.WriteLine("==========================================\n");

        // Get the grain
        var grain = grainFactory.GetGrain<IVectorAddGrain>(0);

        // Test with sample data
        var vectorA = new float[] { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f };
        var vectorB = new float[] { 10.0f, 20.0f, 30.0f, 40.0f, 50.0f };

        Console.WriteLine($"Vector A: [{string.Join(", ", vectorA)}]");
        Console.WriteLine($"Vector B: [{string.Join(", ", vectorB)}]");

        var result = await grain.AddVectorsAsync(vectorA, vectorB);

        Console.WriteLine($"Result:   [{string.Join(", ", result)}]");

        // Benchmark performance
        Console.WriteLine("\nRunning performance benchmark...");
        var largeVectorSize = 1_000_000;
        var largeA = Enumerable.Range(0, largeVectorSize).Select(i => (float)i).ToArray();
        var largeB = Enumerable.Range(0, largeVectorSize).Select(i => (float)i * 2).ToArray();

        var sw = System.Diagnostics.Stopwatch.StartNew();
        var largeResult = await grain.AddVectorsAsync(largeA, largeB);
        sw.Stop();

        Console.WriteLine($"Processed {largeVectorSize:N0} elements in {sw.ElapsedMilliseconds}ms");
        Console.WriteLine($"Throughput: {largeVectorSize / sw.Elapsed.TotalSeconds:N0} elements/second");

        await client.StopAsync();
    }
}
```

Update `Program.cs` to include client execution:

```csharp
// Add this to Program.cs after host initialization
var siloTask = host.RunAsync();

// Wait for silo to start
await Task.Delay(5000);

// Run client
await ClientApp.RunClientAsync();

await siloTask;
```

## Step 7: Run Your Application

```bash
# Build the project
dotnet build

# Run the application
dotnet run
```

You should see output similar to:

```
Orleans GPU Bridge - Vector Addition Demo
==========================================

Vector A: [1, 2, 3, 4, 5]
Vector B: [10, 20, 30, 40, 50]
Result:   [11, 22, 33, 44, 55]

Running performance benchmark...
Processed 1,000,000 elements in 45ms
Throughput: 22,222,222 elements/second
```

## Step 8: Enable GPU Acceleration

To use actual GPU execution instead of CPU fallback:

1. **Implement CUDA Kernel**: Create `VectorAdd.cu` with CUDA code
2. **Compile Kernel**: Use `nvcc` to compile to PTX or cubin
3. **Load Kernel**: Update `VectorAddKernel` to load compiled GPU code
4. **Configure Backend**: Ensure DotCompute backend is properly configured

Example with DotCompute backend:

```csharp
public class VectorAddKernel : IGpuKernel<VectorAddInput, float[]>
{
    private readonly IDotComputeDevice _device;
    private readonly IDotComputeKernel _kernel;

    public VectorAddKernel(IDotComputeDevice device)
    {
        _device = device;
        _kernel = _device.LoadKernel("VectorAdd.ptx", "vectorAdd");
    }

    public async Task<float[]> ExecuteAsync(VectorAddInput input)
    {
        var n = input.A.Length;
        var result = new float[n];

        // Allocate GPU memory
        using var gpuA = _device.Allocate(input.A);
        using var gpuB = _device.Allocate(input.B);
        using var gpuC = _device.Allocate<float>(n);

        // Launch kernel
        var blockSize = 256;
        var gridSize = (n + blockSize - 1) / blockSize;

        await _kernel.LaunchAsync(
            gridDim: new Dim3(gridSize),
            blockDim: new Dim3(blockSize),
            args: new object[] { gpuA, gpuB, gpuC, n }
        );

        // Copy result back
        await gpuC.CopyToAsync(result);

        return result;
    }

    public void Dispose()
    {
        _kernel?.Dispose();
    }
}
```

## Next Steps

Now that you have a working GPU-accelerated Orleans application, explore:

1. **[Concepts and Background](concepts.md)** - Learn about GPU-native actors and ring kernels
2. **[Architecture Overview](architecture.md)** - Understand the system design
3. **[Hypergraph Actors](hypergraph-actors/getting-started/README.md)** - Build multi-way relationships
4. **[Temporal Correctness](temporal/introduction/README.md)** - Implement HLC and Vector Clocks
5. **[Advanced Examples](https://github.com/mivertowski/Orleans.GpuBridge.Core/tree/main/samples)** - Production patterns

## Troubleshooting

### GPU Not Detected

If your GPU is not detected:

```bash
# Check GPU drivers
nvidia-smi  # or rocm-smi for AMD

# Verify CUDA installation
nvcc --version

# Check .NET GPU support
dotnet --info
```

### Performance Issues

If performance is lower than expected:

1. Enable ring kernels for persistent GPU state
2. Increase batch sizes for better GPU utilization
3. Profile with NVIDIA Nsight or AMD ROCProfiler
4. Check memory transfer overhead

### Common Errors

**"Kernel not found"**: Ensure kernel ID matches registration
**"GPU memory allocation failed"**: Check available GPU memory
**"Device not available"**: Verify GPU drivers and CUDA installation

## Resources

- **[API Reference](../api/index.md)** - Complete API documentation
- **[GPU-Native Actors Guide](gpu-actors/getting-started/README.md)** - Advanced GPU patterns
- **[GitHub Repository](https://github.com/mivertowski/Orleans.GpuBridge.Core)** - Source code and examples
- **[Orleans Documentation](https://learn.microsoft.com/en-us/dotnet/orleans/)** - Microsoft Orleans reference

---

[← Back to Home](../index.md) | [Next: Concepts →](concepts.md)

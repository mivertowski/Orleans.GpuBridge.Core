# Orleans.GpuBridge.Core Quick Start Guide (RC1)

Welcome to Orleans.GpuBridge.Core! This guide will help you get started with GPU-accelerated computing in your Orleans applications.

## Table of Contents

1. [Introduction](#introduction)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Your First GPU Kernel](#your-first-gpu-kernel)
5. [Ring Kernel API Basics](#ring-kernel-api-basics)
6. [Sample Applications](#sample-applications)
7. [Next Steps](#next-steps)
8. [Troubleshooting](#troubleshooting)
9. [Getting Help](#getting-help)

## Introduction

Orleans.GpuBridge.Core is a comprehensive GPU acceleration framework for Microsoft Orleans that enables distributed GPU computing across Orleans clusters. It bridges the gap between Orleans' powerful distributed actor model and modern GPU computing capabilities.

### What You Can Do With Orleans.GpuBridge.Core

- **Accelerate Compute-Intensive Operations**: Execute vector operations, matrix multiplications, image processing, and more on GPU hardware
- **Seamless Orleans Integration**: Use GPU acceleration directly within your Orleans grains
- **Automatic Fallback**: Intelligently falls back to CPU when GPU resources are unavailable
- **Multiple Backend Support**: Choose between ILGPU, DotCompute, or custom backends
- **Production-Ready**: Built-in monitoring, health checks, and resilience features

### Key Features

- 🚀 **15-74x Performance**: Significant speedups over CPU-only implementations
- 🔄 **Automatic Fallback**: Graceful degradation to CPU when GPU unavailable
- 🎯 **Type-Safe API**: Strongly-typed kernel execution with compile-time safety
- 📊 **Built-in Monitoring**: Real-time metrics and performance tracking
- 🌐 **Distributed Computing**: Scale GPU workloads across Orleans clusters

## Prerequisites

### Minimum Requirements

Before you begin, ensure you have:

- **.NET 9.0 SDK or later**: [Download here](https://dotnet.microsoft.com/download/dotnet/9.0)
- **Orleans 9.0 or later**: Included as NuGet dependencies
- **4GB RAM**: Minimum system memory
- **Any x64 or ARM64 processor**: CPU-only mode always available

### Recommended Requirements (For GPU Acceleration)

For optimal GPU performance, you should have:

- **NVIDIA GPU with 8GB+ VRAM**: For CUDA backend (GTX 1070 or better)
- **AMD GPU with 8GB+ VRAM**: For OpenCL backend
- **16GB System RAM**: For handling large datasets
- **CUDA 12.0+ or OpenCL 2.0+**: Latest GPU drivers installed

> **💡 Note**: Orleans.GpuBridge.Core works without GPU hardware! It automatically falls back to optimized CPU implementations when GPU resources are unavailable.

### Development Tools

- **Visual Studio 2022 (17.8+)** or **Visual Studio Code** with C# Dev Kit
- **Git**: For cloning sample projects
- **.NET CLI**: Included with .NET SDK

## Installation

### Step 1: Create Your Orleans Project

If you don't already have an Orleans project, create one:

```bash
# Create a new console application
dotnet new console -n MyGpuOrleansApp
cd MyGpuOrleansApp

# Add Orleans packages
dotnet add package Microsoft.Orleans.Server
dotnet add package Microsoft.Orleans.Client
```

### Step 2: Install Orleans.GpuBridge.Core Packages

Install the core packages:

```bash
# Core runtime (required)
dotnet add package Orleans.GpuBridge.Runtime

# Pre-built GPU grains (recommended)
dotnet add package Orleans.GpuBridge.Grains

# Choose a backend (at least one required)
dotnet add package Orleans.GpuBridge.Backends.ILGPU      # For CUDA/OpenCL
# OR
dotnet add package Orleans.GpuBridge.Backends.DotCompute # Alternative backend
```

> **🔍 Package Selection Guide**:
> - **ILGPU Backend**: Best for NVIDIA/AMD GPUs, mature and well-tested
> - **DotCompute Backend**: Cross-platform, supports multiple GPU APIs
> - You can install both and let the framework choose the best available

### Step 3: Verify Installation

Create a simple test to verify everything is installed:

```bash
dotnet build
```

If the build succeeds, you're ready to go! 🎉

## Your First GPU Kernel

Let's create a simple vector addition application that demonstrates GPU acceleration.

### Step 1: Configure Orleans with GPU Bridge

Open your `Program.cs` and add the GPU Bridge configuration:

```csharp
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.DependencyInjection;
using Orleans.GpuBridge.Runtime.Extensions;
using Orleans.GpuBridge.Abstractions.Providers;

var builder = Host.CreateDefaultBuilder(args)
    .UseOrleans(siloBuilder =>
    {
        siloBuilder
            .UseLocalhostClustering()
            .AddMemoryGrainStorage("gpu-storage");
    })
    .ConfigureServices((context, services) =>
    {
        // Add GPU Bridge with configuration
        services.AddGpuBridge(options =>
        {
            options.PreferGpu = true;           // Try GPU first
            options.FallbackToCpu = true;       // Fall back to CPU if needed
            options.BatchSize = 1024;           // Default batch size
            options.MaxRetries = 3;             // Retry failed operations
        })
        // Add available backends
        .AddAllAvailableBackends()              // Auto-discover ILGPU/DotCompute
        .AddCpuFallbackBackend()                // Always add CPU fallback
        .ConfigureBackendSelection(selection =>
        {
            // Prefer GPU backends, fallback to CPU
            selection.PreferredBackends = new List<GpuBackend>
            {
                GpuBackend.Cuda,
                GpuBackend.OpenCL,
                GpuBackend.Cpu
            };
            selection.AllowCpuFallback = true;
            selection.MinimumDeviceMemory = 256 * 1024 * 1024; // 256 MB minimum
        });
    });

var host = builder.Build();
await host.RunAsync();
```

### Step 2: Create Your First Kernel

Create a simple vector addition kernel. Create a new file `VectorAddKernel.cs`:

```csharp
using Orleans.GpuBridge.Abstractions;
using Orleans.GpuBridge.Abstractions.Kernels;

namespace MyGpuOrleansApp;

/// <summary>
/// Vector addition kernel - adds two vectors element-wise
/// </summary>
[GpuKernel("vector-add")]
public class VectorAddKernel : IGpuKernel<VectorPair, float[]>
{
    public string Id => "vector-add";

    public ValueTask<float[]> ExecuteAsync(
        VectorPair input,
        CancellationToken cancellationToken = default)
    {
        // This implementation serves as CPU fallback
        // GPU backends will automatically optimize this
        var result = new float[input.A.Length];

        for (int i = 0; i < input.A.Length; i++)
        {
            result[i] = input.A[i] + input.B[i];
        }

        return ValueTask.FromResult(result);
    }
}

/// <summary>
/// Input data structure for vector addition
/// </summary>
public record VectorPair(float[] A, float[] B);
```

> **💡 How It Works**:
> - The `[GpuKernel]` attribute registers this kernel with the framework
> - The CPU implementation is used as a fallback when GPU is unavailable
> - GPU backends automatically optimize the execution path

### Step 3: Create a GPU-Accelerated Grain

Create a grain that uses your kernel. Create `VectorGrain.cs`:

```csharp
using Orleans;
using Orleans.GpuBridge.Abstractions;
using Orleans.GpuBridge.Abstractions.Grains;

namespace MyGpuOrleansApp;

public interface IVectorGrain : IGrainWithStringKey
{
    Task<float[]> AddVectorsAsync(float[] a, float[] b);
    Task<GpuDeviceInfo> GetDeviceInfoAsync();
}

[GpuAccelerated] // Marks this grain for GPU-aware placement
public class VectorGrain : Grain, IVectorGrain
{
    private readonly IGpuBridge _gpuBridge;

    public VectorGrain(IGpuBridge gpuBridge)
    {
        _gpuBridge = gpuBridge;
    }

    public async Task<float[]> AddVectorsAsync(float[] a, float[] b)
    {
        // Create input
        var input = new VectorPair(a, b);

        // Execute on GPU (or CPU fallback)
        var kernel = await _gpuBridge.GetKernelAsync<VectorPair, float[]>(
            new KernelId("vector-add"));

        return await kernel.ExecuteAsync(input);
    }

    public async Task<GpuDeviceInfo> GetDeviceInfoAsync()
    {
        var info = await _gpuBridge.GetInfoAsync();
        var devices = await _gpuBridge.GetDevicesAsync();

        return new GpuDeviceInfo
        {
            DeviceCount = devices.Count,
            PrimaryDevice = devices.Count > 0 ? devices[0].Name : "CPU Fallback",
            IsGpuAvailable = devices.Count > 0
        };
    }
}

public record GpuDeviceInfo
{
    public int DeviceCount { get; init; }
    public string PrimaryDevice { get; init; } = string.Empty;
    public bool IsGpuAvailable { get; init; }
}
```

### Step 4: Use Your GPU-Accelerated Grain

Now let's use the grain in your application. Update `Program.cs`:

```csharp
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Orleans;
using Orleans.Hosting;
using Orleans.GpuBridge.Runtime.Extensions;
using Orleans.GpuBridge.Abstractions.Providers;
using MyGpuOrleansApp;

var builder = Host.CreateDefaultBuilder(args)
    .UseOrleans(siloBuilder =>
    {
        siloBuilder
            .UseLocalhostClustering()
            .AddMemoryGrainStorage("gpu-storage");
    })
    .ConfigureServices((context, services) =>
    {
        services.AddGpuBridge(options =>
        {
            options.PreferGpu = true;
            options.FallbackToCpu = true;
            options.BatchSize = 1024;
        })
        .AddAllAvailableBackends()
        .AddCpuFallbackBackend();
    })
    .ConfigureLogging(logging =>
    {
        logging.AddConsole();
        logging.SetMinimumLevel(LogLevel.Information);
    });

var host = builder.Build();
await host.StartAsync();

// Get the grain factory
var grainFactory = host.Services.GetRequiredService<IGrainFactory>();

// Create a vector grain
var vectorGrain = grainFactory.GetGrain<IVectorGrain>("my-vector-grain");

// Check GPU availability
var deviceInfo = await vectorGrain.GetDeviceInfoAsync();
Console.WriteLine($"GPU Available: {deviceInfo.IsGpuAvailable}");
Console.WriteLine($"Device: {deviceInfo.PrimaryDevice}");
Console.WriteLine($"Device Count: {deviceInfo.DeviceCount}");
Console.WriteLine();

// Generate test vectors
var vectorSize = 1_000_000;
var a = GenerateRandomVector(vectorSize);
var b = GenerateRandomVector(vectorSize);

Console.WriteLine($"Adding two vectors of size {vectorSize:N0}...");

// Execute vector addition
var stopwatch = System.Diagnostics.Stopwatch.StartNew();
var result = await vectorGrain.AddVectorsAsync(a, b);
stopwatch.Stop();

Console.WriteLine($"Completed in {stopwatch.ElapsedMilliseconds} ms");
Console.WriteLine($"Throughput: {vectorSize / stopwatch.Elapsed.TotalSeconds:N0} elements/sec");
Console.WriteLine();

// Verify correctness
var isCorrect = VerifyResult(a, b, result);
Console.WriteLine($"Result verification: {(isCorrect ? "✓ PASSED" : "✗ FAILED")}");

await host.StopAsync();

static float[] GenerateRandomVector(int size)
{
    var random = new Random(42);
    var vector = new float[size];
    for (int i = 0; i < size; i++)
    {
        vector[i] = (float)random.NextDouble() * 100;
    }
    return vector;
}

static bool VerifyResult(float[] a, float[] b, float[] result)
{
    const float epsilon = 0.0001f;
    for (int i = 0; i < Math.Min(100, a.Length); i++)
    {
        var expected = a[i] + b[i];
        if (Math.Abs(result[i] - expected) > epsilon)
            return false;
    }
    return true;
}
```

### Step 5: Run Your Application

```bash
dotnet run
```

You should see output similar to:

```
GPU Available: True
Device: NVIDIA GeForce RTX 4090
Device Count: 1

Adding two vectors of size 1,000,000...
Completed in 3 ms
Throughput: 333,333,333 elements/sec

Result verification: ✓ PASSED
```

🎉 **Congratulations!** You've created your first GPU-accelerated Orleans application!

## Ring Kernel API Basics

The Ring Kernel API is the core abstraction for GPU kernel execution in Orleans.GpuBridge.Core.

### Core Concepts

#### 1. Kernel Definition

Kernels are defined using the `IGpuKernel<TIn, TOut>` interface:

```csharp
[GpuKernel("my-kernel-id")]
public class MyKernel : IGpuKernel<InputType, OutputType>
{
    public string Id => "my-kernel-id";

    public ValueTask<OutputType> ExecuteAsync(
        InputType input,
        CancellationToken cancellationToken = default)
    {
        // CPU fallback implementation
        // GPU backends will optimize this automatically
    }
}
```

#### 2. Kernel Execution

Execute kernels through the `IGpuBridge` interface:

```csharp
// Get a kernel instance
var kernel = await _gpuBridge.GetKernelAsync<InputType, OutputType>(
    new KernelId("my-kernel-id"));

// Execute the kernel
var result = await kernel.ExecuteAsync(input);
```

#### 3. GPU-Accelerated Grains

Mark grains for GPU-aware placement:

```csharp
[GpuAccelerated] // Orleans will place on nodes with GPU resources
public class MyGrain : Grain, IMyGrain
{
    private readonly IGpuBridge _gpuBridge;

    public MyGrain(IGpuBridge gpuBridge)
    {
        _gpuBridge = gpuBridge;
    }
}
```

### Common Kernel Patterns

#### Vector Operations

```csharp
[GpuKernel("vector-dot")]
public class DotProductKernel : IGpuKernel<VectorPair, float>
{
    public string Id => "vector-dot";

    public ValueTask<float> ExecuteAsync(
        VectorPair input,
        CancellationToken cancellationToken = default)
    {
        float sum = 0f;
        for (int i = 0; i < input.A.Length; i++)
        {
            sum += input.A[i] * input.B[i];
        }
        return ValueTask.FromResult(sum);
    }
}
```

#### Matrix Operations

```csharp
[GpuKernel("matrix-multiply")]
public class MatrixMultiplyKernel : IGpuKernel<MatrixPair, float[,]>
{
    public string Id => "matrix-multiply";

    public ValueTask<float[,]> ExecuteAsync(
        MatrixPair input,
        CancellationToken cancellationToken = default)
    {
        // Matrix multiplication implementation
        // GPU backends optimize this automatically
    }
}
```

#### Batch Processing

```csharp
[GpuKernel("batch-process")]
public class BatchProcessKernel : IGpuKernel<float[][], float[]>
{
    public string Id => "batch-process";

    public ValueTask<float[]> ExecuteAsync(
        float[][] batch,
        CancellationToken cancellationToken = default)
    {
        // Process entire batch on GPU
        // More efficient than individual operations
    }
}
```

### Advanced Features

#### Execution Hints

Control execution behavior with hints:

```csharp
var hints = new GpuExecutionHints
{
    PreferGpu = true,           // Prefer GPU over CPU
    MaxBatchSize = 2048,        // Maximum batch size
    AllowCpuFallback = true,    // Fall back to CPU if GPU fails
    Timeout = TimeSpan.FromSeconds(30)
};

var result = await kernel.ExecuteAsync(input, hints);
```

#### Device Selection

Choose specific GPU devices:

```csharp
var devices = await _gpuBridge.GetDevicesAsync();

// Get device with most memory
var bestDevice = devices
    .OrderByDescending(d => d.TotalMemoryBytes)
    .First();

var kernel = await _gpuBridge.GetKernelAsync<TIn, TOut>(
    kernelId,
    deviceId: bestDevice.Id);
```

#### Performance Monitoring

Monitor kernel execution:

```csharp
var info = await _gpuBridge.GetInfoAsync();

Console.WriteLine($"Total Executions: {info.TotalExecutions}");
Console.WriteLine($"Avg Execution Time: {info.AverageExecutionTime}");
Console.WriteLine($"GPU Utilization: {info.GpuUtilization:P}");
Console.WriteLine($"Memory Usage: {info.MemoryUsageBytes / (1024*1024)} MB");
```

## Sample Applications

Orleans.GpuBridge.Core includes several comprehensive sample applications demonstrating different use cases.

### Available Samples

#### 1. Vector Operations (`samples/Orleans.GpuBridge.Samples`)

Demonstrates basic vector operations including:
- Vector addition
- Dot product
- Normalization
- Reduction operations

```bash
cd samples/Orleans.GpuBridge.Samples
dotnet run -- vector --size 1000000 --batch 10
```

#### 2. Matrix Operations

Shows matrix computations:
- Matrix multiplication
- Matrix transpose
- Matrix inversion

```bash
dotnet run -- matrix --size 1024 --count 5
```

#### 3. Image Processing

GPU-accelerated image operations:
- Image resizing
- Convolution filters
- Edge detection

```bash
dotnet run -- image --path myimage.jpg --operation all
```

#### 4. Graph Processing

Large-scale graph algorithms:
- PageRank
- Shortest path
- Graph traversal

```bash
dotnet run -- graph --nodes 10000 --edges 50000 --algorithm pagerank
```

#### 5. Performance Benchmarks

Compare CPU vs GPU performance:

```bash
dotnet run -- benchmark --type all --duration 30
```

#### 6. Backend Provider Demo

Demonstrates backend selection and switching:

```bash
dotnet run -- backend
```

### Interactive Mode

Run samples interactively:

```bash
cd samples/Orleans.GpuBridge.Samples
dotnet run -- interactive
```

This launches an interactive menu where you can:
- Choose different samples
- Configure parameters
- See real-time performance metrics
- Compare GPU vs CPU execution

## Next Steps

Now that you have Orleans.GpuBridge.Core up and running, here are some recommended next steps:

### 📚 Learn More

1. **Architecture Deep Dive**: Read the [Architecture Overview](starter-kit/DESIGN.md)
2. **Kernel Development**: Study the [Kernel Development Guide](starter-kit/KERNELS.md)
3. **Performance Tuning**: Explore [Performance Optimization](starter-kit/OPERATIONS.md)
4. **API Reference**: Browse the full [API Documentation](api/index.md)

### 🔨 Build Something

1. **Create Custom Kernels**: Implement kernels for your specific use case
2. **Optimize Placement**: Configure GPU-aware grain placement strategies
3. **Monitor Performance**: Set up telemetry and monitoring
4. **Scale Out**: Deploy across multiple GPU-equipped nodes

### 📖 Advanced Topics

- **Multi-Backend Configuration**: Use multiple GPU backends simultaneously
- **Persistent Kernel Hosts**: Keep kernels loaded for faster execution
- **Stream Processing**: GPU-accelerate Orleans streams
- **Custom Memory Pools**: Optimize memory management for your workload

### 🎯 Example Projects

Try building these projects:

1. **Real-time Image Processing Pipeline**: Process video frames using GPU-accelerated grains
2. **Machine Learning Inference Service**: Deploy ML models with GPU acceleration
3. **Scientific Computing Cluster**: Distributed numerical simulations
4. **Financial Analytics Engine**: High-performance financial calculations

## Troubleshooting

### Common Issues and Solutions

#### Issue: "No GPU devices found"

**Symptoms**: Application falls back to CPU, logs show no GPU devices.

**Solutions**:
1. **Check GPU drivers**: Ensure latest GPU drivers are installed
   ```bash
   # NVIDIA
   nvidia-smi

   # AMD (Linux)
   rocm-smi
   ```

2. **Verify CUDA/OpenCL**: Check that GPU APIs are accessible
   ```bash
   # CUDA
   nvcc --version

   # OpenCL
   clinfo
   ```

3. **Check backend installation**: Ensure backend packages are installed
   ```bash
   dotnet list package | grep GpuBridge
   ```

4. **Try explicit backend**: Configure specific backend
   ```csharp
   services.AddGpuBridge()
       .AddILGPUBackend(); // Explicit ILGPU
   ```

#### Issue: "Kernel execution timeout"

**Symptoms**: Operations fail with timeout errors.

**Solutions**:
1. **Increase timeout**: Adjust execution timeout
   ```csharp
   options.ExecutionTimeout = TimeSpan.FromMinutes(5);
   ```

2. **Reduce batch size**: Process smaller batches
   ```csharp
   options.BatchSize = 512; // Smaller batches
   ```

3. **Check GPU memory**: Ensure sufficient GPU memory
   ```csharp
   var devices = await _gpuBridge.GetDevicesAsync();
   foreach (var device in devices)
   {
       Console.WriteLine($"Memory: {device.AvailableMemoryBytes / (1024*1024)} MB");
   }
   ```

#### Issue: "CPU fallback too slow"

**Symptoms**: Performance is poor when GPU unavailable.

**Solutions**:
1. **Optimize CPU implementation**: Use SIMD and parallelization
   ```csharp
   Parallel.For(0, input.Length, i =>
   {
       result[i] = Process(input[i]);
   });
   ```

2. **Enable CPU optimizations**: Configure CPU backend
   ```csharp
   .AddCpuFallbackBackend(options =>
   {
       options.EnableParallelProcessing = true;
       options.EnableSimd = true;
   });
   ```

3. **Consider hybrid approach**: Use GPU for large batches, CPU for small ones
   ```csharp
   if (input.Length > 10000)
       result = await ExecuteOnGpu(input);
   else
       result = await ExecuteOnCpu(input);
   ```

#### Issue: "Out of GPU memory"

**Symptoms**: Operations fail with memory allocation errors.

**Solutions**:
1. **Process in smaller batches**: Reduce batch size
   ```csharp
   options.BatchSize = 256; // Smaller batches
   options.MaxMemoryUsageBytes = 1024 * 1024 * 512; // 512 MB limit
   ```

2. **Enable memory pooling**: Reuse GPU memory
   ```csharp
   options.EnableMemoryPooling = true;
   options.PoolSize = 1024 * 1024 * 256; // 256 MB pool
   ```

3. **Monitor memory usage**: Track memory consumption
   ```csharp
   var info = await _gpuBridge.GetInfoAsync();
   Console.WriteLine($"Memory: {info.MemoryUsageBytes / (1024*1024)} MB");
   ```

#### Issue: "Build errors with backend packages"

**Symptoms**: Compilation fails after adding backend packages.

**Solutions**:
1. **Clear NuGet cache**:
   ```bash
   dotnet nuget locals all --clear
   dotnet restore
   ```

2. **Check package versions**: Ensure compatible versions
   ```bash
   dotnet list package --include-transitive
   ```

3. **Update all packages**:
   ```bash
   dotnet add package Orleans.GpuBridge.Runtime --version [latest]
   dotnet add package Orleans.GpuBridge.Backends.ILGPU --version [latest]
   ```

### Performance Issues

#### Slow Execution

1. **Profile kernel execution**: Use built-in profiling
   ```csharp
   options.EnableProfiling = true;
   ```

2. **Check data transfer overhead**: Minimize CPU↔GPU transfers
3. **Optimize batch sizes**: Experiment with different batch sizes
4. **Use persistent kernels**: Keep kernels loaded

#### High Latency

1. **Enable kernel caching**: Reuse compiled kernels
   ```csharp
   options.EnableKernelCaching = true;
   ```

2. **Pre-warm kernels**: Initialize kernels at startup
   ```csharp
   var kernel = await _gpuBridge.GetKernelAsync<TIn, TOut>(kernelId);
   // Kernel is now loaded and ready
   ```

3. **Use async patterns**: Don't block on GPU operations

### Getting Logs

Enable detailed logging for troubleshooting:

```csharp
.ConfigureLogging(logging =>
{
    logging.AddConsole();
    logging.SetMinimumLevel(LogLevel.Debug); // Verbose logs
    logging.AddFilter("Orleans.GpuBridge", LogLevel.Trace);
});
```

## Getting Help

### Documentation

- **Quick Start Guide**: You're reading it! 📖
- **Architecture Overview**: [docs/starter-kit/DESIGN.md](starter-kit/DESIGN.md)
- **Kernel Development**: [docs/starter-kit/KERNELS.md](starter-kit/KERNELS.md)
- **API Reference**: [docs/api/index.md](api/index.md)

### Community & Support

- **GitHub Issues**: [Report bugs or request features](https://github.com/yourusername/Orleans.GpuBridge.Core/issues)
- **GitHub Discussions**: [Ask questions and share ideas](https://github.com/yourusername/Orleans.GpuBridge.Core/discussions)
- **Stack Overflow**: Tag questions with `orleans-gpubridge`

### Contributing

We welcome contributions! See our [Contributing Guide](CONTRIBUTING.md) for:
- Code contributions
- Documentation improvements
- Bug reports
- Feature requests
- Sample applications

### Commercial Support

For commercial support, enterprise features, and consulting services:
- **Email**: support@orleans-gpubridge.dev
- **Website**: https://orleans-gpubridge.dev

---

## Quick Reference Card

### Installation

```bash
# Core packages
dotnet add package Orleans.GpuBridge.Runtime
dotnet add package Orleans.GpuBridge.Grains

# Backend (choose one or both)
dotnet add package Orleans.GpuBridge.Backends.ILGPU
dotnet add package Orleans.GpuBridge.Backends.DotCompute
```

### Basic Setup

```csharp
services.AddGpuBridge(options =>
{
    options.PreferGpu = true;
    options.FallbackToCpu = true;
})
.AddAllAvailableBackends()
.AddCpuFallbackBackend();
```

### Define a Kernel

```csharp
[GpuKernel("kernel-id")]
public class MyKernel : IGpuKernel<TIn, TOut>
{
    public string Id => "kernel-id";
    public ValueTask<TOut> ExecuteAsync(TIn input, CancellationToken ct = default)
    {
        // Implementation
    }
}
```

### Use in a Grain

```csharp
[GpuAccelerated]
public class MyGrain : Grain, IMyGrain
{
    private readonly IGpuBridge _gpuBridge;

    public MyGrain(IGpuBridge gpuBridge) => _gpuBridge = gpuBridge;

    public async Task<TOut> ProcessAsync(TIn input)
    {
        var kernel = await _gpuBridge.GetKernelAsync<TIn, TOut>(kernelId);
        return await kernel.ExecuteAsync(input);
    }
}
```

### Check GPU Status

```csharp
var info = await _gpuBridge.GetInfoAsync();
var devices = await _gpuBridge.GetDevicesAsync();
```

---

**Built with ❤️ for the Orleans and GPU computing communities**

*Orleans.GpuBridge.Core RC1 - Production-ready GPU acceleration for Orleans*

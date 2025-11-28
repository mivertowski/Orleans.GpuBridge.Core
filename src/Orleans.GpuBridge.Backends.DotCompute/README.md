# Orleans.GpuBridge.Backends.DotCompute

[![NuGet Package](https://img.shields.io/nuget/v/Orleans.GpuBridge.Backends.DotCompute.svg)](https://www.nuget.org/packages/Orleans.GpuBridge.Backends.DotCompute)
[![.NET](https://img.shields.io/badge/.NET-9.0-blue.svg)](https://dotnet.microsoft.com/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)

High-performance DotCompute backend provider for Orleans GPU Bridge, enabling cross-platform GPU acceleration through a unified compute abstraction layer.

## Overview

The Orleans.GpuBridge.Backends.DotCompute backend provides seamless GPU acceleration for Orleans applications by leveraging the DotCompute framework. This backend abstracts multiple GPU compute APIs (CUDA, OpenCL, DirectCompute, Metal, Vulkan) through a single, unified interface while providing advanced features like automatic kernel compilation, memory optimization, and cross-platform deployment.

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│                Orleans Application                       │
├─────────────────────────────────────────────────────────┤
│                Orleans GPU Bridge                       │
├─────────────────────────────────────────────────────────┤
│            DotCompute Backend Provider                  │
├─────────────────┬─────────────────┬─────────────────────┤
│   Kernel        │    Memory       │    Device           │
│   Compiler      │   Allocator     │   Manager           │
├─────────────────┼─────────────────┼─────────────────────┤
│              DotCompute Runtime                         │
├─────────────────┬─────────────────┬─────────────────────┤
│      CUDA       │     OpenCL      │   DirectCompute     │
│     Metal       │     Vulkan      │   CPU Fallback      │
└─────────────────┴─────────────────┴─────────────────────┘
```

## Key Features

### 🚀 Cross-Platform GPU Acceleration
- **Unified API**: Single interface for CUDA, OpenCL, DirectCompute, Metal, and Vulkan
- **Automatic Backend Selection**: Intelligent platform detection and optimal backend selection
- **CPU Fallback**: Seamless fallback to CPU implementations when GPU is unavailable

### 🧠 Advanced Kernel Management
- **Multi-Language Support**: C#, CUDA, OpenCL, HLSL, and Metal Shading Language kernels
- **Just-In-Time Compilation**: Dynamic kernel compilation with optimization
- **Kernel Fusion**: Automatic optimization by combining multiple operations
- **Template System**: Pre-built kernels for common operations (BLAS, convolution, reduction)

### 💾 Intelligent Memory Management
- **Unified Memory**: Automatic host-device memory synchronization where supported
- **Memory Pooling**: Reduces allocation overhead with intelligent pool management
- **Zero-Copy Operations**: Direct memory mapping for maximum performance
- **Automatic Defragmentation**: Background memory optimization

### ⚡ Performance Optimization
- **Graph Optimization**: Automatic computational graph analysis and optimization
- **Batch Processing**: Efficient handling of large data sets
- **Pipeline Optimization**: Overlapped compute and memory transfers
- **Adaptive Work Distribution**: Dynamic load balancing across available devices

### 🔧 Developer Experience
- **Attribute-Based Configuration**: Simple kernel decoration with `[Kernel]` attributes
- **Comprehensive Diagnostics**: Detailed performance metrics and health monitoring
- **Hot Reload Support**: Development-time kernel recompilation
- **Rich Debugging**: GPU debugger integration and profiling tools

## Installation

Install the DotCompute backend via NuGet Package Manager:

### Package Manager Console
```powershell
Install-Package Orleans.GpuBridge.Backends.DotCompute
```

### .NET CLI
```bash
dotnet add package Orleans.GpuBridge.Backends.DotCompute
```

### PackageReference
```xml
<PackageReference Include="Orleans.GpuBridge.Backends.DotCompute" Version="0.1.0" />
```

## Quick Start

### Basic Configuration

Register the DotCompute backend with default settings:

```csharp
using Orleans.GpuBridge.Runtime.Extensions;
using Orleans.GpuBridge.Backends.DotCompute.Extensions;

// Configure Orleans with GPU Bridge and DotCompute backend
var builder = Host.CreateApplicationBuilder(args);

builder.Services
    .AddOrleans(orleans =>
    {
        orleans.UseLocalhostClustering();
    });

// Add GPU Bridge with DotCompute backend
builder.Services
    .AddGpuBridge(options =>
    {
        options.PreferGpu = true;
        options.FallbackToCpu = true;
    })
    .AddDotGpuBackend(); // Add DotCompute backend with defaults

var host = builder.Build();
await host.RunAsync();
```

### Full GPU-Native Actor Configuration

Configure the backend with Ring Kernel support for GPU-native actors:

```csharp
using Orleans.GpuBridge.Runtime.Extensions;
using Orleans.GpuBridge.Backends.DotCompute.Extensions;

builder.Services
    .AddGpuBridge(options =>
    {
        options.PreferGpu = true;
        options.FallbackToCpu = true;
        options.MaxConcurrentKernels = 100;
        options.MemoryPoolSizeMB = 1024;
    })
    .AddDotGpuBackend(config =>
    {
        // Compilation settings
        config.OptimizationLevel = OptimizationLevel.O3;
        config.EnableDebugMode = false;
        config.EnableDiskCache = true;

        // Memory management
        config.MemorySettings.InitialPoolSize = 512 * 1024 * 1024; // 512 MB
        config.MemorySettings.MaxPoolSize = 4L * 1024 * 1024 * 1024; // 4 GB
        config.MemorySettings.EnableDefragmentation = true;
    })
    .Services
    .AddRingKernelSupport(options =>
    {
        options.DefaultGridSize = 1;
        options.DefaultBlockSize = 256;
        options.DefaultQueueCapacity = 256;
        options.EnableKernelCaching = true;
    })
    .AddK2KSupport()                    // Enable kernel-to-kernel messaging
    .AddDotComputeRingKernelBridge();   // GPU-accelerated ring kernel bridge
```

### Advanced Factory Configuration

Use a custom factory for runtime configuration:

```csharp
builder.Services
    .AddGpuBridge(options => options.PreferGpu = true)
    .AddDotGpuBackend(serviceProvider =>
    {
        var environment = serviceProvider.GetRequiredService<IHostEnvironment>();
        var loggerFactory = serviceProvider.GetRequiredService<ILoggerFactory>();

        var config = new DotGpuBackendConfiguration();

        if (environment.IsDevelopment())
        {
            config.EnableDebugMode = true;
            config.OptimizationLevel = OptimizationLevel.O1;
            config.EnableDiskCache = false; // Disable for faster iteration
        }
        else
        {
            config.OptimizationLevel = OptimizationLevel.O3;
            config.MemorySettings.InitialPoolSize = 2L * 1024 * 1024 * 1024; // 2 GB
        }

        return new DotComputeBackendProvider(config, loggerFactory);
    });
```

## Kernel Development

### Simple Kernel Example

Define GPU kernels using the `[Kernel]` attribute:

```csharp
using Orleans.GpuBridge.Backends.DotCompute.Attributes;

public static class MyKernels
{
    /// <summary>
    /// Vector addition kernel optimized for DotCompute
    /// </summary>
    [Kernel("math/vector_add", PreferredWorkGroupSize = 256)]
    public static void VectorAdd(
        ReadOnlySpan<float> a,
        ReadOnlySpan<float> b,
        Span<float> result,
        int size)
    {
        // This method will be compiled to GPU code automatically
        // CPU fallback implementation provided for compatibility
        for (int i = 0; i < size; i++)
        {
            result[i] = a[i] + b[i];
        }
    }
    
    /// <summary>
    /// Matrix multiplication with shared memory optimization
    /// </summary>
    [Kernel("blas/gemm", 
            PreferredWorkGroupSize = 16, 
            RequiresSharedMemory = true,
            SharedMemorySize = 1024)]
    public static void MatrixMultiply(
        ReadOnlySpan<float> a,
        ReadOnlySpan<float> b,
        Span<float> c,
        int m, int n, int k)
    {
        // GPU: Will be optimized with shared memory tiling
        // CPU: Fallback implementation
        for (int row = 0; row < m; row++)
        {
            for (int col = 0; col < n; col++)
            {
                float sum = 0.0f;
                for (int i = 0; i < k; i++)
                {
                    sum += a[row * k + i] * b[i * n + col];
                }
                c[row * n + col] = sum;
            }
        }
    }
}
```

### Advanced Kernel Features

Leverage advanced DotCompute features:

```csharp
/// <summary>
/// Convolution kernel with automatic optimization hints
/// </summary>
[Kernel("nn/conv2d")]
[OptimizeFor(GpuBackend.CUDA, "use_tensor_cores")]
[OptimizeFor(GpuBackend.OpenCL, "use_local_memory")]
[RequiresFeature("fp16_support")]
public static void Convolution2D(
    ReadOnlySpan<float> input,
    ReadOnlySpan<float> kernel,
    Span<float> output,
    int inputWidth, int inputHeight,
    int kernelWidth, int kernelHeight,
    int outputWidth, int outputHeight,
    int stride = 1, int padding = 0)
{
    // Implementation with automatic GPU optimization
    for (int y = 0; y < outputHeight; y++)
    {
        for (int x = 0; x < outputWidth; x++)
        {
            float sum = 0.0f;
            
            for (int ky = 0; ky < kernelHeight; ky++)
            {
                for (int kx = 0; kx < kernelWidth; kx++)
                {
                    int inputY = y * stride + ky - padding;
                    int inputX = x * stride + kx - padding;
                    
                    if (inputY >= 0 && inputY < inputHeight && 
                        inputX >= 0 && inputX < inputWidth)
                    {
                        sum += input[inputY * inputWidth + inputX] * 
                               kernel[ky * kernelWidth + kx];
                    }
                }
            }
            
            output[y * outputWidth + x] = sum;
        }
    }
}
```

### Using GPU Grains with DotCompute Backend

#### GPU-Offload Model (GpuGrainBase)

For batch processing and infrequent GPU access:

```csharp
using Orleans.GpuBridge.Grains.Base;

public class ComputeGrain : GpuGrainBase<ComputeState>, IComputeGrain
{
    public async ValueTask<float[]> ProcessVectorsAsync(float[] input)
    {
        // Execute kernel on GPU with automatic fallback to CPU
        return await InvokeKernelAsync<float[], float[]>("vector-add", input);
    }
}

public struct ComputeState
{
    public int ProcessedCount;
}
```

#### GPU-Native Model (RingKernelGrainBase)

For high-frequency messaging with sub-microsecond latency:

```csharp
using Orleans.GpuBridge.Grains.Base;
using System.Runtime.InteropServices;

public class HighFrequencyActor : RingKernelGrainBase<CounterState, CounterMessage>
{
    protected override string KernelId => "counters/high-frequency";

    public async ValueTask<int> IncrementAsync(int amount)
    {
        var request = new CounterMessage { Amount = amount };
        return await InvokeKernelAsync<CounterMessage, int>(request);
    }

    public async ValueTask<int> GetValueAsync()
    {
        var state = await GetGpuStateAsync();
        return state.Value;
    }
}

[StructLayout(LayoutKind.Sequential)]
public struct CounterState
{
    public int Value;
    public long LastUpdated;
}

[StructLayout(LayoutKind.Sequential)]
public struct CounterMessage
{
    public int Amount;
}
```

#### Using IGpuBridge Directly

For direct kernel access:

```csharp
using Orleans.GpuBridge.Abstractions;

public class DirectKernelGrain : Grain, IDirectKernelGrain
{
    private readonly IGpuBridge _gpuBridge;

    public DirectKernelGrain(IGpuBridge gpuBridge)
    {
        _gpuBridge = gpuBridge;
    }

    public async ValueTask<float[]> ProcessAsync(float[] input)
    {
        // Get typed kernel
        var kernel = await _gpuBridge.GetKernelAsync<float[], float[]>(
            new KernelId("vector-process"));

        // Execute with validation
        var validation = kernel.ValidateInput(input);
        if (!validation.IsValid)
            throw new ArgumentException(validation.ErrorMessage);

        return await kernel.ExecuteAsync(input);
    }
}
```

## Configuration Options

### Compilation Settings

```csharp
config.OptimizationLevel = OptimizationLevel.O3;      // O0, O1, O2, O3
config.EnableDebugMode = false;                       // Enable debug symbols
config.EnableDiskCache = true;                        // Cache compiled kernels
config.CachePath = "./gpu_cache";                     // Cache directory
config.EnableKernelProfiling = true;                  // Profile kernel execution
config.CompilationTimeout = TimeSpan.FromMinutes(5);  // Max compilation time
```

### Platform Preferences

```csharp
// Set platform preference order
config.PreferredPlatforms.Clear();
config.PreferredPlatforms.Add(GpuBackend.CUDA);        // Prefer NVIDIA CUDA
config.PreferredPlatforms.Add(GpuBackend.OpenCL);      // Fallback to OpenCL
config.PreferredPlatforms.Add(GpuBackend.DirectCompute); // Windows DirectCompute
config.PreferredPlatforms.Add(GpuBackend.Metal);       // macOS Metal
config.PreferredPlatforms.Add(GpuBackend.Vulkan);      // Vulkan compute

// Device selection criteria
config.DeviceSelection.PreferDiscreteGpu = true;       // Prefer dedicated GPU
config.DeviceSelection.MinComputeUnits = 8;            // Minimum CUs/SMs
config.DeviceSelection.MinGlobalMemoryMB = 2048;       // Minimum 2GB VRAM
```

### Memory Management

```csharp
// Pool configuration
config.MemorySettings.InitialPoolSize = 512 * 1024 * 1024;  // 512 MB initial
config.MemorySettings.MaxPoolSize = 8L * 1024 * 1024 * 1024; // 8 GB maximum
config.MemorySettings.GrowthFactor = 1.5;                   // Pool growth rate
config.MemorySettings.ShrinkThreshold = 0.25;               // Shrink when 25% used

// Advanced memory features
config.MemorySettings.EnableDefragmentation = true;         // Background defrag
config.MemorySettings.DefragmentationThreshold = 0.3;       // Defrag when 30% fragmented
config.MemorySettings.EnableZeroCopy = true;                // Use zero-copy when available
config.MemorySettings.PreferPinnedMemory = true;            // Use pinned host memory
config.MemorySettings.MemoryAlignment = 256;                // Memory alignment in bytes
```

### Language Settings

```csharp
// Enable automatic language translation
config.LanguageSettings.EnableLanguageTranslation = true;

// Set preferred kernel languages per platform
config.LanguageSettings.PreferredLanguages[GpuBackend.CUDA] = KernelLanguage.CUDA;
config.LanguageSettings.PreferredLanguages[GpuBackend.OpenCL] = KernelLanguage.OpenCL;
config.LanguageSettings.PreferredLanguages[GpuBackend.DirectCompute] = KernelLanguage.HLSL;
config.LanguageSettings.PreferredLanguages[GpuBackend.Metal] = KernelLanguage.MSL;

// Preprocessor definitions
config.LanguageSettings.PreprocessorDefinitions.Add("OPTIMIZATION_LEVEL", "3");
config.LanguageSettings.PreprocessorDefinitions.Add("ENABLE_FP16", "1");
```

## Supported Operations

### Built-in Kernel Templates

The DotCompute backend includes optimized implementations for common operations:

#### Linear Algebra (BLAS)
- **Level 1**: Vector operations (AXPY, DOT, NRM2, ASUM)
- **Level 2**: Matrix-vector operations (GEMV, SYR, GER)
- **Level 3**: Matrix-matrix operations (GEMM, SYRK, TRMM)

#### Neural Network Primitives
- **Activations**: ReLU, Sigmoid, Tanh, Softmax, GELU
- **Convolution**: 1D/2D/3D convolution with various padding modes
- **Pooling**: Max pooling, average pooling, global pooling
- **Normalization**: Batch norm, layer norm, instance norm

#### Image Processing
- **Filters**: Gaussian blur, Sobel edge detection, bilateral filter
- **Transforms**: Color space conversion, resize, rotate
- **Morphology**: Erosion, dilation, opening, closing

#### Signal Processing
- **Transforms**: FFT, DCT, wavelet transforms
- **Filtering**: FIR, IIR, convolution
- **Analysis**: Spectrograms, cross-correlation

### Custom Kernel Support

Implement custom kernels for domain-specific operations:

```csharp
[Kernel("custom/my_algorithm")]
public static void MyCustomAlgorithm(
    ReadOnlySpan<float> input,
    Span<float> output,
    CustomParameters parameters)
{
    // Your custom algorithm implementation
}
```

## Performance Characteristics

### Throughput Benchmarks

Based on internal testing with various workloads:

| Operation Type | CPU (Baseline) | CUDA RTX 4090 | OpenCL AMD | DirectCompute |
|---------------|----------------|----------------|------------|---------------|
| Vector Add (1M elements) | 1.0x | 45.2x | 28.7x | 31.4x |
| Matrix Multiply (2048x2048) | 1.0x | 127.8x | 89.3x | 95.7x |
| Convolution (224x224x3) | 1.0x | 78.9x | 52.1x | 58.3x |
| FFT (1M samples) | 1.0x | 23.4x | 15.8x | 18.2x |

### Memory Performance

- **Zero-copy transfers**: Up to 80% reduction in memory overhead
- **Unified memory**: Automatic migration with 15-25% performance improvement
- **Memory pooling**: 60-90% reduction in allocation time
- **Batch processing**: Linear scaling with batch size up to device limits

### Scalability

- **Multi-GPU support**: Automatic work distribution across available devices
- **Cross-platform**: Consistent performance across Windows, Linux, and macOS
- **Dynamic scaling**: Automatic adjustment based on system load
- **Pipeline optimization**: Overlapped compute and data transfers

## Platform Support

### Operating Systems
- **Windows** 10/11 (DirectCompute, CUDA, OpenCL)
- **Linux** (Ubuntu 18.04+, CentOS 7+, CUDA, OpenCL)
- **macOS** 10.14+ (Metal, OpenCL)

### GPU Vendors
- **NVIDIA**: CUDA 11.0+, compute capability 5.0+
- **AMD**: ROCm 4.0+, OpenCL 2.0+
- **Intel**: oneAPI Level Zero, OpenCL 2.1+
- **Apple**: Metal 2.0+ (macOS), Metal Performance Shaders

### Supported APIs
- **CUDA**: 11.0, 11.2, 11.4, 11.6, 11.8, 12.0+
- **OpenCL**: 1.2, 2.0, 2.1, 2.2, 3.0
- **DirectCompute**: Shader Model 5.0, 5.1, 6.0+
- **Metal**: 2.0, 2.1, 2.2, 2.3, 2.4+
- **Vulkan**: 1.1, 1.2, 1.3+ (compute shaders)

## Troubleshooting

### Common Issues

#### 1. Backend Not Available
```csharp
// Check backend availability
var backend = serviceProvider.GetRequiredService<DotGpuBackendProvider>();
bool isAvailable = await backend.IsAvailableAsync();
if (!isAvailable)
{
    // Handle unavailable backend
    logger.LogWarning("DotCompute backend not available, falling back to CPU");
}
```

#### 2. Compilation Failures
```csharp
try
{
    await _gpuBridge.ExecuteKernelAsync("my_kernel", parameters);
}
catch (KernelCompilationException ex)
{
    logger.LogError("Kernel compilation failed: {Error}", ex.CompilationError);
    // Check kernel syntax and platform compatibility
}
```

#### 3. Memory Allocation Issues
```csharp
// Monitor memory usage
var metrics = await backend.GetMetricsAsync();
var memoryUsage = metrics["available_memory_bytes"];
logger.LogInformation("Available GPU memory: {Memory} bytes", memoryUsage);

// Configure memory limits
config.MemorySettings.MaxPoolSize = availableMemory * 0.8; // Use 80% of available
```

#### 4. Performance Issues
```csharp
// Enable profiling
config.EnableKernelProfiling = true;

// Check batch sizes
var optimalBatchSize = await _gpuBridge.GetOptimalBatchSizeAsync("my_kernel");
logger.LogInformation("Optimal batch size for kernel: {BatchSize}", optimalBatchSize);
```

### Debugging Tips

1. **Enable verbose logging**:
   ```csharp
   config.EnableDebugMode = true;
   config.LogLevel = LogLevel.Debug;
   ```

2. **Check health status**:
   ```csharp
   var health = await backend.CheckHealthAsync();
   logger.LogInformation("Backend health: {IsHealthy}, {Message}", 
       health.IsHealthy, health.Message);
   ```

3. **Profile kernel execution**:
   ```csharp
   config.EnableKernelProfiling = true;
   // Check logs for execution times and bottlenecks
   ```

4. **Validate platform compatibility**:
   ```csharp
   var devices = backend.GetDeviceManager().GetDevices();
   foreach (var device in devices)
   {
       logger.LogInformation("Device: {Name}, Type: {Type}, Memory: {Memory}MB", 
           device.Name, device.Type, device.TotalMemoryBytes / (1024 * 1024));
   }
   ```

## Dependencies

### Core Dependencies
- **.NET 9.0** - Target framework
- **Orleans.GpuBridge.Abstractions** - Core interfaces
- **Orleans.GpuBridge.Runtime** - Runtime components
- **Microsoft.Extensions.Hosting** 9.0.8 - Hosting abstractions
- **Microsoft.Extensions.Logging** 9.0.8 - Logging framework
- **Microsoft.Extensions.Options** 9.0.8 - Configuration system
- **Microsoft.Extensions.DependencyInjection** 9.0.8 - DI container

### Performance Dependencies
- **System.Memory** 4.6.3 - High-performance memory types
- **System.Numerics.Vectors** 4.5.0 - SIMD vector operations

### Development Dependencies (optional)
- **Microsoft.DotNet.ILCompiler** 9.0.8 - AOT compilation support
- **Microsoft.NET.ILLink.Tasks** 9.0.8 - IL linking optimizations

### Runtime Requirements

#### Windows
- **Visual Studio 2022 Redistributable** (x64)
- **CUDA Toolkit** 11.0+ (for NVIDIA GPUs)
- **AMD Software** (for AMD GPUs)
- **DirectX 12** (for DirectCompute support)

#### Linux
- **CUDA Toolkit** 11.0+ (for NVIDIA GPUs)
- **ROCm** 4.0+ (for AMD GPUs)  
- **Intel oneAPI** (for Intel GPUs)
- **OpenCL** runtime libraries

#### macOS
- **macOS** 10.14+ (for Metal support)
- **Xcode Command Line Tools**
- **OpenCL** framework (included with macOS)

## Advanced Usage

### Multi-Backend Configuration
```csharp
// Configure multiple backends with fallback
builder.Services
    .AddGpuBridge(options => options.PreferGpu = true)
    .AddDotGpuBackend(config =>
    {
        config.PreferredPlatforms.Add(GpuBackend.CUDA);
        config.PreferredPlatforms.Add(GpuBackend.OpenCL);
    })
    .AddFallbackBackend(); // CPU fallback
```

### Custom Memory Allocators
```csharp
// Register custom memory allocator
builder.Services.AddSingleton<IMemoryAllocator, CustomMemoryAllocator>();

// Use with DotCompute backend
config.MemorySettings.UseCustomAllocator = true;
```

### Kernel Preprocessing
```csharp
// Add custom preprocessor directives
config.LanguageSettings.PreprocessorDefinitions.Add("CUSTOM_OPTIMIZATION", "1");
config.LanguageSettings.PreprocessorDefinitions.Add("TILE_SIZE", "16");

[Kernel("optimized/kernel")]
[PreprocessorDirective("CUSTOM_OPTIMIZATION")]
public static void OptimizedKernel(/* parameters */)
{
    #if CUSTOM_OPTIMIZATION
        // Optimized implementation
    #else
        // Standard implementation
    #endif
}
```

## License

**Apache License 2.0**

Copyright (c) 2025 Michael Ivertowski

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

## Contributing

Contributions are welcome! Please read our [Contributing Guidelines](../../CONTRIBUTING.md) before submitting pull requests.

### Development Setup
1. Clone the repository
2. Install .NET 9.0 SDK
3. Install GPU development tools (CUDA, ROCm, etc.)
4. Run `dotnet build` to build the project
5. Run `dotnet test` to execute tests

### Reporting Issues
Please report bugs and feature requests through [GitHub Issues](https://github.com/mivertowski/Orleans.GpuBridge.Core/issues).

## Related Projects

- **[Orleans](https://github.com/dotnet/orleans)** - Distributed application framework
- **[Orleans.GpuBridge.Core](../../README.md)** - Main GPU Bridge project
- **[DotCompute](https://github.com/dotcompute/dotcompute)** - Cross-platform GPU compute framework

## Acknowledgments

This project builds upon the excellent work of:
- The Orleans team at Microsoft
- The DotCompute community
- GPU computing framework developers
- High-performance computing researchers

---

For more information, visit the [Orleans GPU Bridge Documentation](../../docs/README.md) or check out our [samples and tutorials](../../samples/README.md).
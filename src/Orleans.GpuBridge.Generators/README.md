# Orleans.GpuBridge.Generators

Roslyn source generators for GPU-native actor code generation in Orleans GPU Bridge.

## Overview

Orleans.GpuBridge.Generators provides compile-time code generation for GPU-native actors. It automatically generates message structs, kernel implementations, and grain wrappers from annotated interfaces, enabling seamless GPU acceleration with minimal boilerplate.

## Key Features

- **Automatic Message Struct Generation**: Generates GPU-friendly message structures
- **Kernel Code Generation**: Creates CUDA/OpenCL kernel boilerplate
- **Grain Implementation Generation**: Generates Orleans grain implementations
- **Compile-Time Validation**: Catches configuration errors during build
- **Message Size Calculation**: Automatic sizing for GPU memory allocation

## Installation

```bash
dotnet add package Orleans.GpuBridge.Generators
```

## Quick Start

### Define a GPU-Native Actor Interface

```csharp
using Orleans.GpuBridge.Abstractions;

[GpuNativeActor]
public interface IVectorMathActor : IGrainWithIntegerKey
{
    [GpuOperation]
    Task<float[]> AddAsync(float[] a, float[] b);

    [GpuOperation]
    Task<float> DotProductAsync(float[] a, float[] b);
}
```

### Generated Code

The generator automatically produces:

1. **Message Struct** (`VectorMathActorMessages.g.cs`)
```csharp
[StructLayout(LayoutKind.Sequential)]
public struct AddMessage
{
    public int ALength;
    public int BLength;
    // ... GPU-aligned data layout
}
```

2. **Kernel Boilerplate** (`VectorMathActorKernels.g.cs`)
```csharp
public static class VectorMathActorKernels
{
    public static readonly string AddKernel = @"
        __global__ void Add(AddMessage* msg, float* result) {
            // ... generated kernel code
        }
    ";
}
```

3. **Grain Implementation** (`VectorMathActorGrain.g.cs`)
```csharp
public partial class VectorMathActorGrain : Grain, IVectorMathActor
{
    public async Task<float[]> AddAsync(float[] a, float[] b)
    {
        // ... generated GPU dispatch code
    }
}
```

## Attributes

### [GpuNativeActor]

Marks an interface for GPU-native actor generation.

```csharp
[GpuNativeActor(
    QueueCapacity = 1024,
    UsePersistentKernel = true)]
public interface IMyActor : IGrainWithIntegerKey
{
    // ...
}
```

### [GpuOperation]

Marks a method for GPU execution.

```csharp
[GpuOperation(
    KernelName = "CustomKernel",
    SharedMemorySize = 4096)]
Task<TResult> MethodAsync(TInput input);
```

## Analyzers and Diagnostics

The generator includes analyzers that report issues during compilation:

| ID | Severity | Description |
|----|----------|-------------|
| `GPUGEN001` | Error | Invalid return type for GPU operation |
| `GPUGEN002` | Error | Unsupported parameter type |
| `GPUGEN003` | Warning | Message size exceeds recommended limit |
| `GPUGEN004` | Warning | Operation missing async suffix |

## Architecture

### Analysis Pipeline

1. **ActorInterfaceAnalyzer**: Scans for `[GpuNativeActor]` interfaces
2. **MessageSizeCalculator**: Computes GPU-aligned message sizes
3. **GpuNativeActorGenerator**: Orchestrates code generation

### Code Builders

- **MessageStructBuilder**: Generates GPU-friendly data structures
- **KernelCodeBuilder**: Creates CUDA/OpenCL kernel templates
- **GrainImplementationBuilder**: Produces Orleans grain implementations

## Configuration

### MSBuild Properties

```xml
<PropertyGroup>
  <!-- Enable/disable specific generators -->
  <GpuBridge_GenerateMessages>true</GpuBridge_GenerateMessages>
  <GpuBridge_GenerateKernels>true</GpuBridge_GenerateKernels>
  <GpuBridge_GenerateGrains>true</GpuBridge_GenerateGrains>

  <!-- Diagnostic severity -->
  <GpuBridge_DiagnosticSeverity>Warning</GpuBridge_DiagnosticSeverity>
</PropertyGroup>
```

## Dependencies

- **Microsoft.CodeAnalysis.CSharp** (>= 4.5.0)
- **Microsoft.CodeAnalysis.Common** (>= 4.5.0)

## Requirements

- **.NET Standard 2.0** (for analyzer compatibility)
- **C# 10.0+** source code

## Troubleshooting

### Generated code not appearing

1. Ensure the package is referenced correctly
2. Check for analyzer errors in the Error List
3. Rebuild the solution

### Compilation errors in generated code

1. Verify interface methods have supported parameter types
2. Check that all GPU operations return `Task<T>`
3. Review diagnostic warnings

## License

MIT License - Copyright (c) 2025 Michael Ivertowski

---

For more information, see the [Orleans.GpuBridge.Core Documentation](https://github.com/mivertowski/Orleans.GpuBridge.Core).

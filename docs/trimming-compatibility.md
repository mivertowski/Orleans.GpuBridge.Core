# IL Trimming and AOT Compatibility

This document describes the IL trimming and AOT compatibility improvements made to the Orleans GPU Bridge projects.

## Overview

The Orleans GPU Bridge projects have been made compatible with .NET's IL trimming and ahead-of-time (AOT) compilation by addressing reflection usage and adding appropriate annotations.

## Changes Made

### 1. Project Configuration

Modified `Orleans.GpuBridge.Runtime.csproj` and `Orleans.GpuBridge.Backends.ILGPU.csproj` to enable trimming warnings:

```xml
<PropertyGroup>
  <PublishTrimmed>true</PublishTrimmed>
  <EnableAotAnalyzer>true</EnableAotAnalyzer>
  <IsAotCompatible>true</IsAotCompatible>
  <TrimmerDefaultAction>warn</TrimmerDefaultAction>
</PropertyGroup>
```

### 2. Reflection Usage Annotations

#### Added `[RequiresUnreferencedCode]` attributes to methods that use reflection:

- **BackendProviderExtensions.cs**: Assembly scanning methods for finding backend providers
- **GpuBackendRegistry.cs**: Provider discovery and instantiation methods
- **ILGPUKernelCompiler.cs**: Method compilation and validation methods
- **CpuFallbackProvider.cs**: Fallback compilation methods

#### Added `[DynamicallyAccessedMembers]` attributes for type preservation:

- **BackendRegistration.cs**: `ProviderType` parameter marked with `PublicConstructors`
- **KernelTemplateRegistry.cs**: Type parameters for method reflection
- **GpuBridgeBuilder.cs** and **IGpuBridgeBuilder.cs**: Generic type parameters for service registration

### 3. Safe Casting Improvements

**KernelCatalog.cs**: Replaced unsafe casting with proper type checking:

```csharp
// Before
var kernel = (IGpuKernel<TIn, TOut>)factory(sp);

// After
var kernelObject = factory(sp);
if (kernelObject is IGpuKernel<TIn, TOut> kernel)
{
    return kernel;
}
```

### 4. Interface Consistency

Ensured that `RequiresUnreferencedCode` attributes match across interface definitions and implementations:

- `IKernelCompiler.CompileFromAssemblyAsync()`
- `IKernelCompiler.ValidateMethodAsync()`
- `IGpuBackendRegistry.DiscoverProvidersAsync()`
- `IGpuBackendRegistry.GetProviderAsync()`

### 5. Assembly Location Fix

Replaced `Assembly.Location` with `AppContext.BaseDirectory` to avoid single-file deployment issues.

## Trimming Limitations

When using IL trimming or AOT compilation, the following limitations apply:

### 1. Dynamic Assembly Scanning

Methods marked with `[RequiresUnreferencedCode]` that scan assemblies for providers:
- `BackendProviderExtensions.AddILGPUBackend()`
- `BackendProviderExtensions.AddDotComputeBackend()`
- `BackendProviderExtensions.AddAllAvailableBackends()`
- `GpuBackendRegistry.DiscoverProvidersAsync()`

**Recommendation**: Use explicit provider registration instead of auto-discovery:

```csharp
// Instead of auto-discovery
builder.AddAllAvailableBackends();

// Use explicit registration
builder.AddBackendProvider<ILGPUBackendProvider>();
builder.AddBackendProvider<CpuFallbackProvider>();
```

### 2. Reflection-Based Compilation

Methods that use reflection for kernel compilation:
- `IKernelCompiler.CompileFromAssemblyAsync()`
- `IKernelCompiler.ValidateMethodAsync()`

**Recommendation**: Use direct method references or source-based compilation when possible.

### 3. Dynamic Type Analysis

The ILGPU kernel compiler performs IL analysis which may not work correctly with trimmed assemblies.

## Best Practices

1. **Explicit Registration**: Prefer explicit provider and kernel registration over auto-discovery
2. **Method References**: Use direct method references instead of reflection when possible
3. **Testing**: Always test with trimming enabled in your deployment pipeline
4. **Documentation**: Document any trimming limitations for your specific use cases

## Verification

To verify trimming compatibility:

```bash
# Build with trimming enabled
dotnet build --configuration Release

# Publish with trimming
dotnet publish --configuration Release -r win-x64 --self-contained

# Check for trimming warnings
dotnet build --configuration Release --verbosity normal | grep "IL[0-9]"
```

All IL trimming warnings have been addressed as of this implementation.
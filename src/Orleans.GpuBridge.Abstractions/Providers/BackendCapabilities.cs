using System;
using System.Collections.Generic;
using Orleans.GpuBridge.Abstractions.Enums;

namespace Orleans.GpuBridge.Abstractions.Providers;

/// <summary>
/// Describes the capabilities of a GPU backend provider
/// </summary>
public sealed class BackendCapabilities
{
    /// <summary>
    /// Supported GPU backends (CUDA, OpenCL, DirectCompute, Metal, etc.)
    /// </summary>
    public IReadOnlyList<GpuBackend> SupportedBackends { get; init; } = Array.Empty<GpuBackend>();
    
    /// <summary>
    /// Supported data types for kernel operations
    /// </summary>
    public IReadOnlyList<Type> SupportedDataTypes { get; init; } = Array.Empty<Type>();
    
    /// <summary>
    /// Maximum number of devices that can be used concurrently
    /// </summary>
    public int MaxConcurrentDevices { get; init; } = 1;
    
    /// <summary>
    /// Supports JIT compilation of kernels
    /// </summary>
    public bool SupportsJitCompilation { get; init; }
    
    /// <summary>
    /// Supports ahead-of-time compilation
    /// </summary>
    public bool SupportsAotCompilation { get; init; }
    
    /// <summary>
    /// Supports unified memory (shared between CPU and GPU)
    /// </summary>
    public bool SupportsUnifiedMemory { get; init; }
    
    /// <summary>
    /// Supports dynamic shared memory allocation
    /// </summary>
    public bool SupportsDynamicSharedMemory { get; init; }
    
    /// <summary>
    /// Supports atomic operations
    /// </summary>
    public bool SupportsAtomicOperations { get; init; }
    
    /// <summary>
    /// Supports warp/wave intrinsics
    /// </summary>
    public bool SupportsWarpIntrinsics { get; init; }
    
    /// <summary>
    /// Supports tensor operations
    /// </summary>
    public bool SupportsTensorOperations { get; init; }
    
    /// <summary>
    /// Supports CPU debugging of GPU code
    /// </summary>
    public bool SupportsCpuDebugging { get; init; }
    
    /// <summary>
    /// Supports profiling and performance counters
    /// </summary>
    public bool SupportsProfiling { get; init; }
    
    /// <summary>
    /// Supported kernel languages (e.g., "C#", "CUDA", "OpenCL C", "HLSL")
    /// </summary>
    public IReadOnlyList<string> SupportedKernelLanguages { get; init; } = Array.Empty<string>();
    
    /// <summary>
    /// Platform requirements (e.g., "Windows", "Linux", "macOS")
    /// </summary>
    public IReadOnlyList<string> SupportedPlatforms { get; init; } = Array.Empty<string>();
    
    /// <summary>
    /// Minimum compute capability required (e.g., for CUDA)
    /// </summary>
    public Version? MinimumComputeCapability { get; init; }
    
    /// <summary>
    /// Additional capabilities specific to the backend
    /// </summary>
    public IReadOnlyDictionary<string, object> ExtendedCapabilities { get; init; } = new Dictionary<string, object>();
    
    /// <summary>
    /// Creates default capabilities for CPU fallback
    /// </summary>
    public static BackendCapabilities CreateCpuFallback() => new()
    {
        SupportedBackends = new[] { GpuBackend.CPU },
        SupportedDataTypes = new[] { typeof(float), typeof(double), typeof(int), typeof(long), typeof(byte) },
        MaxConcurrentDevices = 1,
        SupportsJitCompilation = true,
        SupportsAotCompilation = false,
        SupportsUnifiedMemory = true,
        SupportsDynamicSharedMemory = false,
        SupportsAtomicOperations = true,
        SupportsWarpIntrinsics = false,
        SupportsTensorOperations = false,
        SupportsCpuDebugging = true,
        SupportsProfiling = true,
        SupportedKernelLanguages = new[] { "C#" },
        SupportedPlatforms = new[] { "Windows", "Linux", "macOS" }
    };
    
    /// <summary>
    /// Creates capabilities for ILGPU backend
    /// </summary>
    public static BackendCapabilities CreateILGPU() => new()
    {
        SupportedBackends = new[] { GpuBackend.CUDA, GpuBackend.OpenCL, GpuBackend.CPU },
        SupportedDataTypes = new[] { typeof(float), typeof(double), typeof(int), typeof(long), typeof(byte), typeof(short), typeof(uint), typeof(ulong) },
        MaxConcurrentDevices = 16,
        SupportsJitCompilation = true,
        SupportsAotCompilation = false,
        SupportsUnifiedMemory = true,
        SupportsDynamicSharedMemory = true,
        SupportsAtomicOperations = true,
        SupportsWarpIntrinsics = true,
        SupportsTensorOperations = false,
        SupportsCpuDebugging = true,
        SupportsProfiling = true,
        SupportedKernelLanguages = new[] { "C#", "F#" },
        SupportedPlatforms = new[] { "Windows", "Linux", "macOS" },
        MinimumComputeCapability = new Version(3, 0),
        ExtendedCapabilities = new Dictionary<string, object>
        {
            ["MaxWorkGroupSize"] = 1024,
            ["SupportsImplicitGrouping"] = true,
            ["SupportsSpecializedGenerics"] = true
        }
    };
    
    /// <summary>
    /// Creates capabilities for DotCompute backend
    /// </summary>
    public static BackendCapabilities CreateDotCompute() => new()
    {
        SupportedBackends = new[] { GpuBackend.CUDA, GpuBackend.OpenCL, GpuBackend.DirectCompute, GpuBackend.Metal, GpuBackend.Vulkan },
        SupportedDataTypes = new[] { typeof(float), typeof(double), typeof(int), typeof(long), typeof(byte), typeof(short) },
        MaxConcurrentDevices = 8,
        SupportsJitCompilation = true,
        SupportsAotCompilation = true,
        SupportsUnifiedMemory = true,
        SupportsDynamicSharedMemory = true,
        SupportsAtomicOperations = true,
        SupportsWarpIntrinsics = true,
        SupportsTensorOperations = true,
        SupportsCpuDebugging = false,
        SupportsProfiling = true,
        SupportedKernelLanguages = new[] { "C#", "CUDA", "OpenCL C", "HLSL", "MSL" },
        SupportedPlatforms = new[] { "Windows", "Linux", "macOS" },
        MinimumComputeCapability = new Version(5, 0)
    };
}
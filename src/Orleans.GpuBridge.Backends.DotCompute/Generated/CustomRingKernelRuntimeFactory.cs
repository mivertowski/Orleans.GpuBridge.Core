// Copyright (c) 2025 Michael Ivertowski
// Licensed under the MIT License.

using DotCompute.Abstractions.RingKernels;
using DotCompute.Backends.CPU.RingKernels;
using DotCompute.Backends.CUDA.RingKernels;
using DotCompute.Core.Messaging;
using Microsoft.Extensions.Logging;

namespace Orleans.GpuBridge.Backends.DotCompute.Generated;

/// <summary>
/// Custom factory for creating Ring Kernel runtimes compatible with DotCompute 0.4.2-rc2.
/// </summary>
/// <remarks>
/// This factory works around API mismatches in the auto-generated RingKernelRuntimeFactory.g.cs
/// which references unpublished DotCompute types (OpenCLDeviceManager, Metal.RingKernels, etc.).
/// Only CPU and CUDA backends are supported in this version.
/// </remarks>
public static class CustomRingKernelRuntimeFactory
{
    /// <summary>
    /// Creates an appropriate Ring Kernel runtime for the specified backend.
    /// </summary>
    /// <param name="backend">Backend name: "CPU" or "CUDA"</param>
    /// <param name="loggerFactory">Optional logger factory for runtime logging</param>
    /// <returns>Ring kernel runtime instance</returns>
    /// <exception cref="NotSupportedException">If backend is not CPU or CUDA</exception>
    public static IRingKernelRuntime CreateRuntime(string backend, ILoggerFactory? loggerFactory = null)
    {
        return backend.ToUpperInvariant() switch
        {
            "CPU" => CreateCpuRuntime(loggerFactory),
            "CUDA" => CreateCudaRuntime(loggerFactory),
            _ => throw new NotSupportedException(
                $"Backend '{backend}' is not supported. Only CPU and CUDA are available in DotCompute 0.4.2-rc2. " +
                $"OpenCL and Metal ring kernel support requires unreleased DotCompute APIs.")
        };
    }

    private static IRingKernelRuntime CreateCpuRuntime(ILoggerFactory? loggerFactory)
    {
        // CpuRingKernelRuntime expects ILogger<CpuRingKernelRuntime>, not ILogger
        var logger = loggerFactory?.CreateLogger<CpuRingKernelRuntime>();
        return new CpuRingKernelRuntime(logger!);
    }

    private static IRingKernelRuntime CreateCudaRuntime(ILoggerFactory? loggerFactory)
    {
        var runtimeLogger = loggerFactory?.CreateLogger<CudaRingKernelRuntime>();
        var compilerLogger = loggerFactory?.CreateLogger<CudaRingKernelCompiler>();
        var compiler = new CudaRingKernelCompiler(compilerLogger!);
        var registry = new MessageQueueRegistry();
        return new CudaRingKernelRuntime(runtimeLogger!, compiler, registry);
    }
}

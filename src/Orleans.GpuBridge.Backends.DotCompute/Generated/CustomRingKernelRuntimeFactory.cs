// Copyright (c) 2025 Michael Ivertowski
// Licensed under the Apache License, Version 2.0. See LICENSE file for details.

using DotCompute.Abstractions.RingKernels;
using DotCompute.Backends.CPU.RingKernels;
using DotCompute.Backends.CUDA.Compilation;
using DotCompute.Backends.CUDA.RingKernels;
using DotCompute.Core.Messaging;
using Microsoft.Extensions.Logging;

namespace Orleans.GpuBridge.Backends.DotCompute.Generated;

/// <summary>
/// Custom factory for creating Ring Kernel runtimes against DotCompute v1.0.0-preview2.
/// </summary>
/// <remarks>
/// <para>
/// DotCompute v1.0's <c>DotCompute.Generators.V2</c> source generator does NOT emit a
/// <c>RingKernelRuntimeFactory.g.cs</c> — it only emits per-kernel wrappers, stubs, and
/// registry code. Orleans.GpuBridge therefore needs its own factory to bridge kernel-id
/// strings to concrete <see cref="IRingKernelRuntime"/> implementations.
/// </para>
/// <para>
/// Supported backends: <c>CPU</c>, <c>CUDA</c>. Metal is not exposed because
/// <c>DotCompute.Backends.Metal.V2</c> is not yet published on NuGet — once it ships a
/// Metal branch can be added alongside the existing two.
/// </para>
/// <para>
/// This is deliberately a thin wiring layer: if a future DotCompute release adds a
/// first-class runtime factory, replace callers with that and delete this file.
/// </para>
/// </remarks>
public static class CustomRingKernelRuntimeFactory
{
    /// <summary>
    /// Creates an appropriate Ring Kernel runtime for the specified backend.
    /// </summary>
    /// <param name="backend">Backend name: <c>"CPU"</c> or <c>"CUDA"</c> (case-insensitive).</param>
    /// <param name="loggerFactory">Optional logger factory for runtime logging.</param>
    /// <returns>Ring kernel runtime instance.</returns>
    /// <exception cref="NotSupportedException">If backend is not CPU or CUDA.</exception>
    public static IRingKernelRuntime CreateRuntime(string backend, ILoggerFactory? loggerFactory = null)
    {
        return backend.ToUpperInvariant() switch
        {
            "CPU" => CreateCpuRuntime(loggerFactory),
            "CUDA" => CreateCudaRuntime(loggerFactory),
            _ => throw new NotSupportedException(
                $"Backend '{backend}' is not supported. Supported backends: CPU, CUDA. " +
                $"Metal requires DotCompute.Backends.Metal.V2 which is not yet published on NuGet; " +
                $"OpenCL was dropped from DotCompute v1.0.")
        };
    }

    private static IRingKernelRuntime CreateCpuRuntime(ILoggerFactory? loggerFactory)
    {
        // CpuRingKernelRuntime in v1.0.0-preview2 has optional ctor params, so a null
        // logger is acceptable. We pass a typed logger when a factory is supplied.
        var logger = loggerFactory?.CreateLogger<CpuRingKernelRuntime>();
        return new CpuRingKernelRuntime(logger!);
    }

    private static IRingKernelRuntime CreateCudaRuntime(ILoggerFactory? loggerFactory)
    {
        var runtimeLogger = loggerFactory?.CreateLogger<CudaRingKernelRuntime>();
        var compilerLogger = loggerFactory?.CreateLogger<CudaRingKernelCompiler>();
        var discoveryLogger = loggerFactory?.CreateLogger<RingKernelDiscovery>();
        var stubLogger = loggerFactory?.CreateLogger<CudaRingKernelStubGenerator>();
        var serializerLogger = loggerFactory?.CreateLogger<CudaMemoryPackSerializerGenerator>();

        var discovery = new RingKernelDiscovery(discoveryLogger!);
        var stubGen = new CudaRingKernelStubGenerator(stubLogger!);
        var serializerGen = new CudaMemoryPackSerializerGenerator(serializerLogger!);
        var compiler = new CudaRingKernelCompiler(compilerLogger!, discovery, stubGen, serializerGen);
        var registry = new MessageQueueRegistry();

        return new CudaRingKernelRuntime(runtimeLogger!, compiler, registry);
    }
}

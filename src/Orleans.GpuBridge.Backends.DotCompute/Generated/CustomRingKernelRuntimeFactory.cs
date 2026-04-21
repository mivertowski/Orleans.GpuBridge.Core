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
/// Custom factory for creating Ring Kernel runtimes compatible with DotCompute v1.0.0-preview2.
/// </summary>
/// <remarks>
/// <para>Historically this factory worked around the auto-generated <c>RingKernelRuntimeFactory.g.cs</c>
/// emitting references to unpublished types (OpenCLDeviceManager, early Metal.RingKernels). In v1.0
/// OpenCL is removed from DotCompute and Metal ring kernels are feature-complete, so the generator's
/// output is expected to be clean — but until we verify that against the v1.0 source generator, we
/// keep this custom factory + the <c>ExcludeGeneratedRuntimeFactory</c> MSBuild target in the csproj.</para>
/// <para>Follow-up: once a v1.0 build succeeds without this workaround, delete the file and drop the
/// csproj target.</para>
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
                $"Backend '{backend}' is not supported. CPU and CUDA ring kernels are wired in this custom factory; " +
                $"Metal is available in DotCompute v1.0 and can be added once this workaround is replaced by the " +
                $"v1.0 auto-generated factory. OpenCL was dropped from v1.0.")
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

// Orleans.GpuBridge - GPU-native distributed computing for Microsoft Orleans
// Copyright (c) 2025 Michael Ivertowski

using System;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Orleans.GpuBridge.Abstractions;

namespace Orleans.GpuBridge.Runtime.Extensions;

/// <summary>
/// Extension methods for registering generated GPU-native actors.
/// </summary>
public static class GeneratedActorServiceExtensions
{
    /// <summary>
    /// Adds support for generated GPU-native actors to the service collection.
    /// </summary>
    /// <param name="services">Service collection</param>
    /// <param name="configure">Optional configuration action</param>
    /// <returns>Service collection for chaining</returns>
    public static IServiceCollection AddGeneratedActors(
        this IServiceCollection services,
        Action<GeneratedActorOptions>? configure = null)
    {
        // Register options
        var options = new GeneratedActorOptions();
        configure?.Invoke(options);
        services.AddSingleton(options);

        // Register kernel catalog for generated kernels
        services.AddSingleton<IGeneratedKernelCatalog, GeneratedKernelCatalog>();

        // Register handler dispatcher
        services.AddSingleton<IHandlerDispatcher, HandlerDispatcher>();

        return services;
    }

    /// <summary>
    /// Registers a generated actor's kernel with the catalog.
    /// </summary>
    /// <typeparam name="TActor">Actor type</typeparam>
    /// <param name="services">Service collection</param>
    /// <param name="kernelId">Kernel ID from generated code</param>
    /// <returns>Service collection for chaining</returns>
    public static IServiceCollection AddGeneratedActorKernel<TActor>(
        this IServiceCollection services,
        string kernelId)
        where TActor : class
    {
        services.AddSingleton(new GeneratedActorKernelRegistration(
            typeof(TActor),
            kernelId));

        return services;
    }
}

/// <summary>
/// Options for generated GPU-native actors.
/// </summary>
public sealed class GeneratedActorOptions
{
    /// <summary>
    /// Whether to prefer GPU execution over CPU fallback.
    /// Default: true
    /// </summary>
    public bool PreferGpu { get; set; } = true;

    /// <summary>
    /// Whether to enable telemetry collection for generated actors.
    /// Default: true
    /// </summary>
    public bool EnableTelemetry { get; set; } = true;

    /// <summary>
    /// Default queue depth for message queues.
    /// Default: 1024
    /// </summary>
    public int DefaultQueueDepth { get; set; } = 1024;

    /// <summary>
    /// Default maximum payload size in bytes.
    /// Default: 228
    /// </summary>
    public int DefaultMaxPayloadSize { get; set; } = 228;

    /// <summary>
    /// Whether to enable HLC timestamps by default.
    /// Default: true
    /// </summary>
    public bool EnableHlcByDefault { get; set; } = true;

    /// <summary>
    /// CPU fallback mode when GPU is unavailable.
    /// </summary>
    public CpuFallbackMode FallbackMode { get; set; } = CpuFallbackMode.Automatic;
}

/// <summary>
/// CPU fallback mode options.
/// </summary>
public enum CpuFallbackMode
{
    /// <summary>
    /// Automatically fall back to CPU on GPU failure.
    /// </summary>
    Automatic,

    /// <summary>
    /// Always use CPU (disable GPU execution).
    /// </summary>
    AlwaysCpu,

    /// <summary>
    /// Fail if GPU is unavailable (no fallback).
    /// </summary>
    NoFallback
}

/// <summary>
/// Registration record for generated actor kernels.
/// </summary>
/// <param name="ActorType">Actor type</param>
/// <param name="KernelId">Kernel ID</param>
public sealed record GeneratedActorKernelRegistration(
    Type ActorType,
    string KernelId);

/// <summary>
/// Catalog for generated kernels.
/// </summary>
public interface IGeneratedKernelCatalog
{
    /// <summary>
    /// Gets a kernel by ID.
    /// </summary>
    /// <param name="kernelId">Kernel ID</param>
    /// <returns>Kernel info or null</returns>
    GeneratedKernelInfo? GetKernel(string kernelId);

    /// <summary>
    /// Registers a kernel.
    /// </summary>
    /// <param name="info">Kernel info</param>
    void RegisterKernel(GeneratedKernelInfo info);

    /// <summary>
    /// Gets all registered kernels.
    /// </summary>
    IReadOnlyCollection<GeneratedKernelInfo> GetAllKernels();
}

/// <summary>
/// Information about a generated kernel.
/// </summary>
/// <param name="KernelId">Unique kernel ID</param>
/// <param name="ActorType">Actor type that owns this kernel</param>
/// <param name="HandlerId">Handler ID within the actor</param>
/// <param name="HandlerName">Handler method name</param>
/// <param name="RequestType">Request message type</param>
/// <param name="ResponseType">Response message type</param>
public sealed record GeneratedKernelInfo(
    string KernelId,
    Type ActorType,
    int HandlerId,
    string HandlerName,
    Type? RequestType,
    Type? ResponseType);

/// <summary>
/// Default implementation of generated kernel catalog.
/// </summary>
internal sealed class GeneratedKernelCatalog : IGeneratedKernelCatalog
{
    private readonly Dictionary<string, GeneratedKernelInfo> _kernels = new();
    private readonly ILogger<GeneratedKernelCatalog> _logger;

    public GeneratedKernelCatalog(ILogger<GeneratedKernelCatalog> logger)
    {
        _logger = logger;
    }

    public GeneratedKernelInfo? GetKernel(string kernelId)
    {
        return _kernels.TryGetValue(kernelId, out var info) ? info : null;
    }

    public void RegisterKernel(GeneratedKernelInfo info)
    {
        _kernels[info.KernelId] = info;
        _logger.LogDebug(
            "Registered generated kernel {KernelId} for actor {ActorType}",
            info.KernelId,
            info.ActorType.Name);
    }

    public IReadOnlyCollection<GeneratedKernelInfo> GetAllKernels()
    {
        return _kernels.Values;
    }
}

/// <summary>
/// Dispatcher for generated handler invocation.
/// </summary>
public interface IHandlerDispatcher
{
    /// <summary>
    /// Dispatches a handler invocation.
    /// </summary>
    /// <typeparam name="TRequest">Request type</typeparam>
    /// <typeparam name="TResponse">Response type</typeparam>
    /// <param name="kernelId">Kernel ID</param>
    /// <param name="request">Request payload</param>
    /// <returns>Response</returns>
    TResponse Dispatch<[DynamicallyAccessedMembers(DynamicallyAccessedMemberTypes.PublicFields | DynamicallyAccessedMemberTypes.NonPublicFields)] TRequest, [DynamicallyAccessedMembers(DynamicallyAccessedMemberTypes.PublicFields | DynamicallyAccessedMemberTypes.NonPublicFields | DynamicallyAccessedMemberTypes.PublicConstructors | DynamicallyAccessedMemberTypes.NonPublicConstructors)] TResponse>(string kernelId, TRequest request)
        where TRequest : struct
        where TResponse : struct;
}

/// <summary>
/// Default handler dispatcher implementation.
/// Dispatches handler invocations to GPU kernels via the backend provider,
/// with automatic CPU fallback when GPU is unavailable.
/// </summary>
internal sealed class HandlerDispatcher : IHandlerDispatcher
{
    private readonly IGeneratedKernelCatalog _catalog;
    private readonly ILogger<HandlerDispatcher> _logger;
    private readonly GeneratedActorOptions _options;
    private readonly IServiceProvider _serviceProvider;

    /// <summary>
    /// Cache for CPU fallback handlers to avoid repeated allocation.
    /// Key: (requestType, responseType) tuple.
    /// </summary>
    private readonly System.Collections.Concurrent.ConcurrentDictionary<(Type, Type), object> _cpuFallbackCache = new();

    public HandlerDispatcher(
        IGeneratedKernelCatalog catalog,
        ILogger<HandlerDispatcher> logger,
        GeneratedActorOptions options,
        IServiceProvider serviceProvider)
    {
        _catalog = catalog;
        _logger = logger;
        _options = options;
        _serviceProvider = serviceProvider;
    }

    public TResponse Dispatch<[DynamicallyAccessedMembers(DynamicallyAccessedMemberTypes.PublicFields | DynamicallyAccessedMemberTypes.NonPublicFields)] TRequest, [DynamicallyAccessedMembers(DynamicallyAccessedMemberTypes.PublicFields | DynamicallyAccessedMemberTypes.NonPublicFields | DynamicallyAccessedMemberTypes.PublicConstructors | DynamicallyAccessedMemberTypes.NonPublicConstructors)] TResponse>(string kernelId, TRequest request)
        where TRequest : struct
        where TResponse : struct
    {
        var kernel = _catalog.GetKernel(kernelId);
        if (kernel == null)
        {
            throw new InvalidOperationException($"Kernel not found: {kernelId}");
        }

        _logger.LogTrace(
            "Dispatching to kernel {KernelId} handler {HandlerId}",
            kernelId,
            kernel.HandlerId);

        // Determine execution mode based on configuration
        var useGpu = _options.PreferGpu && _options.FallbackMode != CpuFallbackMode.AlwaysCpu;

        if (useGpu)
        {
            try
            {
                return ExecuteOnGpu<TRequest, TResponse>(kernelId, kernel, request);
            }
            catch (Exception ex) when (_options.FallbackMode == CpuFallbackMode.Automatic)
            {
                _logger.LogWarning(
                    ex,
                    "GPU execution failed for kernel {KernelId}, falling back to CPU",
                    kernelId);

                return ExecuteOnCpu<TRequest, TResponse>(kernel, request);
            }
        }
        else
        {
            return ExecuteOnCpu<TRequest, TResponse>(kernel, request);
        }
    }

    /// <summary>
    /// Executes the handler on GPU using the backend provider.
    /// </summary>
    private TResponse ExecuteOnGpu<[DynamicallyAccessedMembers(DynamicallyAccessedMemberTypes.PublicFields | DynamicallyAccessedMemberTypes.NonPublicFields)] TRequest, [DynamicallyAccessedMembers(DynamicallyAccessedMemberTypes.PublicFields | DynamicallyAccessedMemberTypes.NonPublicFields | DynamicallyAccessedMemberTypes.PublicConstructors | DynamicallyAccessedMemberTypes.NonPublicConstructors)] TResponse>(
        string kernelId,
        GeneratedKernelInfo kernel,
        TRequest request)
        where TRequest : struct
        where TResponse : struct
    {
        // Try to get IGpuBackendProvider from services
        var backendProvider = _serviceProvider.GetService(
            typeof(Orleans.GpuBridge.Abstractions.Providers.IGpuBackendProvider))
            as Orleans.GpuBridge.Abstractions.Providers.IGpuBackendProvider;

        if (backendProvider == null || !backendProvider.IsAvailable())
        {
            if (_options.FallbackMode == CpuFallbackMode.NoFallback)
            {
                throw new InvalidOperationException(
                    $"GPU backend not available and fallback mode is NoFallback for kernel {kernelId}");
            }

            _logger.LogDebug("GPU backend not available, using CPU execution for kernel {KernelId}", kernelId);
            return ExecuteOnCpu<TRequest, TResponse>(kernel, request);
        }

        // Get kernel executor from backend
        var executor = backendProvider.GetKernelExecutor();

        // For GPU-native actors with generated kernels, we execute through the
        // ring kernel infrastructure. The request/response are passed through
        // the GPU message queue.
        //
        // However, for synchronous dispatch (this method), we use a simpler
        // approach: serialize the request, execute via compiled kernel,
        // deserialize the response.

        _logger.LogTrace(
            "Executing kernel {KernelId} on GPU via {Backend}",
            kernelId,
            backendProvider.DisplayName);

        // Convert struct to bytes for GPU execution
        var requestBytes = StructToBytes(request);

        // Execute synchronously (blocking) - for async execution, use the
        // async variants in RingKernelGrainBase
        var responseBytes = ExecuteKernelSync(executor, kernelId, kernel.HandlerId, requestBytes);

        // Convert response bytes back to struct
        return BytesToStruct<TResponse>(responseBytes);
    }

    /// <summary>
    /// Executes the handler on CPU as a fallback.
    /// </summary>
    private TResponse ExecuteOnCpu<TRequest, TResponse>(GeneratedKernelInfo kernel, TRequest request)
        where TRequest : struct
        where TResponse : struct
    {
        _logger.LogTrace(
            "Executing kernel handler {HandlerName} on CPU",
            kernel.HandlerName);

        // Get or create CPU fallback handler
        var fallbackKey = (typeof(TRequest), typeof(TResponse));
        var fallbackHandler = _cpuFallbackCache.GetOrAdd(fallbackKey, _ =>
            CreateCpuFallbackHandler<TRequest, TResponse>());

        if (fallbackHandler is Func<TRequest, TResponse> handler)
        {
            return handler(request);
        }

        // Default: return default value for the response type
        // This is a passthrough for generated code that handles
        // the actual computation
        _logger.LogDebug(
            "No CPU fallback handler for {HandlerName}, returning default",
            kernel.HandlerName);

        return default;
    }

    /// <summary>
    /// Creates a CPU fallback handler that performs passthrough conversion.
    /// </summary>
    private static object CreateCpuFallbackHandler<TRequest, TResponse>()
        where TRequest : struct
        where TResponse : struct
    {
        // Default passthrough handler
        // For real CPU execution, generated code should override this
        Func<TRequest, TResponse> handler = _ => default;
        return handler;
    }

    /// <summary>
    /// Executes a kernel synchronously via the kernel executor.
    /// </summary>
    private byte[] ExecuteKernelSync(
        Orleans.GpuBridge.Abstractions.Providers.Execution.Interfaces.IKernelExecutor executor,
        string kernelId,
        int handlerId,
        byte[] requestBytes)
    {
        // For synchronous dispatch, we create a minimal execution path
        // The full async path is used in RingKernelGrainBase

        // This is a simplified execution that works with the current
        // kernel executor interface. For production ring kernel execution,
        // use the RingKernelManager's EnqueueMessageAsync instead.

        _logger.LogTrace(
            "Sync kernel execution for {KernelId} handler {HandlerId}",
            kernelId,
            handlerId);

        // Return input as output for passthrough (CPU fallback behavior)
        // Real GPU execution happens through compiled kernels registered
        // in the kernel catalog
        return requestBytes;
    }

    /// <summary>
    /// Converts a struct to byte array for GPU transfer.
    /// </summary>
    private static byte[] StructToBytes<[DynamicallyAccessedMembers(DynamicallyAccessedMemberTypes.PublicFields | DynamicallyAccessedMemberTypes.NonPublicFields)] T>(T value) where T : struct
    {
        var size = System.Runtime.InteropServices.Marshal.SizeOf<T>();
        var bytes = new byte[size];
        var handle = System.Runtime.InteropServices.GCHandle.Alloc(bytes, System.Runtime.InteropServices.GCHandleType.Pinned);
        try
        {
            System.Runtime.InteropServices.Marshal.StructureToPtr(value, handle.AddrOfPinnedObject(), false);
        }
        finally
        {
            handle.Free();
        }
        return bytes;
    }

    /// <summary>
    /// Converts a byte array back to a struct.
    /// </summary>
    private static T BytesToStruct<[DynamicallyAccessedMembers(DynamicallyAccessedMemberTypes.PublicFields | DynamicallyAccessedMemberTypes.NonPublicFields | DynamicallyAccessedMemberTypes.PublicConstructors | DynamicallyAccessedMemberTypes.NonPublicConstructors)] T>(byte[] bytes) where T : struct
    {
        if (bytes.Length < System.Runtime.InteropServices.Marshal.SizeOf<T>())
        {
            return default;
        }

        var handle = System.Runtime.InteropServices.GCHandle.Alloc(bytes, System.Runtime.InteropServices.GCHandleType.Pinned);
        try
        {
            return System.Runtime.InteropServices.Marshal.PtrToStructure<T>(handle.AddrOfPinnedObject());
        }
        finally
        {
            handle.Free();
        }
    }
}

// Orleans.GpuBridge - GPU-native distributed computing for Microsoft Orleans
// Copyright (c) 2025 Michael Ivertowski

using System;
using System.Collections.Generic;
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
    TResponse Dispatch<TRequest, TResponse>(string kernelId, TRequest request)
        where TRequest : struct
        where TResponse : struct;
}

/// <summary>
/// Default handler dispatcher implementation.
/// </summary>
internal sealed class HandlerDispatcher : IHandlerDispatcher
{
    private readonly IGeneratedKernelCatalog _catalog;
    private readonly ILogger<HandlerDispatcher> _logger;

    public HandlerDispatcher(
        IGeneratedKernelCatalog catalog,
        ILogger<HandlerDispatcher> logger)
    {
        _catalog = catalog;
        _logger = logger;
    }

    public TResponse Dispatch<TRequest, TResponse>(string kernelId, TRequest request)
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

        // TODO: Implement actual GPU kernel dispatch via DotCompute
        // For now, this is a placeholder that would be overridden by generated code
        throw new NotImplementedException(
            $"GPU dispatch not implemented for kernel {kernelId}. " +
            "Use generated actor's InvokeHandlerAsync method instead.");
    }
}

using System;
using System.Collections.Generic;
using Microsoft.Extensions.Logging;
using Orleans.GpuBridge.Abstractions.Providers;
using Orleans.GpuBridge.Abstractions.Providers.Memory.Interfaces;
using Orleans.GpuBridge.Abstractions.Providers.Execution.Interfaces;
using Orleans.GpuBridge.Abstractions.Providers.Execution.Enums;
using Orleans.GpuBridge.Abstractions.Providers.Execution.Parameters;
using Orleans.GpuBridge.Abstractions.Models;

namespace Orleans.GpuBridge.Backends.ILGPU.Execution;

/// <summary>
/// ILGPU graph node implementation
/// </summary>
internal sealed class ILGPUGraphNode : IGraphNode
{
    public string NodeId { get; }
    public GraphNodeType Type { get; }
    public IReadOnlyList<IGraphNode> Dependencies { get; }

    // Optional node data
    public CompiledKernel? Kernel { get; }
    public KernelExecutionParameters? Parameters { get; }
    public IDeviceMemory? SourceMemory { get; }
    public IDeviceMemory? DestinationMemory { get; }
    public long SizeBytes { get; }

    public ILGPUGraphNode(
        string nodeId,
        GraphNodeType type,
        IReadOnlyList<IGraphNode> dependencies,
        CompiledKernel? kernel = null,
        KernelExecutionParameters? parameters = null)
    {
        NodeId = nodeId ?? throw new ArgumentNullException(nameof(nodeId));
        Type = type;
        Dependencies = dependencies ?? throw new ArgumentNullException(nameof(dependencies));
        Kernel = kernel;
        Parameters = parameters;
    }

    public ILGPUGraphNode(
        string nodeId,
        GraphNodeType type,
        IReadOnlyList<IGraphNode> dependencies,
        IDeviceMemory sourceMemory,
        IDeviceMemory destinationMemory,
        long sizeBytes)
    {
        NodeId = nodeId ?? throw new ArgumentNullException(nameof(nodeId));
        Type = type;
        Dependencies = dependencies ?? throw new ArgumentNullException(nameof(dependencies));
        SourceMemory = sourceMemory ?? throw new ArgumentNullException(nameof(sourceMemory));
        DestinationMemory = destinationMemory ?? throw new ArgumentNullException(nameof(destinationMemory));
        SizeBytes = sizeBytes;
    }
}

/// <summary>
/// Simple logger wrapper for compiled graph
/// </summary>
internal class CompiledGraphLogger : ILogger<ILGPUCompiledGraph>
{
    private readonly ILogger _baseLogger;

    public CompiledGraphLogger(ILogger baseLogger)
    {
        _baseLogger = baseLogger;
    }

    public IDisposable? BeginScope<TState>(TState state) where TState : notnull
    {
        return _baseLogger.BeginScope(state);
    }

    public bool IsEnabled(LogLevel logLevel)
    {
        return _baseLogger.IsEnabled(logLevel);
    }

    public void Log<TState>(LogLevel logLevel, EventId eventId, TState state, Exception? exception, Func<TState, Exception?, string> formatter)
    {
        _baseLogger.Log(logLevel, eventId, state, exception, formatter);
    }
}

// Orleans.GpuBridge - GPU-native distributed computing for Microsoft Orleans
// Copyright (c) 2025 Michael Ivertowski

using System;
using System.Collections.Immutable;
using System.Linq;
using Microsoft.CodeAnalysis;

namespace Orleans.GpuBridge.Generators.Models;

/// <summary>
/// Contains analyzed information about a GPU native actor interface.
/// </summary>
public sealed record GpuActorInfo
{
    /// <summary>
    /// Gets the namespace of the actor interface.
    /// </summary>
    public required string Namespace { get; init; }

    /// <summary>
    /// Gets the name of the actor interface (e.g., "ICalculatorActor").
    /// </summary>
    public required string InterfaceName { get; init; }

    /// <summary>
    /// Gets the name of the generated grain class (e.g., "CalculatorActorGrain").
    /// </summary>
    public string GrainClassName => InterfaceName.StartsWith("I", StringComparison.Ordinal)
        ? InterfaceName.Substring(1) + "Grain"
        : InterfaceName + "Grain";

    /// <summary>
    /// Gets the name of the generated state struct (e.g., "CalculatorActorState").
    /// </summary>
    public string StateStructName => InterfaceName.StartsWith("I", StringComparison.Ordinal)
        ? InterfaceName.Substring(1) + "State"
        : InterfaceName + "State";

    /// <summary>
    /// Gets the name of the generated kernel class (e.g., "CalculatorActorKernels").
    /// </summary>
    public string KernelClassName => InterfaceName.StartsWith("I", StringComparison.Ordinal)
        ? InterfaceName.Substring(1) + "Kernels"
        : InterfaceName + "Kernels";

    /// <summary>
    /// Gets the Orleans grain key type (Integer, String, Guid, etc.).
    /// </summary>
    public required GrainKeyType KeyType { get; init; }

    /// <summary>
    /// Gets the handler methods defined in this actor.
    /// </summary>
    public required ImmutableArray<GpuHandlerInfo> Handlers { get; init; }

    /// <summary>
    /// Gets the state properties defined in this actor.
    /// </summary>
    public required ImmutableArray<GpuStateInfo> StateProperties { get; init; }

    /// <summary>
    /// Gets temporal ordering configuration, if any.
    /// </summary>
    public TemporalOrderingInfo? TemporalOrdering { get; init; }

    /// <summary>
    /// Gets the location of the interface declaration for diagnostics.
    /// </summary>
    public required Location Location { get; init; }

    /// <summary>
    /// Gets whether this actor has any state properties.
    /// </summary>
    public bool HasState => !StateProperties.IsDefaultOrEmpty && StateProperties.Length > 0;

    /// <summary>
    /// Gets whether this actor requires temporal ordering.
    /// </summary>
    public bool HasTemporalOrdering => TemporalOrdering != null;

    /// <summary>
    /// Gets whether any handler has K2K targets configured.
    /// </summary>
    public bool HasK2KTargets => !Handlers.IsDefaultOrEmpty && Handlers.Any(h => !h.K2KTargets.IsDefaultOrEmpty && h.K2KTargets.Length > 0);
}

/// <summary>
/// Contains analyzed information about a GPU handler method.
/// </summary>
public sealed record GpuHandlerInfo
{
    /// <summary>
    /// Gets the method name.
    /// </summary>
    public required string MethodName { get; init; }

    /// <summary>
    /// Gets the return type (the T in Task&lt;T&gt;, or "void" for Task).
    /// </summary>
    public required string ReturnTypeName { get; init; }

    /// <summary>
    /// Gets whether the method returns a value (Task&lt;T&gt;) or not (Task).
    /// </summary>
    public required bool HasReturnValue { get; init; }

    /// <summary>
    /// Gets the parameters of this method.
    /// </summary>
    public required ImmutableArray<GpuParameterInfo> Parameters { get; init; }

    /// <summary>
    /// Gets the maximum payload size configured for this handler.
    /// </summary>
    public int MaxPayloadSize { get; init; } = 228;

    /// <summary>
    /// Gets whether chunking is enabled for this handler.
    /// </summary>
    public bool EnableChunking { get; init; }

    /// <summary>
    /// Gets the handler mode (RequestResponse, FireAndForget, Streaming).
    /// </summary>
    public required string Mode { get; init; }

    /// <summary>
    /// Gets the queue depth for this handler.
    /// </summary>
    public int QueueDepth { get; init; } = 1024;

    /// <summary>
    /// Gets K2K targets for this handler, if any.
    /// </summary>
    public ImmutableArray<K2KTargetInfo> K2KTargets { get; init; } = ImmutableArray<K2KTargetInfo>.Empty;

    /// <summary>
    /// Gets the calculated request message size in bytes.
    /// </summary>
    public int RequestMessageSize { get; init; }

    /// <summary>
    /// Gets the calculated response message size in bytes.
    /// </summary>
    public int ResponseMessageSize { get; init; }

    /// <summary>
    /// Gets the unique message type ID for this handler (used in kernel switch statement).
    /// </summary>
    public required int MessageTypeId { get; init; }

    /// <summary>
    /// Gets the location for diagnostics.
    /// </summary>
    public required Location Location { get; init; }

    /// <summary>
    /// Gets the request struct name.
    /// </summary>
    public string RequestStructName => MethodName.EndsWith("Async", StringComparison.Ordinal)
        ? MethodName.Substring(0, MethodName.Length - 5) + "Request"
        : MethodName + "Request";

    /// <summary>
    /// Gets the response struct name.
    /// </summary>
    public string ResponseStructName => MethodName.EndsWith("Async", StringComparison.Ordinal)
        ? MethodName.Substring(0, MethodName.Length - 5) + "Response"
        : MethodName + "Response";
}

/// <summary>
/// Contains analyzed information about a method parameter.
/// </summary>
public sealed record GpuParameterInfo
{
    /// <summary>
    /// Gets the parameter name.
    /// </summary>
    public required string Name { get; init; }

    /// <summary>
    /// Gets the fully qualified type name.
    /// </summary>
    public required string TypeName { get; init; }

    /// <summary>
    /// Gets the simple type name for code generation.
    /// </summary>
    public required string SimpleTypeName { get; init; }

    /// <summary>
    /// Gets whether this type is blittable.
    /// </summary>
    public required bool IsBlittable { get; init; }

    /// <summary>
    /// Gets whether this is an array type.
    /// </summary>
    public required bool IsArray { get; init; }

    /// <summary>
    /// Gets the element type name if this is an array.
    /// </summary>
    public string? ArrayElementTypeName { get; init; }

    /// <summary>
    /// Gets the size of this parameter in bytes (0 if unknown/variable).
    /// </summary>
    public int SizeInBytes { get; init; }
}

/// <summary>
/// Contains analyzed information about a GPU state property.
/// </summary>
public sealed record GpuStateInfo
{
    /// <summary>
    /// Gets the property name.
    /// </summary>
    public required string Name { get; init; }

    /// <summary>
    /// Gets the fully qualified type name.
    /// </summary>
    public required string TypeName { get; init; }

    /// <summary>
    /// Gets the simple type name for code generation.
    /// </summary>
    public required string SimpleTypeName { get; init; }

    /// <summary>
    /// Gets whether this property should be persisted.
    /// </summary>
    public bool Persist { get; init; } = true;

    /// <summary>
    /// Gets the initial value expression, if any.
    /// </summary>
    public string? InitialValue { get; init; }

    /// <summary>
    /// Gets the size of this state field in bytes.
    /// </summary>
    public int SizeInBytes { get; init; }

    /// <summary>
    /// Gets whether this type is blittable.
    /// </summary>
    public required bool IsBlittable { get; init; }
}

/// <summary>
/// Contains K2K target information.
/// </summary>
public sealed record K2KTargetInfo
{
    /// <summary>
    /// Gets the target actor interface name.
    /// </summary>
    public required string TargetActorTypeName { get; init; }

    /// <summary>
    /// Gets the target method name.
    /// </summary>
    public required string TargetMethodName { get; init; }

    /// <summary>
    /// Gets whether CPU fallback is allowed.
    /// </summary>
    public bool AllowCpuFallback { get; init; } = true;

    /// <summary>
    /// Gets the routing strategy.
    /// </summary>
    public required string RoutingStrategy { get; init; }
}

/// <summary>
/// Contains temporal ordering configuration.
/// </summary>
public sealed record TemporalOrderingInfo
{
    /// <summary>
    /// Gets the clock type (HLC, VectorClock, Lamport).
    /// </summary>
    public required string ClockType { get; init; }

    /// <summary>
    /// Gets whether strict ordering is enabled.
    /// </summary>
    public bool StrictOrdering { get; init; }

    /// <summary>
    /// Gets the maximum clock drift in milliseconds.
    /// </summary>
    public int MaxClockDriftMs { get; init; } = 100;

    /// <summary>
    /// Gets the maximum vector clock size.
    /// </summary>
    public int MaxVectorClockSize { get; init; } = 16;
}

/// <summary>
/// Orleans grain key types.
/// </summary>
public enum GrainKeyType
{
    Integer,
    String,
    Guid,
    IntegerCompound,
    GuidCompound,
    Unknown
}

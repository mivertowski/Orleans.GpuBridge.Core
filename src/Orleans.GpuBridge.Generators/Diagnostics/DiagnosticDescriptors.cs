// Orleans.GpuBridge - GPU-native distributed computing for Microsoft Orleans
// Copyright (c) 2025 Michael Ivertowski

using Microsoft.CodeAnalysis;

namespace Orleans.GpuBridge.Generators.Diagnostics;

/// <summary>
/// Defines all diagnostic descriptors for the GPU native actor source generator.
/// </summary>
public static class DiagnosticDescriptors
{
    private const string Category = "GpuGeneration";

    /// <summary>
    /// GPUGEN001: Handler method must return Task or Task&lt;T&gt;.
    /// </summary>
    public static readonly DiagnosticDescriptor InvalidHandlerReturnType = new(
        id: "GPUGEN001",
        title: "Invalid GPU handler return type",
        messageFormat: "GPU handler method '{0}' must return Task or Task<T>, but returns '{1}'",
        category: Category,
        defaultSeverity: DiagnosticSeverity.Error,
        isEnabledByDefault: true,
        description: "GPU handler methods must return Task or Task<T> to support async message processing.");

    /// <summary>
    /// GPUGEN002: Parameter type is not blittable.
    /// </summary>
    public static readonly DiagnosticDescriptor NonBlittableParameter = new(
        id: "GPUGEN002",
        title: "Non-blittable parameter type",
        messageFormat: "Parameter '{0}' of type '{1}' in method '{2}' is not blittable and cannot be used in GPU kernel messages",
        category: Category,
        defaultSeverity: DiagnosticSeverity.Error,
        isEnabledByDefault: true,
        description: "All GPU handler parameters must be blittable types (value types without references) to be copied to GPU memory.");

    /// <summary>
    /// GPUGEN003: Message payload exceeds maximum size.
    /// </summary>
    public static readonly DiagnosticDescriptor PayloadExceedsMaxSize = new(
        id: "GPUGEN003",
        title: "Message payload exceeds maximum size",
        messageFormat: "Method '{0}' has a payload size of {1} bytes, which exceeds the maximum of {2} bytes. Consider enabling chunking or reducing payload size.",
        category: Category,
        defaultSeverity: DiagnosticSeverity.Error,
        isEnabledByDefault: true,
        description: "Ring kernel messages have a maximum payload size (default 228 bytes). Enable chunking for larger payloads.");

    /// <summary>
    /// GPUGEN004: Reference type in payload.
    /// </summary>
    public static readonly DiagnosticDescriptor ReferenceTypeInPayload = new(
        id: "GPUGEN004",
        title: "Reference type in payload",
        messageFormat: "Type '{0}' used in method '{1}' contains reference type '{2}'. GPU payloads must be value types only.",
        category: Category,
        defaultSeverity: DiagnosticSeverity.Error,
        isEnabledByDefault: true,
        description: "GPU kernels cannot process reference types. All types in GPU messages must be value types.");

    /// <summary>
    /// GPUGEN005: Chunking required but not enabled.
    /// </summary>
    public static readonly DiagnosticDescriptor ChunkingRequired = new(
        id: "GPUGEN005",
        title: "Chunking required for large payload",
        messageFormat: "Method '{0}' uses array type '{1}' which requires chunking, but EnableChunking is not set",
        category: Category,
        defaultSeverity: DiagnosticSeverity.Warning,
        isEnabledByDefault: true,
        description: "Array parameters typically require chunking to fit within message size limits.");

    /// <summary>
    /// GPUGEN006: Inefficient struct layout.
    /// </summary>
    public static readonly DiagnosticDescriptor InefficientStructLayout = new(
        id: "GPUGEN006",
        title: "Inefficient struct layout",
        messageFormat: "Struct '{0}' has {1} bytes of padding due to field alignment. Consider reordering fields for better packing.",
        category: Category,
        defaultSeverity: DiagnosticSeverity.Info,
        isEnabledByDefault: true,
        description: "Struct fields should be ordered by size (largest first) to minimize padding and improve GPU memory efficiency.");

    /// <summary>
    /// GPUGEN007: Interface must inherit from IGrainWithIntegerKey or similar.
    /// </summary>
    public static readonly DiagnosticDescriptor MissingGrainInterface = new(
        id: "GPUGEN007",
        title: "Missing grain interface inheritance",
        messageFormat: "Interface '{0}' marked with [GpuNativeActor] must inherit from an Orleans grain interface (e.g., IGrainWithIntegerKey)",
        category: Category,
        defaultSeverity: DiagnosticSeverity.Error,
        isEnabledByDefault: true,
        description: "GPU native actors must be Orleans grains. The interface must inherit from IGrainWithIntegerKey, IGrainWithStringKey, or IGrainWithGuidKey.");

    /// <summary>
    /// GPUGEN008: No handlers defined.
    /// </summary>
    public static readonly DiagnosticDescriptor NoHandlersDefined = new(
        id: "GPUGEN008",
        title: "No GPU handlers defined",
        messageFormat: "Interface '{0}' marked with [GpuNativeActor] has no methods marked with [GpuHandler]",
        category: Category,
        defaultSeverity: DiagnosticSeverity.Warning,
        isEnabledByDefault: true,
        description: "A GPU native actor should have at least one method marked with [GpuHandler] to be useful.");

    /// <summary>
    /// GPUGEN010: Non-blittable state type.
    /// </summary>
    public static readonly DiagnosticDescriptor NonBlittableState = new(
        id: "GPUGEN010",
        title: "Non-blittable state type",
        messageFormat: "State property '{0}' of type '{1}' is not blittable and cannot be stored in GPU memory",
        category: Category,
        defaultSeverity: DiagnosticSeverity.Error,
        isEnabledByDefault: true,
        description: "GPU state properties must be blittable types to be stored in GPU memory.");

    /// <summary>
    /// GPUGEN020: K2K target is not a GPU actor.
    /// </summary>
    public static readonly DiagnosticDescriptor K2KTargetNotGpuActor = new(
        id: "GPUGEN020",
        title: "K2K target is not a GPU actor",
        messageFormat: "K2K target '{0}' is not marked with [GpuNativeActor]. K2K messaging requires both actors to be GPU-native.",
        category: Category,
        defaultSeverity: DiagnosticSeverity.Error,
        isEnabledByDefault: true,
        description: "Kernel-to-kernel messaging requires both the source and target actors to be GPU-native actors.");

    /// <summary>
    /// GPUGEN021: K2K message size mismatch.
    /// </summary>
    public static readonly DiagnosticDescriptor K2KMessageSizeMismatch = new(
        id: "GPUGEN021",
        title: "K2K message size mismatch",
        messageFormat: "K2K target method '{0}' expects {1} bytes but source provides {2} bytes",
        category: Category,
        defaultSeverity: DiagnosticSeverity.Error,
        isEnabledByDefault: true,
        description: "K2K messages must have compatible sizes between source and target methods.");

    /// <summary>
    /// GPUGEN030: Missing temporal dependency.
    /// </summary>
    public static readonly DiagnosticDescriptor MissingTemporalDependency = new(
        id: "GPUGEN030",
        title: "Missing temporal dependency",
        messageFormat: "Interface '{0}' is marked with [TemporalOrdered] but the project does not reference Orleans.GpuBridge.Abstractions.Temporal",
        category: Category,
        defaultSeverity: DiagnosticSeverity.Error,
        isEnabledByDefault: true,
        description: "Temporal ordering requires the temporal abstractions package.");

    /// <summary>
    /// GPUGEN099: Internal generator error.
    /// </summary>
    public static readonly DiagnosticDescriptor InternalError = new(
        id: "GPUGEN099",
        title: "Internal generator error",
        messageFormat: "An internal error occurred during GPU actor code generation: {0}",
        category: Category,
        defaultSeverity: DiagnosticSeverity.Error,
        isEnabledByDefault: true,
        description: "An unexpected error occurred in the source generator. Please report this issue.");
}

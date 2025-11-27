// Orleans.GpuBridge - GPU-native distributed computing for Microsoft Orleans
// Copyright (c) 2025 Michael Ivertowski

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Threading;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Orleans.GpuBridge.Generators.Models;

namespace Orleans.GpuBridge.Generators.Analysis;

/// <summary>
/// Analyzes interface declarations to extract GPU native actor information.
/// </summary>
public static class ActorInterfaceAnalyzer
{
    private const string GpuNativeActorAttributeName = "GpuNativeActorAttribute";
    private const string GpuHandlerAttributeName = "GpuHandlerAttribute";
    private const string GpuStateAttributeName = "GpuStateAttribute";
    private const string TemporalOrderedAttributeName = "TemporalOrderedAttribute";
    private const string K2KTargetAttributeName = "K2KTargetAttribute";

    /// <summary>
    /// Analyzes an interface declaration and extracts GPU actor information.
    /// </summary>
    /// <param name="interfaceDeclaration">The interface syntax node.</param>
    /// <param name="semanticModel">The semantic model.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>GPU actor info if this is a valid GPU native actor, otherwise null.</returns>
    public static GpuActorInfo? AnalyzeInterface(
        InterfaceDeclarationSyntax interfaceDeclaration,
        SemanticModel semanticModel,
        CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();

        var interfaceSymbol = semanticModel.GetDeclaredSymbol(interfaceDeclaration, cancellationToken) as INamedTypeSymbol;
        if (interfaceSymbol == null)
        {
            return null;
        }

        // Check for [GpuNativeActor] attribute
        if (!HasGpuNativeActorAttribute(interfaceSymbol))
        {
            return null;
        }

        // Determine grain key type
        var keyType = DetermineGrainKeyType(interfaceSymbol);

        // Analyze handler methods
        var handlers = AnalyzeHandlers(interfaceSymbol, semanticModel, cancellationToken);

        // Analyze state properties
        var stateProperties = AnalyzeStateProperties(interfaceSymbol, semanticModel, cancellationToken);

        // Analyze temporal ordering
        var temporalOrdering = AnalyzeTemporalOrdering(interfaceSymbol);

        return new GpuActorInfo
        {
            Namespace = interfaceSymbol.ContainingNamespace.ToDisplayString(),
            InterfaceName = interfaceSymbol.Name,
            KeyType = keyType,
            Handlers = handlers,
            StateProperties = stateProperties,
            TemporalOrdering = temporalOrdering,
            Location = interfaceDeclaration.GetLocation()
        };
    }

    private static bool HasGpuNativeActorAttribute(INamedTypeSymbol symbol)
    {
        return symbol.GetAttributes().Any(attr =>
            attr.AttributeClass?.Name == GpuNativeActorAttributeName ||
            attr.AttributeClass?.Name == "GpuNativeActor");
    }

    private static GrainKeyType DetermineGrainKeyType(INamedTypeSymbol interfaceSymbol)
    {
        foreach (var implementedInterface in interfaceSymbol.AllInterfaces)
        {
            var name = implementedInterface.Name;
            if (name == "IGrainWithIntegerKey")
                return GrainKeyType.Integer;
            if (name == "IGrainWithStringKey")
                return GrainKeyType.String;
            if (name == "IGrainWithGuidKey")
                return GrainKeyType.Guid;
            if (name == "IGrainWithIntegerCompoundKey")
                return GrainKeyType.IntegerCompound;
            if (name == "IGrainWithGuidCompoundKey")
                return GrainKeyType.GuidCompound;
        }

        return GrainKeyType.Unknown;
    }

    private static ImmutableArray<GpuHandlerInfo> AnalyzeHandlers(
        INamedTypeSymbol interfaceSymbol,
        SemanticModel semanticModel,
        CancellationToken cancellationToken)
    {
        var handlers = ImmutableArray.CreateBuilder<GpuHandlerInfo>();
        int messageTypeId = 1;

        foreach (var member in interfaceSymbol.GetMembers())
        {
            if (member is not IMethodSymbol method)
                continue;

            var handlerAttr = method.GetAttributes()
                .FirstOrDefault(attr =>
                    attr.AttributeClass?.Name == GpuHandlerAttributeName ||
                    attr.AttributeClass?.Name == "GpuHandler");

            if (handlerAttr == null)
                continue;

            var handlerInfo = AnalyzeHandler(method, handlerAttr, messageTypeId++, cancellationToken);
            if (handlerInfo != null)
            {
                handlers.Add(handlerInfo);
            }
        }

        return handlers.ToImmutable();
    }

    private static GpuHandlerInfo? AnalyzeHandler(
        IMethodSymbol method,
        AttributeData handlerAttr,
        int messageTypeId,
        CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();

        // Extract return type
        var (returnTypeName, hasReturnValue) = AnalyzeReturnType(method.ReturnType);

        // Extract parameters
        var parameters = ImmutableArray.CreateBuilder<GpuParameterInfo>();
        int totalRequestSize = 4; // Start with 4 bytes for message type ID

        foreach (var param in method.Parameters)
        {
            var paramInfo = AnalyzeParameter(param);
            parameters.Add(paramInfo);
            totalRequestSize += paramInfo.SizeInBytes;
        }

        // Calculate response size
        int responseSize = 4; // Message type ID
        if (hasReturnValue)
        {
            responseSize += MessageSizeCalculator.GetTypeSize(method.ReturnType, out _);
        }

        // Extract attribute properties
        int maxPayloadSize = GetAttributeValue<int>(handlerAttr, "MaxPayloadSize", 228);
        bool enableChunking = GetAttributeValue<bool>(handlerAttr, "EnableChunking", false);
        int queueDepth = GetAttributeValue<int>(handlerAttr, "QueueDepth", 1024);
        string mode = GetAttributeEnumValue(handlerAttr, "Mode", "RequestResponse");

        // Analyze K2K targets
        var k2kTargets = AnalyzeK2KTargets(method);

        return new GpuHandlerInfo
        {
            MethodName = method.Name,
            ReturnTypeName = returnTypeName,
            HasReturnValue = hasReturnValue,
            Parameters = parameters.ToImmutable(),
            MaxPayloadSize = maxPayloadSize,
            EnableChunking = enableChunking,
            Mode = mode,
            QueueDepth = queueDepth,
            K2KTargets = k2kTargets,
            RequestMessageSize = totalRequestSize,
            ResponseMessageSize = responseSize,
            MessageTypeId = messageTypeId,
            Location = method.Locations.FirstOrDefault() ?? Location.None
        };
    }

    private static (string typeName, bool hasValue) AnalyzeReturnType(ITypeSymbol returnType)
    {
        // Check if it's Task<T>
        if (returnType is INamedTypeSymbol namedType &&
            namedType.OriginalDefinition.ToDisplayString() == "System.Threading.Tasks.Task<TResult>")
        {
            var typeArg = namedType.TypeArguments.FirstOrDefault();
            return (typeArg?.ToDisplayString() ?? "void", true);
        }

        // Check if it's Task
        if (returnType.ToDisplayString() == "System.Threading.Tasks.Task")
        {
            return ("void", false);
        }

        // Fallback - assume it's the actual return type
        return (returnType.ToDisplayString(), true);
    }

    private static GpuParameterInfo AnalyzeParameter(IParameterSymbol param)
    {
        var isBlittable = MessageSizeCalculator.IsBlittable(param.Type);
        var isArray = param.Type.TypeKind == TypeKind.Array;
        string? arrayElementType = null;

        if (isArray && param.Type is IArrayTypeSymbol arrayType)
        {
            arrayElementType = arrayType.ElementType.ToDisplayString();
        }

        var size = MessageSizeCalculator.GetTypeSize(param.Type, out _);

        return new GpuParameterInfo
        {
            Name = param.Name,
            TypeName = param.Type.ToDisplayString(),
            SimpleTypeName = GetSimpleTypeName(param.Type),
            IsBlittable = isBlittable,
            IsArray = isArray,
            ArrayElementTypeName = arrayElementType,
            SizeInBytes = size
        };
    }

    private static ImmutableArray<GpuStateInfo> AnalyzeStateProperties(
        INamedTypeSymbol interfaceSymbol,
        SemanticModel semanticModel,
        CancellationToken cancellationToken)
    {
        var properties = ImmutableArray.CreateBuilder<GpuStateInfo>();

        foreach (var member in interfaceSymbol.GetMembers())
        {
            if (member is not IPropertySymbol property)
                continue;

            var stateAttr = property.GetAttributes()
                .FirstOrDefault(attr =>
                    attr.AttributeClass?.Name == GpuStateAttributeName ||
                    attr.AttributeClass?.Name == "GpuState");

            if (stateAttr == null)
                continue;

            var isBlittable = MessageSizeCalculator.IsBlittable(property.Type);
            var size = MessageSizeCalculator.GetTypeSize(property.Type, out _);
            var persist = GetAttributeValue<bool>(stateAttr, "Persist", true);
            var initialValue = GetAttributeValue<string?>(stateAttr, "InitialValue", null);

            properties.Add(new GpuStateInfo
            {
                Name = property.Name,
                TypeName = property.Type.ToDisplayString(),
                SimpleTypeName = GetSimpleTypeName(property.Type),
                Persist = persist,
                InitialValue = initialValue,
                SizeInBytes = size,
                IsBlittable = isBlittable
            });
        }

        return properties.ToImmutable();
    }

    private static TemporalOrderingInfo? AnalyzeTemporalOrdering(INamedTypeSymbol interfaceSymbol)
    {
        var temporalAttr = interfaceSymbol.GetAttributes()
            .FirstOrDefault(attr =>
                attr.AttributeClass?.Name == TemporalOrderedAttributeName ||
                attr.AttributeClass?.Name == "TemporalOrdered");

        if (temporalAttr == null)
            return null;

        return new TemporalOrderingInfo
        {
            ClockType = GetAttributeEnumValue(temporalAttr, "ClockType", "HLC"),
            StrictOrdering = GetAttributeValue<bool>(temporalAttr, "StrictOrdering", false),
            MaxClockDriftMs = GetAttributeValue<int>(temporalAttr, "MaxClockDriftMs", 100),
            MaxVectorClockSize = GetAttributeValue<int>(temporalAttr, "MaxVectorClockSize", 16)
        };
    }

    private static ImmutableArray<K2KTargetInfo> AnalyzeK2KTargets(IMethodSymbol method)
    {
        var targets = ImmutableArray.CreateBuilder<K2KTargetInfo>();

        foreach (var attr in method.GetAttributes())
        {
            if (attr.AttributeClass?.Name != K2KTargetAttributeName &&
                attr.AttributeClass?.Name != "K2KTarget")
                continue;

            if (attr.ConstructorArguments.Length < 2)
                continue;

            var targetType = attr.ConstructorArguments[0].Value as INamedTypeSymbol;
            var targetMethod = attr.ConstructorArguments[1].Value as string;

            if (targetType == null || targetMethod == null)
                continue;

            targets.Add(new K2KTargetInfo
            {
                TargetActorTypeName = targetType.ToDisplayString(),
                TargetMethodName = targetMethod,
                AllowCpuFallback = GetAttributeValue<bool>(attr, "AllowCpuFallback", true),
                RoutingStrategy = GetAttributeEnumValue(attr, "RoutingStrategy", "Direct")
            });
        }

        return targets.ToImmutable();
    }

    private static string GetSimpleTypeName(ITypeSymbol type)
    {
        // Return the simple name for code generation
        if (type is INamedTypeSymbol namedType)
        {
            return namedType.Name;
        }
        return type.ToDisplayString(SymbolDisplayFormat.MinimallyQualifiedFormat);
    }

    private static T GetAttributeValue<T>(AttributeData attr, string propertyName, T defaultValue)
    {
        var namedArg = attr.NamedArguments.FirstOrDefault(na => na.Key == propertyName);
        if (!namedArg.Equals(default(KeyValuePair<string, TypedConstant>)) && namedArg.Value.Value is T value)
        {
            return value;
        }
        return defaultValue;
    }

    private static string GetAttributeEnumValue(AttributeData attr, string propertyName, string defaultValue)
    {
        var namedArg = attr.NamedArguments.FirstOrDefault(na => na.Key == propertyName);
        if (!namedArg.Equals(default(KeyValuePair<string, TypedConstant>)))
        {
            // Enum values come as the underlying type, we need the name
            if (namedArg.Value.Value != null)
            {
                return namedArg.Value.Value.ToString() ?? defaultValue;
            }
        }
        return defaultValue;
    }
}

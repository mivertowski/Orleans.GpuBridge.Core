using System;
using System.Collections.Immutable;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Microsoft.CodeAnalysis.Diagnostics;

namespace Orleans.GpuBridge.Analyzers.Analyzers.Correctness;

/// <summary>
/// Analyzer that validates power-of-2 requirements for GPU-native actor configurations.
/// Queue capacity and message size must be powers of 2 for efficient GPU indexing.
/// </summary>
[DiagnosticAnalyzer(LanguageNames.CSharp)]
public sealed class PowerOfTwoAnalyzer : DiagnosticAnalyzer
{
    private static readonly DiagnosticDescriptor QueueCapacityRule = new(
        id: DiagnosticIds.QueueCapacityNotPowerOfTwo,
        title: "Queue capacity must be a power of 2",
        messageFormat: "Queue capacity {0} is not a power of 2. GPU-native actors require power-of-2 sizing for efficient indexing. Use {1} or {2} instead",
        category: DiagnosticCategories.Correctness,
        defaultSeverity: DiagnosticSeverity.Error,
        isEnabledByDefault: true,
        description: "GPU-native message queues require power-of-2 capacity for efficient modulo operations " +
                    "using bitwise AND. Non-power-of-2 sizes cause runtime failures.",
        helpLinkUri: "https://github.com/yourusername/Orleans.GpuBridge.Core/docs/gpu-native-actors.md");

    private static readonly DiagnosticDescriptor MessageSizeRule = new(
        id: DiagnosticIds.MessageSizeNotPowerOfTwo,
        title: "Message size must be a power of 2",
        messageFormat: "Message size {0} bytes is not a power of 2. GPU memory transfers require power-of-2 alignment. Use {1} or {2} bytes instead",
        category: DiagnosticCategories.Correctness,
        defaultSeverity: DiagnosticSeverity.Error,
        isEnabledByDefault: true,
        description: "GPU-native actors require power-of-2 message sizes for optimal memory transfer and alignment. " +
                    "Non-power-of-2 sizes cause inefficient GPU memory access patterns.",
        helpLinkUri: "https://github.com/yourusername/Orleans.GpuBridge.Core/docs/gpu-native-actors.md");

    public override ImmutableArray<DiagnosticDescriptor> SupportedDiagnostics =>
        ImmutableArray.Create(QueueCapacityRule, MessageSizeRule);

    public override void Initialize(AnalysisContext context)
    {
        context.ConfigureGeneratedCodeAnalysis(GeneratedCodeAnalysisFlags.None);
        context.EnableConcurrentExecution();

        context.RegisterSyntaxNodeAction(AnalyzeObjectInitializer, SyntaxKind.ObjectInitializerExpression);
    }

    private static void AnalyzeObjectInitializer(SyntaxNodeAnalysisContext context)
    {
        var initializer = (InitializerExpressionSyntax)context.Node;

        // Check if parent is object creation for GpuNativeActorConfiguration or VertexConfiguration
        if (initializer.Parent is not ObjectCreationExpressionSyntax objectCreation)
        {
            return;
        }

        var typeInfo = context.SemanticModel.GetTypeInfo(objectCreation);
        if (typeInfo.Type == null)
        {
            return;
        }

        var typeName = typeInfo.Type.ToDisplayString();
        if (!IsGpuNativeActorConfiguration(typeName))
        {
            return;
        }

        // Analyze each assignment in the initializer
        foreach (var expression in initializer.Expressions)
        {
            if (expression is not AssignmentExpressionSyntax assignment)
            {
                continue;
            }

            if (assignment.Left is not IdentifierNameSyntax propertyName)
            {
                continue;
            }

            var propertyNameText = propertyName.Identifier.Text;

            // Check MessageQueueCapacity
            if (propertyNameText == "MessageQueueCapacity")
            {
                ValidateProperty(context, assignment.Right, QueueCapacityRule);
            }
            // Check MessageSize
            else if (propertyNameText == "MessageSize")
            {
                ValidateProperty(context, assignment.Right, MessageSizeRule);
            }
        }
    }

    private static void ValidateProperty(
        SyntaxNodeAnalysisContext context,
        ExpressionSyntax valueExpression,
        DiagnosticDescriptor rule)
    {
        // Try to get constant value
        var constantValue = context.SemanticModel.GetConstantValue(valueExpression);
        if (!constantValue.HasValue || constantValue.Value is not int intValue)
        {
            return;
        }

        // Check if power of 2
        if (IsPowerOfTwo(intValue))
        {
            return;
        }

        // Calculate nearest powers of 2
        var (lower, upper) = GetNearestPowersOfTwo(intValue);

        // Report diagnostic
        var diagnostic = Diagnostic.Create(
            rule,
            valueExpression.GetLocation(),
            intValue,
            lower,
            upper);

        context.ReportDiagnostic(diagnostic);
    }

    private static bool IsGpuNativeActorConfiguration(string typeName)
    {
        return typeName == "Orleans.GpuBridge.Grains.GpuNative.GpuNativeActorConfiguration" ||
               typeName == "Orleans.GpuBridge.Grains.GpuNative.VertexConfiguration" ||
               typeName.EndsWith(".GpuNativeActorConfiguration", StringComparison.Ordinal) ||
               typeName.EndsWith(".VertexConfiguration", StringComparison.Ordinal);
    }

    private static bool IsPowerOfTwo(int value)
    {
        return value > 0 && (value & (value - 1)) == 0;
    }

    private static (int Lower, int Upper) GetNearestPowersOfTwo(int value)
    {
        if (value <= 1)
        {
            return (1, 2);
        }

        // Find the nearest lower power of 2
        int log = (int)Math.Floor(Math.Log(value, 2));
        int lower = 1 << log;

        // Find the nearest upper power of 2
        int upper = 1 << (log + 1);

        return (lower, upper);
    }
}

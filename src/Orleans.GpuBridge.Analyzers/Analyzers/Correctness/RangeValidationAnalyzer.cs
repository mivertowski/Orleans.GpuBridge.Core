using System;
using System.Collections.Immutable;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Microsoft.CodeAnalysis.Diagnostics;

namespace Orleans.GpuBridge.Analyzers.Analyzers.Correctness;

/// <summary>
/// Analyzer that validates range constraints for GPU-native actor configuration values.
/// </summary>
[DiagnosticAnalyzer(LanguageNames.CSharp)]
public sealed class RangeValidationAnalyzer : DiagnosticAnalyzer
{
    private static readonly DiagnosticDescriptor QueueCapacityRangeRule = new(
        id: DiagnosticIds.QueueCapacityOutOfRange,
        title: "Queue capacity out of valid range",
        messageFormat: "Queue capacity {0} is out of range. Valid range is 256 to 1,048,576. {1}",
        category: DiagnosticCategories.Correctness,
        defaultSeverity: DiagnosticSeverity.Error,
        isEnabledByDefault: true,
        description: "GPU-native message queues must have capacity between 256 and 1,048,576. " +
                    "Smaller queues risk constant overflow. Larger queues may exhaust GPU memory.",
        helpLinkUri: "https://github.com/yourusername/Orleans.GpuBridge.Core/docs/gpu-native-actors.md");

    private static readonly DiagnosticDescriptor MessageSizeRangeRule = new(
        id: DiagnosticIds.MessageSizeOutOfRange,
        title: "Message size out of valid range",
        messageFormat: "Message size {0} bytes is out of range. Valid range is 256 to 4,096 bytes. {1}",
        category: DiagnosticCategories.Correctness,
        defaultSeverity: DiagnosticSeverity.Error,
        isEnabledByDefault: true,
        description: "GPU-native actor messages must be between 256 and 4,096 bytes. " +
                    "Smaller messages waste GPU bandwidth. Larger messages reduce cache effectiveness.",
        helpLinkUri: "https://github.com/yourusername/Orleans.GpuBridge.Core/docs/gpu-native-actors.md");

    private static readonly DiagnosticDescriptor ThreadsPerActorRule = new(
        id: DiagnosticIds.ThreadsPerActorExceedsLimit,
        title: "Threads per actor exceeds CUDA limit",
        messageFormat: "Threads per actor {0} exceeds the maximum of 1,024. This is the CUDA block size limit. Most actors only need 1 thread",
        category: DiagnosticCategories.Correctness,
        defaultSeverity: DiagnosticSeverity.Error,
        isEnabledByDefault: true,
        description: "CUDA restricts block size to 1,024 threads. GPU-native actors that exceed this limit " +
                    "will fail to launch. Most actors only need 1 thread per actor.",
        helpLinkUri: "https://github.com/yourusername/Orleans.GpuBridge.Core/docs/gpu-native-actors.md");

    private const int MinQueueCapacity = 256;
    private const int MaxQueueCapacity = 1_048_576;
    private const int MinMessageSize = 256;
    private const int MaxMessageSize = 4096;
    private const int MaxThreadsPerActor = 1024;

    public override ImmutableArray<DiagnosticDescriptor> SupportedDiagnostics =>
        ImmutableArray.Create(QueueCapacityRangeRule, MessageSizeRangeRule, ThreadsPerActorRule);

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
                ValidateQueueCapacity(context, assignment.Right);
            }
            // Check MessageSize
            else if (propertyNameText == "MessageSize")
            {
                ValidateMessageSize(context, assignment.Right);
            }
            // Check ThreadsPerActor
            else if (propertyNameText == "ThreadsPerActor")
            {
                ValidateThreadsPerActor(context, assignment.Right);
            }
        }
    }

    private static void ValidateQueueCapacity(SyntaxNodeAnalysisContext context, ExpressionSyntax valueExpression)
    {
        var constantValue = context.SemanticModel.GetConstantValue(valueExpression);
        if (!constantValue.HasValue || constantValue.Value is not int intValue)
        {
            return;
        }

        if (intValue >= MinQueueCapacity && intValue <= MaxQueueCapacity)
        {
            return;
        }

        string suggestion;
        if (intValue < MinQueueCapacity)
        {
            suggestion = $"Use at least {MinQueueCapacity} to avoid constant overflow with GPU-native message rates";
        }
        else
        {
            suggestion = $"Use at most {MaxQueueCapacity} to avoid exhausting GPU memory";
        }

        var diagnostic = Diagnostic.Create(
            QueueCapacityRangeRule,
            valueExpression.GetLocation(),
            intValue,
            suggestion);

        context.ReportDiagnostic(diagnostic);
    }

    private static void ValidateMessageSize(SyntaxNodeAnalysisContext context, ExpressionSyntax valueExpression)
    {
        var constantValue = context.SemanticModel.GetConstantValue(valueExpression);
        if (!constantValue.HasValue || constantValue.Value is not int intValue)
        {
            return;
        }

        if (intValue >= MinMessageSize && intValue <= MaxMessageSize)
        {
            return;
        }

        string suggestion;
        if (intValue < MinMessageSize)
        {
            suggestion = $"Use at least {MinMessageSize} bytes to avoid wasting GPU memory bandwidth due to alignment";
        }
        else
        {
            suggestion = $"Use at most {MaxMessageSize} bytes to maintain GPU cache effectiveness. Consider splitting large messages";
        }

        var diagnostic = Diagnostic.Create(
            MessageSizeRangeRule,
            valueExpression.GetLocation(),
            intValue,
            suggestion);

        context.ReportDiagnostic(diagnostic);
    }

    private static void ValidateThreadsPerActor(SyntaxNodeAnalysisContext context, ExpressionSyntax valueExpression)
    {
        var constantValue = context.SemanticModel.GetConstantValue(valueExpression);
        if (!constantValue.HasValue || constantValue.Value is not int intValue)
        {
            return;
        }

        if (intValue > 0 && intValue <= MaxThreadsPerActor)
        {
            return;
        }

        var diagnostic = Diagnostic.Create(
            ThreadsPerActorRule,
            valueExpression.GetLocation(),
            intValue);

        context.ReportDiagnostic(diagnostic);
    }

    private static bool IsGpuNativeActorConfiguration(string typeName)
    {
        return typeName == "Orleans.GpuBridge.Grains.GpuNative.GpuNativeActorConfiguration" ||
               typeName == "Orleans.GpuBridge.Grains.GpuNative.VertexConfiguration" ||
               typeName.EndsWith(".GpuNativeActorConfiguration", StringComparison.Ordinal) ||
               typeName.EndsWith(".VertexConfiguration", StringComparison.Ordinal);
    }
}

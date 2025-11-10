using System;
using System.Collections.Immutable;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Microsoft.CodeAnalysis.Diagnostics;

namespace Orleans.GpuBridge.Analyzers.Analyzers.Performance;

/// <summary>
/// Analyzer that warns about temporal ordering performance overhead.
/// Temporal ordering adds ~15% performance overhead and should only be enabled when needed.
/// </summary>
[DiagnosticAnalyzer(LanguageNames.CSharp)]
public sealed class TemporalOrderingAnalyzer : DiagnosticAnalyzer
{
    private static readonly DiagnosticDescriptor Rule = new(
        id: DiagnosticIds.TemporalOrderingOverhead,
        title: "Temporal ordering adds 15% performance overhead",
        messageFormat: "Temporal ordering is enabled. This adds ~15% performance overhead. Only enable if you need causal consistency guarantees",
        category: DiagnosticCategories.Performance,
        defaultSeverity: DiagnosticSeverity.Info,
        isEnabledByDefault: true,
        description: "Temporal ordering with Hybrid Logical Clocks provides causal consistency but adds approximately " +
                    "15% performance overhead due to timestamp management and ordering checks. Only enable this feature " +
                    "if your application requires happened-before relationships between messages. Disable for " +
                    "performance-critical applications that don't need causal ordering.",
        helpLinkUri: "https://github.com/yourusername/Orleans.GpuBridge.Core/docs/temporal-ordering.md");

    public override ImmutableArray<DiagnosticDescriptor> SupportedDiagnostics => ImmutableArray.Create(Rule);

    public override void Initialize(AnalysisContext context)
    {
        context.ConfigureGeneratedCodeAnalysis(GeneratedCodeAnalysisFlags.None);
        context.EnableConcurrentExecution();

        context.RegisterSyntaxNodeAction(AnalyzeObjectInitializer, SyntaxKind.ObjectInitializerExpression);
    }

    private static void AnalyzeObjectInitializer(SyntaxNodeAnalysisContext context)
    {
        var initializer = (InitializerExpressionSyntax)context.Node;

        // Check if parent is object creation for GpuNativeActorConfiguration
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

        // Look for EnableTemporalOrdering = true
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

            if (propertyName.Identifier.Text != "EnableTemporalOrdering")
            {
                continue;
            }

            // Check if value is 'true'
            var constantValue = context.SemanticModel.GetConstantValue(assignment.Right);
            if (!constantValue.HasValue || constantValue.Value is not true)
            {
                continue;
            }

            // Report diagnostic
            var diagnostic = Diagnostic.Create(
                Rule,
                assignment.GetLocation());

            context.ReportDiagnostic(diagnostic);
        }
    }

    private static bool IsGpuNativeActorConfiguration(string typeName)
    {
        return typeName == "Orleans.GpuBridge.Grains.GpuNative.GpuNativeActorConfiguration" ||
               typeName == "Orleans.GpuBridge.Grains.GpuNative.VertexConfiguration" ||
               typeName.EndsWith(".GpuNativeActorConfiguration", StringComparison.Ordinal) ||
               typeName.EndsWith(".VertexConfiguration", StringComparison.Ordinal);
    }
}

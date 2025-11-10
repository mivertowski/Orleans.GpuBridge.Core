using System;
using System.Collections.Immutable;
using System.Linq;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Microsoft.CodeAnalysis.Diagnostics;

namespace Orleans.GpuBridge.Analyzers.Analyzers.BestPractices;

/// <summary>
/// Analyzer that detects missing telemetry registration for GPU-native actors.
/// Production systems should always register telemetry for observability.
/// </summary>
[DiagnosticAnalyzer(LanguageNames.CSharp)]
public sealed class TelemetryAnalyzer : DiagnosticAnalyzer
{
    private static readonly DiagnosticDescriptor Rule = new(
        id: DiagnosticIds.MissingTelemetryRegistration,
        title: "Consider registering telemetry for GPU-native actors",
        messageFormat: "No telemetry registration detected. Production systems should register RingKernelTelemetry, TemporalOrderingTelemetry, and MessageQueueTelemetry for observability",
        category: DiagnosticCategories.BestPractices,
        defaultSeverity: DiagnosticSeverity.Info,
        isEnabledByDefault: true,
        description: "GPU-native actors benefit from telemetry to monitor kernel health, temporal ordering accuracy, " +
                    "and message queue utilization. Register Orleans.GpuBridge.Diagnostics telemetry services for " +
                    "comprehensive observability in production environments.",
        helpLinkUri: "https://github.com/yourusername/Orleans.GpuBridge.Core/docs/telemetry.md");

    public override ImmutableArray<DiagnosticDescriptor> SupportedDiagnostics => ImmutableArray.Create(Rule);

    public override void Initialize(AnalysisContext context)
    {
        context.ConfigureGeneratedCodeAnalysis(GeneratedCodeAnalysisFlags.None);
        context.EnableConcurrentExecution();

        // Analyze entire compilation to check for telemetry registration
        context.RegisterCompilationStartAction(compilationContext =>
        {
            var hasGpuBridgeReference = compilationContext.Compilation.ReferencedAssemblyNames
                .Any(name => name.Name.Contains("Orleans.GpuBridge", StringComparison.OrdinalIgnoreCase));

            if (!hasGpuBridgeReference)
            {
                return;
            }

            var telemetryTypes = new[]
            {
                "RingKernelTelemetry",
                "TemporalOrderingTelemetry",
                "MessageQueueTelemetry"
            };

            var foundTelemetryRegistrations = false;

            compilationContext.RegisterSyntaxNodeAction(nodeContext =>
            {
                var invocation = (InvocationExpressionSyntax)nodeContext.Node;

                // Check for AddSingleton or AddScoped calls
                if (invocation.Expression is not MemberAccessExpressionSyntax memberAccess)
                {
                    return;
                }

                var methodName = memberAccess.Name.Identifier.Text;
                if (methodName != "AddSingleton" && methodName != "AddScoped" && methodName != "AddTransient")
                {
                    return;
                }

                // Check if it's registering a telemetry type
                var symbolInfo = nodeContext.SemanticModel.GetSymbolInfo(invocation);
                if (symbolInfo.Symbol is not IMethodSymbol methodSymbol)
                {
                    return;
                }

                // Check type arguments for telemetry types
                if (methodSymbol.TypeArguments.Length > 0)
                {
                    var typeArg = methodSymbol.TypeArguments[0];
                    var typeName = typeArg.Name;

                    if (telemetryTypes.Any(t => typeName.Contains(t, StringComparison.Ordinal)))
                    {
                        foundTelemetryRegistrations = true;
                    }
                }
            }, SyntaxKind.InvocationExpression);

            compilationContext.RegisterCompilationEndAction(endContext =>
            {
                // If we found GPU bridge usage but no telemetry registration,
                // report on a suitable location (typically the first method in the compilation)
                if (!foundTelemetryRegistrations)
                {
                    // Find a suitable location to report the diagnostic
                    var firstMethod = endContext.Compilation.SyntaxTrees
                        .SelectMany(tree => tree.GetRoot().DescendantNodes())
                        .OfType<MethodDeclarationSyntax>()
                        .FirstOrDefault(m => m.Identifier.Text.Contains("Configure", StringComparison.OrdinalIgnoreCase) ||
                                            m.Identifier.Text.Contains("Startup", StringComparison.OrdinalIgnoreCase));

                    if (firstMethod != null)
                    {
                        var diagnostic = Diagnostic.Create(
                            Rule,
                            firstMethod.Identifier.GetLocation());

                        endContext.ReportDiagnostic(diagnostic);
                    }
                }
            });
        });
    }
}

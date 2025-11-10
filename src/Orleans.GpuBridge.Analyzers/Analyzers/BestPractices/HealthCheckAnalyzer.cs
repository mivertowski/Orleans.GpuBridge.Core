using System;
using System.Collections.Immutable;
using System.Linq;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Microsoft.CodeAnalysis.Diagnostics;

namespace Orleans.GpuBridge.Analyzers.Analyzers.BestPractices;

/// <summary>
/// Analyzer that detects missing health check registration for GPU-native actors.
/// Production systems should register health checks for Kubernetes readiness/liveness probes.
/// </summary>
[DiagnosticAnalyzer(LanguageNames.CSharp)]
public sealed class HealthCheckAnalyzer : DiagnosticAnalyzer
{
    private static readonly DiagnosticDescriptor Rule = new(
        id: DiagnosticIds.MissingHealthCheckRegistration,
        title: "Consider registering health checks for GPU-native actors",
        messageFormat: "No health check registration detected. Production systems should register RingKernelHealthCheck for Kubernetes readiness/liveness probes",
        category: DiagnosticCategories.BestPractices,
        defaultSeverity: DiagnosticSeverity.Info,
        isEnabledByDefault: true,
        description: "GPU-native actors should register health checks to enable Kubernetes readiness and liveness probes. " +
                    "RingKernelHealthCheck monitors kernel responsiveness and message queue health. " +
                    "Use services.AddHealthChecks().AddCheck<RingKernelHealthCheck>(\"ring-kernels\") in production.",
        helpLinkUri: "https://github.com/yourusername/Orleans.GpuBridge.Core/docs/health-checks.md");

    public override ImmutableArray<DiagnosticDescriptor> SupportedDiagnostics => ImmutableArray.Create(Rule);

    public override void Initialize(AnalysisContext context)
    {
        context.ConfigureGeneratedCodeAnalysis(GeneratedCodeAnalysisFlags.None);
        context.EnableConcurrentExecution();

        // Analyze entire compilation to check for health check registration
        context.RegisterCompilationStartAction(compilationContext =>
        {
            var hasGpuBridgeReference = compilationContext.Compilation.ReferencedAssemblyNames
                .Any(name => name.Name.Contains("Orleans.GpuBridge", StringComparison.OrdinalIgnoreCase));

            if (!hasGpuBridgeReference)
            {
                return;
            }

            var foundHealthCheckRegistration = false;

            compilationContext.RegisterSyntaxNodeAction(nodeContext =>
            {
                var invocation = (InvocationExpressionSyntax)nodeContext.Node;

                // Check for AddHealthChecks() or AddCheck<T>() calls
                if (invocation.Expression is not MemberAccessExpressionSyntax memberAccess)
                {
                    return;
                }

                var methodName = memberAccess.Name.Identifier.Text;
                if (methodName != "AddHealthChecks" && methodName != "AddCheck")
                {
                    return;
                }

                // Check if it's registering RingKernelHealthCheck
                var symbolInfo = nodeContext.SemanticModel.GetSymbolInfo(invocation);
                if (symbolInfo.Symbol is not IMethodSymbol methodSymbol)
                {
                    return;
                }

                // Check type arguments for health check types
                if (methodSymbol.TypeArguments.Length > 0)
                {
                    var typeArg = methodSymbol.TypeArguments[0];
                    var typeName = typeArg.Name;

                    if (typeName.Contains("RingKernelHealthCheck", StringComparison.Ordinal) ||
                        typeName.Contains("GpuHealthCheck", StringComparison.Ordinal))
                    {
                        foundHealthCheckRegistration = true;
                    }
                }

                // Also check for AddHealthChecks() call which indicates health check setup
                if (methodName == "AddHealthChecks")
                {
                    // Don't mark as found yet, just that health checks are being configured
                    // We want to ensure GPU-specific checks are added
                }
            }, SyntaxKind.InvocationExpression);

            compilationContext.RegisterCompilationEndAction(endContext =>
            {
                // If we found GPU bridge usage but no health check registration,
                // report on a suitable location
                if (!foundHealthCheckRegistration)
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

using System;
using System.Collections.Immutable;
using System.Linq;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Microsoft.CodeAnalysis.Diagnostics;

namespace Orleans.GpuBridge.Analyzers.Analyzers.Correctness;

/// <summary>
/// Analyzer that detects synchronous blocking calls in async grain methods.
/// Blocking calls like .Result, .Wait(), and .GetAwaiter().GetResult() can cause deadlocks in Orleans grains.
/// </summary>
[DiagnosticAnalyzer(LanguageNames.CSharp)]
public sealed class SynchronousBlockingAnalyzer : DiagnosticAnalyzer
{
    private static readonly DiagnosticDescriptor Rule = new(
        id: DiagnosticIds.SynchronousBlockingInGrain,
        title: "Synchronous blocking calls must not be used in async grain methods",
        messageFormat: "Synchronous blocking call '{0}' in async grain method can cause deadlocks. Use 'await' instead",
        category: DiagnosticCategories.Correctness,
        defaultSeverity: DiagnosticSeverity.Error,
        isEnabledByDefault: true,
        description: "Synchronous blocking on asynchronous operations (.Result, .Wait(), .GetAwaiter().GetResult()) " +
                    "in Orleans grain methods can cause deadlocks due to Orleans' single-threaded execution model. " +
                    "Always use 'await' to consume asynchronous operations.",
        helpLinkUri: "https://docs.microsoft.com/orleans/grains/grain-threading");

    public override ImmutableArray<DiagnosticDescriptor> SupportedDiagnostics => ImmutableArray.Create(Rule);

    public override void Initialize(AnalysisContext context)
    {
        context.ConfigureGeneratedCodeAnalysis(GeneratedCodeAnalysisFlags.None);
        context.EnableConcurrentExecution();

        context.RegisterSyntaxNodeAction(AnalyzeMemberAccess, SyntaxKind.SimpleMemberAccessExpression);
        context.RegisterSyntaxNodeAction(AnalyzeInvocation, SyntaxKind.InvocationExpression);
    }

    private static void AnalyzeMemberAccess(SyntaxNodeAnalysisContext context)
    {
        var memberAccess = (MemberAccessExpressionSyntax)context.Node;

        // Check for .Result property access
        if (memberAccess.Name.Identifier.Text != "Result")
        {
            return;
        }

        // Verify it's accessing Task.Result or Task<T>.Result
        var symbolInfo = context.SemanticModel.GetSymbolInfo(memberAccess);
        if (symbolInfo.Symbol is not IPropertySymbol propertySymbol)
        {
            return;
        }

        if (!IsTaskResultProperty(propertySymbol))
        {
            return;
        }

        // Check if we're in an async grain method
        if (!IsInAsyncGrainMethod(context))
        {
            return;
        }

        // Report diagnostic
        var diagnostic = Diagnostic.Create(
            Rule,
            memberAccess.GetLocation(),
            ".Result");

        context.ReportDiagnostic(diagnostic);
    }

    private static void AnalyzeInvocation(SyntaxNodeAnalysisContext context)
    {
        var invocation = (InvocationExpressionSyntax)context.Node;

        var symbolInfo = context.SemanticModel.GetSymbolInfo(invocation);
        if (symbolInfo.Symbol is not IMethodSymbol methodSymbol)
        {
            return;
        }

        // Check for .Wait() or .GetAwaiter().GetResult()
        var methodName = methodSymbol.Name;
        if (methodName != "Wait" && methodName != "GetResult")
        {
            return;
        }

        // Verify it's a blocking Task method
        if (!IsTaskBlockingMethod(methodSymbol))
        {
            return;
        }

        // Check if we're in an async grain method
        if (!IsInAsyncGrainMethod(context))
        {
            return;
        }

        // Report diagnostic
        var blockingCall = methodName == "Wait" ? ".Wait()" : ".GetAwaiter().GetResult()";
        var diagnostic = Diagnostic.Create(
            Rule,
            invocation.GetLocation(),
            blockingCall);

        context.ReportDiagnostic(diagnostic);
    }

    private static bool IsTaskResultProperty(IPropertySymbol propertySymbol)
    {
        var containingType = propertySymbol.ContainingType;
        if (containingType == null)
        {
            return false;
        }

        var fullName = containingType.ToDisplayString();
        return fullName.StartsWith("System.Threading.Tasks.Task<", StringComparison.Ordinal);
    }

    private static bool IsTaskBlockingMethod(IMethodSymbol methodSymbol)
    {
        var containingType = methodSymbol.ContainingType;
        if (containingType == null)
        {
            return false;
        }

        var fullName = containingType.ToDisplayString();

        if (methodSymbol.Name == "Wait")
        {
            return fullName == "System.Threading.Tasks.Task" ||
                   fullName.StartsWith("System.Threading.Tasks.Task<", StringComparison.Ordinal);
        }

        if (methodSymbol.Name == "GetResult")
        {
            // Check for TaskAwaiter or TaskAwaiter<T>
            return fullName == "System.Runtime.CompilerServices.TaskAwaiter" ||
                   fullName.StartsWith("System.Runtime.CompilerServices.TaskAwaiter<", StringComparison.Ordinal);
        }

        return false;
    }

    private static bool IsInAsyncGrainMethod(SyntaxNodeAnalysisContext context)
    {
        // Check if we're in an async method
        if (context.ContainingSymbol is not IMethodSymbol methodSymbol || !methodSymbol.IsAsync)
        {
            return false;
        }

        // Check if the containing type is an Orleans grain
        var containingType = methodSymbol.ContainingType;
        if (containingType == null)
        {
            return false;
        }

        return IsOrleansGrain(containingType, context.Compilation);
    }

    private static bool IsOrleansGrain(INamedTypeSymbol typeSymbol, Compilation compilation)
    {
        // Check if the type inherits from Orleans.Grain or Orleans.Grain<T>
        var grainType = compilation.GetTypeByMetadataName("Orleans.Grain");
        var grainStateType = compilation.GetTypeByMetadataName("Orleans.Grain`1");

        if (grainType == null && grainStateType == null)
        {
            // Orleans assemblies not referenced - skip analysis
            return false;
        }

        // Walk up the inheritance hierarchy
        var current = typeSymbol.BaseType;
        while (current != null)
        {
            var originalDefinition = current.OriginalDefinition;

            if (grainType != null && SymbolEqualityComparer.Default.Equals(originalDefinition, grainType))
            {
                return true;
            }

            if (grainStateType != null && SymbolEqualityComparer.Default.Equals(originalDefinition, grainStateType))
            {
                return true;
            }

            current = current.BaseType;
        }

        // Check for [GrainAttribute]
        var hasGrainAttribute = typeSymbol.GetAttributes().Any(attr =>
        {
            var attrClass = attr.AttributeClass;
            return attrClass != null &&
                   attrClass.Name == "GrainAttribute" &&
                   attrClass.ContainingNamespace?.ToString() == "Orleans";
        });

        if (hasGrainAttribute)
        {
            return true;
        }

        // Check if implements IGrain
        var implementsIGrain = typeSymbol.AllInterfaces.Any(i =>
        {
            var ns = i.ContainingNamespace?.ToString();
            return ns == "Orleans" && i.Name.StartsWith("IGrain", StringComparison.Ordinal);
        });

        return implementsIGrain;
    }
}

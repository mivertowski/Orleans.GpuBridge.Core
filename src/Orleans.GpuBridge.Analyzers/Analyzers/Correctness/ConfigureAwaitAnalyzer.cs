using System;
using System.Collections.Immutable;
using System.Linq;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Microsoft.CodeAnalysis.Diagnostics;

namespace Orleans.GpuBridge.Analyzers.Analyzers.Correctness;

/// <summary>
/// Analyzer that detects ConfigureAwait(false) usage in Orleans grain contexts.
/// ConfigureAwait(false) breaks Orleans' single-threaded execution guarantees and can cause deadlocks.
/// </summary>
[DiagnosticAnalyzer(LanguageNames.CSharp)]
public sealed class ConfigureAwaitAnalyzer : DiagnosticAnalyzer
{
    private static readonly DiagnosticDescriptor Rule = new(
        id: DiagnosticIds.ConfigureAwaitInGrain,
        title: "ConfigureAwait(false) must not be used in Orleans grain contexts",
        messageFormat: "ConfigureAwait(false) breaks Orleans grain execution guarantees and can cause deadlocks. Remove ConfigureAwait(false) in grain method '{0}'",
        category: DiagnosticCategories.Correctness,
        defaultSeverity: DiagnosticSeverity.Error,
        isEnabledByDefault: true,
        description: "Orleans grains have single-threaded execution semantics. Using ConfigureAwait(false) " +
                    "allows continuations to run on different threads, which breaks Orleans' guarantees and " +
                    "can cause race conditions, deadlocks, and state corruption. Always await without " +
                    "ConfigureAwait(false) in grain methods.",
        helpLinkUri: "https://docs.microsoft.com/orleans/grains/grain-threading");

    public override ImmutableArray<DiagnosticDescriptor> SupportedDiagnostics => ImmutableArray.Create(Rule);

    public override void Initialize(AnalysisContext context)
    {
        context.ConfigureGeneratedCodeAnalysis(GeneratedCodeAnalysisFlags.None);
        context.EnableConcurrentExecution();

        context.RegisterSyntaxNodeAction(AnalyzeInvocation, SyntaxKind.InvocationExpression);
    }

    private static void AnalyzeInvocation(SyntaxNodeAnalysisContext context)
    {
        var invocation = (InvocationExpressionSyntax)context.Node;

        // Check if this is a ConfigureAwait invocation
        if (!IsConfigureAwaitInvocation(invocation, context.SemanticModel))
        {
            return;
        }

        // Check if the argument is 'false'
        if (!HasFalseArgument(invocation))
        {
            return;
        }

        // Check if we're in an Orleans grain context
        var containingType = context.ContainingSymbol?.ContainingType;
        if (containingType == null || !IsOrleansGrain(containingType, context.Compilation))
        {
            return;
        }

        // Report diagnostic
        var methodName = context.ContainingSymbol?.Name ?? "unknown";
        var diagnostic = Diagnostic.Create(
            Rule,
            invocation.GetLocation(),
            methodName);

        context.ReportDiagnostic(diagnostic);
    }

    private static bool IsConfigureAwaitInvocation(InvocationExpressionSyntax invocation, SemanticModel semanticModel)
    {
        if (invocation.Expression is not MemberAccessExpressionSyntax memberAccess)
        {
            return false;
        }

        if (memberAccess.Name.Identifier.Text != "ConfigureAwait")
        {
            return false;
        }

        // Verify it's actually the Task.ConfigureAwait method
        var symbolInfo = semanticModel.GetSymbolInfo(invocation);
        if (symbolInfo.Symbol is not IMethodSymbol methodSymbol)
        {
            return false;
        }

        // Check if it's defined on Task or Task<T>
        var containingType = methodSymbol.ContainingType;
        if (containingType == null)
        {
            return false;
        }

        var fullName = containingType.ToDisplayString();
        return fullName == "System.Threading.Tasks.Task" ||
               fullName.StartsWith("System.Threading.Tasks.Task<", StringComparison.Ordinal);
    }

    private static bool HasFalseArgument(InvocationExpressionSyntax invocation)
    {
        if (invocation.ArgumentList.Arguments.Count != 1)
        {
            return false;
        }

        var argument = invocation.ArgumentList.Arguments[0];
        if (argument.Expression is LiteralExpressionSyntax literal)
        {
            return literal.IsKind(SyntaxKind.FalseLiteralExpression);
        }

        return false;
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

        // Check if implements IGrain or IGrainWithGuidKey, etc.
        var implementsIGrain = typeSymbol.AllInterfaces.Any(i =>
        {
            var ns = i.ContainingNamespace?.ToString();
            return ns == "Orleans" && i.Name.StartsWith("IGrain", StringComparison.Ordinal);
        });

        return implementsIGrain;
    }
}

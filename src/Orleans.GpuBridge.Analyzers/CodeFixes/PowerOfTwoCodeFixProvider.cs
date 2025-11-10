using System;
using System.Collections.Immutable;
using System.Composition;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CodeActions;
using Microsoft.CodeAnalysis.CodeFixes;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;

namespace Orleans.GpuBridge.Analyzers.CodeFixes;

/// <summary>
/// Code fix provider that suggests nearest power-of-2 values for queue capacity and message size.
/// </summary>
[ExportCodeFixProvider(LanguageNames.CSharp, Name = nameof(PowerOfTwoCodeFixProvider))]
[Shared]
public sealed class PowerOfTwoCodeFixProvider : CodeFixProvider
{
    public override ImmutableArray<string> FixableDiagnosticIds =>
        ImmutableArray.Create(
            DiagnosticIds.QueueCapacityNotPowerOfTwo,
            DiagnosticIds.MessageSizeNotPowerOfTwo);

    public override FixAllProvider GetFixAllProvider() => WellKnownFixAllProviders.BatchFixer;

    public override async Task RegisterCodeFixesAsync(CodeFixContext context)
    {
        var root = await context.Document.GetSyntaxRootAsync(context.CancellationToken).ConfigureAwait(false);
        if (root == null)
        {
            return;
        }

        var diagnostic = context.Diagnostics.First();
        var diagnosticSpan = diagnostic.Location.SourceSpan;

        // Find the literal expression identified by the diagnostic
        var literalExpression = root.FindToken(diagnosticSpan.Start)
            .Parent?
            .AncestorsAndSelf()
            .OfType<LiteralExpressionSyntax>()
            .FirstOrDefault();

        if (literalExpression == null)
        {
            return;
        }

        var semanticModel = await context.Document.GetSemanticModelAsync(context.CancellationToken).ConfigureAwait(false);
        if (semanticModel == null)
        {
            return;
        }

        // Get the current value
        var constantValue = semanticModel.GetConstantValue(literalExpression);
        if (!constantValue.HasValue || constantValue.Value is not int currentValue)
        {
            return;
        }

        // Calculate nearest powers of 2
        var (lower, upper) = GetNearestPowersOfTwo(currentValue);

        // Register code fix for rounding down
        context.RegisterCodeFix(
            CodeAction.Create(
                title: $"Change to {lower} (round down to power of 2)",
                createChangedDocument: c => ReplaceValueAsync(context.Document, literalExpression, lower, c),
                equivalenceKey: $"{nameof(PowerOfTwoCodeFixProvider)}_RoundDown"),
            diagnostic);

        // Register code fix for rounding up
        context.RegisterCodeFix(
            CodeAction.Create(
                title: $"Change to {upper} (round up to power of 2)",
                createChangedDocument: c => ReplaceValueAsync(context.Document, literalExpression, upper, c),
                equivalenceKey: $"{nameof(PowerOfTwoCodeFixProvider)}_RoundUp"),
            diagnostic);
    }

    private static async Task<Document> ReplaceValueAsync(
        Document document,
        LiteralExpressionSyntax literalExpression,
        int newValue,
        CancellationToken cancellationToken)
    {
        var root = await document.GetSyntaxRootAsync(cancellationToken).ConfigureAwait(false);
        if (root == null)
        {
            return document;
        }

        // Create new literal with the corrected value
        var newLiteral = SyntaxFactory.LiteralExpression(
            SyntaxKind.NumericLiteralExpression,
            SyntaxFactory.Literal(newValue));

        // Preserve trivia
        newLiteral = newLiteral
            .WithLeadingTrivia(literalExpression.GetLeadingTrivia())
            .WithTrailingTrivia(literalExpression.GetTrailingTrivia());

        var newRoot = root.ReplaceNode(literalExpression, newLiteral);

        return document.WithSyntaxRoot(newRoot);
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

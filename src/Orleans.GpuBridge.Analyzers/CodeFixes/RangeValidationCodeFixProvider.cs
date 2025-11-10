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
/// Code fix provider that clamps values to valid ranges for GPU-native actor configuration.
/// </summary>
[ExportCodeFixProvider(LanguageNames.CSharp, Name = nameof(RangeValidationCodeFixProvider))]
[Shared]
public sealed class RangeValidationCodeFixProvider : CodeFixProvider
{
    private const int MinQueueCapacity = 256;
    private const int MaxQueueCapacity = 1_048_576;
    private const int MinMessageSize = 256;
    private const int MaxMessageSize = 4096;

    public override ImmutableArray<string> FixableDiagnosticIds =>
        ImmutableArray.Create(
            DiagnosticIds.QueueCapacityOutOfRange,
            DiagnosticIds.MessageSizeOutOfRange);

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

        // Determine the appropriate fix based on diagnostic ID
        if (diagnostic.Id == DiagnosticIds.QueueCapacityOutOfRange)
        {
            RegisterQueueCapacityFixes(context, diagnostic, literalExpression, currentValue);
        }
        else if (diagnostic.Id == DiagnosticIds.MessageSizeOutOfRange)
        {
            RegisterMessageSizeFixes(context, diagnostic, literalExpression, currentValue);
        }
    }

    private void RegisterQueueCapacityFixes(
        CodeFixContext context,
        Diagnostic diagnostic,
        LiteralExpressionSyntax literalExpression,
        int currentValue)
    {
        if (currentValue < MinQueueCapacity)
        {
            // Suggest minimum value
            context.RegisterCodeFix(
                CodeAction.Create(
                    title: $"Change to {MinQueueCapacity} (minimum queue capacity)",
                    createChangedDocument: c => ReplaceValueAsync(context.Document, literalExpression, MinQueueCapacity, c),
                    equivalenceKey: $"{nameof(RangeValidationCodeFixProvider)}_MinQueueCapacity"),
                diagnostic);

            // Suggest nearest power of 2 above minimum if different
            var nearestPow2 = GetNearestPowerOfTwoAbove(MinQueueCapacity);
            if (nearestPow2 > MinQueueCapacity && nearestPow2 <= MaxQueueCapacity)
            {
                context.RegisterCodeFix(
                    CodeAction.Create(
                        title: $"Change to {nearestPow2} (recommended starting capacity)",
                        createChangedDocument: c => ReplaceValueAsync(context.Document, literalExpression, nearestPow2, c),
                        equivalenceKey: $"{nameof(RangeValidationCodeFixProvider)}_RecommendedQueueCapacity"),
                    diagnostic);
            }
        }
        else if (currentValue > MaxQueueCapacity)
        {
            // Suggest maximum value
            context.RegisterCodeFix(
                CodeAction.Create(
                    title: $"Change to {MaxQueueCapacity} (maximum queue capacity)",
                    createChangedDocument: c => ReplaceValueAsync(context.Document, literalExpression, MaxQueueCapacity, c),
                    equivalenceKey: $"{nameof(RangeValidationCodeFixProvider)}_MaxQueueCapacity"),
                diagnostic);

            // Suggest a reasonable production value
            const int recommendedMax = 65536;
            context.RegisterCodeFix(
                CodeAction.Create(
                    title: $"Change to {recommendedMax} (recommended for most workloads)",
                    createChangedDocument: c => ReplaceValueAsync(context.Document, literalExpression, recommendedMax, c),
                    equivalenceKey: $"{nameof(RangeValidationCodeFixProvider)}_RecommendedMaxQueueCapacity"),
                diagnostic);
        }
    }

    private void RegisterMessageSizeFixes(
        CodeFixContext context,
        Diagnostic diagnostic,
        LiteralExpressionSyntax literalExpression,
        int currentValue)
    {
        if (currentValue < MinMessageSize)
        {
            // Suggest minimum value
            context.RegisterCodeFix(
                CodeAction.Create(
                    title: $"Change to {MinMessageSize} (minimum message size)",
                    createChangedDocument: c => ReplaceValueAsync(context.Document, literalExpression, MinMessageSize, c),
                    equivalenceKey: $"{nameof(RangeValidationCodeFixProvider)}_MinMessageSize"),
                diagnostic);

            // Suggest next power of 2 if different
            var nextPow2 = GetNearestPowerOfTwoAbove(MinMessageSize);
            if (nextPow2 > MinMessageSize && nextPow2 <= MaxMessageSize)
            {
                context.RegisterCodeFix(
                    CodeAction.Create(
                        title: $"Change to {nextPow2} (recommended message size)",
                        createChangedDocument: c => ReplaceValueAsync(context.Document, literalExpression, nextPow2, c),
                        equivalenceKey: $"{nameof(RangeValidationCodeFixProvider)}_RecommendedMessageSize"),
                    diagnostic);
            }
        }
        else if (currentValue > MaxMessageSize)
        {
            // Suggest maximum value
            context.RegisterCodeFix(
                CodeAction.Create(
                    title: $"Change to {MaxMessageSize} (maximum message size)",
                    createChangedDocument: c => ReplaceValueAsync(context.Document, literalExpression, MaxMessageSize, c),
                    equivalenceKey: $"{nameof(RangeValidationCodeFixProvider)}_MaxMessageSize"),
                diagnostic);

            // Suggest a reasonable default
            const int recommendedSize = 1024;
            context.RegisterCodeFix(
                CodeAction.Create(
                    title: $"Change to {recommendedSize} (recommended for most messages)",
                    createChangedDocument: c => ReplaceValueAsync(context.Document, literalExpression, recommendedSize, c),
                    equivalenceKey: $"{nameof(RangeValidationCodeFixProvider)}_RecommendedDefaultMessageSize"),
                diagnostic);
        }
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

    private static int GetNearestPowerOfTwoAbove(int value)
    {
        if (value <= 1)
        {
            return 2;
        }

        int log = (int)Math.Ceiling(Math.Log(value, 2));
        return 1 << log;
    }
}

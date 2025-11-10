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
/// Code fix provider that removes ConfigureAwait(false) calls from Orleans grain methods.
/// </summary>
[ExportCodeFixProvider(LanguageNames.CSharp, Name = nameof(ConfigureAwaitCodeFixProvider))]
[Shared]
public sealed class ConfigureAwaitCodeFixProvider : CodeFixProvider
{
    public override ImmutableArray<string> FixableDiagnosticIds =>
        ImmutableArray.Create(DiagnosticIds.ConfigureAwaitInGrain);

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

        // Find the invocation expression identified by the diagnostic
        var invocation = root.FindToken(diagnosticSpan.Start)
            .Parent?
            .AncestorsAndSelf()
            .OfType<InvocationExpressionSyntax>()
            .FirstOrDefault();

        if (invocation == null)
        {
            return;
        }

        // Register a code action that will remove ConfigureAwait(false)
        context.RegisterCodeFix(
            CodeAction.Create(
                title: "Remove ConfigureAwait(false)",
                createChangedDocument: c => RemoveConfigureAwaitAsync(context.Document, invocation, c),
                equivalenceKey: nameof(ConfigureAwaitCodeFixProvider)),
            diagnostic);
    }

    private static async Task<Document> RemoveConfigureAwaitAsync(
        Document document,
        InvocationExpressionSyntax invocation,
        CancellationToken cancellationToken)
    {
        var root = await document.GetSyntaxRootAsync(cancellationToken).ConfigureAwait(false);
        if (root == null)
        {
            return document;
        }

        // The invocation is: someTask.ConfigureAwait(false)
        // We want to replace it with: someTask
        if (invocation.Expression is not MemberAccessExpressionSyntax memberAccess)
        {
            return document;
        }

        // Get the expression before .ConfigureAwait(false)
        var taskExpression = memberAccess.Expression;

        // Replace the entire invocation with just the task expression
        var newRoot = root.ReplaceNode(invocation, taskExpression);

        return document.WithSyntaxRoot(newRoot);
    }
}

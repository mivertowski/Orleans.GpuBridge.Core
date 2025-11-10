using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CodeFixes;
using Microsoft.CodeAnalysis.CSharp.Testing;
using Microsoft.CodeAnalysis.Diagnostics;
using Microsoft.CodeAnalysis.Testing;
using Microsoft.CodeAnalysis.Testing.Verifiers;

namespace Orleans.GpuBridge.Analyzers.Tests.Verifiers;

/// <summary>
/// Helper class for verifying code fix providers.
/// </summary>
public static class CSharpCodeFixVerifier<TAnalyzer, TCodeFix>
    where TAnalyzer : DiagnosticAnalyzer, new()
    where TCodeFix : CodeFixProvider, new()
{
    /// <summary>
    /// Verifies that the code fix correctly transforms the source.
    /// </summary>
    public static async Task VerifyCodeFixAsync(
        string source,
        string fixedSource,
        params DiagnosticResult[] expected)
    {
        var test = new Test
        {
            TestCode = source,
            FixedCode = fixedSource,
        };

        test.ExpectedDiagnostics.AddRange(expected);
        await test.RunAsync();
    }

    /// <summary>
    /// Verifies that the code fix correctly transforms the source (uses first expected diagnostic automatically).
    /// </summary>
    public static async Task VerifyCodeFixAsync(
        string source,
        DiagnosticResult expected,
        string fixedSource)
    {
        await VerifyCodeFixAsync(source, fixedSource, expected);
    }

    private class Test : CSharpCodeFixTest<TAnalyzer, TCodeFix, XUnitVerifier>
    {
        public Test()
        {
            // Add Orleans reference
            ReferenceAssemblies = ReferenceAssemblies.Net.Net80
                .AddPackages([
                    new PackageIdentity("Microsoft.Orleans.Core", "8.2.0"),
                    new PackageIdentity("Microsoft.Orleans.Runtime", "8.2.0"),
                ]);

            // Add Orleans.GpuBridge reference
            TestState.AdditionalReferences.Add(MetadataReference.CreateFromFile(
                typeof(object).Assembly.Location));
        }

        protected override CompilationOptions CreateCompilationOptions()
        {
            var options = base.CreateCompilationOptions();
            return options.WithSpecificDiagnosticOptions(
                options.SpecificDiagnosticOptions.SetItems(
                    CSharpVerifierHelper.NullableWarnings));
        }
    }

    private static class CSharpVerifierHelper
    {
        internal static ImmutableDictionary<string, ReportDiagnostic> NullableWarnings { get; } =
            GetNullableWarningsFromCompiler();

        private static ImmutableDictionary<string, ReportDiagnostic> GetNullableWarningsFromCompiler()
        {
            var args = new[] { "/warnaserror:nullable" };
            var commandLineArguments = Microsoft.CodeAnalysis.CSharp.CSharpCommandLineParser.Default.Parse(
                args,
                baseDirectory: Environment.CurrentDirectory,
                sdkDirectory: Environment.CurrentDirectory);

            return commandLineArguments.CompilationOptions.SpecificDiagnosticOptions;
        }
    }
}

using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp.Testing;
using Microsoft.CodeAnalysis.Diagnostics;
using Microsoft.CodeAnalysis.Testing;
using Microsoft.CodeAnalysis.Testing.Verifiers;

namespace Orleans.GpuBridge.Analyzers.Tests.Verifiers;

/// <summary>
/// Helper class for verifying analyzer diagnostics.
/// </summary>
public static class CSharpAnalyzerVerifier<TAnalyzer>
    where TAnalyzer : DiagnosticAnalyzer, new()
{
    /// <summary>
    /// Verifies that the given source produces the expected diagnostics.
    /// </summary>
    public static async Task VerifyAnalyzerAsync(string source, params DiagnosticResult[] expected)
    {
        var test = new Test
        {
            TestCode = source,
        };

        test.ExpectedDiagnostics.AddRange(expected);
        await test.RunAsync();
    }

    private class Test : CSharpAnalyzerTest<TAnalyzer, XUnitVerifier>
    {
        public Test()
        {
            // Add Orleans reference
            ReferenceAssemblies = ReferenceAssemblies.Net.Net80
                .AddPackages([
                    new PackageIdentity("Microsoft.Orleans.Core", "8.2.0"),
                    new PackageIdentity("Microsoft.Orleans.Runtime", "8.2.0"),
                ]);

            // Add Orleans.GpuBridge reference (for testing configuration analyzers)
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

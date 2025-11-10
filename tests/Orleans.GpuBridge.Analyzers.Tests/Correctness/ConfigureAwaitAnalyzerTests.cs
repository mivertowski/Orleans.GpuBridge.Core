using Microsoft.CodeAnalysis.Testing;
using Orleans.GpuBridge.Analyzers.Analyzers.Correctness;
using Orleans.GpuBridge.Analyzers.CodeFixes;
using Orleans.GpuBridge.Analyzers.Tests.Verifiers;
using Xunit;

namespace Orleans.GpuBridge.Analyzers.Tests.Correctness;

/// <summary>
/// Tests for ConfigureAwaitAnalyzer and ConfigureAwaitCodeFixProvider.
/// </summary>
public sealed class ConfigureAwaitAnalyzerTests
{
    [Fact]
    public async Task ConfigureAwait_InGrainMethod_ReportsDiagnostic()
    {
        var test = """
            using System.Threading.Tasks;
            using Orleans;

            public class MyGrain : Grain
            {
                public async Task MyMethod()
                {
                    await Task.Delay(100).{|OGBA001:ConfigureAwait(false)|};
                }
            }
            """;

        var expected = DiagnosticResult
            .CompilerError("OGBA001")
            .WithSpan(8, 31, 8, 54)
            .WithArguments("MyMethod");

        await CSharpAnalyzerVerifier<ConfigureAwaitAnalyzer>
            .VerifyAnalyzerAsync(test, expected);
    }

    [Fact]
    public async Task ConfigureAwait_InGrainStateMethod_ReportsDiagnostic()
    {
        var test = """
            using System.Threading.Tasks;
            using Orleans;

            public class MyState { }

            public class MyGrain : Grain<MyState>
            {
                public async Task MyMethod()
                {
                    await Task.Delay(100).{|OGBA001:ConfigureAwait(false)|};
                }
            }
            """;

        var expected = DiagnosticResult
            .CompilerError("OGBA001")
            .WithSpan(10, 31, 10, 54)
            .WithArguments("MyMethod");

        await CSharpAnalyzerVerifier<ConfigureAwaitAnalyzer>
            .VerifyAnalyzerAsync(test, expected);
    }

    [Fact]
    public async Task ConfigureAwait_OutsideGrain_NoDiagnostic()
    {
        var test = """
            using System.Threading.Tasks;

            public class NotAGrain
            {
                public async Task MyMethod()
                {
                    await Task.Delay(100).ConfigureAwait(false); // OK - not a grain
                }
            }
            """;

        await CSharpAnalyzerVerifier<ConfigureAwaitAnalyzer>
            .VerifyAnalyzerAsync(test);
    }

    [Fact]
    public async Task ConfigureAwait_True_NoDiagnostic()
    {
        var test = """
            using System.Threading.Tasks;
            using Orleans;

            public class MyGrain : Grain
            {
                public async Task MyMethod()
                {
                    await Task.Delay(100).ConfigureAwait(true); // OK - ConfigureAwait(true) is safe
                }
            }
            """;

        await CSharpAnalyzerVerifier<ConfigureAwaitAnalyzer>
            .VerifyAnalyzerAsync(test);
    }

    [Fact]
    public async Task ConfigureAwait_CodeFix_RemovesConfigureAwait()
    {
        var test = """
            using System.Threading.Tasks;
            using Orleans;

            public class MyGrain : Grain
            {
                public async Task MyMethod()
                {
                    await Task.Delay(100).{|OGBA001:ConfigureAwait(false)|};
                }
            }
            """;

        var fixedCode = """
            using System.Threading.Tasks;
            using Orleans;

            public class MyGrain : Grain
            {
                public async Task MyMethod()
                {
                    await Task.Delay(100);
                }
            }
            """;

        var expected = DiagnosticResult
            .CompilerError("OGBA001")
            .WithSpan(8, 31, 8, 54)
            .WithArguments("MyMethod");

        await CSharpCodeFixVerifier<ConfigureAwaitAnalyzer, ConfigureAwaitCodeFixProvider>
            .VerifyCodeFixAsync(test, expected, fixedCode);
    }

    [Fact]
    public async Task ConfigureAwait_MultipleOccurrences_ReportsAll()
    {
        var test = """
            using System.Threading.Tasks;
            using Orleans;

            public class MyGrain : Grain
            {
                public async Task MyMethod()
                {
                    await Task.Delay(100).{|OGBA001:ConfigureAwait(false)|};
                    await Task.Delay(200).{|OGBA001:ConfigureAwait(false)|};
                    await Task.Delay(300).{|OGBA001:ConfigureAwait(false)|};
                }
            }
            """;

        var expected1 = DiagnosticResult
            .CompilerError("OGBA001")
            .WithSpan(8, 31, 8, 54)
            .WithArguments("MyMethod");

        var expected2 = DiagnosticResult
            .CompilerError("OGBA001")
            .WithSpan(9, 31, 9, 54)
            .WithArguments("MyMethod");

        var expected3 = DiagnosticResult
            .CompilerError("OGBA001")
            .WithSpan(10, 31, 10, 54)
            .WithArguments("MyMethod");

        await CSharpAnalyzerVerifier<ConfigureAwaitAnalyzer>
            .VerifyAnalyzerAsync(test, expected1, expected2, expected3);
    }

    [Fact]
    public async Task ConfigureAwait_InGrainTask_ReportsDiagnostic()
    {
        var test = """
            using System.Threading.Tasks;
            using Orleans;

            public class MyGrain : Grain
            {
                public async Task<int> MyMethod()
                {
                    var result = await GetValueAsync().{|OGBA001:ConfigureAwait(false)|};
                    return result;
                }

                private Task<int> GetValueAsync() => Task.FromResult(42);
            }
            """;

        var expected = DiagnosticResult
            .CompilerError("OGBA001")
            .WithSpan(8, 48, 8, 71)
            .WithArguments("MyMethod");

        await CSharpAnalyzerVerifier<ConfigureAwaitAnalyzer>
            .VerifyAnalyzerAsync(test, expected);
    }
}

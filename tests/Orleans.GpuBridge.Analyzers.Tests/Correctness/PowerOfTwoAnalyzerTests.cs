using Microsoft.CodeAnalysis.Testing;
using Orleans.GpuBridge.Analyzers.Analyzers.Correctness;
using Orleans.GpuBridge.Analyzers.CodeFixes;
using Orleans.GpuBridge.Analyzers.Tests.Verifiers;
using Xunit;

namespace Orleans.GpuBridge.Analyzers.Tests.Correctness;

/// <summary>
/// Tests for PowerOfTwoAnalyzer and PowerOfTwoCodeFixProvider.
/// </summary>
public sealed class PowerOfTwoAnalyzerTests
{
    [Fact]
    public async Task QueueCapacity_NotPowerOfTwo_ReportsDiagnostic()
    {
        var test = """
            namespace Orleans.GpuBridge.Grains.GpuNative
            {
                public class GpuNativeActorConfiguration
                {
                    public int MessageQueueCapacity { get; set; }
                    public int MessageSize { get; set; }
                }

                public class TestClass
                {
                    public void TestMethod()
                    {
                        var config = new GpuNativeActorConfiguration
                        {
                            MessageQueueCapacity = {|OGBA002:1000|}
                        };
                    }
                }
            }
            """;

        var expected = DiagnosticResult
            .CompilerError("OGBA002")
            .WithSpan(15, 44, 15, 48)
            .WithArguments(1000, 512, 1024);

        await CSharpAnalyzerVerifier<PowerOfTwoAnalyzer>
            .VerifyAnalyzerAsync(test, expected);
    }

    [Fact]
    public async Task MessageSize_NotPowerOfTwo_ReportsDiagnostic()
    {
        var test = """
            namespace Orleans.GpuBridge.Grains.GpuNative
            {
                public class GpuNativeActorConfiguration
                {
                    public int MessageQueueCapacity { get; set; }
                    public int MessageSize { get; set; }
                }

                public class TestClass
                {
                    public void TestMethod()
                    {
                        var config = new GpuNativeActorConfiguration
                        {
                            MessageSize = {|OGBA003:300|}
                        };
                    }
                }
            }
            """;

        var expected = DiagnosticResult
            .CompilerError("OGBA003")
            .WithSpan(15, 31, 15, 34)
            .WithArguments(300, 256, 512);

        await CSharpAnalyzerVerifier<PowerOfTwoAnalyzer>
            .VerifyAnalyzerAsync(test, expected);
    }

    [Fact]
    public async Task QueueCapacity_PowerOfTwo_NoDiagnostic()
    {
        var test = """
            namespace Orleans.GpuBridge.Grains.GpuNative
            {
                public class GpuNativeActorConfiguration
                {
                    public int MessageQueueCapacity { get; set; }
                    public int MessageSize { get; set; }
                }

                public class TestClass
                {
                    public void TestMethod()
                    {
                        var config = new GpuNativeActorConfiguration
                        {
                            MessageQueueCapacity = 1024 // OK - power of 2
                        };
                    }
                }
            }
            """;

        await CSharpAnalyzerVerifier<PowerOfTwoAnalyzer>
            .VerifyAnalyzerAsync(test);
    }

    [Fact]
    public async Task QueueCapacity_CodeFix_OffersRoundDown()
    {
        var test = """
            namespace Orleans.GpuBridge.Grains.GpuNative
            {
                public class GpuNativeActorConfiguration
                {
                    public int MessageQueueCapacity { get; set; }
                    public int MessageSize { get; set; }
                }

                public class TestClass
                {
                    public void TestMethod()
                    {
                        var config = new GpuNativeActorConfiguration
                        {
                            MessageQueueCapacity = {|OGBA002:1000|}
                        };
                    }
                }
            }
            """;

        var fixedCode = """
            namespace Orleans.GpuBridge.Grains.GpuNative
            {
                public class GpuNativeActorConfiguration
                {
                    public int MessageQueueCapacity { get; set; }
                    public int MessageSize { get; set; }
                }

                public class TestClass
                {
                    public void TestMethod()
                    {
                        var config = new GpuNativeActorConfiguration
                        {
                            MessageQueueCapacity = 512
                        };
                    }
                }
            }
            """;

        var expected = DiagnosticResult
            .CompilerError("OGBA002")
            .WithSpan(15, 44, 15, 48)
            .WithArguments(1000, 512, 1024);

        await CSharpCodeFixVerifier<PowerOfTwoAnalyzer, PowerOfTwoCodeFixProvider>
            .VerifyCodeFixAsync(test, expected, fixedCode);
    }

    [Fact]
    public async Task MessageSize_CodeFix_OffersRoundUp()
    {
        var test = """
            namespace Orleans.GpuBridge.Grains.GpuNative
            {
                public class GpuNativeActorConfiguration
                {
                    public int MessageQueueCapacity { get; set; }
                    public int MessageSize { get; set; }
                }

                public class TestClass
                {
                    public void TestMethod()
                    {
                        var config = new GpuNativeActorConfiguration
                        {
                            MessageSize = {|OGBA003:300|}
                        };
                    }
                }
            }
            """;

        var fixedCode = """
            namespace Orleans.GpuBridge.Grains.GpuNative
            {
                public class GpuNativeActorConfiguration
                {
                    public int MessageQueueCapacity { get; set; }
                    public int MessageSize { get; set; }
                }

                public class TestClass
                {
                    public void TestMethod()
                    {
                        var config = new GpuNativeActorConfiguration
                        {
                            MessageSize = 512
                        };
                    }
                }
            }
            """;

        var expected = DiagnosticResult
            .CompilerError("OGBA003")
            .WithSpan(15, 31, 15, 34)
            .WithArguments(300, 256, 512);

        // Use index 1 for the second code fix (round up)
        await CSharpCodeFixVerifier<PowerOfTwoAnalyzer, PowerOfTwoCodeFixProvider>
            .VerifyCodeFixAsync(test, expected, fixedCode);
    }

    [Fact]
    public async Task BothProperties_NotPowerOfTwo_ReportsBothDiagnostics()
    {
        var test = """
            namespace Orleans.GpuBridge.Grains.GpuNative
            {
                public class GpuNativeActorConfiguration
                {
                    public int MessageQueueCapacity { get; set; }
                    public int MessageSize { get; set; }
                }

                public class TestClass
                {
                    public void TestMethod()
                    {
                        var config = new GpuNativeActorConfiguration
                        {
                            MessageQueueCapacity = {|OGBA002:1000|},
                            MessageSize = {|OGBA003:300|}
                        };
                    }
                }
            }
            """;

        var expected1 = DiagnosticResult
            .CompilerError("OGBA002")
            .WithSpan(15, 44, 15, 48)
            .WithArguments(1000, 512, 1024);

        var expected2 = DiagnosticResult
            .CompilerError("OGBA003")
            .WithSpan(16, 31, 16, 34)
            .WithArguments(300, 256, 512);

        await CSharpAnalyzerVerifier<PowerOfTwoAnalyzer>
            .VerifyAnalyzerAsync(test, expected1, expected2);
    }
}

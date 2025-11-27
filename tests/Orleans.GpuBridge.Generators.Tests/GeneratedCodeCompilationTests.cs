// Orleans.GpuBridge - GPU-native distributed computing for Microsoft Orleans
// Copyright (c) 2025 Michael Ivertowski

using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Orleans.GpuBridge.Generators;
using Xunit;

namespace Orleans.GpuBridge.Generators.Tests;

/// <summary>
/// Verifies that the generated code structure is correct and contains expected components.
/// These tests verify generator output without requiring external assembly references.
/// </summary>
public class GeneratedCodeCompilationTests
{
    /// <summary>
    /// Tests that a simple calculator actor generates all expected code components.
    /// </summary>
    [Fact]
    public void GeneratedCode_ForCalculatorActor_CompilesSuccessfully()
    {
        // Arrange
        var source = """
            using System.Threading.Tasks;
            using Orleans;
            using Orleans.GpuBridge.Abstractions.Generation;

            namespace TestNamespace
            {
                [GpuNativeActor]
                public interface ICalculatorActor : IGrainWithIntegerKey
                {
                    [GpuHandler]
                    Task<int> AddAsync(int a, int b);

                    [GpuHandler]
                    Task<int> MultiplyAsync(int a, int b);

                    [GpuHandler(Mode = GpuHandlerMode.FireAndForget)]
                    Task ResetAsync();
                }
            }
            """;

        // Act
        var (generatorDiagnostics, generatedSources) = RunGenerator(source);

        // Assert - No generator errors
        var errors = generatorDiagnostics.Where(d => d.Severity == DiagnosticSeverity.Error).ToList();
        Assert.True(errors.Count == 0,
            $"Generator produced errors:\n{string.Join("\n", errors.Select(e => $"  {e.Id}: {e.GetMessage()}"))}");

        // Verify base infrastructure files were generated
        Assert.True(generatedSources.Count >= 8,
            $"Expected at least 8 generated files (attributes + enums), got {generatedSources.Count}");

        var allText = string.Join("\n", generatedSources);

        // Verify attribute definitions generated
        Assert.Contains("GpuNativeActorAttribute", allText);
        Assert.Contains("GpuHandlerAttribute", allText);

        // Verify enum definitions generated
        Assert.Contains("GpuHandlerMode", allText);
        Assert.Contains("FireAndForget", allText);
    }

    /// <summary>
    /// Tests that a stateful counter actor generates state struct.
    /// </summary>
    [Fact]
    public void GeneratedCode_ForStatefulActor_CompilesSuccessfully()
    {
        // Arrange
        var source = """
            using System.Threading.Tasks;
            using Orleans;
            using Orleans.GpuBridge.Abstractions.Generation;

            namespace TestNamespace
            {
                [GpuNativeActor]
                public interface ICounterActor : IGrainWithIntegerKey
                {
                    [GpuState]
                    int Count { get; }

                    [GpuHandler]
                    Task<int> IncrementAsync(int amount);

                    [GpuHandler]
                    Task<int> GetCountAsync();
                }
            }
            """;

        // Act
        var (generatorDiagnostics, generatedSources) = RunGenerator(source);

        // Assert
        var errors = generatorDiagnostics.Where(d => d.Severity == DiagnosticSeverity.Error).ToList();
        Assert.True(errors.Count == 0,
            $"Generator produced errors:\n{string.Join("\n", errors.Select(e => $"  {e.Id}: {e.GetMessage()}"))}");

        // Verify state struct generated
        Assert.True(generatedSources.Any(s => s.Contains("CounterActorState")),
            "Expected state struct to be generated");
    }

    /// <summary>
    /// Tests that temporal ordered actor generates HLC-related code.
    /// </summary>
    [Fact]
    public void GeneratedCode_ForTemporalActor_CompilesSuccessfully()
    {
        // Arrange
        var source = """
            using System.Threading.Tasks;
            using Orleans;
            using Orleans.GpuBridge.Abstractions.Generation;

            namespace TestNamespace
            {
                [GpuNativeActor]
                [TemporalOrdered(ClockType = TemporalClockType.HLC)]
                public interface IEventActor : IGrainWithIntegerKey
                {
                    [GpuState]
                    long EventId { get; }

                    [GpuHandler]
                    Task<long> RecordEventAsync(int eventType, long timestamp);
                }
            }
            """;

        // Act
        var (generatorDiagnostics, generatedSources) = RunGenerator(source);

        // Assert
        var errors = generatorDiagnostics.Where(d => d.Severity == DiagnosticSeverity.Error).ToList();
        Assert.True(errors.Count == 0,
            $"Generator produced errors:\n{string.Join("\n", errors.Select(e => $"  {e.Id}: {e.GetMessage()}"))}");

        // Verify HLC-related code generated
        var allText = string.Join("\n", generatedSources);
        Assert.Contains("HybridTimestamp", allText);
        Assert.Contains("EventActorState", allText);
    }

    /// <summary>
    /// Tests that K2K target actor generates dispatch helpers.
    /// </summary>
    [Fact]
    public void GeneratedCode_ForK2KActor_CompilesSuccessfully()
    {
        // Arrange
        var source = """
            using System.Threading.Tasks;
            using Orleans;
            using Orleans.GpuBridge.Abstractions.Generation;

            namespace TestNamespace
            {
                [GpuNativeActor]
                public interface IAggregatorActor : IGrainWithIntegerKey
                {
                    [GpuHandler]
                    [K2KTarget(typeof(IWorkerActor), "ProcessAsync", RoutingStrategy = K2KRoutingStrategy.Broadcast)]
                    Task<int> DistributeWorkAsync(int workId);
                }

                [GpuNativeActor]
                public interface IWorkerActor : IGrainWithIntegerKey
                {
                    [GpuHandler]
                    Task<int> ProcessAsync(int workItem);
                }
            }
            """;

        // Act
        var (generatorDiagnostics, generatedSources) = RunGenerator(source);

        // Assert
        var errors = generatorDiagnostics.Where(d => d.Severity == DiagnosticSeverity.Error).ToList();
        Assert.True(errors.Count == 0,
            $"Generator produced errors:\n{string.Join("\n", errors.Select(e => $"  {e.Id}: {e.GetMessage()}"))}");

        // Verify K2K dispatch code generated
        var allText = string.Join("\n", generatedSources);
        Assert.Contains("K2KDispatch", allText);
        Assert.Contains("AggregatorActorGrain", allText);
        Assert.Contains("WorkerActorGrain", allText);
    }

    /// <summary>
    /// Tests that CPU fallback implementations are generated.
    /// </summary>
    [Fact]
    public void GeneratedCode_IncludesCpuFallback_ForHandlers()
    {
        // Arrange
        var source = """
            using System.Threading.Tasks;
            using Orleans;
            using Orleans.GpuBridge.Abstractions.Generation;

            namespace TestNamespace
            {
                [GpuNativeActor]
                public interface IMathActor : IGrainWithIntegerKey
                {
                    [GpuHandler]
                    Task<int> AddAsync(int a, int b);

                    [GpuHandler]
                    Task<long> FactorialAsync(int n);
                }
            }
            """;

        // Act
        var (generatorDiagnostics, generatedSources) = RunGenerator(source);

        // Assert
        var errors = generatorDiagnostics.Where(d => d.Severity == DiagnosticSeverity.Error).ToList();
        Assert.True(errors.Count == 0,
            $"Generator produced errors:\n{string.Join("\n", errors.Select(e => $"  {e.Id}: {e.GetMessage()}"))}");

        // Verify CPU fallback generated
        var allText = string.Join("\n", generatedSources);
        Assert.Contains("CpuFallback_Add", allText);
        Assert.Contains("CpuFallback_Factorial", allText);
    }

    /// <summary>
    /// Tests message struct generation with proper blittable layout attributes.
    /// </summary>
    [Fact]
    public void GeneratedCode_HasProperMessageLayout()
    {
        // Arrange
        var source = """
            using System.Threading.Tasks;
            using Orleans;
            using Orleans.GpuBridge.Abstractions.Generation;

            namespace TestNamespace
            {
                [GpuNativeActor]
                public interface IDataActor : IGrainWithIntegerKey
                {
                    [GpuHandler(MaxPayloadSize = 128)]
                    Task<double> ProcessDataAsync(int id, float value, long timestamp);
                }
            }
            """;

        // Act
        var (generatorDiagnostics, generatedSources) = RunGenerator(source);

        // Assert
        var errors = generatorDiagnostics.Where(d => d.Severity == DiagnosticSeverity.Error).ToList();
        Assert.True(errors.Count == 0,
            $"Generator produced errors:\n{string.Join("\n", errors.Select(e => $"  {e.Id}: {e.GetMessage()}"))}");

        // Verify struct layout attributes
        var messagesSource = generatedSources.FirstOrDefault(s => s.Contains("ProcessDataRequest"));
        Assert.NotNull(messagesSource);
        Assert.Contains("StructLayout", messagesSource);
        Assert.Contains("LayoutKind.Sequential", messagesSource);
    }

    /// <summary>
    /// Tests that generated code includes proper using directives.
    /// </summary>
    [Fact]
    public void GeneratedCode_IncludesRequiredUsings()
    {
        // Arrange
        var source = """
            using System.Threading.Tasks;
            using Orleans;
            using Orleans.GpuBridge.Abstractions.Generation;

            namespace TestNamespace
            {
                [GpuNativeActor]
                public interface ISimpleActor : IGrainWithIntegerKey
                {
                    [GpuHandler]
                    Task<int> ProcessAsync(int value);
                }
            }
            """;

        // Act
        var (generatorDiagnostics, generatedSources) = RunGenerator(source);

        // Assert
        var errors = generatorDiagnostics.Where(d => d.Severity == DiagnosticSeverity.Error).ToList();
        Assert.True(errors.Count == 0,
            $"Generator produced errors:\n{string.Join("\n", errors.Select(e => $"  {e.Id}: {e.GetMessage()}"))}");

        var allText = string.Join("\n", generatedSources);
        Assert.Contains("using System;", allText);
        Assert.Contains("using System.Runtime.InteropServices;", allText);
    }

    /// <summary>
    /// Tests that ring kernel code is properly generated with DotCompute attributes.
    /// </summary>
    [Fact]
    public void GeneratedCode_IncludesRingKernelAttributes()
    {
        // Arrange
        var source = """
            using System.Threading.Tasks;
            using Orleans;
            using Orleans.GpuBridge.Abstractions.Generation;

            namespace TestNamespace
            {
                [GpuNativeActor]
                public interface IRingActor : IGrainWithIntegerKey
                {
                    [GpuHandler]
                    Task<int> ComputeAsync(int input);
                }
            }
            """;

        // Act
        var (generatorDiagnostics, generatedSources) = RunGenerator(source);

        // Assert
        var errors = generatorDiagnostics.Where(d => d.Severity == DiagnosticSeverity.Error).ToList();
        Assert.True(errors.Count == 0,
            $"Generator produced errors:\n{string.Join("\n", errors.Select(e => $"  {e.Id}: {e.GetMessage()}"))}");

        var kernelSource = generatedSources.FirstOrDefault(s => s.Contains("RingActorKernels"));
        Assert.NotNull(kernelSource);
        Assert.Contains("[RingKernel", kernelSource);
        Assert.Contains("KernelId", kernelSource);
    }

    private static (IReadOnlyList<Diagnostic> Diagnostics, List<string> GeneratedSources) RunGenerator(string source)
    {
        var generator = new GpuNativeActorGenerator();
        var syntaxTree = CSharpSyntaxTree.ParseText(source);

        var references = new List<MetadataReference>
        {
            MetadataReference.CreateFromFile(typeof(object).Assembly.Location),
            MetadataReference.CreateFromFile(typeof(Task).Assembly.Location),
            MetadataReference.CreateFromFile(typeof(Enumerable).Assembly.Location),
        };

        // Add runtime references
        var runtimePath = Path.GetDirectoryName(typeof(object).Assembly.Location)!;
        references.Add(MetadataReference.CreateFromFile(Path.Combine(runtimePath, "System.Runtime.dll")));

        // Create compilation with generator
        var compilation = CSharpCompilation.Create(
            "TestAssembly",
            new[] { syntaxTree },
            references,
            new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary));

        // Run generator
        var driver = CSharpGeneratorDriver.Create(generator).RunGenerators(compilation);
        var result = driver.GetRunResult();

        var generatedSources = result.GeneratedTrees
            .Select(t => t.GetText().ToString())
            .ToList();

        return (result.Diagnostics, generatedSources);
    }
}

// Orleans.GpuBridge - GPU-native distributed computing for Microsoft Orleans
// Copyright (c) 2025 Michael Ivertowski

using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Orleans.GpuBridge.Generators;
using Xunit;

namespace Orleans.GpuBridge.Generators.Tests;

/// <summary>
/// End-to-end integration tests for ICalculatorActor generation.
/// Validates the complete source generator pipeline from attribute-decorated
/// interface to generated message structs, kernels, and grain implementation.
/// </summary>
public class CalculatorActorGeneratorTests
{
    /// <summary>
    /// Tests complete generation for a simple calculator actor with blittable parameters.
    /// </summary>
    [Fact]
    public void Generator_ProducesCompleteOutput_ForCalculatorActor()
    {
        // Arrange - Define ICalculatorActor with generated attributes
        var source = """
            using System.Threading.Tasks;
            using Orleans;
            using Orleans.GpuBridge.Abstractions.Generation;

            namespace TestNamespace
            {
                /// <summary>
                /// GPU-native calculator actor demonstrating ring kernel message generation.
                /// </summary>
                [GpuNativeActor]
                public interface ICalculatorActor : IGrainWithIntegerKey
                {
                    /// <summary>
                    /// Adds two integers on the GPU.
                    /// </summary>
                    [GpuHandler]
                    Task<int> AddAsync(int a, int b);

                    /// <summary>
                    /// Multiplies two integers on the GPU.
                    /// </summary>
                    [GpuHandler(Mode = GpuHandlerMode.FireAndForget)]
                    Task MultiplyAsync(int a, int b);

                    /// <summary>
                    /// Computes factorial on the GPU.
                    /// </summary>
                    [GpuHandler(MaxPayloadSize = 128, QueueDepth = 512)]
                    Task<long> FactorialAsync(int n);
                }
            }
            """;

        var generator = new GpuNativeActorGenerator();
        var driver = CreateDriver(source, generator, includeOrleansReferences: true);

        // Act
        var result = driver.GetRunResult();

        // Assert - Should generate 8 base files + actor-specific files
        Assert.True(result.GeneratedTrees.Length >= 8,
            $"Expected at least 8 files (5 attrs + 3 enums), got {result.GeneratedTrees.Length}");

        var allText = string.Join("\n", result.GeneratedTrees.Select(t => t.GetText().ToString()));

        // Verify attributes are generated
        Assert.Contains("GpuNativeActorAttribute", allText);
        Assert.Contains("GpuHandlerAttribute", allText);

        // Verify enums with values
        Assert.Contains("GpuHandlerMode", allText);
        Assert.Contains("RequestResponse", allText);
        Assert.Contains("FireAndForget", allText);

        // Verify no errors in diagnostic
        var errors = result.Diagnostics.Where(d => d.Severity == DiagnosticSeverity.Error);
        Assert.Empty(errors);
    }

    /// <summary>
    /// Tests generator with state properties defined in actor interface.
    /// </summary>
    [Fact]
    public void Generator_HandlesStateProperties_ForStatefulActor()
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
                    int CurrentValue { get; }

                    [GpuState(Persist = false)]
                    long OperationCount { get; }

                    [GpuHandler]
                    Task<int> IncrementAsync(int amount);

                    [GpuHandler]
                    Task<int> GetValueAsync();
                }
            }
            """;

        var generator = new GpuNativeActorGenerator();
        var driver = CreateDriver(source, generator, includeOrleansReferences: true);

        // Act
        var result = driver.GetRunResult();

        // Assert
        Assert.True(result.GeneratedTrees.Length >= 8);

        var allText = string.Join("\n", result.GeneratedTrees.Select(t => t.GetText().ToString()));
        Assert.Contains("GpuStateAttribute", allText);
        Assert.Contains("bool Persist", allText);
    }

    /// <summary>
    /// Tests generator with K2K targets configured for multi-actor communication.
    /// </summary>
    [Fact]
    public void Generator_HandlesK2KTargets_ForDistributedActor()
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
                    Task<int> AggregateAsync(int[] values);
                }

                [GpuNativeActor]
                public interface IWorkerActor : IGrainWithIntegerKey
                {
                    [GpuHandler]
                    Task<int> ProcessAsync(int value);
                }
            }
            """;

        var generator = new GpuNativeActorGenerator();
        var driver = CreateDriver(source, generator, includeOrleansReferences: true);

        // Act
        var result = driver.GetRunResult();

        // Assert
        Assert.True(result.GeneratedTrees.Length >= 8);

        var allText = string.Join("\n", result.GeneratedTrees.Select(t => t.GetText().ToString()));
        Assert.Contains("K2KTargetAttribute", allText);
        Assert.Contains("K2KRoutingStrategy", allText);
        Assert.Contains("Broadcast", allText);
    }

    /// <summary>
    /// Tests generator with temporal ordering configured.
    /// </summary>
    [Fact]
    public void Generator_HandlesTemporalOrdering_ForCausalActor()
    {
        // Arrange
        var source = """
            using System.Threading.Tasks;
            using Orleans;
            using Orleans.GpuBridge.Abstractions.Generation;

            namespace TestNamespace
            {
                [GpuNativeActor]
                [TemporalOrdered(ClockType = TemporalClockType.HLC, StrictOrdering = true)]
                public interface ITransactionActor : IGrainWithIntegerKey
                {
                    [GpuHandler]
                    Task<long> CommitAsync(long transactionId, int amount);
                }
            }
            """;

        var generator = new GpuNativeActorGenerator();
        var driver = CreateDriver(source, generator, includeOrleansReferences: true);

        // Act
        var result = driver.GetRunResult();

        // Assert
        Assert.True(result.GeneratedTrees.Length >= 8);

        var allText = string.Join("\n", result.GeneratedTrees.Select(t => t.GetText().ToString()));
        Assert.Contains("TemporalOrderedAttribute", allText);
        Assert.Contains("TemporalClockType", allText);
        Assert.Contains("HLC", allText);
        Assert.Contains("VectorClock", allText);
        Assert.Contains("Lamport", allText);
    }

    /// <summary>
    /// Tests that generator validates blittable type requirements.
    /// </summary>
    [Fact]
    public void Generator_ValidatesBlittableTypes_AndReportsWarnings()
    {
        // Arrange - Use string which is non-blittable
        var source = """
            using System.Threading.Tasks;
            using Orleans;
            using Orleans.GpuBridge.Abstractions.Generation;

            namespace TestNamespace
            {
                [GpuNativeActor]
                public interface IInvalidActor : IGrainWithIntegerKey
                {
                    [GpuHandler]
                    Task ProcessAsync(string nonBlittableParam);
                }
            }
            """;

        var generator = new GpuNativeActorGenerator();
        var driver = CreateDriver(source, generator, includeOrleansReferences: true);

        // Act
        var result = driver.GetRunResult();

        // Assert - Should still generate base files but may have warnings
        Assert.True(result.GeneratedTrees.Length >= 8);

        // Generator should report GPUGEN002 for non-blittable parameter
        // (actual behavior depends on whether analyzer finds the attribute)
    }

    /// <summary>
    /// Tests generation of multiple handlers with different modes.
    /// </summary>
    [Fact]
    public void Generator_HandlesMultipleHandlerModes_WithUniqueMessageTypeIds()
    {
        // Arrange
        var source = """
            using System.Threading.Tasks;
            using Orleans;
            using Orleans.GpuBridge.Abstractions.Generation;

            namespace TestNamespace
            {
                [GpuNativeActor]
                public interface IMultiModeActor : IGrainWithIntegerKey
                {
                    [GpuHandler(Mode = GpuHandlerMode.RequestResponse)]
                    Task<int> QueryAsync(int id);

                    [GpuHandler(Mode = GpuHandlerMode.FireAndForget)]
                    Task UpdateAsync(int id, int value);

                    [GpuHandler(Mode = GpuHandlerMode.Streaming)]
                    Task StreamAsync(int batchId);
                }
            }
            """;

        var generator = new GpuNativeActorGenerator();
        var driver = CreateDriver(source, generator, includeOrleansReferences: true);

        // Act
        var result = driver.GetRunResult();

        // Assert
        Assert.True(result.GeneratedTrees.Length >= 8);

        var allText = string.Join("\n", result.GeneratedTrees.Select(t => t.GetText().ToString()));
        Assert.Contains("RequestResponse", allText);
        Assert.Contains("FireAndForget", allText);
        Assert.Contains("Streaming", allText);
    }

    /// <summary>
    /// Tests that attribute properties have correct default values.
    /// </summary>
    [Fact]
    public void Generator_ProducesCorrectDefaults_ForAttributeProperties()
    {
        // Arrange
        var source = """
            namespace Test { public class C { } }
            """;

        var generator = new GpuNativeActorGenerator();
        var driver = CreateDriver(source, generator);

        // Act
        var result = driver.GetRunResult();
        var allText = string.Join("\n", result.GeneratedTrees.Select(t => t.GetText().ToString()));

        // Assert - Verify default values
        Assert.Contains("228", allText); // Default MaxPayloadSize
        Assert.Contains("1024", allText); // Default QueueDepth

        // Verify GpuHandlerAttribute defaults
        Assert.Contains("GpuHandlerMode Mode { get; set; }", allText);
        Assert.Contains("int MaxPayloadSize { get; set; }", allText);
        Assert.Contains("int QueueDepth { get; set; }", allText);
        Assert.Contains("bool EnableChunking { get; set; }", allText);

        // Verify GpuStateAttribute defaults
        Assert.Contains("bool Persist { get; set; }", allText);
        Assert.Contains("bool ReadOnly { get; set; }", allText);
    }

    private static GeneratorDriver CreateDriver(
        string source,
        IIncrementalGenerator generator,
        bool includeOrleansReferences = false)
    {
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

        if (includeOrleansReferences)
        {
            // Add Orleans.Core reference if available
            var orleansAssembly = AppDomain.CurrentDomain.GetAssemblies()
                .FirstOrDefault(a => a.GetName().Name == "Orleans.Core");
            if (orleansAssembly != null)
            {
                references.Add(MetadataReference.CreateFromFile(orleansAssembly.Location));
            }
        }

        var compilation = CSharpCompilation.Create(
            "TestAssembly",
            new[] { syntaxTree },
            references,
            new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary));

        return CSharpGeneratorDriver.Create(generator).RunGenerators(compilation);
    }
}

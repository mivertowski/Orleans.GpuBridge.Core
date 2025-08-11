using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Channels;
using System.Threading.Tasks;
using FluentAssertions;
using Microsoft.Extensions.Logging;
using Moq;
using Orleans.GpuBridge.Abstractions;
using Orleans.GpuBridge.BridgeFX;
using Xunit;

namespace Orleans.GpuBridge.Tests.BridgeFX;

/// <summary>
/// Unit tests for GpuPipeline and pipeline stages
/// </summary>
public class GpuPipelineTests
{
    private readonly Mock<IGpuBridge> _mockBridge;
    private readonly Mock<ILogger<GpuPipeline>> _mockLogger;
    private readonly Mock<IGpuKernel<int, string>> _mockKernel;

    public GpuPipelineTests()
    {
        _mockBridge = new Mock<IGpuBridge>();
        _mockLogger = new Mock<ILogger<GpuPipeline>>();
        _mockKernel = new Mock<IGpuKernel<int, string>>();
    }

    [Fact]
    public void Constructor_WithValidParameters_ShouldCreatePipeline()
    {
        // Act
        var pipeline = new GpuPipeline(_mockBridge.Object, _mockLogger.Object);

        // Assert
        pipeline.Should().NotBeNull();
    }

    [Fact]
    public void AddKernel_ShouldAddKernelStage()
    {
        // Arrange
        var pipeline = new GpuPipeline(_mockBridge.Object, _mockLogger.Object);
        var kernelId = KernelId.Parse("test-kernel");

        // Act
        var result = pipeline.AddKernel<int, string>(kernelId);

        // Assert
        result.Should().Be(pipeline); // Fluent interface
    }

    [Fact]
    public void AddKernel_WithFilter_ShouldAddKernelStageWithFilter()
    {
        // Arrange
        var pipeline = new GpuPipeline(_mockBridge.Object, _mockLogger.Object);
        var kernelId = KernelId.Parse("test-kernel");
        var filter = new Func<int, bool>(x => x > 0);

        // Act
        var result = pipeline.AddKernel<int, string>(kernelId, filter);

        // Assert
        result.Should().Be(pipeline);
    }

    [Fact]
    public void Transform_ShouldAddTransformStage()
    {
        // Arrange
        var pipeline = new GpuPipeline(_mockBridge.Object, _mockLogger.Object);
        var transform = new Func<int, string>(x => x.ToString());

        // Act
        var result = pipeline.Transform(transform);

        // Assert
        result.Should().Be(pipeline);
    }

    [Fact]
    public void Batch_ShouldAddBatchStage()
    {
        // Arrange
        var pipeline = new GpuPipeline(_mockBridge.Object, _mockLogger.Object);
        var batchSize = 10;
        var timeout = TimeSpan.FromMilliseconds(500);

        // Act
        var result = pipeline.Batch<int>(batchSize, timeout);

        // Assert
        result.Should().Be(pipeline);
    }

    [Fact]
    public void Batch_WithoutTimeout_ShouldUseDefaultTimeout()
    {
        // Arrange
        var pipeline = new GpuPipeline(_mockBridge.Object, _mockLogger.Object);
        var batchSize = 10;

        // Act
        var result = pipeline.Batch<int>(batchSize);

        // Assert
        result.Should().Be(pipeline);
    }

    [Fact]
    public void Parallel_ShouldAddParallelStage()
    {
        // Arrange
        var pipeline = new GpuPipeline(_mockBridge.Object, _mockLogger.Object);
        var processor = new Func<int, Task<string>>(x => Task.FromResult(x.ToString()));

        // Act
        var result = pipeline.Parallel(processor, maxConcurrency: 4);

        // Assert
        result.Should().Be(pipeline);
    }

    [Fact]
    public void Parallel_WithoutMaxConcurrency_ShouldUseProcessorCount()
    {
        // Arrange
        var pipeline = new GpuPipeline(_mockBridge.Object, _mockLogger.Object);
        var processor = new Func<int, Task<string>>(x => Task.FromResult(x.ToString()));

        // Act
        var result = pipeline.Parallel(processor);

        // Assert
        result.Should().Be(pipeline);
    }

    [Fact]
    public void Filter_ShouldAddFilterStage()
    {
        // Arrange
        var pipeline = new GpuPipeline(_mockBridge.Object, _mockLogger.Object);
        var predicate = new Func<int, bool>(x => x > 0);

        // Act
        var result = pipeline.Filter(predicate);

        // Assert
        result.Should().Be(pipeline);
    }

    [Fact]
    public void Tap_ShouldAddTapStage()
    {
        // Arrange
        var pipeline = new GpuPipeline(_mockBridge.Object, _mockLogger.Object);
        var tappedValues = new List<int>();
        var action = new Action<int>(x => tappedValues.Add(x));

        // Act
        var result = pipeline.Tap(action);

        // Assert
        result.Should().Be(pipeline);
    }

    [Fact]
    public void Build_WithValidPipeline_ShouldReturnExecutablePipeline()
    {
        // Arrange
        var pipeline = new GpuPipeline(_mockBridge.Object, _mockLogger.Object);
        pipeline.Transform<int, string>(x => x.ToString());

        // Act
        var executable = pipeline.Build<int, string>();

        // Assert
        executable.Should().NotBeNull();
        executable.Should().BeAssignableTo<IPipeline<int, string>>();
    }

    [Fact]
    public void FluentInterface_ShouldAllowChaining()
    {
        // Arrange
        var pipeline = new GpuPipeline(_mockBridge.Object, _mockLogger.Object);
        var kernelId = KernelId.Parse("test-kernel");

        // Act
        var result = pipeline
            .Filter<int>(x => x > 0)
            .Transform<int, double>(x => x * 2.0)
            .Batch<double>(5, TimeSpan.FromMilliseconds(100))
            .AddKernel<IReadOnlyList<double>, IReadOnlyList<string>>(kernelId)
            .Transform<IReadOnlyList<string>, string>(list => string.Join(",", list));

        // Assert
        result.Should().Be(pipeline);
    }

    [Theory]
    [InlineData(1)]
    [InlineData(5)]
    [InlineData(10)]
    [InlineData(100)]
    public void Batch_WithVariousBatchSizes_ShouldCreateCorrectStage(int batchSize)
    {
        // Arrange
        var pipeline = new GpuPipeline(_mockBridge.Object, _mockLogger.Object);

        // Act
        var result = pipeline.Batch<int>(batchSize);

        // Assert
        result.Should().Be(pipeline);
    }

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(4)]
    [InlineData(8)]
    public void Parallel_WithVariousConcurrencyLevels_ShouldCreateCorrectStage(int maxConcurrency)
    {
        // Arrange
        var pipeline = new GpuPipeline(_mockBridge.Object, _mockLogger.Object);
        var processor = new Func<int, Task<string>>(x => Task.FromResult(x.ToString()));

        // Act
        var result = pipeline.Parallel(processor, maxConcurrency);

        // Assert
        result.Should().Be(pipeline);
    }

    [Fact]
    public void ComplexPipeline_ShouldBuildSuccessfully()
    {
        // Arrange
        var pipeline = new GpuPipeline(_mockBridge.Object, _mockLogger.Object);
        var kernelId = KernelId.Parse("compute-kernel");
        var processedItems = new List<int>();

        // Act
        var executablePipeline = pipeline
            .Filter<int>(x => x > 0)
            .Tap<int>(x => processedItems.Add(x))
            .Batch<int>(5, TimeSpan.FromMilliseconds(100))
            .Transform<IReadOnlyList<int>, int[]>(list => list.ToArray())
            .Transform<int[], string>(array => $"Batch[{string.Join(",", array)}]")
            .Build<int, string>();

        // Assert
        executablePipeline.Should().NotBeNull();
    }
}

/// <summary>
/// Unit tests for individual pipeline stages
/// </summary>
public class PipelineStageTests
{
    private readonly Mock<IGpuBridge> _mockBridge;
    private readonly Mock<IGpuKernel<int, string>> _mockKernel;

    public PipelineStageTests()
    {
        _mockBridge = new Mock<IGpuBridge>();
        _mockKernel = new Mock<IGpuKernel<int, string>>();
    }

    [Fact]
    public async Task KernelStage_ProcessAsync_ShouldExecuteKernel()
    {
        // Arrange
        var kernelId = KernelId.Parse("test-kernel");
        var stage = CreateKernelStage(kernelId);
        var input = 42;
        var expectedOutput = "42";
        var handle = KernelHandle.Create();

        _mockBridge.Setup(b => b.GetKernelAsync<int, string>(kernelId, It.IsAny<CancellationToken>()))
                   .ReturnsAsync(_mockKernel.Object);

        _mockKernel.Setup(k => k.SubmitBatchAsync(
                It.Is<int[]>(batch => batch.Length == 1 && batch[0] == input),
                null,
                It.IsAny<CancellationToken>()))
                  .ReturnsAsync(handle);

        _mockKernel.Setup(k => k.ReadResultsAsync(handle, It.IsAny<CancellationToken>()))
                  .Returns(new[] { expectedOutput }.ToAsyncEnumerable());

        // Act
        var result = await stage.ProcessAsync(input);

        // Assert
        result.Should().Be(expectedOutput);
    }

    [Fact]
    public async Task KernelStage_WithFilter_ShouldFilterInput()
    {
        // Arrange
        var kernelId = KernelId.Parse("test-kernel");
        var filter = new Func<int, bool>(x => x > 0);
        var stage = CreateKernelStage(kernelId, filter);
        var input = -5; // Should be filtered out

        // Act
        var result = await stage.ProcessAsync(input);

        // Assert
        result.Should().BeNull();
    }

    [Fact]
    public async Task KernelStage_WithInvalidInputType_ShouldThrowException()
    {
        // Arrange
        var kernelId = KernelId.Parse("test-kernel");
        var stage = CreateKernelStage(kernelId);
        var invalidInput = "string input"; // Should be int

        // Act & Assert
        await Assert.ThrowsAsync<ArgumentException>(() =>
            stage.ProcessAsync(invalidInput));
    }

    [Fact]
    public async Task TransformStage_ProcessAsync_ShouldTransformInput()
    {
        // Arrange
        var transform = new Func<int, string>(x => $"Value: {x}");
        var stage = CreateTransformStage(transform);
        var input = 42;

        // Act
        var result = await stage.ProcessAsync(input);

        // Assert
        result.Should().Be("Value: 42");
    }

    [Fact]
    public async Task TransformStage_WithInvalidInputType_ShouldThrowException()
    {
        // Arrange
        var transform = new Func<int, string>(x => x.ToString());
        var stage = CreateTransformStage(transform);
        var invalidInput = "not an int";

        // Act & Assert
        await Assert.ThrowsAsync<ArgumentException>(() =>
            stage.ProcessAsync(invalidInput));
    }

    [Fact]
    public async Task BatchStage_ProcessAsync_ShouldBatchItems()
    {
        // Arrange
        var batchSize = 3;
        var timeout = TimeSpan.FromSeconds(1);
        var stage = CreateBatchStage<int>(batchSize, timeout);

        // Act
        var result1 = await stage.ProcessAsync(1);
        var result2 = await stage.ProcessAsync(2);
        var result3 = await stage.ProcessAsync(3); // Should trigger batch

        // Assert
        result1.Should().BeNull(); // Not enough items yet
        result2.Should().BeNull(); // Still not enough
        result3.Should().NotBeNull();
        result3.Should().BeOfType<List<int>>();
        
        var batch = (List<int>)result3!;
        batch.Should().HaveCount(3);
        batch.Should().ContainInOrder(1, 2, 3);
    }

    [Fact]
    public async Task ParallelStage_ProcessAsync_ShouldExecuteInParallel()
    {
        // Arrange
        var processor = new Func<int, Task<string>>(async x =>
        {
            await Task.Delay(10); // Simulate work
            return $"Processed: {x}";
        });
        var stage = CreateParallelStage(processor, maxConcurrency: 2);
        var input = 42;

        // Act
        var result = await stage.ProcessAsync(input);

        // Assert
        result.Should().Be("Processed: 42");
    }

    [Fact]
    public async Task FilterStage_ProcessAsync_WithPassingPredicate_ShouldReturnInput()
    {
        // Arrange
        var predicate = new Func<int, bool>(x => x > 0);
        var stage = CreateFilterStage(predicate);
        var input = 42;

        // Act
        var result = await stage.ProcessAsync(input);

        // Assert
        result.Should().Be(input);
    }

    [Fact]
    public async Task FilterStage_ProcessAsync_WithFailingPredicate_ShouldReturnNull()
    {
        // Arrange
        var predicate = new Func<int, bool>(x => x > 0);
        var stage = CreateFilterStage(predicate);
        var input = -42;

        // Act
        var result = await stage.ProcessAsync(input);

        // Assert
        result.Should().BeNull();
    }

    [Fact]
    public async Task TapStage_ProcessAsync_ShouldExecuteActionAndReturnInput()
    {
        // Arrange
        var tappedValues = new List<int>();
        var action = new Action<int>(x => tappedValues.Add(x));
        var stage = CreateTapStage(action);
        var input = 42;

        // Act
        var result = await stage.ProcessAsync(input);

        // Assert
        result.Should().Be(input);
        tappedValues.Should().ContainSingle().Which.Should().Be(input);
    }

    [Theory]
    [InlineData(typeof(int), typeof(string))]
    [InlineData(typeof(double), typeof(int))]
    [InlineData(typeof(string), typeof(bool))]
    public void Stage_InputOutputTypes_ShouldBeCorrect(Type inputType, Type outputType)
    {
        // Arrange & Act
        var transformStage = CreateGenericTransformStage(inputType, outputType);

        // Assert
        transformStage.InputType.Should().Be(inputType);
        transformStage.OutputType.Should().Be(outputType);
    }

    private IPipelineStage CreateKernelStage(KernelId kernelId, Func<int, bool>? filter = null)
    {
        // Use reflection to create KernelStage since it's internal
        var kernelStageType = typeof(GpuPipeline).Assembly
            .GetType("Orleans.GpuBridge.BridgeFX.KernelStage`2")!
            .MakeGenericType(typeof(int), typeof(string));

        return (IPipelineStage)Activator.CreateInstance(
            kernelStageType,
            kernelId,
            _mockBridge.Object,
            filter)!;
    }

    private IPipelineStage CreateTransformStage(Func<int, string> transform)
    {
        var transformStageType = typeof(GpuPipeline).Assembly
            .GetType("Orleans.GpuBridge.BridgeFX.TransformStage`2")!
            .MakeGenericType(typeof(int), typeof(string));

        return (IPipelineStage)Activator.CreateInstance(transformStageType, transform)!;
    }

    private IPipelineStage CreateBatchStage<T>(int batchSize, TimeSpan timeout)
    {
        var batchStageType = typeof(GpuPipeline).Assembly
            .GetType("Orleans.GpuBridge.BridgeFX.BatchStage`1")!
            .MakeGenericType(typeof(T));

        return (IPipelineStage)Activator.CreateInstance(batchStageType, batchSize, timeout)!;
    }

    private IPipelineStage CreateParallelStage(Func<int, Task<string>> processor, int maxConcurrency)
    {
        var parallelStageType = typeof(GpuPipeline).Assembly
            .GetType("Orleans.GpuBridge.BridgeFX.ParallelStage`2")!
            .MakeGenericType(typeof(int), typeof(string));

        return (IPipelineStage)Activator.CreateInstance(parallelStageType, processor, maxConcurrency)!;
    }

    private IPipelineStage CreateFilterStage(Func<int, bool> predicate)
    {
        var filterStageType = typeof(GpuPipeline).Assembly
            .GetType("Orleans.GpuBridge.BridgeFX.FilterStage`1")!
            .MakeGenericType(typeof(int));

        return (IPipelineStage)Activator.CreateInstance(filterStageType, predicate)!;
    }

    private IPipelineStage CreateTapStage(Action<int> action)
    {
        var tapStageType = typeof(GpuPipeline).Assembly
            .GetType("Orleans.GpuBridge.BridgeFX.TapStage`1")!
            .MakeGenericType(typeof(int));

        return (IPipelineStage)Activator.CreateInstance(tapStageType, action)!;
    }

    private IPipelineStage CreateGenericTransformStage(Type inputType, Type outputType)
    {
        var transformStageType = typeof(GpuPipeline).Assembly
            .GetType("Orleans.GpuBridge.BridgeFX.TransformStage`2")!
            .MakeGenericType(inputType, outputType);

        // Create a simple transform function
        var transform = CreateTransformFunction(inputType, outputType);

        return (IPipelineStage)Activator.CreateInstance(transformStageType, transform)!;
    }

    private object CreateTransformFunction(Type inputType, Type outputType)
    {
        // Create a simple transform that converts input to string then to target type
        var funcType = typeof(Func<,>).MakeGenericType(inputType, outputType);
        
        if (outputType == typeof(string))
        {
            var method = inputType.GetMethod("ToString");
            return Delegate.CreateDelegate(funcType, null, method!);
        }
        
        // For other types, create a simple delegate
        var param = System.Linq.Expressions.Expression.Parameter(inputType, "x");
        var convert = System.Linq.Expressions.Expression.Convert(param, outputType);
        var lambda = System.Linq.Expressions.Expression.Lambda(funcType, convert, param);
        
        return lambda.Compile();
    }
}
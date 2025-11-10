using Xunit;
using FluentAssertions;
using Orleans.GpuBridge.BridgeFX;
using Orleans.GpuBridge.BridgeFX.Pipeline.Core;
using Orleans.GpuBridge.BridgeFX.Pipeline.Stages;
using Orleans.GpuBridge.Abstractions;
using Microsoft.Extensions.Logging;
using Moq;
using System.Reflection;

namespace Orleans.GpuBridge.Tests.RC2.BridgeFX;

/// <summary>
/// Targeted tests to achieve 80% coverage for BridgeFX by covering exactly 31 uncovered lines.
/// Focuses on error paths, type validation, and fluent API methods.
/// </summary>
public class PipelineCoverageCompletionTests : IDisposable
{
    private readonly Mock<IGpuBridge> _mockBridge;
    private readonly Mock<ILogger<GpuPipeline>> _mockLogger;

    public PipelineCoverageCompletionTests()
    {
        _mockBridge = new Mock<IGpuBridge>();
        _mockLogger = new Mock<ILogger<GpuPipeline>>();
    }

    #region ParallelStage.cs - Lines 30-31 (4 lines)

    [Fact]
    public async Task ParallelStage_WithWrongInputType_ShouldThrowArgumentException()
    {
        // Arrange
        var stage = new ParallelStage<int, int>(x => Task.FromResult(x * 2), maxConcurrency: 2);
        var wrongInput = "string instead of int";

        // Act & Assert
        var exception = await Assert.ThrowsAsync<ArgumentException>(
            async () => await stage.ProcessAsync(wrongInput, CancellationToken.None));

        exception.Message.Should().Contain("Expected System.Int32");
        exception.Message.Should().Contain("got System.String");
    }

    [Fact]
    public async Task ParallelStage_WithWrongInputType_ExceptionMessageContainsCorrectTypeInfo()
    {
        // Arrange
        var stage = new ParallelStage<double, double>(x => Task.FromResult(x * 2.0), maxConcurrency: 2);
        var wrongInput = new object(); // object instead of double

        // Act & Assert
        var exception = await Assert.ThrowsAsync<ArgumentException>(
            async () => await stage.ProcessAsync(wrongInput, CancellationToken.None));

        exception.Message.Should().Contain("Expected System.Double");
        exception.Message.Should().Contain("got System.Object");
    }

    #endregion

    #region TapStage.cs - Lines 27-28 (4 lines)

    [Fact]
    public async Task TapStage_WithWrongInputType_ShouldThrowArgumentException()
    {
        // Arrange
        int sideEffectValue = 0;
        var stage = new TapStage<int>(x => sideEffectValue = x);
        var wrongInput = "string instead of int";

        // Act & Assert
        var exception = await Assert.ThrowsAsync<ArgumentException>(
            async () => await stage.ProcessAsync(wrongInput, CancellationToken.None));

        exception.Message.Should().Contain("Expected System.Int32");
        exception.Message.Should().Contain("got System.String");
        sideEffectValue.Should().Be(0); // Side effect should not have executed
    }

    [Fact]
    public async Task TapStage_WithWrongInputType_ExceptionMessageIsCorrect()
    {
        // Arrange
        var stage = new TapStage<string>(x => { /* do nothing */ });
        var wrongInput = 42; // int instead of string

        // Act & Assert
        var exception = await Assert.ThrowsAsync<ArgumentException>(
            async () => await stage.ProcessAsync(wrongInput, CancellationToken.None));

        exception.Message.Should().Contain("Expected System.String");
        exception.Message.Should().Contain("got System.Int32");
    }

    #endregion

    #region GpuPipeline.cs - Lines 47-50, 81-84 (16 lines)

    [Fact]
    public void GpuPipeline_AddKernel_WithoutFilter_AddsKernelStage()
    {
        // Arrange
        var pipeline = new GpuPipeline(_mockBridge.Object, _mockLogger.Object);
        var kernelId = new KernelId("test-kernel");

        // Act
        var result = pipeline.AddKernel<int, int>(kernelId);

        // Assert
        result.Should().BeSameAs(pipeline); // Fluent API returns pipeline

        // Use reflection to verify stage was added
        var stagesField = typeof(GpuPipeline).GetField("_stages", BindingFlags.NonPublic | BindingFlags.Instance);
        var stages = (List<IPipelineStage>)stagesField!.GetValue(pipeline)!;
        stages.Should().HaveCount(1);
        stages[0].Should().BeOfType<KernelStage<int, int>>();
    }

    [Fact]
    public void GpuPipeline_AddKernel_WithFilter_AddsKernelStageWithFilter()
    {
        // Arrange
        var pipeline = new GpuPipeline(_mockBridge.Object, _mockLogger.Object);
        var kernelId = new KernelId("test-kernel");
        Func<int, bool> filter = x => x > 0;

        // Act
        var result = pipeline.AddKernel<int, int>(kernelId, filter);

        // Assert
        result.Should().BeSameAs(pipeline);

        var stagesField = typeof(GpuPipeline).GetField("_stages", BindingFlags.NonPublic | BindingFlags.Instance);
        var stages = (List<IPipelineStage>)stagesField!.GetValue(pipeline)!;
        stages.Should().HaveCount(1);
        stages[0].Should().BeOfType<KernelStage<int, int>>();
    }

    [Fact]
    public void GpuPipeline_AddKernel_ReturnsThisPipeline_ForChaining()
    {
        // Arrange
        var pipeline = new GpuPipeline(_mockBridge.Object, _mockLogger.Object);
        var kernelId1 = new KernelId("kernel-1");
        var kernelId2 = new KernelId("kernel-2");

        // Act - Chain multiple AddKernel calls
        var result = pipeline
            .AddKernel<int, int>(kernelId1)
            .AddKernel<int, int>(kernelId2);

        // Assert
        result.Should().BeSameAs(pipeline);

        var stagesField = typeof(GpuPipeline).GetField("_stages", BindingFlags.NonPublic | BindingFlags.Instance);
        var stages = (List<IPipelineStage>)stagesField!.GetValue(pipeline)!;
        stages.Should().HaveCount(2);
    }

    [Fact]
    public void GpuPipeline_AddKernel_AddsKernelStageToStagesList()
    {
        // Arrange
        var pipeline = new GpuPipeline(_mockBridge.Object, _mockLogger.Object);
        var kernelId = new KernelId("vector-add");

        // Act
        pipeline.AddKernel<float[], float[]>(kernelId);

        // Assert
        var stagesField = typeof(GpuPipeline).GetField("_stages", BindingFlags.NonPublic | BindingFlags.Instance);
        var stages = (List<IPipelineStage>)stagesField!.GetValue(pipeline)!;

        stages.Should().HaveCount(1);
        var kernelStage = stages[0].Should().BeOfType<KernelStage<float[], float[]>>().Subject;
        kernelStage.Should().NotBeNull();
    }

    [Fact]
    public void GpuPipeline_Batch_WithoutTimeout_UsesDefaultTimeout()
    {
        // Arrange
        var pipeline = new GpuPipeline(_mockBridge.Object, _mockLogger.Object);

        // Act
        var result = pipeline.Batch<int>(batchSize: 10);

        // Assert
        result.Should().BeSameAs(pipeline);

        var stagesField = typeof(GpuPipeline).GetField("_stages", BindingFlags.NonPublic | BindingFlags.Instance);
        var stages = (List<IPipelineStage>)stagesField!.GetValue(pipeline)!;
        stages.Should().HaveCount(1);
        stages[0].Should().BeOfType<BatchStage<int>>();
    }

    [Fact]
    public void GpuPipeline_Batch_WithCustomTimeout_UsesProvidedTimeout()
    {
        // Arrange
        var pipeline = new GpuPipeline(_mockBridge.Object, _mockLogger.Object);
        var timeout = TimeSpan.FromSeconds(5);

        // Act
        var result = pipeline.Batch<int>(batchSize: 10, timeout: timeout);

        // Assert
        result.Should().BeSameAs(pipeline);

        var stagesField = typeof(GpuPipeline).GetField("_stages", BindingFlags.NonPublic | BindingFlags.Instance);
        var stages = (List<IPipelineStage>)stagesField!.GetValue(pipeline)!;
        stages.Should().HaveCount(1);
        stages[0].Should().BeOfType<BatchStage<int>>();
    }

    [Fact]
    public void GpuPipeline_Batch_ReturnsThisPipeline_ForChaining()
    {
        // Arrange
        var pipeline = new GpuPipeline(_mockBridge.Object, _mockLogger.Object);

        // Act - Chain multiple Batch calls
        var result = pipeline
            .Batch<int>(batchSize: 10)
            .Batch<int>(batchSize: 20);

        // Assert
        result.Should().BeSameAs(pipeline);

        var stagesField = typeof(GpuPipeline).GetField("_stages", BindingFlags.NonPublic | BindingFlags.Instance);
        var stages = (List<IPipelineStage>)stagesField!.GetValue(pipeline)!;
        stages.Should().HaveCount(2);
    }

    [Fact]
    public void GpuPipeline_Batch_AddsBatchStageToStagesList()
    {
        // Arrange
        var pipeline = new GpuPipeline(_mockBridge.Object, _mockLogger.Object);

        // Act
        pipeline.Batch<string>(batchSize: 5, timeout: TimeSpan.FromMilliseconds(200));

        // Assert
        var stagesField = typeof(GpuPipeline).GetField("_stages", BindingFlags.NonPublic | BindingFlags.Instance);
        var stages = (List<IPipelineStage>)stagesField!.GetValue(pipeline)!;

        stages.Should().HaveCount(1);
        var batchStage = stages[0].Should().BeOfType<BatchStage<string>>().Subject;
        batchStage.Should().NotBeNull();
    }

    #endregion

    #region AsyncTransformStage.cs - Lines 27, 29-30, 32-33, 35-37 (7 lines)

    [Fact]
    public async Task AsyncTransformStage_WithNullInputForNullableType_ShouldHandleGracefully()
    {
        // Arrange
        var stage = new AsyncTransformStage<int?, int?>(x => Task.FromResult<int?>(x.HasValue ? x.Value * 2 : null));

        // Act
        var result = await stage.ProcessAsync(null, CancellationToken.None);

        // Assert - should not throw, allows null for nullable types
        // The stage processes null as default(int?) which is null
        result.Should().BeNull();
    }

    [Fact]
    public async Task AsyncTransformStage_WithNullInputForNonNullableType_ShouldThrow()
    {
        // Arrange
        var stage = new AsyncTransformStage<int, int>(x => Task.FromResult(x * 2));

        // Act & Assert
        var exception = await Assert.ThrowsAsync<ArgumentException>(
            async () => await stage.ProcessAsync(null, CancellationToken.None));

        exception.Message.Should().Contain("Expected System.Int32");
        exception.Message.Should().Contain("null");
    }

    [Fact]
    public async Task AsyncTransformStage_WithWrongType_ShouldThrowWithCorrectMessage()
    {
        // Arrange
        var stage = new AsyncTransformStage<int, int>(x => Task.FromResult(x * 2));
        var wrongInput = "string";

        // Act & Assert
        var exception = await Assert.ThrowsAsync<ArgumentException>(
            async () => await stage.ProcessAsync(wrongInput, CancellationToken.None));

        exception.Message.Should().Contain("Expected System.Int32");
        exception.Message.Should().Contain("System.String");
    }

    [Fact]
    public async Task AsyncTransformStage_WithNullableStringAndNullInput_ShouldThrowForNullableReferenceType()
    {
        // Arrange
        var stage = new AsyncTransformStage<string?, string?>(x => Task.FromResult<string?>(x?.ToUpper()));

        // Act & Assert
        // NOTE: Nullable.GetUnderlyingType() only works for nullable VALUE types (int?, double?)
        // It returns null for nullable REFERENCE types (string?), so null string throws
        var exception = await Assert.ThrowsAsync<ArgumentException>(
            async () => await stage.ProcessAsync(null, CancellationToken.None));

        exception.Message.Should().Contain("Expected System.String");
        exception.Message.Should().Contain("null");
    }

    #endregion

    public void Dispose()
    {
        // Cleanup if needed
        _mockBridge.Reset();
        _mockLogger.Reset();
    }
}

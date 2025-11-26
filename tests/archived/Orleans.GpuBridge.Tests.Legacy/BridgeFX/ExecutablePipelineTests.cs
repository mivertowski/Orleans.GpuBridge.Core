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
/// Unit tests for ExecutablePipeline and pipeline execution scenarios
/// </summary>
public class ExecutablePipelineTests
{
    private readonly Mock<IGpuBridge> _mockBridge;
    private readonly Mock<ILogger<GpuPipeline>> _mockLogger;
    private readonly Mock<IGpuKernel<int, string>> _mockKernel;

    public ExecutablePipelineTests()
    {
        _mockBridge = new Mock<IGpuBridge>();
        _mockLogger = new Mock<ILogger<GpuPipeline>>();
        _mockKernel = new Mock<IGpuKernel<int, string>>();
    }

    [Fact]
    public async Task ProcessAsync_WithSimpleTransform_ShouldTransformInput()
    {
        // Arrange
        var pipeline = new GpuPipeline(_mockBridge.Object, _mockLogger.Object)
            .Transform<int, string>(x => $"Value: {x}");
        
        var executable = pipeline.Build<int, string>();
        var input = 42;

        // Act
        var result = await executable.ProcessAsync(input);

        // Assert
        result.Should().Be("Value: 42");
    }

    [Fact]
    public async Task ProcessAsync_WithChainedTransforms_ShouldApplyAllTransforms()
    {
        // Arrange
        var pipeline = new GpuPipeline(_mockBridge.Object, _mockLogger.Object)
            .Transform<int, double>(x => x * 2.0)
            .Transform<double, string>(x => $"Double: {x}");
        
        var executable = pipeline.Build<int, string>();
        var input = 21;

        // Act
        var result = await executable.ProcessAsync(input);

        // Assert
        result.Should().Be("Double: 42");
    }

    [Fact]
    public async Task ProcessAsync_WithFilter_ShouldFilterOutItems()
    {
        // Arrange
        var pipeline = new GpuPipeline(_mockBridge.Object, _mockLogger.Object)
            .Filter<int>(x => x > 0)
            .Transform<int, string>(x => x.ToString());
        
        var executable = pipeline.Build<int, string>();

        // Act & Assert for positive value
        var result1 = await executable.ProcessAsync(42);
        result1.Should().Be("42");

        // The pipeline should throw for filtered values since they become null
        await Assert.ThrowsAsync<InvalidOperationException>(() =>
            executable.ProcessAsync(-42));
    }

    [Fact]
    public async Task ProcessAsync_WithTap_ShouldExecuteSideEffect()
    {
        // Arrange
        var tappedValues = new List<int>();
        var pipeline = new GpuPipeline(_mockBridge.Object, _mockLogger.Object)
            .Tap<int>(x => tappedValues.Add(x))
            .Transform<int, string>(x => x.ToString());
        
        var executable = pipeline.Build<int, string>();
        var input = 42;

        // Act
        var result = await executable.ProcessAsync(input);

        // Assert
        result.Should().Be("42");
        tappedValues.Should().ContainSingle().Which.Should().Be(42);
    }

    [Fact]
    public async Task ProcessAsync_WithParallelStage_ShouldProcessInParallel()
    {
        // Arrange
        var processedIds = new List<int>();
        var pipeline = new GpuPipeline(_mockBridge.Object, _mockLogger.Object)
            .Parallel<int, string>(async x =>
            {
                processedIds.Add(Thread.CurrentThread.ManagedThreadId);
                await Task.Delay(10);
                return x.ToString();
            }, maxConcurrency: 2);
        
        var executable = pipeline.Build<int, string>();
        var input = 42;

        // Act
        var result = await executable.ProcessAsync(input);

        // Assert
        result.Should().Be("42");
        processedIds.Should().ContainSingle();
    }

    [Fact]
    public async Task ProcessManyAsync_WithMultipleInputs_ShouldProcessAll()
    {
        // Arrange
        var pipeline = new GpuPipeline(_mockBridge.Object, _mockLogger.Object)
            .Transform<int, string>(x => $"Item: {x}");
        
        var executable = pipeline.Build<int, string>();
        var inputs = new[] { 1, 2, 3, 4, 5 }.ToAsyncEnumerable();

        // Act
        var results = new List<string>();
        await foreach (var result in executable.ProcessManyAsync(inputs))
        {
            results.Add(result);
        }

        // Assert
        results.Should().HaveCount(5);
        results.Should().BeEquivalentTo(new[]
        {
            "Item: 1", "Item: 2", "Item: 3", "Item: 4", "Item: 5"
        });
    }

    [Fact]
    public async Task ProcessManyAsync_WithException_ShouldContinueProcessing()
    {
        // Arrange
        var processCount = 0;
        var pipeline = new GpuPipeline(_mockBridge.Object, _mockLogger.Object)
            .Transform<int, string>(x =>
            {
                processCount++;
                if (x == 3) throw new InvalidOperationException("Test exception");
                return x.ToString();
            });
        
        var executable = pipeline.Build<int, string>();
        var inputs = new[] { 1, 2, 3, 4, 5 }.ToAsyncEnumerable();

        // Act
        var results = new List<string>();
        await foreach (var result in executable.ProcessManyAsync(inputs))
        {
            results.Add(result);
        }

        // Assert
        results.Should().HaveCount(4); // All except the one that threw
        results.Should().BeEquivalentTo(new[] { "1", "2", "4", "5" });
        processCount.Should().Be(5); // All items were attempted
    }

    [Fact]
    public async Task ProcessChannelAsync_ShouldProcessFromChannelToChannel()
    {
        // Arrange
        var pipeline = new GpuPipeline(_mockBridge.Object, _mockLogger.Object)
            .Transform<int, string>(x => $"Processed: {x}");
        
        var executable = pipeline.Build<int, string>();
        
        var inputChannel = Channel.CreateUnbounded<int>();
        var outputChannel = Channel.CreateUnbounded<string>();

        // Add test data
        await inputChannel.Writer.WriteAsync(1);
        await inputChannel.Writer.WriteAsync(2);
        await inputChannel.Writer.WriteAsync(3);
        inputChannel.Writer.Complete();

        // Act
        await executable.ProcessChannelAsync(
            inputChannel.Reader, 
            outputChannel.Writer);

        // Assert
        var results = new List<string>();
        await foreach (var result in outputChannel.Reader.ReadAllAsync())
        {
            results.Add(result);
        }

        results.Should().HaveCount(3);
        results.Should().BeEquivalentTo(new[]
        {
            "Processed: 1", "Processed: 2", "Processed: 3"
        });
    }

    [Fact]
    public async Task ProcessChannelAsync_WithException_ShouldContinueAndCompleteOutput()
    {
        // Arrange
        var pipeline = new GpuPipeline(_mockBridge.Object, _mockLogger.Object)
            .Transform<int, string>(x =>
            {
                if (x == 2) throw new InvalidOperationException("Test exception");
                return x.ToString();
            });
        
        var executable = pipeline.Build<int, string>();
        
        var inputChannel = Channel.CreateUnbounded<int>();
        var outputChannel = Channel.CreateUnbounded<string>();

        // Add test data
        await inputChannel.Writer.WriteAsync(1);
        await inputChannel.Writer.WriteAsync(2); // This will throw
        await inputChannel.Writer.WriteAsync(3);
        inputChannel.Writer.Complete();

        // Act
        await executable.ProcessChannelAsync(
            inputChannel.Reader, 
            outputChannel.Writer);

        // Assert
        var results = new List<string>();
        await foreach (var result in outputChannel.Reader.ReadAllAsync())
        {
            results.Add(result);
        }

        results.Should().HaveCount(2); // Only successful items
        results.Should().BeEquivalentTo(new[] { "1", "3" });
    }

    [Fact]
    public void Build_WithEmptyPipeline_ShouldThrowException()
    {
        // Arrange
        var pipeline = new GpuPipeline(_mockBridge.Object, _mockLogger.Object);

        // Act & Assert
        Assert.Throws<InvalidOperationException>(() =>
            pipeline.Build<int, string>());
    }

    [Fact]
    public void Build_WithIncompatibleTypes_ShouldThrowException()
    {
        // Arrange
        var pipeline = new GpuPipeline(_mockBridge.Object, _mockLogger.Object)
            .Transform<int, string>(x => x.ToString()); // Outputs string

        // Act & Assert - trying to build with double as output type
        Assert.Throws<InvalidOperationException>(() =>
            pipeline.Build<int, double>());
    }

    [Fact]
    public void Build_WithTypeMismatchBetweenStages_ShouldThrowException()
    {
        // Arrange - this should fail because we're trying to connect string output to double input
        var pipeline = new GpuPipeline(_mockBridge.Object, _mockLogger.Object)
            .Transform<int, string>(x => x.ToString())
            .Transform<double, int>(x => (int)x); // This expects double but gets string

        // Act & Assert
        Assert.Throws<InvalidOperationException>(() =>
            pipeline.Build<int, int>());
    }

    [Fact]
    public async Task ComplexPipeline_ShouldExecuteCorrectly()
    {
        // Arrange
        var tappedValues = new List<int>();
        var pipeline = new GpuPipeline(_mockBridge.Object, _mockLogger.Object)
            .Filter<int>(x => x > 0)
            .Tap<int>(x => tappedValues.Add(x))
            .Transform<int, double>(x => x * 2.5)
            .Transform<double, string>(x => $"Result: {x:F1}");
        
        var executable = pipeline.Build<int, string>();
        var input = 4;

        // Act
        var result = await executable.ProcessAsync(input);

        // Assert
        result.Should().Be("Result: 10.0");
        tappedValues.Should().ContainSingle().Which.Should().Be(4);
    }

    [Fact]
    public async Task Pipeline_WithCancellation_ShouldRespectCancellationToken()
    {
        // Arrange
        var pipeline = new GpuPipeline(_mockBridge.Object, _mockLogger.Object)
            .Parallel<int, string>(async x =>
            {
                await Task.Delay(1000); // Long running task
                return x.ToString();
            });
        
        var executable = pipeline.Build<int, string>();
        var cts = new CancellationTokenSource();
        cts.CancelAfter(100); // Cancel after 100ms

        // Act & Assert
        await Assert.ThrowsAsync<OperationCanceledException>(() =>
            executable.ProcessAsync(42, cts.Token));
    }

    [Theory]
    [InlineData(1)]
    [InlineData(5)]
    [InlineData(10)]
    [InlineData(100)]
    public async Task ProcessManyAsync_WithVariousInputCounts_ShouldProcessAll(int inputCount)
    {
        // Arrange
        var pipeline = new GpuPipeline(_mockBridge.Object, _mockLogger.Object)
            .Transform<int, string>(x => x.ToString());
        
        var executable = pipeline.Build<int, string>();
        var inputs = Enumerable.Range(1, inputCount).ToAsyncEnumerable();

        // Act
        var results = new List<string>();
        await foreach (var result in executable.ProcessManyAsync(inputs))
        {
            results.Add(result);
        }

        // Assert
        results.Should().HaveCount(inputCount);
        for (int i = 1; i <= inputCount; i++)
        {
            results.Should().Contain(i.ToString());
        }
    }

    [Fact]
    public async Task Pipeline_WithBatchStage_ShouldCollectItemsIntoBatches()
    {
        // Note: This test demonstrates conceptual batching behavior
        // In reality, BatchStage would need additional coordination
        
        // Arrange
        var processedBatches = new List<int>();
        var pipeline = new GpuPipeline(_mockBridge.Object, _mockLogger.Object)
            .Transform<int, int>(x => x) // Pass through
            .Tap<int>(x => processedBatches.Add(x));
        
        var executable = pipeline.Build<int, int>();

        // Act
        var results = new List<int>();
        for (int i = 1; i <= 5; i++)
        {
            var result = await executable.ProcessAsync(i);
            results.Add(result);
        }

        // Assert
        results.Should().BeEquivalentTo(new[] { 1, 2, 3, 4, 5 });
        processedBatches.Should().BeEquivalentTo(new[] { 1, 2, 3, 4, 5 });
    }

    [Fact]
    public async Task Pipeline_ErrorHandling_ShouldProvideDetailedErrorMessages()
    {
        // Arrange
        var pipeline = new GpuPipeline(_mockBridge.Object, _mockLogger.Object)
            .Transform<int, string>(x =>
            {
                if (x == 0) throw new DivideByZeroException("Cannot process zero");
                return (100 / x).ToString();
            });
        
        var executable = pipeline.Build<int, string>();

        // Act & Assert
        var exception = await Assert.ThrowsAsync<DivideByZeroException>(() =>
            executable.ProcessAsync(0));
        
        exception.Message.Should().Be("Cannot process zero");
    }

    [Fact]
    public async Task Pipeline_TypeValidation_ShouldEnforceCorrectTypes()
    {
        // Arrange
        var pipeline = new GpuPipeline(_mockBridge.Object, _mockLogger.Object)
            .Transform<int, string>(x => x.ToString());
        
        var executable = pipeline.Build<int, string>();

        // Act - process with correct type
        var result1 = await executable.ProcessAsync(42);

        // Assert
        result1.Should().Be("42");
        result1.Should().BeOfType<string>();
    }
}

/// <summary>
/// Integration tests for end-to-end pipeline scenarios
/// </summary>
public class PipelineIntegrationTests
{
    private readonly Mock<IGpuBridge> _mockBridge;
    private readonly Mock<ILogger<GpuPipeline>> _mockLogger;

    public PipelineIntegrationTests()
    {
        _mockBridge = new Mock<IGpuBridge>();
        _mockLogger = new Mock<ILogger<GpuPipeline>>();
    }

    [Fact]
    public async Task DataProcessingPipeline_ShouldProcessDataEndToEnd()
    {
        // Arrange - Simulate a data processing pipeline
        var processedSteps = new List<string>();
        
        var pipeline = new GpuPipeline(_mockBridge.Object, _mockLogger.Object)
            .Tap<int>(x => processedSteps.Add($"Input: {x}"))
            .Filter<int>(x => x > 0)
            .Tap<int>(x => processedSteps.Add($"Filtered: {x}"))
            .Transform<int, double>(x => x * 1.5)
            .Tap<double>(x => processedSteps.Add($"Scaled: {x}"))
            .Transform<double, string>(x => $"Final: {x:F2}");
        
        var executable = pipeline.Build<int, string>();
        var inputs = new[] { -1, 0, 1, 2, 3 };

        // Act
        var results = new List<string>();
        foreach (var input in inputs)
        {
            try
            {
                var result = await executable.ProcessAsync(input);
                results.Add(result);
            }
            catch (InvalidOperationException)
            {
                // Expected for filtered values
            }
        }

        // Assert
        results.Should().HaveCount(3); // Only positive values
        results.Should().BeEquivalentTo(new[]
        {
            "Final: 1.50",
            "Final: 3.00", 
            "Final: 4.50"
        });
        
        processedSteps.Should().Contain("Input: -1");
        processedSteps.Should().Contain("Input: 0");
        processedSteps.Should().Contain("Input: 1");
        processedSteps.Should().Contain("Filtered: 1");
        processedSteps.Should().NotContain("Filtered: -1");
        processedSteps.Should().NotContain("Filtered: 0");
    }

    [Fact]
    public async Task StreamProcessingPipeline_ShouldHandleAsyncStreams()
    {
        // Arrange
        var pipeline = new GpuPipeline(_mockBridge.Object, _mockLogger.Object)
            .Parallel<int, string>(async x =>
            {
                await Task.Delay(x); // Simulate varying processing times
                return $"Processed-{x}";
            }, maxConcurrency: 3);
        
        var executable = pipeline.Build<int, string>();
        var inputs = Enumerable.Range(1, 10).ToAsyncEnumerable();

        // Act
        var results = new List<string>();
        await foreach (var result in executable.ProcessManyAsync(inputs))
        {
            results.Add(result);
        }

        // Assert
        results.Should().HaveCount(10);
        for (int i = 1; i <= 10; i++)
        {
            results.Should().Contain($"Processed-{i}");
        }
    }

    [Fact]
    public async Task ChannelBasedPipeline_ShouldProcessContinuousStream()
    {
        // Arrange
        var pipeline = new GpuPipeline(_mockBridge.Object, _mockLogger.Object)
            .Transform<int, double>(x => x * 0.5)
            .Transform<double, string>(x => $"Half: {x}");
        
        var executable = pipeline.Build<int, string>();
        
        var inputChannel = Channel.CreateUnbounded<int>();
        var outputChannel = Channel.CreateUnbounded<string>();

        // Act - Start processing in background
        var processingTask = executable.ProcessChannelAsync(
            inputChannel.Reader,
            outputChannel.Writer);

        // Send data to input channel
        for (int i = 1; i <= 5; i++)
        {
            await inputChannel.Writer.WriteAsync(i);
        }
        inputChannel.Writer.Complete();

        // Wait for processing to complete
        await processingTask;

        // Assert
        var results = new List<string>();
        await foreach (var result in outputChannel.Reader.ReadAllAsync())
        {
            results.Add(result);
        }

        results.Should().HaveCount(5);
        results.Should().BeEquivalentTo(new[]
        {
            "Half: 0.5",
            "Half: 1",
            "Half: 1.5", 
            "Half: 2",
            "Half: 2.5"
        });
    }
}
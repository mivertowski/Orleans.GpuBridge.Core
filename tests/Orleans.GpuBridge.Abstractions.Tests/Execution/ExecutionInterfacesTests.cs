using Orleans.GpuBridge.Abstractions.Models;
using Orleans.GpuBridge.Abstractions.Providers.Execution.Interfaces;
using Orleans.GpuBridge.Abstractions.Providers.Execution.Parameters;
using Orleans.GpuBridge.Abstractions.Providers.Execution.Results;
using Orleans.GpuBridge.Abstractions.Providers.Execution.Results.Statistics;

namespace Orleans.GpuBridge.Abstractions.Tests.Execution;

/// <summary>
/// Tests for execution-related interfaces: IKernelExecutor, IKernelExecution, IKernelGraph
/// </summary>
public class ExecutionInterfacesTests
{
    #region IKernelExecutor Tests

    [Fact]
    public async Task IKernelExecutor_ExecuteAsync_WithValidKernel_ReturnsResult()
    {
        // Arrange
        var mockExecutor = new Mock<IKernelExecutor>();
        var kernel = new CompiledKernel
        {
            KernelId = "test-kernel",
            Name = "TestKernel",
            CompiledCode = new byte[] { 0x01, 0x02 }
        };
        var parameters = new KernelExecutionParameters
        {
            GlobalWorkSize = new[] { 1024 },
            LocalWorkSize = new[] { 256 }
        };
        var expectedResult = new KernelExecutionResult(
            Success: true,
            ErrorMessage: null,
            Timing: new KernelTiming(TimeSpan.FromMilliseconds(0.5), TimeSpan.FromMilliseconds(1.0), TimeSpan.FromMilliseconds(1.5))
        );

        mockExecutor
            .Setup(e => e.ExecuteAsync(kernel, parameters, default))
            .ReturnsAsync(expectedResult);

        // Act
        var result = await mockExecutor.Object.ExecuteAsync(kernel, parameters);

        // Assert
        result.Should().Be(expectedResult);
        result.Success.Should().BeTrue();
        result.Timing!.TotalTime.TotalMilliseconds.Should().Be(1.5);
    }

    [Fact]
    public async Task IKernelExecutor_ExecuteAsync_WithCancellation_PropagatesToken()
    {
        // Arrange
        var mockExecutor = new Mock<IKernelExecutor>();
        var kernel = new CompiledKernel { KernelId = "test", CompiledCode = Array.Empty<byte>() };
        var parameters = new KernelExecutionParameters { GlobalWorkSize = new[] { 1 } };
        var cts = new CancellationTokenSource();

        mockExecutor
            .Setup(e => e.ExecuteAsync(kernel, parameters, cts.Token))
            .ThrowsAsync(new OperationCanceledException());

        // Act & Assert
        await Assert.ThrowsAsync<OperationCanceledException>(
            async () => await mockExecutor.Object.ExecuteAsync(kernel, parameters, cts.Token));
    }

    [Fact]
    public async Task IKernelExecutor_ExecuteAsyncNonBlocking_ReturnsExecution()
    {
        // Arrange
        var mockExecutor = new Mock<IKernelExecutor>();
        var mockExecution = new Mock<IKernelExecution>();
        var kernel = new CompiledKernel { KernelId = "test", CompiledCode = Array.Empty<byte>() };
        var parameters = new KernelExecutionParameters { GlobalWorkSize = new[] { 1 } };

        mockExecutor
            .Setup(e => e.ExecuteAsyncNonBlocking(kernel, parameters, default))
            .ReturnsAsync(mockExecution.Object);

        // Act
        var execution = await mockExecutor.Object.ExecuteAsyncNonBlocking(kernel, parameters);

        // Assert
        execution.Should().NotBeNull();
        execution.Should().Be(mockExecution.Object);
    }

    [Fact]
    public async Task IKernelExecutor_ExecuteBatchAsync_WithMultipleKernels_ReturnsResults()
    {
        // Arrange
        var mockExecutor = new Mock<IKernelExecutor>();
        var batch = new List<KernelBatchItem>
        {
            new KernelBatchItem(new CompiledKernel { KernelId = "k1", CompiledCode = Array.Empty<byte>() }, new KernelExecutionParameters { GlobalWorkSize = new[] { 1 } }),
            new KernelBatchItem(new CompiledKernel { KernelId = "k2", CompiledCode = Array.Empty<byte>() }, new KernelExecutionParameters { GlobalWorkSize = new[] { 2 } })
        };
        var options = new BatchExecutionOptions(ExecuteInParallel: true, MaxParallelism: 2);
        var expectedResult = new BatchExecutionResult(
            SuccessCount: 2,
            FailureCount: 0,
            Results: new List<KernelExecutionResult>
            {
                new KernelExecutionResult(true),
                new KernelExecutionResult(true)
            },
            TotalExecutionTime: TimeSpan.FromMilliseconds(3.5)
        );

        mockExecutor
            .Setup(e => e.ExecuteBatchAsync(batch, options, default))
            .ReturnsAsync(expectedResult);

        // Act
        var result = await mockExecutor.Object.ExecuteBatchAsync(batch, options);

        // Assert
        result.Should().Be(expectedResult);
        result.SuccessCount.Should().Be(2);
        result.FailureCount.Should().Be(0);
    }

    [Fact]
    public void IKernelExecutor_CreateGraph_ReturnsKernelGraph()
    {
        // Arrange
        var mockExecutor = new Mock<IKernelExecutor>();
        var mockGraph = new Mock<IKernelGraph>();

        mockExecutor
            .Setup(e => e.CreateGraph("test-graph"))
            .Returns(mockGraph.Object);

        // Act
        var graph = mockExecutor.Object.CreateGraph("test-graph");

        // Assert
        graph.Should().NotBeNull();
        graph.Should().Be(mockGraph.Object);
    }

    [Fact]
    public async Task IKernelExecutor_ProfileAsync_ReturnsPerformanceMetrics()
    {
        // Arrange
        var mockExecutor = new Mock<IKernelExecutor>();
        var kernel = new CompiledKernel { KernelId = "test", CompiledCode = Array.Empty<byte>() };
        var parameters = new KernelExecutionParameters { GlobalWorkSize = new[] { 1 } };
        var expectedProfile = new KernelProfile(
            AverageExecutionTime: TimeSpan.FromMilliseconds(1.2),
            MinExecutionTime: TimeSpan.FromMilliseconds(1.0),
            MaxExecutionTime: TimeSpan.FromMilliseconds(1.5),
            StandardDeviation: 0.1,
            MemoryBandwidthBytesPerSecond: 1000000,
            ComputeThroughputGFlops: 1.5,
            OptimalBlockSize: 256
        );

        mockExecutor
            .Setup(e => e.ProfileAsync(kernel, parameters, 100, default))
            .ReturnsAsync(expectedProfile);

        // Act
        var profile = await mockExecutor.Object.ProfileAsync(kernel, parameters, 100);

        // Assert
        profile.Should().Be(expectedProfile);
        profile.AverageExecutionTime.TotalMilliseconds.Should().Be(1.2);
    }

    [Fact]
    public void IKernelExecutor_GetStatistics_ReturnsExecutionStats()
    {
        // Arrange
        var mockExecutor = new Mock<IKernelExecutor>();
        var expectedStats = new ExecutionStatistics(
            TotalKernelsExecuted: 1000,
            TotalBatchesExecuted: 100,
            TotalGraphsExecuted: 10,
            TotalExecutionTime: TimeSpan.FromSeconds(250),
            AverageKernelTime: TimeSpan.FromMilliseconds(2.5),
            TotalBytesTransferred: 1024 * 1024,
            TotalErrors: 10,
            KernelExecutionCounts: new Dictionary<string, long> { ["test-kernel"] = 1000 }
        );

        mockExecutor
            .Setup(e => e.GetStatistics())
            .Returns(expectedStats);

        // Act
        var stats = mockExecutor.Object.GetStatistics();

        // Assert
        stats.Should().Be(expectedStats);
        stats.TotalKernelsExecuted.Should().Be(1000);
    }

    [Fact]
    public void IKernelExecutor_ResetStatistics_ClearsStats()
    {
        // Arrange
        var mockExecutor = new Mock<IKernelExecutor>();
        mockExecutor.Setup(e => e.ResetStatistics()).Verifiable();

        // Act
        mockExecutor.Object.ResetStatistics();

        // Assert
        mockExecutor.Verify(e => e.ResetStatistics(), Times.Once);
    }

    #endregion

    #region IKernelExecution Tests

    [Fact]
    public async Task IKernelExecution_WaitForCompletionAsync_CompletesSuccessfully()
    {
        // Arrange
        var mockExecution = new Mock<IKernelExecution>();
        var expectedResult = new KernelExecutionResult(true);

        mockExecution
            .Setup(e => e.WaitForCompletionAsync(default))
            .ReturnsAsync(expectedResult);

        // Act
        var result = await mockExecution.Object.WaitForCompletionAsync();

        // Assert
        result.Should().Be(expectedResult);
        result.Success.Should().BeTrue();
    }

    [Fact]
    public void IKernelExecution_IsComplete_ReflectsExecutionState()
    {
        // Arrange
        var mockExecution = new Mock<IKernelExecution>();
        mockExecution.Setup(e => e.IsComplete).Returns(true);

        // Act
        var isComplete = mockExecution.Object.IsComplete;

        // Assert
        isComplete.Should().BeTrue();
    }

    [Fact]
    public async Task IKernelExecution_WaitForCompletionAsync_WithCancellation_PropagatesToken()
    {
        // Arrange
        var mockExecution = new Mock<IKernelExecution>();
        var cts = new CancellationTokenSource();

        mockExecution
            .Setup(e => e.WaitForCompletionAsync(cts.Token))
            .ThrowsAsync(new OperationCanceledException());

        // Act & Assert
        await Assert.ThrowsAsync<OperationCanceledException>(
            async () => await mockExecution.Object.WaitForCompletionAsync(cts.Token));
    }

    #endregion

    #region IKernelGraph Tests

    [Fact]
    public void IKernelGraph_AddKernel_AddsKernelToGraph()
    {
        // Arrange
        var mockGraph = new Mock<IKernelGraph>();
        var mockNode = new Mock<IGraphNode>();

        mockGraph
            .Setup(g => g.AddKernel(It.IsAny<CompiledKernel>(), It.IsAny<KernelExecutionParameters>(), null))
            .Returns(mockNode.Object);

        var kernel = new CompiledKernel { KernelId = "test", CompiledCode = Array.Empty<byte>() };
        var parameters = new KernelExecutionParameters { GlobalWorkSize = new[] { 1 } };

        // Act
        var node = mockGraph.Object.AddKernel(kernel, parameters);

        // Assert
        node.Should().NotBeNull();
        node.Should().Be(mockNode.Object);
    }

    [Fact]
    public async Task IKernelGraph_CompileAsync_ReturnsCompiledGraph()
    {
        // Arrange
        var mockGraph = new Mock<IKernelGraph>();
        var mockCompiled = new Mock<ICompiledGraph>();

        mockGraph
            .Setup(g => g.CompileAsync(default))
            .ReturnsAsync(mockCompiled.Object);

        // Act
        var compiled = await mockGraph.Object.CompileAsync();

        // Assert
        compiled.Should().NotBeNull();
        compiled.Should().Be(mockCompiled.Object);
    }

    [Fact]
    public void IKernelGraph_Validate_ChecksGraphValidity()
    {
        // Arrange
        var mockGraph = new Mock<IKernelGraph>();
        var validationResult = new GraphValidationResult(
            IsValid: true,
            Errors: null,
            Warnings: null
        );

        mockGraph
            .Setup(g => g.Validate())
            .Returns(validationResult);

        // Act
        var result = mockGraph.Object.Validate();

        // Assert
        result.IsValid.Should().BeTrue();
        result.Errors.Should().BeNullOrEmpty();
    }

    #endregion

    #region ICompiledGraph Tests

    [Fact]
    public async Task ICompiledGraph_ExecuteAsync_RunsGraph()
    {
        // Arrange
        var mockCompiled = new Mock<ICompiledGraph>();
        var expectedResult = new GraphExecutionResult(
            Success: true,
            NodesExecuted: 5,
            ExecutionTime: TimeSpan.FromMilliseconds(10.5)
        );

        mockCompiled
            .Setup(g => g.ExecuteAsync(default))
            .ReturnsAsync(expectedResult);

        // Act
        var result = await mockCompiled.Object.ExecuteAsync();

        // Assert
        result.Should().Be(expectedResult);
        result.Success.Should().BeTrue();
        result.NodesExecuted.Should().Be(5);
    }

    [Fact]
    public void IKernelGraph_Validate_ReturnsValidationResult()
    {
        // Arrange
        var mockGraph = new Mock<IKernelGraph>();
        var validationResult = new GraphValidationResult(
            IsValid: true,
            Errors: null
        );

        mockGraph
            .Setup(g => g.Validate())
            .Returns(validationResult);

        // Act
        var result = mockGraph.Object.Validate();

        // Assert
        result.IsValid.Should().BeTrue();
        result.Errors.Should().BeNullOrEmpty();
    }

    #endregion

    #region Edge Cases

    [Fact]
    public async Task IKernelExecutor_ExecuteBatchAsync_WithEmptyBatch_ReturnsEmptyResult()
    {
        // Arrange
        var mockExecutor = new Mock<IKernelExecutor>();
        var emptyBatch = Array.Empty<KernelBatchItem>();
        var options = new BatchExecutionOptions();
        var expectedResult = new BatchExecutionResult(0, 0, Array.Empty<KernelExecutionResult>(), TimeSpan.Zero);

        mockExecutor
            .Setup(e => e.ExecuteBatchAsync(emptyBatch, options, default))
            .ReturnsAsync(expectedResult);

        // Act
        var result = await mockExecutor.Object.ExecuteBatchAsync(emptyBatch, options);

        // Assert
        result.SuccessCount.Should().Be(0);
        result.FailureCount.Should().Be(0);
    }

    [Fact]
    public async Task IKernelExecutor_ExecuteAsync_WithFailure_ReturnsFailedResult()
    {
        // Arrange
        var mockExecutor = new Mock<IKernelExecutor>();
        var kernel = new CompiledKernel { KernelId = "test", CompiledCode = Array.Empty<byte>() };
        var parameters = new KernelExecutionParameters { GlobalWorkSize = new[] { 1 } };
        var failedResult = new KernelExecutionResult(
            Success: false,
            ErrorMessage: "Execution failed",
            Timing: null
        );

        mockExecutor
            .Setup(e => e.ExecuteAsync(kernel, parameters, default))
            .ReturnsAsync(failedResult);

        // Act
        var result = await mockExecutor.Object.ExecuteAsync(kernel, parameters);

        // Assert
        result.Success.Should().BeFalse();
        result.ErrorMessage.Should().Be("Execution failed");
    }

    #endregion
}

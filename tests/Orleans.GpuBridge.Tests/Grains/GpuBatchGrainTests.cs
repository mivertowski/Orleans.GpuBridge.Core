using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using FluentAssertions;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Moq;
using Orleans;
using Orleans.GpuBridge.Abstractions;
using Orleans.GpuBridge.Grains;
using Orleans.GpuBridge.Runtime;
using Orleans.TestingHost;
using Xunit;

namespace Orleans.GpuBridge.Tests.Grains;

/// <summary>
/// Unit tests for GpuBatchGrain
/// </summary>
public class GpuBatchGrainTests : IClassFixture<ClusterFixture>
{
    private readonly ClusterFixture _fixture;
    private readonly Mock<IGpuBridge> _mockBridge;
    private readonly Mock<IGpuKernel<int, string>> _mockKernel;
    private readonly Mock<ILogger<GpuBatchGrain<int, string>>> _mockLogger;

    public GpuBatchGrainTests(ClusterFixture fixture)
    {
        _fixture = fixture;
        _mockBridge = new Mock<IGpuBridge>();
        _mockKernel = new Mock<IGpuKernel<int, string>>();
        _mockLogger = new Mock<ILogger<GpuBatchGrain<int, string>>>();
    }

    [Fact]
    public async Task ExecuteAsync_WithValidBatch_ShouldReturnSuccessResult()
    {
        // Arrange
        var kernelId = KernelId.Parse("test-kernel");
        var batch = new List<int> { 1, 2, 3, 4, 5 };
        var expectedResults = new List<string> { "1", "2", "3", "4", "5" };
        var handle = KernelHandle.Create();

        _mockBridge.Setup(b => b.GetKernelAsync<int, string>(kernelId, It.IsAny<CancellationToken>()))
                   .ReturnsAsync(_mockKernel.Object);

        _mockKernel.Setup(k => k.SubmitBatchAsync(batch, null, It.IsAny<CancellationToken>()))
                  .ReturnsAsync(handle);

        _mockKernel.Setup(k => k.ReadResultsAsync(handle, It.IsAny<CancellationToken>()))
                  .Returns(expectedResults.ToAsyncEnumerable());

        var grain = new GpuBatchGrain<int, string>(_mockLogger.Object);
        
        // Mock the service provider
        var serviceProviderMock = new Mock<IServiceProvider>();
        serviceProviderMock.Setup(sp => sp.GetRequiredService<IGpuBridge>())
                          .Returns(_mockBridge.Object);

        // Use reflection to set the service provider
        var serviceProviderField = typeof(Grain).GetProperty("ServiceProvider");
        serviceProviderField?.SetValue(grain, serviceProviderMock.Object);
        
        // Mock GetPrimaryKeyString method
        var grainMock = new Mock<GpuBatchGrain<int, string>>(_mockLogger.Object) { CallBase = true };
        grainMock.Setup(g => g.GetPrimaryKeyString()).Returns(kernelId.Value);

        // Act
        await grain.OnActivateAsync(CancellationToken.None);
        var result = await grain.ExecuteAsync(batch);

        // Assert
        result.Should().NotBeNull();
        result.Success.Should().BeTrue();
        result.Results.Should().BeEquivalentTo(expectedResults);
        result.KernelId.Should().Be(kernelId);
        result.HandleId.Should().Be(handle.Id);
        result.ExecutionTime.Should().BeGreaterThan(TimeSpan.Zero);
        result.Error.Should().BeNull();

        _mockKernel.Verify(k => k.SubmitBatchAsync(batch, null, It.IsAny<CancellationToken>()), Times.Once);
        _mockKernel.Verify(k => k.ReadResultsAsync(handle, It.IsAny<CancellationToken>()), Times.Once);
    }

    [Fact]
    public async Task ExecuteAsync_WithEmptyBatch_ShouldReturnEmptyResult()
    {
        // Arrange
        var kernelId = KernelId.Parse("test-kernel");
        var batch = new List<int>();
        var handle = KernelHandle.Create();

        _mockBridge.Setup(b => b.GetKernelAsync<int, string>(kernelId, It.IsAny<CancellationToken>()))
                   .ReturnsAsync(_mockKernel.Object);

        _mockKernel.Setup(k => k.SubmitBatchAsync(batch, null, It.IsAny<CancellationToken>()))
                  .ReturnsAsync(handle);

        _mockKernel.Setup(k => k.ReadResultsAsync(handle, It.IsAny<CancellationToken>()))
                  .Returns(System.Linq.AsyncEnumerable.Empty<string>());

        var grain = new GpuBatchGrain<int, string>(_mockLogger.Object);

        // Mock the service provider
        var serviceProviderMock = new Mock<IServiceProvider>();
        serviceProviderMock.Setup(sp => sp.GetRequiredService<IGpuBridge>())
                          .Returns(_mockBridge.Object);

        var serviceProviderField = typeof(Grain).GetProperty("ServiceProvider");
        serviceProviderField?.SetValue(grain, serviceProviderMock.Object);

        // Act
        await grain.OnActivateAsync(CancellationToken.None);
        var result = await grain.ExecuteAsync(batch);

        // Assert
        result.Should().NotBeNull();
        result.Success.Should().BeTrue();
        result.Results.Should().BeEmpty();
        result.KernelId.Should().Be(kernelId);
        result.ExecutionTime.Should().BeGreaterThan(TimeSpan.Zero);
    }

    [Fact]
    public async Task ExecuteAsync_WithKernelException_ShouldReturnFailureResult()
    {
        // Arrange
        var kernelId = KernelId.Parse("test-kernel");
        var batch = new List<int> { 1, 2, 3 };
        var expectedError = "GPU kernel execution failed";

        _mockBridge.Setup(b => b.GetKernelAsync<int, string>(kernelId, It.IsAny<CancellationToken>()))
                   .ReturnsAsync(_mockKernel.Object);

        _mockKernel.Setup(k => k.SubmitBatchAsync(batch, null, It.IsAny<CancellationToken>()))
                  .ThrowsAsync(new InvalidOperationException(expectedError));

        var grain = new GpuBatchGrain<int, string>(_mockLogger.Object);

        var serviceProviderMock = new Mock<IServiceProvider>();
        serviceProviderMock.Setup(sp => sp.GetRequiredService<IGpuBridge>())
                          .Returns(_mockBridge.Object);

        var serviceProviderField = typeof(Grain).GetProperty("ServiceProvider");
        serviceProviderField?.SetValue(grain, serviceProviderMock.Object);

        // Act
        await grain.OnActivateAsync(CancellationToken.None);
        var result = await grain.ExecuteAsync(batch);

        // Assert
        result.Should().NotBeNull();
        result.Success.Should().BeFalse();
        result.Results.Should().BeEmpty();
        result.Error.Should().Be(expectedError);
        result.ExecutionTime.Should().Be(TimeSpan.Zero);
    }

    [Fact]
    public async Task ExecuteAsync_WithHints_ShouldPassHintsToKernel()
    {
        // Arrange
        var kernelId = KernelId.Parse("test-kernel");
        var batch = new List<int> { 1, 2, 3 };
        var hints = new GpuExecutionHints { MaxMicroBatch = 2 };
        var handle = KernelHandle.Create();

        _mockBridge.Setup(b => b.GetKernelAsync<int, string>(kernelId, It.IsAny<CancellationToken>()))
                   .ReturnsAsync(_mockKernel.Object);

        _mockKernel.Setup(k => k.SubmitBatchAsync(batch, hints, It.IsAny<CancellationToken>()))
                  .ReturnsAsync(handle);

        _mockKernel.Setup(k => k.ReadResultsAsync(handle, It.IsAny<CancellationToken>()))
                  .Returns(new[] { "1", "2", "3" }.ToAsyncEnumerable());

        var grain = new GpuBatchGrain<int, string>(_mockLogger.Object);

        var serviceProviderMock = new Mock<IServiceProvider>();
        serviceProviderMock.Setup(sp => sp.GetRequiredService<IGpuBridge>())
                          .Returns(_mockBridge.Object);

        var serviceProviderField = typeof(Grain).GetProperty("ServiceProvider");
        serviceProviderField?.SetValue(grain, serviceProviderMock.Object);

        // Act
        await grain.OnActivateAsync(CancellationToken.None);
        var result = await grain.ExecuteAsync(batch, hints);

        // Assert
        result.Should().NotBeNull();
        result.Success.Should().BeTrue();
        _mockKernel.Verify(k => k.SubmitBatchAsync(batch, hints, It.IsAny<CancellationToken>()), Times.Once);
    }

    [Fact]
    public async Task ExecuteWithCallbackAsync_WithSuccessfulExecution_ShouldCallObserverMethods()
    {
        // Arrange
        var kernelId = KernelId.Parse("test-kernel");
        var batch = new List<int> { 1, 2, 3 };
        var expectedResults = new List<string> { "1", "2", "3" };
        var handle = KernelHandle.Create();
        var mockObserver = new Mock<IGpuResultObserver<string>>();

        _mockBridge.Setup(b => b.GetKernelAsync<int, string>(kernelId, It.IsAny<CancellationToken>()))
                   .ReturnsAsync(_mockKernel.Object);

        _mockKernel.Setup(k => k.SubmitBatchAsync(batch, null, It.IsAny<CancellationToken>()))
                  .ReturnsAsync(handle);

        _mockKernel.Setup(k => k.ReadResultsAsync(handle, It.IsAny<CancellationToken>()))
                  .Returns(expectedResults.ToAsyncEnumerable());

        var grain = new GpuBatchGrain<int, string>(_mockLogger.Object);

        var serviceProviderMock = new Mock<IServiceProvider>();
        serviceProviderMock.Setup(sp => sp.GetRequiredService<IGpuBridge>())
                          .Returns(_mockBridge.Object);

        var serviceProviderField = typeof(Grain).GetProperty("ServiceProvider");
        serviceProviderField?.SetValue(grain, serviceProviderMock.Object);

        // Act
        await grain.OnActivateAsync(CancellationToken.None);
        var result = await grain.ExecuteWithCallbackAsync(batch, mockObserver.Object);

        // Assert
        result.Should().NotBeNull();
        result.Success.Should().BeTrue();
        result.Results.Should().BeEquivalentTo(expectedResults);

        // Verify observer was called for each result
        foreach (var expectedResult in expectedResults)
        {
            mockObserver.Verify(o => o.OnNextAsync(expectedResult), Times.Once);
        }
        mockObserver.Verify(o => o.OnCompletedAsync(), Times.Once);
        mockObserver.Verify(o => o.OnErrorAsync(It.IsAny<Exception>()), Times.Never);
    }

    [Fact]
    public async Task ExecuteWithCallbackAsync_WithFailure_ShouldCallOnError()
    {
        // Arrange
        var kernelId = KernelId.Parse("test-kernel");
        var batch = new List<int> { 1, 2, 3 };
        var expectedError = "GPU execution failed";
        var mockObserver = new Mock<IGpuResultObserver<string>>();

        _mockBridge.Setup(b => b.GetKernelAsync<int, string>(kernelId, It.IsAny<CancellationToken>()))
                   .ReturnsAsync(_mockKernel.Object);

        _mockKernel.Setup(k => k.SubmitBatchAsync(batch, null, It.IsAny<CancellationToken>()))
                  .ThrowsAsync(new InvalidOperationException(expectedError));

        var grain = new GpuBatchGrain<int, string>(_mockLogger.Object);

        var serviceProviderMock = new Mock<IServiceProvider>();
        serviceProviderMock.Setup(sp => sp.GetRequiredService<IGpuBridge>())
                          .Returns(_mockBridge.Object);

        var serviceProviderField = typeof(Grain).GetProperty("ServiceProvider");
        serviceProviderField?.SetValue(grain, serviceProviderMock.Object);

        // Act
        await grain.OnActivateAsync(CancellationToken.None);
        var result = await grain.ExecuteWithCallbackAsync(batch, mockObserver.Object);

        // Assert
        result.Should().NotBeNull();
        result.Success.Should().BeFalse();
        result.Error.Should().Be(expectedError);

        mockObserver.Verify(o => o.OnErrorAsync(It.Is<Exception>(ex => ex.Message == expectedError)), Times.Once);
        mockObserver.Verify(o => o.OnNextAsync(It.IsAny<string>()), Times.Never);
        mockObserver.Verify(o => o.OnCompletedAsync(), Times.Never);
    }

    [Fact]
    public async Task ExecuteWithCallbackAsync_WithObserverException_ShouldPropagateException()
    {
        // Arrange
        var kernelId = KernelId.Parse("test-kernel");
        var batch = new List<int> { 1 };
        var expectedResults = new List<string> { "1" };
        var handle = KernelHandle.Create();
        var mockObserver = new Mock<IGpuResultObserver<string>>();
        var observerException = new InvalidOperationException("Observer failed");

        _mockBridge.Setup(b => b.GetKernelAsync<int, string>(kernelId, It.IsAny<CancellationToken>()))
                   .ReturnsAsync(_mockKernel.Object);

        _mockKernel.Setup(k => k.SubmitBatchAsync(batch, null, It.IsAny<CancellationToken>()))
                  .ReturnsAsync(handle);

        _mockKernel.Setup(k => k.ReadResultsAsync(handle, It.IsAny<CancellationToken>()))
                  .Returns(expectedResults.ToAsyncEnumerable());

        mockObserver.Setup(o => o.OnNextAsync("1"))
                   .ThrowsAsync(observerException);

        var grain = new GpuBatchGrain<int, string>(_mockLogger.Object);

        var serviceProviderMock = new Mock<IServiceProvider>();
        serviceProviderMock.Setup(sp => sp.GetRequiredService<IGpuBridge>())
                          .Returns(_mockBridge.Object);

        var serviceProviderField = typeof(Grain).GetProperty("ServiceProvider");
        serviceProviderField?.SetValue(grain, serviceProviderMock.Object);

        // Act & Assert
        await grain.OnActivateAsync(CancellationToken.None);
        await Assert.ThrowsAsync<InvalidOperationException>(
            () => grain.ExecuteWithCallbackAsync(batch, mockObserver.Object));

        mockObserver.Verify(o => o.OnErrorAsync(observerException), Times.Once);
    }

    [Fact]
    public void GpuBatchResult_Success_ShouldReturnTrueWhenNoError()
    {
        // Arrange
        var result = new GpuBatchResult<string>(
            new[] { "test" },
            TimeSpan.FromMilliseconds(100),
            "handle-123",
            KernelId.Parse("kernel-1"));

        // Act & Assert
        result.Success.Should().BeTrue();
    }

    [Fact]
    public void GpuBatchResult_Success_ShouldReturnFalseWhenHasError()
    {
        // Arrange
        var result = new GpuBatchResult<string>(
            Array.Empty<string>(),
            TimeSpan.Zero,
            "handle-123",
            KernelId.Parse("kernel-1"),
            "Some error occurred");

        // Act & Assert
        result.Success.Should().BeFalse();
    }

    [Theory]
    [InlineData(1)]
    [InlineData(10)]
    [InlineData(100)]
    [InlineData(1000)]
    public async Task ExecuteAsync_WithVariousBatchSizes_ShouldHandleCorrectly(int batchSize)
    {
        // Arrange
        var kernelId = KernelId.Parse("test-kernel");
        var batch = Enumerable.Range(1, batchSize).ToList();
        var expectedResults = batch.Select(x => x.ToString()).ToList();
        var handle = KernelHandle.Create();

        _mockBridge.Setup(b => b.GetKernelAsync<int, string>(kernelId, It.IsAny<CancellationToken>()))
                   .ReturnsAsync(_mockKernel.Object);

        _mockKernel.Setup(k => k.SubmitBatchAsync(batch, null, It.IsAny<CancellationToken>()))
                  .ReturnsAsync(handle);

        _mockKernel.Setup(k => k.ReadResultsAsync(handle, It.IsAny<CancellationToken>()))
                  .Returns(expectedResults.ToAsyncEnumerable());

        var grain = new GpuBatchGrain<int, string>(_mockLogger.Object);

        var serviceProviderMock = new Mock<IServiceProvider>();
        serviceProviderMock.Setup(sp => sp.GetRequiredService<IGpuBridge>())
                          .Returns(_mockBridge.Object);

        var serviceProviderField = typeof(Grain).GetProperty("ServiceProvider");
        serviceProviderField?.SetValue(grain, serviceProviderMock.Object);

        // Act
        await grain.OnActivateAsync(CancellationToken.None);
        var result = await grain.ExecuteAsync(batch);

        // Assert
        result.Should().NotBeNull();
        result.Success.Should().BeTrue();
        result.Results.Should().HaveCount(batchSize);
        result.Results.Should().BeEquivalentTo(expectedResults);
    }
}

/// <summary>
/// Test fixture for Orleans cluster
/// </summary>
public class ClusterFixture : IDisposable
{
    public TestCluster Cluster { get; private set; }

    public ClusterFixture()
    {
        var builder = new TestClusterBuilder();
        builder.AddSiloBuilderConfigurator<TestSiloConfigurations>();
        Cluster = builder.Build();
        Cluster.Deploy();
    }

    public void Dispose()
    {
        Cluster?.StopAllSilos();
    }
}

/// <summary>
/// Test silo configurations
/// </summary>
public class TestSiloConfigurations : ISiloConfigurator
{
    public void Configure(ISiloBuilder siloBuilder)
    {
        siloBuilder.ConfigureServices(services =>
        {
            // Add test services if needed
        });
    }
}

/// <summary>
/// Extension methods for testing
/// </summary>
public static class TestExtensions
{
    public static IAsyncEnumerable<T> ToAsyncEnumerable<T>(this IEnumerable<T> source)
    {
        return source.ToAsyncEnumerable();
    }
}
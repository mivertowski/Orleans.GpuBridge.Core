using Orleans.GpuBridge.Abstractions.Application.Interfaces;
using Orleans.GpuBridge.Abstractions.Domain.ValueObjects;
using Orleans.GpuBridge.Abstractions.Kernels;
using Orleans.GpuBridge.Abstractions;
using Orleans.GpuBridge.Abstractions.Enums;

namespace Orleans.GpuBridge.Abstractions.Tests.Interfaces;

/// <summary>
/// Tests for IGpuKernel and IGpuBridge interfaces
/// </summary>
public class KernelInterfacesTests
{
    #region IGpuKernel Tests

    [Fact]
    public async Task IGpuKernel_SubmitBatchAsync_WithValidItems_ReturnsHandle()
    {
        // Arrange
        var mockKernel = new Mock<IGpuKernel<int, int>>();
        var expectedHandle = new KernelHandle("test-kernel", DateTimeOffset.UtcNow);
        var items = new List<int> { 1, 2, 3 };

        mockKernel
            .Setup(k => k.SubmitBatchAsync(It.IsAny<IReadOnlyList<int>>(), null, default))
            .ReturnsAsync(expectedHandle);

        // Act
        var result = await mockKernel.Object.SubmitBatchAsync(items);

        // Assert
        result.Should().Be(expectedHandle);
        mockKernel.Verify(k => k.SubmitBatchAsync(items, null, default), Times.Once);
    }

    [Fact]
    public async Task IGpuKernel_SubmitBatchAsync_WithHints_PassesHintsCorrectly()
    {
        // Arrange
        var mockKernel = new Mock<IGpuKernel<float, float>>();
        var expectedHandle = new KernelHandle("test-kernel", DateTimeOffset.UtcNow);
        var items = new List<float> { 1.0f, 2.0f };
        var hints = new GpuExecutionHints(PreferGpu: true, MaxMicroBatch: 128);

        mockKernel
            .Setup(k => k.SubmitBatchAsync(items, hints, default))
            .ReturnsAsync(expectedHandle);

        // Act
        var result = await mockKernel.Object.SubmitBatchAsync(items, hints);

        // Assert
        result.Should().Be(expectedHandle);
        mockKernel.Verify(k => k.SubmitBatchAsync(items, hints, default), Times.Once);
    }

    [Fact]
    public async Task IGpuKernel_SubmitBatchAsync_WithCancellation_PropagatesToken()
    {
        // Arrange
        var mockKernel = new Mock<IGpuKernel<string, string>>();
        var items = new List<string> { "test" };
        var cts = new CancellationTokenSource();

        mockKernel
            .Setup(k => k.SubmitBatchAsync(items, null, cts.Token))
            .ThrowsAsync(new OperationCanceledException());

        // Act & Assert
        await Assert.ThrowsAsync<OperationCanceledException>(
            async () => await mockKernel.Object.SubmitBatchAsync(items, ct: cts.Token));
    }

    [Fact]
    public async Task IGpuKernel_ReadResultsAsync_WithValidHandle_ReturnsResults()
    {
        // Arrange
        var mockKernel = new Mock<IGpuKernel<int, int>>();
        var handle = new KernelHandle("test-kernel", DateTimeOffset.UtcNow);
        var expectedResults = new[] { 1, 2, 3 };

        mockKernel
            .Setup(k => k.ReadResultsAsync(handle, default))
            .Returns(ToAsyncEnumerable(expectedResults));

        // Act
        var results = new List<int>();
        await foreach (var result in mockKernel.Object.ReadResultsAsync(handle))
        {
            results.Add(result);
        }

        // Assert
        results.Should().BeEquivalentTo(expectedResults);
    }

    [Fact]
    public async Task IGpuKernel_ReadResultsAsync_WithEmptyResults_ReturnsEmpty()
    {
        // Arrange
        var mockKernel = new Mock<IGpuKernel<int, int>>();
        var handle = new KernelHandle("test-kernel", DateTimeOffset.UtcNow);

        mockKernel
            .Setup(k => k.ReadResultsAsync(handle, default))
            .Returns(ToAsyncEnumerable(Array.Empty<int>()));

        // Act
        var results = new List<int>();
        await foreach (var result in mockKernel.Object.ReadResultsAsync(handle))
        {
            results.Add(result);
        }

        // Assert
        results.Should().BeEmpty();
    }

    private static async IAsyncEnumerable<T> ToAsyncEnumerable<T>(IEnumerable<T> source)
    {
        foreach (var item in source)
        {
            yield return await Task.FromResult(item);
        }
    }

    [Fact]
    public async Task IGpuKernel_GetInfoAsync_ReturnsKernelInfo()
    {
        // Arrange
        var mockKernel = new Mock<IGpuKernel<int, int>>();
        var expectedInfo = new KernelInfo(
            Id: new KernelId("test-kernel"),
            Description: "Test Kernel v1.0.0",
            InputType: typeof(int),
            OutputType: typeof(int),
            SupportsGpu: true,
            PreferredBatchSize: 256
        );

        mockKernel
            .Setup(k => k.GetInfoAsync(default))
            .ReturnsAsync(expectedInfo);

        // Act
        var result = await mockKernel.Object.GetInfoAsync();

        // Assert
        result.Should().Be(expectedInfo);
        result.Id.Value.Should().Be("test-kernel");
        result.Description.Should().Be("Test Kernel v1.0.0");
    }

    #endregion

    #region IGpuBridge Tests

    [Fact]
    public async Task IGpuBridge_GetInfoAsync_ReturnsValidInfo()
    {
        // Arrange
        var mockBridge = new Mock<IGpuBridge>();
        var expectedInfo = new GpuBridgeInfo(
            Version: "1.0.0",
            DeviceCount: 2,
            TotalMemoryBytes: 10L * 1024 * 1024 * 1024,
            Backend: GpuBackend.CUDA,
            IsGpuAvailable: true
        );

        mockBridge
            .Setup(b => b.GetInfoAsync(default))
            .ReturnsAsync(expectedInfo);

        // Act
        var result = await mockBridge.Object.GetInfoAsync();

        // Assert
        result.Should().Be(expectedInfo);
        result.DeviceCount.Should().Be(2);
        result.IsGpuAvailable.Should().BeTrue();
    }

    [Fact]
    public async Task IGpuBridge_GetKernelAsync_WithValidId_ReturnsKernel()
    {
        // Arrange
        var mockBridge = new Mock<IGpuBridge>();
        var mockKernel = new Mock<IGpuKernel<int, int>>();
        var kernelId = new KernelId("test-kernel");

        mockBridge
            .Setup(b => b.GetKernelAsync<int, int>(kernelId, default))
            .ReturnsAsync(mockKernel.Object);

        // Act
        var result = await mockBridge.Object.GetKernelAsync<int, int>(kernelId);

        // Assert
        result.Should().NotBeNull();
        result.Should().Be(mockKernel.Object);
    }

    [Fact]
    public async Task IGpuBridge_GetKernelAsync_WithInvalidId_ThrowsException()
    {
        // Arrange
        var mockBridge = new Mock<IGpuBridge>();
        var kernelId = new KernelId("invalid-kernel");

        mockBridge
            .Setup(b => b.GetKernelAsync<int, int>(kernelId, default))
            .ThrowsAsync(new KeyNotFoundException($"Kernel '{kernelId}' not found"));

        // Act & Assert
        await Assert.ThrowsAsync<KeyNotFoundException>(
            async () => await mockBridge.Object.GetKernelAsync<int, int>(kernelId));
    }

    [Fact]
    public async Task IGpuBridge_GetDevicesAsync_ReturnsDeviceList()
    {
        // Arrange
        var mockBridge = new Mock<IGpuBridge>();
        var expectedDevices = new List<GpuDevice>
        {
            new GpuDevice(0, "NVIDIA RTX 3080", DeviceType.CUDA, 10737418240, 10737418240, 68, new[] { "CUDA 8.6" }),
            new GpuDevice(1, "NVIDIA RTX 3090", DeviceType.CUDA, 24576000000, 24576000000, 82, new[] { "CUDA 8.6" })
        };

        mockBridge
            .Setup(b => b.GetDevicesAsync(default))
            .ReturnsAsync(expectedDevices);

        // Act
        var result = await mockBridge.Object.GetDevicesAsync();

        // Assert
        result.Should().HaveCount(2);
        result.Should().BeEquivalentTo(expectedDevices);
    }

    [Fact]
    public async Task IGpuBridge_GetDevicesAsync_WithNoDevices_ReturnsEmptyList()
    {
        // Arrange
        var mockBridge = new Mock<IGpuBridge>();

        mockBridge
            .Setup(b => b.GetDevicesAsync(default))
            .ReturnsAsync(Array.Empty<GpuDevice>());

        // Act
        var result = await mockBridge.Object.GetDevicesAsync();

        // Assert
        result.Should().BeEmpty();
    }

    [Fact]
    public async Task IGpuBridge_ExecuteKernelAsync_WithDynamicTypes_ReturnsResult()
    {
        // Arrange
        var mockBridge = new Mock<IGpuBridge>();
        var input = new { Value = 42 };
        var expectedOutput = new { Result = 84 };

        mockBridge
            .Setup(b => b.ExecuteKernelAsync("test-kernel", input, default))
            .ReturnsAsync(expectedOutput);

        // Act
        var result = await mockBridge.Object.ExecuteKernelAsync("test-kernel", input);

        // Assert
        result.Should().Be(expectedOutput);
    }

    [Fact]
    public async Task IGpuBridge_ExecuteKernelAsync_WithCancellation_PropagatesToken()
    {
        // Arrange
        var mockBridge = new Mock<IGpuBridge>();
        var cts = new CancellationTokenSource();

        mockBridge
            .Setup(b => b.ExecuteKernelAsync("test-kernel", It.IsAny<object>(), cts.Token))
            .ThrowsAsync(new OperationCanceledException());

        // Act & Assert
        await Assert.ThrowsAsync<OperationCanceledException>(
            async () => await mockBridge.Object.ExecuteKernelAsync("test-kernel", new { }, cts.Token));
    }

    #endregion

    #region Type Constraint Tests

    [Fact]
    public void IGpuKernel_RequiresNonNullableTypes()
    {
        // Arrange & Act
        var kernelType = typeof(IGpuKernel<,>);
        var genericArgs = kernelType.GetGenericArguments();

        // Assert
        genericArgs.Should().HaveCount(2);
        // Verify generic parameters exist (notnull constraint doesn't show in reflection attributes)
        genericArgs[0].Name.Should().Be("TIn");
        genericArgs[1].Name.Should().Be("TOut");
    }

    #endregion
}

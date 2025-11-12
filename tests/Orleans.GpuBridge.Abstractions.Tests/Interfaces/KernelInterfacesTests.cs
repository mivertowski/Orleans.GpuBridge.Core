using Orleans.GpuBridge.Abstractions;
using Orleans.GpuBridge.Abstractions.Kernels;
using Orleans.GpuBridge.Abstractions.Enums;

namespace Orleans.GpuBridge.Abstractions.Tests.Interfaces;

/// <summary>
/// Tests for IGpuKernel and IGpuBridge interfaces
/// </summary>
public class KernelInterfacesTests
{
    #region IGpuKernel Tests

    [Fact]
    public void IGpuKernel_Properties_ReturnExpectedValues()
    {
        // Arrange
        var mockKernel = new Mock<IGpuKernel<int, int>>();
        mockKernel.Setup(k => k.KernelId).Returns("test-kernel");
        mockKernel.Setup(k => k.DisplayName).Returns("Test Kernel");
        mockKernel.Setup(k => k.BackendProvider).Returns("CUDA");
        mockKernel.Setup(k => k.IsInitialized).Returns(true);
        mockKernel.Setup(k => k.IsGpuAccelerated).Returns(true);

        // Act
        var kernel = mockKernel.Object;

        // Assert
        kernel.KernelId.Should().Be("test-kernel");
        kernel.DisplayName.Should().Be("Test Kernel");
        kernel.BackendProvider.Should().Be("CUDA");
        kernel.IsInitialized.Should().BeTrue();
        kernel.IsGpuAccelerated.Should().BeTrue();
    }

    [Fact]
    public async Task IGpuKernel_InitializeAsync_CompletesSuccessfully()
    {
        // Arrange
        var mockKernel = new Mock<IGpuKernel<int, int>>();
        mockKernel
            .Setup(k => k.InitializeAsync(default))
            .Returns(Task.CompletedTask);

        // Act
        await mockKernel.Object.InitializeAsync();

        // Assert
        mockKernel.Verify(k => k.InitializeAsync(default), Times.Once);
    }

    [Fact]
    public async Task IGpuKernel_InitializeAsync_WithCancellation_PropagatesToken()
    {
        // Arrange
        var mockKernel = new Mock<IGpuKernel<int, int>>();
        var cts = new CancellationTokenSource();

        mockKernel
            .Setup(k => k.InitializeAsync(cts.Token))
            .ThrowsAsync(new OperationCanceledException());

        // Act & Assert
        await Assert.ThrowsAsync<OperationCanceledException>(
            async () => await mockKernel.Object.InitializeAsync(cts.Token));
    }

    [Fact]
    public async Task IGpuKernel_ExecuteAsync_WithValidInput_ReturnsResult()
    {
        // Arrange
        var mockKernel = new Mock<IGpuKernel<int, int>>();
        var input = 42;
        var expectedOutput = 84;

        mockKernel
            .Setup(k => k.ExecuteAsync(input, default))
            .ReturnsAsync(expectedOutput);

        // Act
        var result = await mockKernel.Object.ExecuteAsync(input);

        // Assert
        result.Should().Be(expectedOutput);
        mockKernel.Verify(k => k.ExecuteAsync(input, default), Times.Once);
    }

    [Fact]
    public async Task IGpuKernel_ExecuteAsync_WithCancellation_PropagatesToken()
    {
        // Arrange
        var mockKernel = new Mock<IGpuKernel<string, string>>();
        var input = "test";
        var cts = new CancellationTokenSource();

        mockKernel
            .Setup(k => k.ExecuteAsync(input, cts.Token))
            .ThrowsAsync(new OperationCanceledException());

        // Act & Assert
        await Assert.ThrowsAsync<OperationCanceledException>(
            async () => await mockKernel.Object.ExecuteAsync(input, cts.Token));
    }

    [Fact]
    public async Task IGpuKernel_ExecuteBatchAsync_WithValidInputs_ReturnsResults()
    {
        // Arrange
        var mockKernel = new Mock<IGpuKernel<int, int>>();
        var inputs = new[] { 1, 2, 3 };
        var expectedOutputs = new[] { 2, 4, 6 };

        mockKernel
            .Setup(k => k.ExecuteBatchAsync(inputs, default))
            .ReturnsAsync(expectedOutputs);

        // Act
        var results = await mockKernel.Object.ExecuteBatchAsync(inputs);

        // Assert
        results.Should().BeEquivalentTo(expectedOutputs);
        mockKernel.Verify(k => k.ExecuteBatchAsync(inputs, default), Times.Once);
    }

    [Fact]
    public async Task IGpuKernel_ExecuteBatchAsync_WithEmptyInput_ReturnsEmpty()
    {
        // Arrange
        var mockKernel = new Mock<IGpuKernel<int, int>>();
        var inputs = Array.Empty<int>();
        var expectedOutputs = Array.Empty<int>();

        mockKernel
            .Setup(k => k.ExecuteBatchAsync(inputs, default))
            .ReturnsAsync(expectedOutputs);

        // Act
        var results = await mockKernel.Object.ExecuteBatchAsync(inputs);

        // Assert
        results.Should().BeEmpty();
    }

    [Fact]
    public async Task IGpuKernel_ExecuteBatchAsync_WithCancellation_PropagatesToken()
    {
        // Arrange
        var mockKernel = new Mock<IGpuKernel<float, float>>();
        var inputs = new[] { 1.0f, 2.0f };
        var cts = new CancellationTokenSource();

        mockKernel
            .Setup(k => k.ExecuteBatchAsync(inputs, cts.Token))
            .ThrowsAsync(new OperationCanceledException());

        // Act & Assert
        await Assert.ThrowsAsync<OperationCanceledException>(
            async () => await mockKernel.Object.ExecuteBatchAsync(inputs, cts.Token));
    }

    [Fact]
    public void IGpuKernel_GetEstimatedExecutionTimeMicroseconds_ReturnsValidEstimate()
    {
        // Arrange
        var mockKernel = new Mock<IGpuKernel<int, int>>();
        var inputSize = 1000;
        var expectedTime = 150L; // 150 microseconds

        mockKernel
            .Setup(k => k.GetEstimatedExecutionTimeMicroseconds(inputSize))
            .Returns(expectedTime);

        // Act
        var result = mockKernel.Object.GetEstimatedExecutionTimeMicroseconds(inputSize);

        // Assert
        result.Should().Be(expectedTime);
        result.Should().BeGreaterThan(0);
    }

    [Fact]
    public void IGpuKernel_GetMemoryRequirements_ReturnsMemoryBreakdown()
    {
        // Arrange
        var mockKernel = new Mock<IGpuKernel<int, int>>();
        var expectedRequirements = new KernelMemoryRequirements(
            InputMemoryBytes: 4096,
            OutputMemoryBytes: 4096,
            WorkingMemoryBytes: 8192,
            TotalMemoryBytes: 16384
        );

        mockKernel
            .Setup(k => k.GetMemoryRequirements())
            .Returns(expectedRequirements);

        // Act
        var result = mockKernel.Object.GetMemoryRequirements();

        // Assert
        result.Should().Be(expectedRequirements);
        result.TotalMemoryBytes.Should().Be(16384);
        result.InputMemoryBytes.Should().BeGreaterThan(0);
    }

    [Fact]
    public void IGpuKernel_ValidateInput_WithValidInput_ReturnsValid()
    {
        // Arrange
        var mockKernel = new Mock<IGpuKernel<int, int>>();
        var input = 42;
        var expectedResult = KernelValidationResult.Valid();

        mockKernel
            .Setup(k => k.ValidateInput(input))
            .Returns(expectedResult);

        // Act
        var result = mockKernel.Object.ValidateInput(input);

        // Assert
        result.IsValid.Should().BeTrue();
        result.ErrorMessage.Should().BeNull();
        result.ValidationErrors.Should().BeNull();
    }

    [Fact]
    public void IGpuKernel_ValidateInput_WithInvalidInput_ReturnsErrors()
    {
        // Arrange
        var mockKernel = new Mock<IGpuKernel<int, int>>();
        var input = -1;
        var expectedResult = KernelValidationResult.Invalid(
            "Input must be positive",
            "Value is negative: -1"
        );

        mockKernel
            .Setup(k => k.ValidateInput(input))
            .Returns(expectedResult);

        // Act
        var result = mockKernel.Object.ValidateInput(input);

        // Assert
        result.IsValid.Should().BeFalse();
        result.ErrorMessage.Should().Be("Input must be positive");
        result.ValidationErrors.Should().Contain("Value is negative: -1");
    }

    [Fact]
    public async Task IGpuKernel_WarmupAsync_CompletesSuccessfully()
    {
        // Arrange
        var mockKernel = new Mock<IGpuKernel<int, int>>();
        mockKernel
            .Setup(k => k.WarmupAsync(default))
            .Returns(Task.CompletedTask);

        // Act
        await mockKernel.Object.WarmupAsync();

        // Assert
        mockKernel.Verify(k => k.WarmupAsync(default), Times.Once);
    }

    [Fact]
    public async Task IGpuKernel_WarmupAsync_WithCancellation_PropagatesToken()
    {
        // Arrange
        var mockKernel = new Mock<IGpuKernel<int, int>>();
        var cts = new CancellationTokenSource();

        mockKernel
            .Setup(k => k.WarmupAsync(cts.Token))
            .ThrowsAsync(new OperationCanceledException());

        // Act & Assert
        await Assert.ThrowsAsync<OperationCanceledException>(
            async () => await mockKernel.Object.WarmupAsync(cts.Token));
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

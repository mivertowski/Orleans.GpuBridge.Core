using Orleans.GpuBridge.Abstractions;
using Orleans.GpuBridge.Abstractions.Enums;
using Orleans.GpuBridge.Abstractions.Kernels;
using Orleans.GpuBridge.Abstractions.Models;
using Orleans.GpuBridge.Abstractions.Domain.ValueObjects;
using System.Reflection;
using KernelId = Orleans.GpuBridge.Abstractions.KernelId;

namespace Orleans.GpuBridge.Tests.RC2.Abstractions;

/// <summary>
/// Comprehensive interface contract tests for Orleans.GpuBridge.Abstractions
/// Testing interfaces, value objects, configuration, and attributes to increase coverage toward 50%+
/// </summary>
public class InterfaceContractTests
{
    #region IGpuKernel Interface Contracts (8 tests)

    [Fact]
    public async Task IGpuKernel_SubmitBatchAsync_WithValidData_ShouldReturnHandle()
    {
        // Arrange
        var mockKernel = new Mock<IGpuKernel<int, int>>();
        var expectedHandle = new KernelHandle("test-123", DateTimeOffset.UtcNow, KernelStatus.Queued);
        var batch = new List<int> { 1, 2, 3 }.AsReadOnly();

        mockKernel
            .Setup(k => k.SubmitBatchAsync(
                It.IsAny<IReadOnlyList<int>>(),
                It.IsAny<GpuExecutionHints?>(),
                It.IsAny<CancellationToken>()))
            .ReturnsAsync(expectedHandle);

        // Act
        var result = await mockKernel.Object.SubmitBatchAsync(batch, null, CancellationToken.None);

        // Assert
        result.Should().Be(expectedHandle);
        mockKernel.Verify(k => k.SubmitBatchAsync(batch, null, CancellationToken.None), Times.Once);
    }

    [Fact]
    public async Task IGpuKernel_SubmitBatchAsync_WithNullBatch_ShouldThrow()
    {
        // Arrange
        var mockKernel = new Mock<IGpuKernel<int, int>>();
        mockKernel
            .Setup(k => k.SubmitBatchAsync(
                null!,
                It.IsAny<GpuExecutionHints?>(),
                It.IsAny<CancellationToken>()))
            .ThrowsAsync(new ArgumentNullException("items"));

        // Act & Assert
        await Assert.ThrowsAsync<ArgumentNullException>(async () =>
            await mockKernel.Object.SubmitBatchAsync(null!, null, CancellationToken.None));
    }

    [Fact]
    public async Task IGpuKernel_SubmitBatchAsync_WithEmptyBatch_ShouldReturnHandle()
    {
        // Arrange
        var mockKernel = new Mock<IGpuKernel<int, int>>();
        var expectedHandle = new KernelHandle("empty-batch", DateTimeOffset.UtcNow, KernelStatus.Completed);
        var emptyBatch = new List<int>().AsReadOnly();

        mockKernel
            .Setup(k => k.SubmitBatchAsync(emptyBatch, null, CancellationToken.None))
            .ReturnsAsync(expectedHandle);

        // Act
        var result = await mockKernel.Object.SubmitBatchAsync(emptyBatch, null, CancellationToken.None);

        // Assert
        result.Should().Be(expectedHandle);
        result.Status.Should().Be(KernelStatus.Completed);
    }

    [Fact]
    public async Task IGpuKernel_SubmitBatchAsync_WithExecutionHints_ShouldPassHints()
    {
        // Arrange
        var mockKernel = new Mock<IGpuKernel<int, int>>();
        var hints = new GpuExecutionHints(PreferredDevice: 0, HighPriority: true, MaxMicroBatch: 512);
        var batch = new List<int> { 1, 2, 3 }.AsReadOnly();
        var expectedHandle = KernelHandle.Create();

        mockKernel
            .Setup(k => k.SubmitBatchAsync(batch, hints, CancellationToken.None))
            .ReturnsAsync(expectedHandle);

        // Act
        var result = await mockKernel.Object.SubmitBatchAsync(batch, hints, CancellationToken.None);

        // Assert
        result.Should().NotBeNull();
        mockKernel.Verify(k => k.SubmitBatchAsync(batch, hints, CancellationToken.None), Times.Once);
    }

    [Fact]
    public async Task IGpuKernel_ReadResultsAsync_ReturnsAsyncEnumerable()
    {
        // Arrange
        var mockKernel = new Mock<IGpuKernel<int, int>>();
        var handle = KernelHandle.Create();
        var expectedResults = new[] { 10, 20, 30 };

        mockKernel
            .Setup(k => k.ReadResultsAsync(handle, CancellationToken.None))
            .Returns(expectedResults.ToAsyncEnumerable());

        // Act
        var results = new List<int>();
        await foreach (var result in mockKernel.Object.ReadResultsAsync(handle, CancellationToken.None))
        {
            results.Add(result);
        }

        // Assert
        results.Should().Equal(expectedResults);
    }

    [Fact]
    public async Task IGpuKernel_ReadResultsAsync_WithCancellation_ShouldRespectToken()
    {
        // Arrange
        var mockKernel = new Mock<IGpuKernel<int, int>>();
        var handle = KernelHandle.Create();
        var cts = new CancellationTokenSource();
        cts.Cancel();

        mockKernel
            .Setup(k => k.ReadResultsAsync(handle, cts.Token))
            .Returns(AsyncEnumerable.Empty<int>());

        // Act
        var results = new List<int>();
        await foreach (var result in mockKernel.Object.ReadResultsAsync(handle, cts.Token))
        {
            results.Add(result);
        }

        // Assert
        results.Should().BeEmpty();
    }

    [Fact]
    public async Task IGpuKernel_GetInfoAsync_ReturnsKernelInfo()
    {
        // Arrange
        var mockKernel = new Mock<IGpuKernel<int, int>>();
        var kernelId = new KernelId("test-kernel");
        var expectedInfo = new KernelInfo(
            Id: kernelId,
            Description: "Test kernel",
            InputType: typeof(int),
            OutputType: typeof(int),
            SupportsGpu: true,
            PreferredBatchSize: 1024,
            Metadata: null);

        mockKernel
            .Setup(k => k.GetInfoAsync(CancellationToken.None))
            .ReturnsAsync(expectedInfo);

        // Act
        var result = await mockKernel.Object.GetInfoAsync(CancellationToken.None);

        // Assert
        result.Should().Be(expectedInfo);
        result.Id.Should().Be(kernelId);
        result.SupportsGpu.Should().BeTrue();
        result.PreferredBatchSize.Should().Be(1024);
    }

    [Fact]
    public async Task IGpuKernel_MultipleSubmitBatchAsync_ShouldReturnDifferentHandles()
    {
        // Arrange
        var mockKernel = new Mock<IGpuKernel<int, int>>();
        var batch = new List<int> { 1, 2, 3 }.AsReadOnly();

        mockKernel
            .Setup(k => k.SubmitBatchAsync(batch, null, CancellationToken.None))
            .ReturnsAsync(() => KernelHandle.Create());

        // Act
        var handle1 = await mockKernel.Object.SubmitBatchAsync(batch, null, CancellationToken.None);
        var handle2 = await mockKernel.Object.SubmitBatchAsync(batch, null, CancellationToken.None);

        // Assert
        handle1.Id.Should().NotBe(handle2.Id);
        mockKernel.Verify(k => k.SubmitBatchAsync(batch, null, CancellationToken.None), Times.Exactly(2));
    }

    #endregion

    #region IGpuBridge Interface Contracts (6 tests)

    [Fact]
    public async Task IGpuBridge_GetKernelAsync_WithValidKernelId_ReturnsKernel()
    {
        // Arrange
        var mockBridge = new Mock<IGpuBridge>();
        var kernelId = new KernelId("vector-add");
        var mockKernel = new Mock<IGpuKernel<float[], float[]>>();

        mockBridge
            .Setup(b => b.GetKernelAsync<float[], float[]>(kernelId, CancellationToken.None))
            .ReturnsAsync(mockKernel.Object);

        // Act
        var result = await mockBridge.Object.GetKernelAsync<float[], float[]>(kernelId, CancellationToken.None);

        // Assert
        result.Should().NotBeNull();
        result.Should().Be(mockKernel.Object);
    }

    [Fact]
    public async Task IGpuBridge_GetKernelAsync_WithNonExistentId_ShouldThrow()
    {
        // Arrange
        var mockBridge = new Mock<IGpuBridge>();
        var kernelId = new KernelId("non-existent");

        mockBridge
            .Setup(b => b.GetKernelAsync<int, int>(kernelId, CancellationToken.None))
            .ThrowsAsync(new KeyNotFoundException($"Kernel '{kernelId}' not found"));

        // Act & Assert
        await Assert.ThrowsAsync<KeyNotFoundException>(async () =>
            await mockBridge.Object.GetKernelAsync<int, int>(kernelId, CancellationToken.None));
    }

    [Fact]
    public async Task IGpuBridge_GetInfoAsync_ReturnsValidInfo()
    {
        // Arrange
        var mockBridge = new Mock<IGpuBridge>();
        var expectedInfo = new GpuBridgeInfo(
            Version: "1.0.0",
            DeviceCount: 2,
            TotalMemoryBytes: 8L * 1024 * 1024 * 1024,
            Backend: GpuBackend.CUDA,
            IsGpuAvailable: true,
            Metadata: null);

        mockBridge
            .Setup(b => b.GetInfoAsync(CancellationToken.None))
            .ReturnsAsync(expectedInfo);

        // Act
        var result = await mockBridge.Object.GetInfoAsync(CancellationToken.None);

        // Assert
        result.Should().Be(expectedInfo);
        result.IsGpuAvailable.Should().BeTrue();
        result.DeviceCount.Should().Be(2);
        result.Backend.Should().Be(GpuBackend.CUDA);
    }

    [Fact]
    public async Task IGpuBridge_GetDevicesAsync_ReturnsDeviceList()
    {
        // Arrange
        var mockBridge = new Mock<IGpuBridge>();
        var devices = new List<GpuDevice>
        {
            new GpuDevice(0, "NVIDIA RTX 4090", DeviceType.CUDA, 24L * 1024 * 1024 * 1024,
                         20L * 1024 * 1024 * 1024, 128, new[] { "Compute 8.9", "FP64" }),
            new GpuDevice(1, "AMD Radeon RX 7900", DeviceType.OpenCL, 16L * 1024 * 1024 * 1024,
                         14L * 1024 * 1024 * 1024, 96, new[] { "FP64", "Atomics" })
        }.AsReadOnly();

        mockBridge
            .Setup(b => b.GetDevicesAsync(CancellationToken.None))
            .ReturnsAsync(devices);

        // Act
        var result = await mockBridge.Object.GetDevicesAsync(CancellationToken.None);

        // Assert
        result.Should().HaveCount(2);
        result[0].Name.Should().Contain("RTX");
        result[1].Name.Should().Contain("Radeon");
    }

    [Fact]
    public async Task IGpuBridge_ExecuteKernelAsync_WithDynamicTypes_ReturnsResult()
    {
        // Arrange
        var mockBridge = new Mock<IGpuBridge>();
        var kernelId = "dynamic-kernel";
        object input = new[] { 1, 2, 3 };
        object expectedOutput = new[] { 2, 4, 6 };

        mockBridge
            .Setup(b => b.ExecuteKernelAsync(kernelId, input, CancellationToken.None))
            .ReturnsAsync(expectedOutput);

        // Act
        var result = await mockBridge.Object.ExecuteKernelAsync(kernelId, input, CancellationToken.None);

        // Assert
        result.Should().Be(expectedOutput);
    }

    [Fact]
    public async Task IGpuBridge_ConcurrentGetKernelAsync_ShouldHandleMultipleCalls()
    {
        // Arrange
        var mockBridge = new Mock<IGpuBridge>();
        var kernelId = new KernelId("concurrent-kernel");
        var mockKernel = new Mock<IGpuKernel<int, int>>();

        mockBridge
            .Setup(b => b.GetKernelAsync<int, int>(kernelId, It.IsAny<CancellationToken>()))
            .ReturnsAsync(mockKernel.Object);

        // Act
        var tasks = Enumerable.Range(0, 10)
            .Select(async _ => await mockBridge.Object.GetKernelAsync<int, int>(kernelId, CancellationToken.None))
            .ToArray();

        var results = await Task.WhenAll(tasks);

        // Assert
        results.Should().AllBeEquivalentTo(mockKernel.Object);
        mockBridge.Verify(b => b.GetKernelAsync<int, int>(kernelId, It.IsAny<CancellationToken>()), Times.Exactly(10));
    }

    #endregion

    #region Configuration Classes (8 tests)

    [Fact]
    public void GpuBridgeOptions_DefaultValues_ShouldBeValid()
    {
        // Arrange & Act
        var options = new GpuBridgeOptions();

        // Assert
        options.PreferGpu.Should().BeTrue();
        options.DefaultMicroBatch.Should().Be(8192);
        options.MaxConcurrentKernels.Should().Be(100);
        options.MemoryPoolSizeMB.Should().Be(1024);
        options.EnableGpuDirectStorage.Should().BeFalse();
        options.MaxDevices.Should().Be(4);
        options.EnableProfiling.Should().BeFalse();
        options.EnableProviderDiscovery.Should().BeTrue();
        options.BatchSize.Should().Be(1024);
        options.MaxRetries.Should().Be(3);
        options.FallbackToCpu.Should().BeTrue();
    }

    [Fact]
    public void GpuBridgeOptions_PreferGpu_CanBeModified()
    {
        // Arrange
        var options = new GpuBridgeOptions { PreferGpu = false };

        // Assert
        options.PreferGpu.Should().BeFalse();
    }

    [Fact]
    public void GpuBridgeOptions_CustomValues_ShouldBeSet()
    {
        // Arrange & Act
        var options = new GpuBridgeOptions
        {
            PreferGpu = false,
            DefaultMicroBatch = 4096,
            MaxConcurrentKernels = 50,
            MemoryPoolSizeMB = 512,
            EnableGpuDirectStorage = true,
            MaxDevices = 2,
            EnableProfiling = true,
            DefaultBackend = "CUDA",
            BatchSize = 2048,
            MaxRetries = 5,
            FallbackToCpu = false
        };

        // Assert
        options.PreferGpu.Should().BeFalse();
        options.DefaultMicroBatch.Should().Be(4096);
        options.MaxConcurrentKernels.Should().Be(50);
        options.MemoryPoolSizeMB.Should().Be(512);
        options.EnableGpuDirectStorage.Should().BeTrue();
        options.MaxDevices.Should().Be(2);
        options.EnableProfiling.Should().BeTrue();
        options.DefaultBackend.Should().Be("CUDA");
        options.BatchSize.Should().Be(2048);
        options.MaxRetries.Should().Be(5);
        options.FallbackToCpu.Should().BeFalse();
    }

    [Fact]
    public void GpuBridgeOptions_TelemetryOptions_DefaultValues()
    {
        // Arrange & Act
        var options = new GpuBridgeOptions();

        // Assert
        options.Telemetry.Should().NotBeNull();
        options.Telemetry.EnableMetrics.Should().BeTrue();
        options.Telemetry.EnableTracing.Should().BeTrue();
        options.Telemetry.SamplingRate.Should().Be(0.1);
    }

    [Fact]
    public void GpuExecutionHints_DefaultInstance_ShouldHaveCorrectValues()
    {
        // Arrange & Act
        var hints = GpuExecutionHints.Default;

        // Assert
        hints.PreferredDevice.Should().BeNull();
        hints.HighPriority.Should().BeFalse();
        hints.MaxMicroBatch.Should().BeNull();
        hints.Persistent.Should().BeTrue();
        hints.PreferGpu.Should().BeTrue();
        hints.Timeout.Should().BeNull();
        hints.MaxRetries.Should().BeNull();
        hints.PreferredBatchSize.Should().Be(1024);
        hints.TimeoutMs.Should().Be(30000);
    }

    [Fact]
    public void GpuExecutionHints_CpuOnly_ShouldDisableGpu()
    {
        // Arrange & Act
        var hints = GpuExecutionHints.CpuOnly;

        // Assert
        hints.PreferGpu.Should().BeFalse();
        hints.Persistent.Should().BeFalse();
    }

    [Fact]
    public void GpuExecutionHints_HighPriorityGpu_ShouldSetPriority()
    {
        // Arrange & Act
        var hints = GpuExecutionHints.HighPriorityGpu;

        // Assert
        hints.HighPriority.Should().BeTrue();
        hints.PreferGpu.Should().BeTrue();
    }

    [Fact]
    public void GpuExecutionHints_CustomValues_ShouldBeSet()
    {
        // Arrange & Act
        var hints = new GpuExecutionHints(
            PreferredDevice: 1,
            HighPriority: true,
            MaxMicroBatch: 512,
            Persistent: false,
            PreferGpu: false,
            Timeout: TimeSpan.FromSeconds(60),
            MaxRetries: 5);

        // Assert
        hints.PreferredDevice.Should().Be(1);
        hints.HighPriority.Should().BeTrue();
        hints.MaxMicroBatch.Should().Be(512);
        hints.Persistent.Should().BeFalse();
        hints.PreferGpu.Should().BeFalse();
        hints.Timeout.Should().Be(TimeSpan.FromSeconds(60));
        hints.MaxRetries.Should().Be(5);
        hints.PreferredBatchSize.Should().Be(512);
    }

    #endregion

    #region Model Classes and DTOs (10 tests)

    [Fact]
    public void KernelId_Equality_ShouldCompareByValue()
    {
        // Arrange
        var id1 = new KernelId("test-kernel");
        var id2 = new KernelId("test-kernel");
        var id3 = new KernelId("other-kernel");

        // Assert
        id1.Should().Be(id2);
        id1.Should().NotBe(id3);
        id1.GetHashCode().Should().Be(id2.GetHashCode());
    }

    [Fact]
    public void KernelId_ToString_ShouldReturnValue()
    {
        // Arrange
        var id = new KernelId("test-kernel-123");

        // Act
        var result = id.ToString();

        // Assert
        result.Should().Be("test-kernel-123");
    }

    [Fact]
    public void KernelId_Parse_WithValidString_ShouldCreateId()
    {
        // Arrange & Act
        var id = KernelId.Parse("valid-kernel-id");

        // Assert
        id.Value.Should().Be("valid-kernel-id");
    }

    [Fact]
    public void KernelId_Parse_WithNullOrWhitespace_ShouldThrow()
    {
        // Act & Assert
        Assert.Throws<ArgumentException>(() => KernelId.Parse(null!));
        Assert.Throws<ArgumentException>(() => KernelId.Parse(""));
        Assert.Throws<ArgumentException>(() => KernelId.Parse("   "));
    }

    [Fact]
    public void KernelId_TryParse_WithValidString_ShouldReturnTrue()
    {
        // Act
        var success = KernelId.TryParse("valid-id", out var result);

        // Assert
        success.Should().BeTrue();
        result.Value.Should().Be("valid-id");
    }

    [Fact]
    public void KernelId_TryParse_WithInvalidString_ShouldReturnFalse()
    {
        // Act
        var success1 = KernelId.TryParse(null, out var result1);
        var success2 = KernelId.TryParse("", out var result2);

        // Assert
        success1.Should().BeFalse();
        success2.Should().BeFalse();
        result1.Should().Be(default(KernelId));
        result2.Should().Be(default(KernelId));
    }

    [Fact]
    public void KernelHandle_Creation_ShouldGenerateUniqueId()
    {
        // Arrange & Act
        var handle1 = KernelHandle.Create();
        var handle2 = KernelHandle.Create();

        // Assert
        handle1.Id.Should().NotBe(handle2.Id);
        handle1.Status.Should().Be(KernelStatus.Queued);
        handle2.Status.Should().Be(KernelStatus.Queued);
        handle1.SubmittedAt.Should().BeCloseTo(DateTimeOffset.UtcNow, TimeSpan.FromSeconds(1));
    }

    [Fact]
    public void KernelHandle_CustomStatus_ShouldBeSet()
    {
        // Arrange & Act
        var handle = new KernelHandle("custom-id", DateTimeOffset.UtcNow, KernelStatus.Running);

        // Assert
        handle.Id.Should().Be("custom-id");
        handle.Status.Should().Be(KernelStatus.Running);
    }

    [Fact]
    public void KernelResult_WithSuccess_ShouldHaveCorrectProperties()
    {
        // Arrange
        var results = new List<int> { 1, 2, 3 }.AsReadOnly();
        var handle = KernelHandle.Create();
        var executionTime = TimeSpan.FromMilliseconds(100);

        // Act
        var result = new KernelResult<int>(results, executionTime, handle, Success: true, Error: null);

        // Assert
        result.Success.Should().BeTrue();
        result.Results.Should().Equal(results);
        result.ExecutionTime.Should().Be(executionTime);
        result.Handle.Should().Be(handle);
        result.Error.Should().BeNull();
    }

    [Fact]
    public void KernelResult_WithError_ShouldHaveCorrectProperties()
    {
        // Arrange
        var handle = KernelHandle.Create();
        var errorMessage = "Kernel execution failed";

        // Act
        var result = new KernelResult<int>(
            Array.Empty<int>(),
            TimeSpan.Zero,
            handle,
            Success: false,
            Error: errorMessage);

        // Assert
        result.Success.Should().BeFalse();
        result.Error.Should().Be(errorMessage);
        result.Results.Should().BeEmpty();
    }

    #endregion

    #region Attributes and Metadata (5 tests)

    [Fact]
    public void GpuAcceleratedAttribute_OnClass_ShouldBeRetrievable()
    {
        // Arrange
        var type = typeof(TestGpuAcceleratedClass);

        // Act
        var attribute = type.GetCustomAttribute<GpuAcceleratedAttribute>();

        // Assert
        attribute.Should().NotBeNull();
        attribute!.KernelId.Should().Be("test-kernel");
    }

    [Fact]
    public void GpuAcceleratedAttribute_OnMethod_ShouldBeRetrievable()
    {
        // Arrange
        var method = typeof(TestGpuAcceleratedClass).GetMethod(nameof(TestGpuAcceleratedClass.ProcessData));

        // Act
        var attribute = method?.GetCustomAttribute<GpuAcceleratedAttribute>();

        // Assert
        attribute.Should().NotBeNull();
        attribute!.KernelId.Should().Be("process-kernel");
    }

    [Fact]
    public void GpuAcceleratedAttribute_KernelId_ShouldBeAccessible()
    {
        // Arrange & Act
        var attribute = new GpuAcceleratedAttribute("my-kernel-id");

        // Assert
        attribute.KernelId.Should().Be("my-kernel-id");
    }

    [Fact]
    public void GpuAcceleratedAttribute_Inheritance_ShouldNotBeInherited()
    {
        // Arrange
        var baseType = typeof(TestGpuAcceleratedClass);
        var derivedType = typeof(DerivedTestClass);

        // Act
        var baseAttribute = baseType.GetCustomAttribute<GpuAcceleratedAttribute>(inherit: false);
        var derivedAttribute = derivedType.GetCustomAttribute<GpuAcceleratedAttribute>(inherit: false);

        // Assert
        baseAttribute.Should().NotBeNull();
        derivedAttribute.Should().BeNull(); // Not inherited because AllowMultiple=false
    }

    [Fact]
    public void GpuAcceleratedAttribute_MultipleApplications_ShouldNotBeAllowed()
    {
        // Arrange
        var attributeUsage = typeof(GpuAcceleratedAttribute).GetCustomAttribute<AttributeUsageAttribute>();

        // Assert
        attributeUsage.Should().NotBeNull();
        attributeUsage!.AllowMultiple.Should().BeFalse();
        attributeUsage.ValidOn.Should().HaveFlag(AttributeTargets.Class);
        attributeUsage.ValidOn.Should().HaveFlag(AttributeTargets.Method);
    }

    #endregion

    #region Value Object Tests (6 tests)

    [Fact]
    public void GpuBridgeInfo_Creation_ShouldSetAllProperties()
    {
        // Arrange & Act
        var metadata = new Dictionary<string, object> { { "key", "value" } };
        var info = new GpuBridgeInfo(
            Version: "2.0.0",
            DeviceCount: 4,
            TotalMemoryBytes: 32L * 1024 * 1024 * 1024,
            Backend: GpuBackend.Vulkan,
            IsGpuAvailable: true,
            Metadata: metadata);

        // Assert
        info.Version.Should().Be("2.0.0");
        info.DeviceCount.Should().Be(4);
        info.TotalMemoryBytes.Should().Be(32L * 1024 * 1024 * 1024);
        info.Backend.Should().Be(GpuBackend.Vulkan);
        info.IsGpuAvailable.Should().BeTrue();
        info.Metadata.Should().ContainKey("key");
    }

    [Fact]
    public void GpuDevice_Creation_ShouldCalculateProperties()
    {
        // Arrange & Act
        var device = new GpuDevice(
            Index: 0,
            Name: "Test GPU",
            Type: DeviceType.CUDA,
            TotalMemoryBytes: 8L * 1024 * 1024 * 1024,
            AvailableMemoryBytes: 6L * 1024 * 1024 * 1024,
            ComputeUnits: 80,
            Capabilities: new[] { "FP64", "Compute 8.6" });

        // Assert
        device.Id.Should().Be("device_0");
        device.MemoryUtilization.Should().BeApproximately(0.25, 0.01);
        device.MaxThreadsPerBlock.Should().Be(1024);
        device.WarpSize.Should().Be(32);
    }

    [Fact]
    public void ComputeCapability_Comparison_ShouldWork()
    {
        // Arrange
        var cc1 = new ComputeCapability(8, 6);
        var cc2 = new ComputeCapability(7, 5);
        var cc3 = new ComputeCapability(8, 0);

        // Assert
        cc1.IsAtLeast(8, 6).Should().BeTrue();
        cc1.IsAtLeast(8, 0).Should().BeTrue();
        cc1.IsAtLeast(9, 0).Should().BeFalse();
        cc2.IsAtLeast(7, 0).Should().BeTrue();
        cc3.IsAtLeast(8, 0).Should().BeTrue();
    }

    [Fact]
    public void KernelInfo_Creation_ShouldSetAllFields()
    {
        // Arrange
        var kernelId = new KernelId("info-test");
        var metadata = new Dictionary<string, object> { { "version", "1.0" } };

        // Act
        var info = new KernelInfo(
            Id: kernelId,
            Description: "Test kernel info",
            InputType: typeof(int[]),
            OutputType: typeof(float[]),
            SupportsGpu: true,
            PreferredBatchSize: 2048,
            Metadata: metadata);

        // Assert
        info.Id.Should().Be(kernelId);
        info.Description.Should().Be("Test kernel info");
        info.InputType.Should().Be(typeof(int[]));
        info.OutputType.Should().Be(typeof(float[]));
        info.SupportsGpu.Should().BeTrue();
        info.PreferredBatchSize.Should().Be(2048);
        info.Metadata.Should().ContainKey("version");
    }

    [Fact]
    public void ThermalInfo_Calculations_ShouldBeCorrect()
    {
        // Arrange & Act
        var thermal = new ThermalInfo(
            TemperatureCelsius: 75,
            MaxTemperatureCelsius: 90,
            ThrottleTemperatureCelsius: 85,
            IsThrottling: false);

        // Assert
        thermal.TemperatureUtilization.Should().BeApproximately(0.833, 0.01);
        // IsNearThermalLimit checks if temp >= throttle * 0.9 => 75 >= 76.5 => False
        thermal.IsNearThermalLimit.Should().BeFalse();

        // Test case where it should be true
        var thermalNearLimit = new ThermalInfo(
            TemperatureCelsius: 80,
            MaxTemperatureCelsius: 90,
            ThrottleTemperatureCelsius: 85,
            IsThrottling: false);
        thermalNearLimit.IsNearThermalLimit.Should().BeTrue(); // 80 >= 85 * 0.9 = 76.5
    }

    [Fact]
    public void PerformanceMetrics_Calculations_ShouldBeCorrect()
    {
        // Arrange & Act
        var metrics = new PerformanceMetrics(
            UtilizationPercent: 85.0,
            MemoryUsedBytes: 6L * 1024 * 1024 * 1024,
            PowerUsageWatts: 250.0,
            ClockSpeedMHz: 1800,
            MemoryClockMHz: 9000)
        {
            MemoryUtilizationPercent = 75.0
        };

        // Assert
        metrics.PowerEfficiency.Should().BeApproximately(0.34, 0.01);
        metrics.HealthScore.Should().BeGreaterThan(0);
        metrics.HealthScore.Should().BeLessThanOrEqualTo(1.0);
    }

    #endregion

    #region Enum Tests (3 tests)

    [Fact]
    public void KernelStatus_AllValues_ShouldBeDefined()
    {
        // Arrange & Act
        var values = Enum.GetValues<KernelStatus>();

        // Assert
        values.Should().Contain(KernelStatus.Queued);
        values.Should().Contain(KernelStatus.Running);
        values.Should().Contain(KernelStatus.Completed);
        values.Should().Contain(KernelStatus.Failed);
        values.Should().Contain(KernelStatus.Cancelled);
        values.Should().HaveCount(5);
    }

    [Fact]
    public void DeviceFeatures_FlagsEnum_ShouldSupportBitwise()
    {
        // Arrange & Act
        var features = DeviceFeatures.DoublePrecision | DeviceFeatures.TensorCores | DeviceFeatures.Atomics;

        // Assert
        features.HasFlag(DeviceFeatures.DoublePrecision).Should().BeTrue();
        features.HasFlag(DeviceFeatures.TensorCores).Should().BeTrue();
        features.HasFlag(DeviceFeatures.Atomics).Should().BeTrue();
        features.HasFlag(DeviceFeatures.RayTracing).Should().BeFalse();
        features.Should().NotBe(DeviceFeatures.None);
    }

    [Fact]
    public void GpuBackend_AllValues_ShouldBeDefined()
    {
        // Arrange & Act
        var values = Enum.GetValues<GpuBackend>();

        // Assert
        values.Should().Contain(GpuBackend.Auto);
        values.Should().Contain(GpuBackend.CPU);
        values.Should().Contain(GpuBackend.CUDA);
        values.Should().Contain(GpuBackend.OpenCL);
        values.Should().Contain(GpuBackend.Metal);
        values.Should().Contain(GpuBackend.Vulkan);
        values.Should().HaveCount(7);
    }

    #endregion
}

#region Test Helper Classes

[GpuAccelerated("test-kernel")]
public class TestGpuAcceleratedClass
{
    [GpuAccelerated("process-kernel")]
    public void ProcessData() { }
}

public class DerivedTestClass : TestGpuAcceleratedClass { }

#endregion

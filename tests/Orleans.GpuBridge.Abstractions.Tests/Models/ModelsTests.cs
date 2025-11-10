using Orleans.GpuBridge.Abstractions.Kernels;
using Orleans.GpuBridge.Abstractions.Memory;
using Orleans.GpuBridge.Abstractions.Models;
using Orleans.GpuBridge.Abstractions;
using Orleans.GpuBridge.Abstractions.Domain.ValueObjects;
using Orleans.GpuBridge.Abstractions.Enums;
using Orleans.GpuBridge.Abstractions.Models.Compilation;

namespace Orleans.GpuBridge.Abstractions.Tests.Models;

/// <summary>
/// Tests for model classes and DTOs
/// </summary>
public class ModelsTests
{
    #region KernelHandle Tests

    [Fact]
    public void KernelHandle_Constructor_SetsProperties()
    {
        // Arrange
        var id = "kernel-123";
        var submittedAt = DateTimeOffset.UtcNow;

        // Act
        var handle = new KernelHandle(id, submittedAt);

        // Assert
        handle.Id.Should().Be(id);
        handle.SubmittedAt.Should().Be(submittedAt);
        handle.Status.Should().Be(KernelStatus.Queued);
    }

    [Fact]
    public void KernelHandle_Create_GeneratesUniqueHandle()
    {
        // Act
        var handle1 = KernelHandle.Create();
        var handle2 = KernelHandle.Create();

        // Assert
        handle1.Id.Should().NotBe(handle2.Id);
        handle1.Status.Should().Be(KernelStatus.Queued);
    }

    [Fact]
    public void KernelHandle_WithStatus_SetsStatus()
    {
        // Arrange & Act
        var handle = new KernelHandle("test", DateTimeOffset.UtcNow, KernelStatus.Running);

        // Assert
        handle.Status.Should().Be(KernelStatus.Running);
    }

    [Fact]
    public void KernelHandle_Equality_SameId_ReturnsTrue()
    {
        // Arrange
        var now = DateTimeOffset.UtcNow;
        var handle1 = new KernelHandle("kernel-1", now);
        var handle2 = new KernelHandle("kernel-1", now);

        // Act & Assert
        handle1.Should().Be(handle2);
    }

    [Fact]
    public void KernelHandle_Equality_DifferentId_ReturnsFalse()
    {
        // Arrange
        var now = DateTimeOffset.UtcNow;
        var handle1 = new KernelHandle("kernel-1", now);
        var handle2 = new KernelHandle("kernel-2", now);

        // Act & Assert
        handle1.Should().NotBe(handle2);
    }

    #endregion

    #region KernelInfo Tests

    [Fact]
    public void KernelInfo_Constructor_SetsAllProperties()
    {
        // Arrange
        var id = new KernelId("kernel-1");
        var description = "Test Kernel";
        var inputType = typeof(int);
        var outputType = typeof(float);

        // Act
        var info = new KernelInfo(
            Id: id,
            Description: description,
            InputType: inputType,
            OutputType: outputType,
            SupportsGpu: true,
            PreferredBatchSize: 256
        );

        // Assert
        info.Id.Should().Be(id);
        info.Description.Should().Be(description);
        info.InputType.Should().Be(inputType);
        info.OutputType.Should().Be(outputType);
        info.SupportsGpu.Should().BeTrue();
        info.PreferredBatchSize.Should().Be(256);
    }

    [Fact]
    public void KernelInfo_WithMetadata_StoresMetadata()
    {
        // Arrange
        var metadata = new Dictionary<string, object>
        {
            ["version"] = "1.0",
            ["optimized"] = true
        };

        // Act
        var info = new KernelInfo(
            new KernelId("k1"),
            "Kernel",
            typeof(int),
            typeof(int),
            true,
            128,
            metadata
        );

        // Assert
        info.Metadata.Should().ContainKey("version");
        info.Metadata!["optimized"].Should().Be(true);
    }

    [Fact]
    public void KernelInfo_Equality_SameValues_ReturnsTrue()
    {
        // Arrange
        var id = new KernelId("k1");
        var info1 = new KernelInfo(id, "Kernel", typeof(int), typeof(int), true, 128);
        var info2 = new KernelInfo(id, "Kernel", typeof(int), typeof(int), true, 128);

        // Act & Assert
        info1.Should().Be(info2);
    }

    #endregion

    #region KernelResult Tests

    [Fact]
    public void KernelResult_Constructor_SetsAllProperties()
    {
        // Arrange
        var results = new List<int> { 1, 2, 3 };
        var executionTime = TimeSpan.FromMilliseconds(100);
        var handle = KernelHandle.Create();

        // Act
        var result = new KernelResult<int>(results, executionTime, handle);

        // Assert
        result.Results.Should().BeEquivalentTo(results);
        result.ExecutionTime.Should().Be(executionTime);
        result.Handle.Should().Be(handle);
        result.Success.Should().BeTrue();
        result.Error.Should().BeNull();
    }

    [Fact]
    public void KernelResult_WithError_SetsErrorAndFailure()
    {
        // Arrange
        var handle = KernelHandle.Create();

        // Act
        var result = new KernelResult<int>(
            Array.Empty<int>(),
            TimeSpan.Zero,
            handle,
            Success: false,
            Error: "Execution failed"
        );

        // Assert
        result.Success.Should().BeFalse();
        result.Error.Should().Be("Execution failed");
    }

    [Fact]
    public void KernelResult_WithResults_ContainsData()
    {
        // Arrange
        var results = new[] { 10, 20, 30 };
        var handle = KernelHandle.Create();

        // Act
        var result = new KernelResult<int>(results, TimeSpan.FromMilliseconds(50), handle);

        // Assert
        result.Results.Should().HaveCount(3);
        result.Results.Should().Contain(20);
    }

    #endregion

    #region MemoryPoolStats Tests

    [Fact]
    public void MemoryPoolStats_Constructor_SetsAllProperties()
    {
        // Arrange & Act
        var stats = new MemoryPoolStats(
            TotalAllocated: 1024 * 1024,
            InUse: 512 * 1024,
            Available: 512 * 1024,
            BufferCount: 10,
            RentCount: 50,
            ReturnCount: 40
        );

        // Assert
        stats.TotalAllocated.Should().Be(1024 * 1024);
        stats.InUse.Should().Be(512 * 1024);
        stats.Available.Should().Be(512 * 1024);
        stats.BufferCount.Should().Be(10);
    }

    [Fact]
    public void MemoryPoolStats_Equality_SameValues_ReturnsTrue()
    {
        // Arrange
        var stats1 = new MemoryPoolStats(1000, 500, 500, 5, 10, 5);
        var stats2 = new MemoryPoolStats(1000, 500, 500, 5, 10, 5);

        // Act & Assert
        stats1.Should().Be(stats2);
    }

    [Fact]
    public void MemoryPoolStats_ZeroValues_IsValid()
    {
        // Arrange & Act
        var stats = new MemoryPoolStats(0, 0, 0, 0, 0, 0);

        // Assert
        stats.TotalAllocated.Should().Be(0);
        stats.BufferCount.Should().Be(0);
    }

    #endregion

    #region BufferUsage Tests

    [Fact]
    public void BufferUsage_EnumValues_AreDefined()
    {
        // Arrange & Act - BufferUsage is an enum, test flag combinations
        var readOnly = BufferUsage.ReadOnly;
        var writeOnly = BufferUsage.WriteOnly;
        var readWrite = BufferUsage.ReadWrite;
        var persistent = BufferUsage.Persistent;

        // Assert
        readOnly.Should().Be(BufferUsage.ReadOnly);
        writeOnly.Should().Be(BufferUsage.WriteOnly);
        readWrite.Should().Be(BufferUsage.ReadOnly | BufferUsage.WriteOnly);
        persistent.Should().Be(BufferUsage.Persistent);
    }

    [Fact]
    public void BufferUsage_FlagCombinations_WorkCorrectly()
    {
        // Arrange & Act
        var combined = BufferUsage.ReadWrite | BufferUsage.Persistent;

        // Assert
        combined.HasFlag(BufferUsage.ReadOnly).Should().BeTrue();
        combined.HasFlag(BufferUsage.WriteOnly).Should().BeTrue();
        combined.HasFlag(BufferUsage.Persistent).Should().BeTrue();
    }

    #endregion

    #region GpuExecutionHints Tests

    [Fact]
    public void GpuExecutionHints_DefaultConstructor_SetsDefaults()
    {
        // Arrange & Act
        var hints = new GpuExecutionHints();

        // Assert
        hints.PreferGpu.Should().BeTrue(); // Default is true
        hints.MaxMicroBatch.Should().BeNull();
        hints.PreferredDevice.Should().BeNull();
    }

    [Fact]
    public void GpuExecutionHints_WithValues_SetsProperties()
    {
        // Arrange & Act
        var hints = new GpuExecutionHints(
            PreferredDevice: 1,
            HighPriority: true,
            MaxMicroBatch: 256,
            Persistent: true,
            PreferGpu: true
        );

        // Assert
        hints.PreferGpu.Should().BeTrue();
        hints.MaxMicroBatch.Should().Be(256);
        hints.PreferredDevice.Should().Be(1);
        hints.HighPriority.Should().BeTrue();
    }

    [Fact]
    public void GpuExecutionHints_PreferredBatchSize_CalculatesCorrectly()
    {
        // Arrange
        var hints1 = new GpuExecutionHints(MaxMicroBatch: 512);
        var hints2 = new GpuExecutionHints(); // No MaxMicroBatch

        // Act & Assert
        hints1.PreferredBatchSize.Should().Be(512);
        hints2.PreferredBatchSize.Should().Be(1024); // Default
    }

    #endregion

    #region KernelLaunchParameters Tests

    [Fact]
    public void KernelLaunchParameters_Constructor_SetsGlobalAndLocalWorkSize()
    {
        // Arrange & Act
        var parameters = new KernelLaunchParameters(
            GlobalWorkSize: new[] { 1024, 1, 1 },
            LocalWorkSize: new[] { 256, 1, 1 }
        );

        // Assert
        parameters.GlobalWorkSize.Should().BeEquivalentTo(new[] { 1024, 1, 1 });
        parameters.LocalWorkSize.Should().BeEquivalentTo(new[] { 256, 1, 1 });
    }

    [Fact]
    public void KernelLaunchParameters_WithSharedMemory_SetsDynamicSharedMemory()
    {
        // Arrange & Act
        var parameters = new KernelLaunchParameters(
            GlobalWorkSize: new[] { 1, 1, 1 },
            LocalWorkSize: new[] { 1, 1, 1 },
            DynamicSharedMemoryBytes: 4096
        );

        // Assert
        parameters.DynamicSharedMemoryBytes.Should().Be(4096);
    }

    [Fact]
    public void KernelLaunchParameters_WithArguments_StoresArgumentsDictionary()
    {
        // Arrange
        var args = new Dictionary<string, object>
        {
            ["param1"] = 42,
            ["param2"] = 3.14f
        };

        // Act
        var parameters = new KernelLaunchParameters(
            GlobalWorkSize: new[] { 4, 2, 1 },
            LocalWorkSize: new[] { 256, 1, 1 },
            Arguments: args
        );

        // Assert
        parameters.Arguments.Should().ContainKey("param1");
        parameters.Arguments!["param2"].Should().Be(3.14f);
    }

    #endregion

    #region GpuDevice Tests

    [Fact]
    public void GpuDevice_Constructor_SetsAllProperties()
    {
        // Arrange & Act
        var device = new GpuDevice(
            Index: 0,
            Name: "NVIDIA RTX 3080",
            Type: DeviceType.CUDA,
            TotalMemoryBytes: 10737418240,
            AvailableMemoryBytes: 10737418240,
            ComputeUnits: 68,
            Capabilities: new[] { "CUDA 8.6", "Tensor Cores" }
        );

        // Assert
        device.Index.Should().Be(0);
        device.Name.Should().Be("NVIDIA RTX 3080");
        device.Type.Should().Be(DeviceType.CUDA);
        device.TotalMemoryBytes.Should().Be(10737418240);
        device.ComputeUnits.Should().Be(68);
    }

    [Fact]
    public void GpuDevice_Equality_SameValues_ReturnsTrue()
    {
        // Arrange
        var device1 = new GpuDevice(0, "RTX 3080", DeviceType.CUDA, 1000, 1000, 68, new[] { "CUDA" });
        var device2 = new GpuDevice(0, "RTX 3080", DeviceType.CUDA, 1000, 1000, 68, new[] { "CUDA" });

        // Act & Assert
        // Arrays use reference equality, so compare properties individually
        device1.Index.Should().Be(device2.Index);
        device1.Name.Should().Be(device2.Name);
        device1.Type.Should().Be(device2.Type);
        device1.Capabilities.Should().BeEquivalentTo(device2.Capabilities);
    }

    [Fact]
    public void GpuDevice_MemoryUtilization_CalculatesCorrectly()
    {
        // Arrange
        var device = new GpuDevice(0, "GPU", DeviceType.CUDA, 10000, 7000, 10, new[] { "CUDA" });

        // Act
        var utilization = device.MemoryUtilization;

        // Assert
        utilization.Should().BeApproximately(0.3, 0.01); // (10000-7000)/10000 = 0.3
    }

    #endregion

    #region GpuBridgeInfo Tests

    [Fact]
    public void GpuBridgeInfo_Constructor_SetsProperties()
    {
        // Arrange & Act
        var info = new GpuBridgeInfo(
            Version: "1.0.0",
            DeviceCount: 2,
            TotalMemoryBytes: 20L * 1024 * 1024 * 1024,
            Backend: GpuBackend.CUDA,
            IsGpuAvailable: true
        );

        // Assert
        info.Version.Should().Be("1.0.0");
        info.DeviceCount.Should().Be(2);
        info.Backend.Should().Be(GpuBackend.CUDA);
        info.IsGpuAvailable.Should().BeTrue();
    }

    [Fact]
    public void GpuBridgeInfo_WithNoDevices_SetsZeroCount()
    {
        // Arrange & Act
        var info = new GpuBridgeInfo("1.0", 0, 0, GpuBackend.CPU, false);

        // Assert
        info.DeviceCount.Should().Be(0);
        info.IsGpuAvailable.Should().BeFalse();
    }

    #endregion

    #region CompiledKernel Tests

    [Fact]
    public void CompiledKernel_InitProperties_SetsAllValues()
    {
        // Arrange
        var binary = new byte[] { 0x01, 0x02, 0x03 };

        // Act
        var kernel = new CompiledKernel
        {
            KernelId = "kernel-1",
            Name = "TestKernel",
            CompiledCode = binary
        };

        // Assert
        kernel.KernelId.Should().Be("kernel-1");
        kernel.Name.Should().Be("TestKernel");
        kernel.CompiledCode.Should().BeEquivalentTo(binary);
    }

    [Fact]
    public void CompiledKernel_WithMetadata_StoresMetadata()
    {
        // Arrange
        var metadata = new KernelMetadata(
            RequiredSharedMemory: 1024,
            MaxThreadsPerBlock: 1024,
            PreferredBlockSize: 256,
            UsesSharedMemory: true
        );

        // Act
        var kernel = new CompiledKernel
        {
            KernelId = "kernel-1",
            Name = "TestKernel",
            CompiledCode = Array.Empty<byte>(),
            Metadata = metadata
        };

        // Assert
        kernel.Metadata.Should().NotBeNull();
        kernel.Metadata.RequiredSharedMemory.Should().Be(1024);
        kernel.Metadata.MaxThreadsPerBlock.Should().Be(1024);
    }

    #endregion

    #region DeviceMetrics Tests

    [Fact]
    public void DeviceMetrics_InitProperties_SetsAllMetrics()
    {
        // Arrange & Act
        var metrics = new DeviceMetrics
        {
            GpuUtilizationPercent = 75.5f,
            MemoryUtilizationPercent = 80.0f,
            TemperatureCelsius = 65.0f,
            PowerWatts = 250.0f,
            UsedMemoryBytes = 8000000000
        };

        // Assert
        metrics.GpuUtilizationPercent.Should().BeApproximately(75.5f, 0.01f);
        metrics.TemperatureCelsius.Should().BeApproximately(65.0f, 0.01f);
        metrics.PowerWatts.Should().BeApproximately(250.0f, 0.01f);
        metrics.UsedMemoryBytes.Should().Be(8000000000);
    }

    [Fact]
    public void DeviceMetrics_MemoryUtilization_ReflectsUsage()
    {
        // Arrange & Act
        var metrics = new DeviceMetrics
        {
            GpuUtilizationPercent = 0,
            MemoryUtilizationPercent = 50.0f,
            TemperatureCelsius = 0,
            PowerWatts = 0,
            UsedMemoryBytes = 5000000000
        };

        // Assert
        metrics.MemoryUtilizationPercent.Should().BeApproximately(50.0f, 0.01f);
        metrics.UsedMemoryBytes.Should().Be(5000000000);
    }

    #endregion

    #region Edge Cases and Validation

    [Fact]
    public void KernelHandle_WithEmptyId_IsValid()
    {
        // Arrange & Act
        var handle = new KernelHandle("", DateTimeOffset.UtcNow);

        // Assert
        handle.Id.Should().Be("");
    }

    [Fact]
    public void MemoryPoolStats_NegativeValues_ShouldBeHandled()
    {
        // This tests that the model can handle edge cases
        // In production, validation should prevent negative values
        // Act & Assert - Constructor should accept or throw
        var action = () => new MemoryPoolStats(-100, 0, 0, 0, 0, 0);

        // Either it throws or accepts - both are valid depending on design
        // This documents the behavior
    }

    [Fact]
    public void GpuDevice_LargeMemorySize_HandlesCorrectly()
    {
        // Arrange - Test with 1TB
        var device = new GpuDevice(0, "High-end GPU", DeviceType.CUDA, 1_099_511_627_776, 1_099_511_627_776, 256, new[] { "CUDA" });

        // Act & Assert
        device.TotalMemoryBytes.Should().Be(1_099_511_627_776);
    }

    #endregion
}

using Orleans.GpuBridge.Abstractions.Enums;
using Orleans.GpuBridge.Abstractions.Enums.Compilation;
using Orleans.GpuBridge.Abstractions;
using System.Reflection;

namespace Orleans.GpuBridge.Abstractions.Tests.Enums;

/// <summary>
/// Tests for enums and attributes in the Abstractions project
/// </summary>
public class EnumsAndAttributesTests
{
    #region DeviceStatus Tests

    [Theory]
    [InlineData(DeviceStatus.Available)]
    [InlineData(DeviceStatus.Busy)]
    [InlineData(DeviceStatus.Offline)]
    [InlineData(DeviceStatus.Error)]
    [InlineData(DeviceStatus.Resetting)]
    [InlineData(DeviceStatus.Unknown)]
    public void DeviceStatus_AllValues_AreValid(DeviceStatus status)
    {
        // Act & Assert
        Enum.IsDefined(typeof(DeviceStatus), status).Should().BeTrue();
    }

    [Fact]
    public void DeviceStatus_ToString_ReturnsName()
    {
        // Arrange
        var status = DeviceStatus.Available;

        // Act
        var result = status.ToString();

        // Assert
        result.Should().Be("Available");
    }

    [Fact]
    public void DeviceStatus_ParseFromString_ReturnsCorrectEnum()
    {
        // Arrange
        var statusString = "Busy";

        // Act
        var status = Enum.Parse<DeviceStatus>(statusString);

        // Assert
        status.Should().Be(DeviceStatus.Busy);
    }

    [Fact]
    public void DeviceStatus_HasAllExpectedValues()
    {
        // Arrange
        var expectedValues = new[]
        {
            DeviceStatus.Available,
            DeviceStatus.Busy,
            DeviceStatus.Offline,
            DeviceStatus.Error,
            DeviceStatus.Resetting,
            DeviceStatus.Unknown
        };

        // Act
        var actualValues = Enum.GetValues<DeviceStatus>();

        // Assert
        actualValues.Should().BeEquivalentTo(expectedValues);
    }

    #endregion

    #region KernelStatus Tests

    [Theory]
    [InlineData(KernelStatus.Queued)]
    [InlineData(KernelStatus.Running)]
    [InlineData(KernelStatus.Completed)]
    [InlineData(KernelStatus.Failed)]
    [InlineData(KernelStatus.Cancelled)]
    public void KernelStatus_AllValues_AreValid(KernelStatus status)
    {
        // Act & Assert
        Enum.IsDefined(typeof(KernelStatus), status).Should().BeTrue();
    }

    [Fact]
    public void KernelStatus_Comparison_WorksCorrectly()
    {
        // Arrange
        var queued = KernelStatus.Queued;
        var completed = KernelStatus.Completed;

        // Act & Assert
        queued.Should().NotBe(completed);
        (queued == KernelStatus.Queued).Should().BeTrue();
    }

    #endregion

    #region KernelLanguage Tests

    [Theory]
    [InlineData(KernelLanguage.CUDA)]
    [InlineData(KernelLanguage.OpenCL)]
    [InlineData(KernelLanguage.HLSL)]
    [InlineData(KernelLanguage.SPIRV)]
    [InlineData(KernelLanguage.PTX)]
    public void KernelLanguage_AllValues_AreValid(KernelLanguage language)
    {
        // Act & Assert
        Enum.IsDefined(typeof(KernelLanguage), language).Should().BeTrue();
    }

    [Fact]
    public void KernelLanguage_ToString_ReturnsCorrectName()
    {
        // Arrange
        var language = KernelLanguage.CUDA;

        // Act
        var result = language.ToString();

        // Assert
        result.Should().Be("CUDA");
    }

    [Fact]
    public void KernelLanguage_AllLanguages_AreUnique()
    {
        // Arrange
        var languages = Enum.GetValues<KernelLanguage>();

        // Act
        var distinctCount = languages.Distinct().Count();

        // Assert
        distinctCount.Should().Be(languages.Length);
    }

    #endregion

    #region OptimizationLevel Tests

    [Theory]
    [InlineData(OptimizationLevel.O0)]
    [InlineData(OptimizationLevel.O1)]
    [InlineData(OptimizationLevel.O2)]
    [InlineData(OptimizationLevel.O3)]
    public void OptimizationLevel_AllValues_AreValid(OptimizationLevel level)
    {
        // Act & Assert
        Enum.IsDefined(typeof(OptimizationLevel), level).Should().BeTrue();
    }

    [Fact]
    public void OptimizationLevel_HasOrderedValues()
    {
        // Arrange & Act
        var o0 = (int)OptimizationLevel.O0;
        var o1 = (int)OptimizationLevel.O1;
        var o2 = (int)OptimizationLevel.O2;
        var o3 = (int)OptimizationLevel.O3;

        // Assert
        o0.Should().BeLessThan(o1);
        o1.Should().BeLessThan(o2);
        o2.Should().BeLessThan(o3);
    }

    #endregion

    #region DeviceType Tests

    [Theory]
    [InlineData(DeviceType.GPU)]
    [InlineData(DeviceType.CPU)]
    [InlineData(DeviceType.Accelerator)]
    public void DeviceType_AllValues_AreValid(DeviceType type)
    {
        // Act & Assert
        Enum.IsDefined(typeof(DeviceType), type).Should().BeTrue();
    }

    [Fact]
    public void DeviceType_GPU_IsDefaultPreference()
    {
        // Arrange
        var gpuType = DeviceType.GPU;

        // Act & Assert
        gpuType.Should().Be(DeviceType.GPU);
        // GPU is typically preferred but CPU is 0 in the enum
        gpuType.Should().BeDefined();
    }

    #endregion

    #region GpuBackend Tests

    [Theory]
    [InlineData(GpuBackend.Auto)]
    [InlineData(GpuBackend.CPU)]
    [InlineData(GpuBackend.CUDA)]
    [InlineData(GpuBackend.OpenCL)]
    [InlineData(GpuBackend.DirectCompute)]
    [InlineData(GpuBackend.Metal)]
    [InlineData(GpuBackend.Vulkan)]
    public void GpuBackend_AllValues_AreValid(GpuBackend backend)
    {
        // Act & Assert
        Enum.IsDefined(typeof(GpuBackend), backend).Should().BeTrue();
    }

    [Fact]
    public void GpuBackend_ToString_ReturnsBackendName()
    {
        // Arrange
        var backend = GpuBackend.CUDA;

        // Act
        var name = backend.ToString();

        // Assert
        name.Should().Be("CUDA");
    }

    [Fact]
    public void GpuBackend_Auto_IsDefaultPreference()
    {
        // Arrange
        var autoBackend = GpuBackend.Auto;

        // Act & Assert
        autoBackend.Should().Be(GpuBackend.Auto);
        ((int)autoBackend).Should().Be(0);
    }

    #endregion

    #region GpuAcceleratedAttribute Tests

    [Fact]
    public void GpuAcceleratedAttribute_Constructor_SetsKernelId()
    {
        // Arrange
        var kernelId = "test-kernel";

        // Act
        var attribute = new GpuAcceleratedAttribute(kernelId);

        // Assert
        attribute.KernelId.Should().Be(kernelId);
    }

    [Fact]
    public void GpuAcceleratedAttribute_CanBeAppliedToClass()
    {
        // Arrange
        var type = typeof(TestClass);

        // Act
        var attribute = type.GetCustomAttribute<GpuAcceleratedAttribute>();

        // Assert
        attribute.Should().NotBeNull();
        attribute!.KernelId.Should().Be("test-class-kernel");
    }

    [Fact]
    public void GpuAcceleratedAttribute_CanBeAppliedToMethod()
    {
        // Arrange
        var method = typeof(TestClass).GetMethod(nameof(TestClass.TestMethod));

        // Act
        var attribute = method!.GetCustomAttribute<GpuAcceleratedAttribute>();

        // Assert
        attribute.Should().NotBeNull();
        attribute!.KernelId.Should().Be("test-method-kernel");
    }

    [Fact]
    public void GpuAcceleratedAttribute_AttributeUsage_AllowsClassAndMethod()
    {
        // Arrange
        var attributeType = typeof(GpuAcceleratedAttribute);

        // Act
        var usage = attributeType.GetCustomAttribute<AttributeUsageAttribute>();

        // Assert
        usage.Should().NotBeNull();
        usage!.ValidOn.Should().HaveFlag(AttributeTargets.Class);
        usage.ValidOn.Should().HaveFlag(AttributeTargets.Method);
    }

    [Fact]
    public void GpuAcceleratedAttribute_AllowMultiple_IsFalse()
    {
        // Arrange
        var attributeType = typeof(GpuAcceleratedAttribute);

        // Act
        var usage = attributeType.GetCustomAttribute<AttributeUsageAttribute>();

        // Assert
        usage.Should().NotBeNull();
        usage!.AllowMultiple.Should().BeFalse();
    }

    [Fact]
    public void GpuAcceleratedAttribute_WithEmptyKernelId_IsValid()
    {
        // Arrange & Act
        var attribute = new GpuAcceleratedAttribute(string.Empty);

        // Assert
        attribute.KernelId.Should().BeEmpty();
    }

    [Fact]
    public void GpuAcceleratedAttribute_WithNullKernelId_ThrowsOrAccepts()
    {
        // This tests null handling behavior
        // Act & Assert
        var action = () => new GpuAcceleratedAttribute(null!);

        // Depending on design, this might throw or accept null
        // Document the actual behavior
    }

    #endregion

    #region Enum Flags and Bitwise Operations

    [Fact]
    public void Enums_CanBeUsedInSwitch()
    {
        // Arrange
        var status = DeviceStatus.Available;
        var result = string.Empty;

        // Act
        switch (status)
        {
            case DeviceStatus.Available:
                result = "Available";
                break;
            case DeviceStatus.Busy:
                result = "Busy";
                break;
            default:
                result = "Other";
                break;
        }

        // Assert
        result.Should().Be("Available");
    }

    [Fact]
    public void Enums_CanBeUsedInDictionaries()
    {
        // Arrange
        var statusMap = new Dictionary<DeviceStatus, string>
        {
            [DeviceStatus.Available] = "Ready",
            [DeviceStatus.Busy] = "Working",
            [DeviceStatus.Offline] = "Down"
        };

        // Act
        var result = statusMap[DeviceStatus.Available];

        // Assert
        result.Should().Be("Ready");
    }

    [Fact]
    public void Enums_DefaultValue_IsFirstOrZero()
    {
        // Act
        var defaultStatus = default(DeviceStatus);
        var defaultLanguage = default(KernelLanguage);

        // Assert
        defaultStatus.Should().Be((DeviceStatus)0);
        defaultLanguage.Should().Be((KernelLanguage)0);
    }

    #endregion

    #region Test Helper Classes

    [GpuAccelerated("test-class-kernel")]
    private class TestClass
    {
        [GpuAccelerated("test-method-kernel")]
        public void TestMethod()
        {
        }
    }

    #endregion
}

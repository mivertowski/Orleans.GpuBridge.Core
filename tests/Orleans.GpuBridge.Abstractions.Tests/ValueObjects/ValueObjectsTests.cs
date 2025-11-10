using Orleans.GpuBridge.Abstractions.Domain.ValueObjects;

namespace Orleans.GpuBridge.Abstractions.Tests.ValueObjects;

/// <summary>
/// Tests for domain value objects
/// </summary>
public class ValueObjectsTests
{
    #region KernelId Tests

    [Fact]
    public void KernelId_Constructor_WithValidValue_CreatesInstance()
    {
        // Arrange
        var value = "test-kernel-123";

        // Act
        var kernelId = new KernelId(value);

        // Assert
        kernelId.Value.Should().Be(value);
    }

    [Fact]
    public void KernelId_ToString_ReturnsValue()
    {
        // Arrange
        var kernelId = new KernelId("my-kernel");

        // Act
        var result = kernelId.ToString();

        // Assert
        result.Should().Be("my-kernel");
    }

    [Fact]
    public void KernelId_Parse_WithValidString_ReturnsKernelId()
    {
        // Arrange
        var idString = "kernel-456";

        // Act
        var kernelId = KernelId.Parse(idString);

        // Assert
        kernelId.Value.Should().Be(idString);
    }

    [Theory]
    [InlineData(null)]
    [InlineData("")]
    [InlineData("   ")]
    public void KernelId_Parse_WithInvalidString_ThrowsException(string? invalidString)
    {
        // Act & Assert
        Assert.Throws<ArgumentException>(() => KernelId.Parse(invalidString!));
    }

    [Fact]
    public void KernelId_TryParse_WithValidString_ReturnsTrue()
    {
        // Arrange
        var idString = "valid-kernel";

        // Act
        var success = KernelId.TryParse(idString, out var result);

        // Assert
        success.Should().BeTrue();
        result.Value.Should().Be(idString);
    }

    [Theory]
    [InlineData(null)]
    [InlineData("")]
    [InlineData("   ")]
    public void KernelId_TryParse_WithInvalidString_ReturnsFalse(string? invalidString)
    {
        // Act
        var success = KernelId.TryParse(invalidString, out var result);

        // Assert
        success.Should().BeFalse();
        result.Should().Be(default(KernelId));
    }

    [Fact]
    public void KernelId_ImplicitConversion_ToString_Works()
    {
        // Arrange
        var kernelId = new KernelId("test-kernel");

        // Act
        string result = kernelId;

        // Assert
        result.Should().Be("test-kernel");
    }

    [Fact]
    public void KernelId_Equality_SameValue_ReturnsTrue()
    {
        // Arrange
        var id1 = new KernelId("kernel-1");
        var id2 = new KernelId("kernel-1");

        // Act & Assert
        id1.Should().Be(id2);
        (id1 == id2).Should().BeTrue();
    }

    [Fact]
    public void KernelId_Equality_DifferentValue_ReturnsFalse()
    {
        // Arrange
        var id1 = new KernelId("kernel-1");
        var id2 = new KernelId("kernel-2");

        // Act & Assert
        id1.Should().NotBe(id2);
        (id1 != id2).Should().BeTrue();
    }

    [Fact]
    public void KernelId_GetHashCode_SameValue_ReturnsSameHash()
    {
        // Arrange
        var id1 = new KernelId("kernel-hash");
        var id2 = new KernelId("kernel-hash");

        // Act
        var hash1 = id1.GetHashCode();
        var hash2 = id2.GetHashCode();

        // Assert
        hash1.Should().Be(hash2);
    }

    #endregion

    #region ComputeCapability Tests

    [Fact]
    public void ComputeCapability_Constructor_SetsProperties()
    {
        // Arrange & Act
        var capability = new ComputeCapability(8, 6);

        // Assert
        capability.Major.Should().Be(8);
        capability.Minor.Should().Be(6);
    }

    [Fact]
    public void ComputeCapability_ToString_ReturnsFormattedString()
    {
        // Arrange
        var capability = new ComputeCapability(7, 5);

        // Act
        var result = capability.ToString();

        // Assert
        result.Should().Be("7.5");
    }

    [Fact]
    public void ComputeCapability_IsAtLeast_WithLowerVersion_ReturnsTrue()
    {
        // Arrange
        var capability = new ComputeCapability(8, 0);

        // Act
        var result = capability.IsAtLeast(7, 5);

        // Assert
        result.Should().BeTrue();
    }

    [Fact]
    public void ComputeCapability_IsAtLeast_WithSameVersion_ReturnsTrue()
    {
        // Arrange
        var capability = new ComputeCapability(7, 5);

        // Act
        var result = capability.IsAtLeast(7, 5);

        // Assert
        result.Should().BeTrue();
    }

    [Fact]
    public void ComputeCapability_IsAtLeast_WithHigherVersion_ReturnsFalse()
    {
        // Arrange
        var capability = new ComputeCapability(7, 0);

        // Act
        var result = capability.IsAtLeast(8, 0);

        // Assert
        result.Should().BeFalse();
    }

    [Fact]
    public void ComputeCapability_IsAtLeast_WithHigherMinor_ReturnsFalse()
    {
        // Arrange
        var capability = new ComputeCapability(7, 0);

        // Act
        var result = capability.IsAtLeast(7, 5);

        // Assert
        result.Should().BeFalse();
    }

    [Fact]
    public void ComputeCapability_Equality_SameValues_ReturnsTrue()
    {
        // Arrange
        var cap1 = new ComputeCapability(8, 6);
        var cap2 = new ComputeCapability(8, 6);

        // Act & Assert
        cap1.Should().Be(cap2);
    }

    [Fact]
    public void ComputeCapability_Equality_DifferentValues_ReturnsFalse()
    {
        // Arrange
        var cap1 = new ComputeCapability(8, 6);
        var cap2 = new ComputeCapability(7, 5);

        // Act & Assert
        cap1.Should().NotBe(cap2);
    }

    [Theory]
    [InlineData(3, 5, "3.5")]
    [InlineData(6, 0, "6.0")]
    [InlineData(8, 9, "8.9")]
    public void ComputeCapability_ToString_VariousVersions_FormatsCorrectly(int major, int minor, string expected)
    {
        // Arrange
        var capability = new ComputeCapability(major, minor);

        // Act
        var result = capability.ToString();

        // Assert
        result.Should().Be(expected);
    }

    #endregion

    #region PerformanceMetrics Tests

    [Fact]
    public void PerformanceMetrics_Constructor_SetsAllProperties()
    {
        // Arrange & Act
        var metrics = new PerformanceMetrics(
            UtilizationPercent: 75.0,
            MemoryUsedBytes: 8_000_000_000,
            PowerUsageWatts: 250.0,
            ClockSpeedMHz: 1800,
            MemoryClockMHz: 7000
        );

        // Assert
        metrics.UtilizationPercent.Should().Be(75.0);
        metrics.MemoryUsedBytes.Should().Be(8_000_000_000);
        metrics.PowerUsageWatts.Should().Be(250.0);
        metrics.ClockSpeedMHz.Should().Be(1800);
        metrics.MemoryClockMHz.Should().Be(7000);
    }

    [Fact]
    public void PerformanceMetrics_PowerEfficiency_CalculatesCorrectly()
    {
        // Arrange
        var metrics = new PerformanceMetrics(
            UtilizationPercent: 80.0,
            MemoryUsedBytes: 5_000_000_000,
            PowerUsageWatts: 200.0,
            ClockSpeedMHz: 1500,
            MemoryClockMHz: 6000
        );

        // Act
        var efficiency = metrics.PowerEfficiency;

        // Assert
        efficiency.Should().BeApproximately(0.4, 0.01); // 80 / 200 = 0.4
    }

    [Fact]
    public void PerformanceMetrics_HealthScore_CalculatesCorrectly()
    {
        // Arrange
        var metrics = new PerformanceMetrics(
            UtilizationPercent: 50.0,
            MemoryUsedBytes: 4_000_000_000,
            PowerUsageWatts: 150.0,
            ClockSpeedMHz: 1500,
            MemoryClockMHz: 6000
        );

        // Act
        var healthScore = metrics.HealthScore;

        // Assert
        healthScore.Should().BeGreaterThan(0.0);
        healthScore.Should().BeLessOrEqualTo(1.0);
    }

    [Fact]
    public void PerformanceMetrics_Equality_SameValues_ReturnsTrue()
    {
        // Arrange
        var metrics1 = new PerformanceMetrics(75.0, 8_000_000_000, 250.0, 1800, 7000);
        var metrics2 = new PerformanceMetrics(75.0, 8_000_000_000, 250.0, 1800, 7000);

        // Act & Assert
        metrics1.Should().Be(metrics2);
    }

    #endregion

    #region ThermalInfo Tests

    [Fact]
    public void ThermalInfo_Constructor_SetsProperties()
    {
        // Arrange & Act
        var thermal = new ThermalInfo(
            TemperatureCelsius: 65,
            MaxTemperatureCelsius: 90,
            ThrottleTemperatureCelsius: 85,
            IsThrottling: false
        );

        // Assert
        thermal.TemperatureCelsius.Should().Be(65);
        thermal.MaxTemperatureCelsius.Should().Be(90);
        thermal.ThrottleTemperatureCelsius.Should().Be(85);
        thermal.IsThrottling.Should().BeFalse();
    }

    [Fact]
    public void ThermalInfo_TemperatureUtilization_CalculatesCorrectly()
    {
        // Arrange
        var thermal = new ThermalInfo(
            TemperatureCelsius: 45,
            MaxTemperatureCelsius: 90,
            ThrottleTemperatureCelsius: 85,
            IsThrottling: false
        );

        // Act
        var utilization = thermal.TemperatureUtilization;

        // Assert
        utilization.Should().BeApproximately(0.5, 0.01); // 45 / 90 = 0.5
    }

    [Fact]
    public void ThermalInfo_IsNearThermalLimit_DetectsCorrectly()
    {
        // Arrange
        var nearLimit = new ThermalInfo(80, 90, 85, false);
        var notNearLimit = new ThermalInfo(60, 90, 85, false);

        // Act & Assert
        nearLimit.IsNearThermalLimit.Should().BeTrue(); // 80 >= 85 * 0.9 (76.5)
        notNearLimit.IsNearThermalLimit.Should().BeFalse();
    }

    [Fact]
    public void ThermalInfo_Equality_SameValues_ReturnsTrue()
    {
        // Arrange
        var thermal1 = new ThermalInfo(65, 90, 85, false);
        var thermal2 = new ThermalInfo(65, 90, 85, false);

        // Act & Assert
        thermal1.Should().Be(thermal2);
    }

    [Fact]
    public void ThermalInfo_Throttling_ReflectsState()
    {
        // Arrange & Act
        var throttling = new ThermalInfo(88, 90, 85, true);

        // Assert
        throttling.IsThrottling.Should().BeTrue();
        throttling.TemperatureCelsius.Should().BeGreaterThanOrEqualTo(throttling.ThrottleTemperatureCelsius);
    }

    #endregion

    #region Value Object Immutability Tests

    [Fact]
    public void KernelId_IsImmutable_RecordStruct()
    {
        // Arrange
        var id = new KernelId("immutable");

        // Act - Cannot modify, it's a record struct
        // This test documents that KernelId is immutable
        var id2 = id with { Value = "new-value" };

        // Assert
        id.Value.Should().Be("immutable");
        id2.Value.Should().Be("new-value");
        id.Should().NotBe(id2);
    }

    [Fact]
    public void ComputeCapability_IsImmutable_Record()
    {
        // Arrange
        var cap = new ComputeCapability(8, 6);

        // Act - Cannot modify, it's a record
        var cap2 = cap with { Major = 9 };

        // Assert
        cap.Major.Should().Be(8);
        cap2.Major.Should().Be(9);
        cap.Should().NotBe(cap2);
    }

    #endregion

    #region Edge Cases

    [Fact]
    public void KernelId_WithSpecialCharacters_IsValid()
    {
        // Arrange
        var specialId = "kernel_test-123.v1@production";

        // Act
        var kernelId = new KernelId(specialId);

        // Assert
        kernelId.Value.Should().Be(specialId);
    }

    [Fact]
    public void ComputeCapability_WithNegativeValues_CreatesInstance()
    {
        // This tests that negative values are accepted
        // Validation should happen at a higher level if needed
        // Arrange & Act
        var capability = new ComputeCapability(-1, -1);

        // Assert
        capability.Major.Should().Be(-1);
        capability.Minor.Should().Be(-1);
    }

    [Fact]
    public void PerformanceMetrics_WithZeroPower_HandlesEfficiencyCalculation()
    {
        // Tests edge case handling for power efficiency
        // Act
        var metrics = new PerformanceMetrics(50.0, 1000000, 0, 1500, 6000);

        // Assert
        metrics.PowerEfficiency.Should().Be(0.0);
    }

    [Fact]
    public void ThermalInfo_WithZeroMaxTemp_HandlesUtilizationCalculation()
    {
        // Tests edge case handling for temperature utilization
        // Arrange & Act
        var thermal = new ThermalInfo(50, 0, 85, false);

        // Assert
        thermal.TemperatureUtilization.Should().Be(0.0);
    }

    #endregion

    #region Value Object Collections

    [Fact]
    public void KernelId_CanBeUsedInHashSet()
    {
        // Arrange
        var ids = new HashSet<KernelId>
        {
            new KernelId("kernel-1"),
            new KernelId("kernel-2"),
            new KernelId("kernel-1") // Duplicate
        };

        // Act & Assert
        ids.Should().HaveCount(2); // Duplicate removed
    }

    [Fact]
    public void ComputeCapability_CanBeUsedInDictionary()
    {
        // Arrange
        var capabilities = new Dictionary<ComputeCapability, string>
        {
            [new ComputeCapability(7, 5)] = "Turing",
            [new ComputeCapability(8, 6)] = "Ampere",
            [new ComputeCapability(9, 0)] = "Hopper"
        };

        // Act
        var architecture = capabilities[new ComputeCapability(8, 6)];

        // Assert
        architecture.Should().Be("Ampere");
        capabilities.Should().HaveCount(3);
    }

    #endregion
}

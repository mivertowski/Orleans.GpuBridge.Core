// Orleans.GpuBridge - GPU-native distributed computing for Microsoft Orleans
// Copyright (c) 2025 Michael Ivertowski

using Xunit.Sdk;

namespace Orleans.GpuBridge.Tests.Common.Attributes;

/// <summary>
/// Marks a test as belonging to the Hardware test category.
/// Use for tests that require GPU hardware.
/// </summary>
/// <example>
/// <code>
/// [HardwareTrait]
/// [SkippableFact]
/// public void MyHardwareTest() { }
/// </code>
/// </example>
[TraitDiscoverer("Xunit.Sdk.TraitDiscoverer", "xunit.core")]
[AttributeUsage(AttributeTargets.Class | AttributeTargets.Method, AllowMultiple = false)]
public sealed class HardwareTraitAttribute : Attribute, ITraitAttribute
{
    /// <summary>
    /// Gets the trait category name.
    /// </summary>
    public const string Category = "Hardware";

    /// <summary>
    /// Initializes a new instance of the <see cref="HardwareTraitAttribute"/> class.
    /// </summary>
    public HardwareTraitAttribute() { }
}

/// <summary>
/// Marks a test as belonging to the Integration test category.
/// </summary>
[TraitDiscoverer("Xunit.Sdk.TraitDiscoverer", "xunit.core")]
[AttributeUsage(AttributeTargets.Class | AttributeTargets.Method, AllowMultiple = false)]
public sealed class IntegrationTraitAttribute : Attribute, ITraitAttribute
{
    /// <summary>
    /// Gets the trait category name.
    /// </summary>
    public const string Category = "Integration";

    /// <summary>
    /// Initializes a new instance of the <see cref="IntegrationTraitAttribute"/> class.
    /// </summary>
    public IntegrationTraitAttribute() { }
}

/// <summary>
/// Marks a test as belonging to the Performance test category.
/// Use for benchmark and performance-sensitive tests.
/// </summary>
[TraitDiscoverer("Xunit.Sdk.TraitDiscoverer", "xunit.core")]
[AttributeUsage(AttributeTargets.Class | AttributeTargets.Method, AllowMultiple = false)]
public sealed class PerformanceTraitAttribute : Attribute, ITraitAttribute
{
    /// <summary>
    /// Gets the trait category name.
    /// </summary>
    public const string Category = "Performance";

    /// <summary>
    /// Initializes a new instance of the <see cref="PerformanceTraitAttribute"/> class.
    /// </summary>
    public PerformanceTraitAttribute() { }
}

/// <summary>
/// Marks a test as belonging to the Unit test category.
/// Use for pure unit tests with no external dependencies.
/// </summary>
[TraitDiscoverer("Xunit.Sdk.TraitDiscoverer", "xunit.core")]
[AttributeUsage(AttributeTargets.Class | AttributeTargets.Method, AllowMultiple = false)]
public sealed class UnitTraitAttribute : Attribute, ITraitAttribute
{
    /// <summary>
    /// Gets the trait category name.
    /// </summary>
    public const string Category = "Unit";

    /// <summary>
    /// Initializes a new instance of the <see cref="UnitTraitAttribute"/> class.
    /// </summary>
    public UnitTraitAttribute() { }
}

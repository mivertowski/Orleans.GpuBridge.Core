// Orleans.GpuBridge - GPU-native distributed computing for Microsoft Orleans
// Copyright (c) 2025 Michael Ivertowski

using Orleans.GpuBridge.Tests.Common.Hardware;
using Xunit;

namespace Orleans.GpuBridge.Tests.Common.Attributes;

/// <summary>
/// Marks a test that requires native Linux (not Windows or WSL2).
/// The test will be automatically skipped on other platforms.
/// </summary>
/// <example>
/// <code>
/// [RequiresNativeLinuxFact]
/// public void MyLinuxOnlyTest()
/// {
///     // Test runs only on native Linux
/// }
/// </code>
/// </example>
public sealed class RequiresNativeLinuxFactAttribute : FactAttribute
{
    /// <summary>
    /// Initializes a new instance of the <see cref="RequiresNativeLinuxFactAttribute"/> class.
    /// </summary>
    public RequiresNativeLinuxFactAttribute()
    {
        if (!HardwareDetection.IsNativeLinux)
        {
            Skip = HardwareDetection.IsWsl2
                ? "Test requires native Linux - WSL2 detected"
                : $"Test requires native Linux - current OS: {System.Runtime.InteropServices.RuntimeInformation.OSDescription}";
        }
    }
}

/// <summary>
/// Marks a theory that requires native Linux.
/// </summary>
public sealed class RequiresNativeLinuxTheoryAttribute : TheoryAttribute
{
    /// <summary>
    /// Initializes a new instance of the <see cref="RequiresNativeLinuxTheoryAttribute"/> class.
    /// </summary>
    public RequiresNativeLinuxTheoryAttribute()
    {
        if (!HardwareDetection.IsNativeLinux)
        {
            Skip = HardwareDetection.IsWsl2
                ? "Test requires native Linux - WSL2 detected"
                : $"Test requires native Linux - current OS: {System.Runtime.InteropServices.RuntimeInformation.OSDescription}";
        }
    }
}

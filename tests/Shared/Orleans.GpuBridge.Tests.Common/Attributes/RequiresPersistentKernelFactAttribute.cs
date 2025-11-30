// Orleans.GpuBridge - GPU-native distributed computing for Microsoft Orleans
// Copyright (c) 2025 Michael Ivertowski

using Orleans.GpuBridge.Tests.Common.Hardware;
using Xunit;

namespace Orleans.GpuBridge.Tests.Common.Attributes;

/// <summary>
/// Marks a test that requires persistent kernel support (native Linux with CUDA).
/// The test will be automatically skipped on WSL2 or non-Linux systems.
/// </summary>
/// <remarks>
/// <para>
/// Persistent kernels require system-scope atomics which are not supported
/// in WSL2's GPU-PV virtualization layer.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// [RequiresPersistentKernelFact]
/// public void MyPersistentKernelTest()
/// {
///     // Test runs only on native Linux with CUDA
/// }
/// </code>
/// </example>
public sealed class RequiresPersistentKernelFactAttribute : FactAttribute
{
    /// <summary>
    /// Initializes a new instance of the <see cref="RequiresPersistentKernelFactAttribute"/> class.
    /// </summary>
    public RequiresPersistentKernelFactAttribute()
    {
        if (!HardwareDetection.IsPersistentKernelSupported)
        {
            Skip = HardwareDetection.GetPersistentKernelUnavailableReason();
        }
    }
}

/// <summary>
/// Marks a theory that requires persistent kernel support (native Linux with CUDA).
/// </summary>
public sealed class RequiresPersistentKernelTheoryAttribute : TheoryAttribute
{
    /// <summary>
    /// Initializes a new instance of the <see cref="RequiresPersistentKernelTheoryAttribute"/> class.
    /// </summary>
    public RequiresPersistentKernelTheoryAttribute()
    {
        if (!HardwareDetection.IsPersistentKernelSupported)
        {
            Skip = HardwareDetection.GetPersistentKernelUnavailableReason();
        }
    }
}

// Orleans.GpuBridge - GPU-native distributed computing for Microsoft Orleans
// Copyright (c) 2025 Michael Ivertowski

using Orleans.GpuBridge.Tests.Common.Hardware;
using Xunit;

namespace Orleans.GpuBridge.Tests.Common.Attributes;

/// <summary>
/// Marks a test that requires CUDA to be available.
/// The test will be automatically skipped if CUDA is not detected.
/// </summary>
/// <example>
/// <code>
/// [RequiresCudaFact]
/// public void MyGpuTest()
/// {
///     // Test runs only if CUDA is available
/// }
/// </code>
/// </example>
public sealed class RequiresCudaFactAttribute : FactAttribute
{
    /// <summary>
    /// Initializes a new instance of the <see cref="RequiresCudaFactAttribute"/> class.
    /// </summary>
    public RequiresCudaFactAttribute()
    {
        if (!HardwareDetection.IsCudaAvailable)
        {
            Skip = HardwareDetection.GetCudaUnavailableReason();
        }
    }
}

/// <summary>
/// Marks a theory (data-driven test) that requires CUDA to be available.
/// The test will be automatically skipped if CUDA is not detected.
/// </summary>
public sealed class RequiresCudaTheoryAttribute : TheoryAttribute
{
    /// <summary>
    /// Initializes a new instance of the <see cref="RequiresCudaTheoryAttribute"/> class.
    /// </summary>
    public RequiresCudaTheoryAttribute()
    {
        if (!HardwareDetection.IsCudaAvailable)
        {
            Skip = HardwareDetection.GetCudaUnavailableReason();
        }
    }
}

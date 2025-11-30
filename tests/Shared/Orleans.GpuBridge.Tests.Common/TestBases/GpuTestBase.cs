// Orleans.GpuBridge - GPU-native distributed computing for Microsoft Orleans
// Copyright (c) 2025 Michael Ivertowski

using Orleans.GpuBridge.Tests.Common.Hardware;
using Xunit;
using Xunit.Abstractions;

namespace Orleans.GpuBridge.Tests.Common.TestBases;

/// <summary>
/// Base class for GPU-dependent tests providing hardware detection and skip helpers.
/// </summary>
/// <remarks>
/// <para>Usage in test classes:</para>
/// <code>
/// public class MyGpuTests : GpuTestBase
/// {
///     public MyGpuTests(ITestOutputHelper output) : base(output) { }
///
///     [SkippableFact]
///     public void TestRequiringCuda()
///     {
///         SkipIfNoCuda();
///         // Test code here
///     }
/// }
/// </code>
/// </remarks>
public abstract class GpuTestBase
{
    /// <summary>
    /// Test output helper for logging hardware info and diagnostics.
    /// </summary>
    protected ITestOutputHelper Output { get; }

    /// <summary>
    /// Indicates whether CUDA is available on this system.
    /// </summary>
    protected static bool IsCudaAvailable => HardwareDetection.IsCudaAvailable;

    /// <summary>
    /// Indicates whether running under WSL2.
    /// </summary>
    protected static bool IsWsl2 => HardwareDetection.IsWsl2;

    /// <summary>
    /// Indicates whether persistent kernel mode is supported.
    /// </summary>
    protected static bool IsPersistentKernelSupported => HardwareDetection.IsPersistentKernelSupported;

    /// <summary>
    /// Gets GPU information if available.
    /// </summary>
    protected static GpuInfo? GpuInfo => HardwareDetection.GpuInfo;

    /// <summary>
    /// Initializes a new instance of the <see cref="GpuTestBase"/> class.
    /// </summary>
    /// <param name="output">Test output helper for logging.</param>
    protected GpuTestBase(ITestOutputHelper output)
    {
        Output = output ?? throw new ArgumentNullException(nameof(output));
        LogHardwareInfo();
    }

    /// <summary>
    /// Skips the test if CUDA is not available.
    /// </summary>
    /// <param name="reason">Optional custom reason message.</param>
    protected static void SkipIfNoCuda(string? reason = null)
    {
        Skip.If(!HardwareDetection.IsCudaAvailable,
            reason ?? HardwareDetection.GetCudaUnavailableReason());
    }

    /// <summary>
    /// Skips the test if running on WSL2.
    /// </summary>
    /// <param name="reason">Optional custom reason message.</param>
    protected static void SkipIfWsl2(string? reason = null)
    {
        Skip.If(HardwareDetection.IsWsl2,
            reason ?? "Test not supported on WSL2 due to GPU-PV limitations");
    }

    /// <summary>
    /// Skips the test if persistent kernel mode is not supported.
    /// </summary>
    /// <param name="reason">Optional custom reason message.</param>
    protected static void SkipIfNoPersistentKernel(string? reason = null)
    {
        Skip.If(!HardwareDetection.IsPersistentKernelSupported,
            reason ?? HardwareDetection.GetPersistentKernelUnavailableReason());
    }

    /// <summary>
    /// Skips the test if not running on native Linux.
    /// </summary>
    /// <param name="reason">Optional custom reason message.</param>
    protected static void SkipIfNotNativeLinux(string? reason = null)
    {
        Skip.If(!HardwareDetection.IsNativeLinux,
            reason ?? "Test requires native Linux (not Windows or WSL2)");
    }

    /// <summary>
    /// Skips the test if GPU memory is below the specified threshold.
    /// </summary>
    /// <param name="minimumMemoryMB">Minimum required GPU memory in megabytes.</param>
    protected static void SkipIfInsufficientGpuMemory(int minimumMemoryMB)
    {
        var gpuInfo = HardwareDetection.GpuInfo;
        Skip.If(gpuInfo == null, "No GPU detected");
        Skip.If(gpuInfo!.MemoryMB < minimumMemoryMB,
            $"Test requires {minimumMemoryMB}MB GPU memory, but only {gpuInfo.MemoryMB}MB available");
    }

    /// <summary>
    /// Gets a timing tolerance multiplier based on the test environment.
    /// WSL2 gets a higher tolerance due to virtualization overhead.
    /// </summary>
    /// <returns>A multiplier for timing assertions (1.0 for native, 2.0 for WSL2).</returns>
    protected static double GetTimingToleranceMultiplier()
    {
        return HardwareDetection.IsWsl2
            ? Wsl2Limitations.Performance.Wsl2TimingToleranceMultiplier
            : 1.0;
    }

    /// <summary>
    /// Sets up the LD_LIBRARY_PATH environment variable for WSL2 CUDA access.
    /// Call this in test fixtures that need direct CUDA library access.
    /// </summary>
    protected static void SetupWsl2CudaEnvironment()
    {
        if (!HardwareDetection.IsWsl2)
            return;

        var currentPath = Environment.GetEnvironmentVariable("LD_LIBRARY_PATH") ?? string.Empty;
        const string wslCudaPath = "/usr/lib/wsl/lib";

        if (!currentPath.Contains(wslCudaPath, StringComparison.Ordinal))
        {
            Environment.SetEnvironmentVariable(
                "LD_LIBRARY_PATH",
                $"{wslCudaPath}:{currentPath}");
        }
    }

    private void LogHardwareInfo()
    {
        Output.WriteLine("=== Hardware Detection ===");
        Output.WriteLine($"CUDA Available: {HardwareDetection.IsCudaAvailable}");
        Output.WriteLine($"WSL2: {HardwareDetection.IsWsl2}");
        Output.WriteLine($"Native Linux: {HardwareDetection.IsNativeLinux}");
        Output.WriteLine($"Persistent Kernel Supported: {HardwareDetection.IsPersistentKernelSupported}");

        if (HardwareDetection.CudaVersion != null)
            Output.WriteLine($"CUDA Version: {HardwareDetection.CudaVersion}");

        if (HardwareDetection.GpuInfo != null)
        {
            var gpu = HardwareDetection.GpuInfo;
            Output.WriteLine($"GPU: {gpu.Name}");
            if (gpu.MemoryMB > 0)
                Output.WriteLine($"GPU Memory: {gpu.MemoryMB} MB");
            if (gpu.ComputeCapability != null)
                Output.WriteLine($"Compute Capability: {gpu.ComputeCapability}");
        }

        Output.WriteLine("==========================");
    }
}

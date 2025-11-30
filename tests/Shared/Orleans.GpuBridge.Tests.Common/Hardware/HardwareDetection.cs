// Orleans.GpuBridge - GPU-native distributed computing for Microsoft Orleans
// Copyright (c) 2025 Michael Ivertowski

using System.Diagnostics;
using System.Runtime.InteropServices;

namespace Orleans.GpuBridge.Tests.Common.Hardware;

/// <summary>
/// Centralized hardware detection for GPU and platform capabilities.
/// Uses lazy caching to avoid repeated expensive system calls.
/// </summary>
public static class HardwareDetection
{
    private static readonly Lazy<bool> _isCudaAvailable = new(DetectCuda, LazyThreadSafetyMode.ExecutionAndPublication);
    private static readonly Lazy<bool> _isWsl2 = new(DetectWsl2, LazyThreadSafetyMode.ExecutionAndPublication);
    private static readonly Lazy<bool> _isNativeCuda = new(DetectNativeCuda, LazyThreadSafetyMode.ExecutionAndPublication);
    private static readonly Lazy<string?> _cudaVersion = new(DetectCudaVersion, LazyThreadSafetyMode.ExecutionAndPublication);
    private static readonly Lazy<GpuInfo?> _gpuInfo = new(DetectGpuInfo, LazyThreadSafetyMode.ExecutionAndPublication);

    /// <summary>
    /// Gets whether CUDA runtime is available on this system.
    /// </summary>
    public static bool IsCudaAvailable => _isCudaAvailable.Value;

    /// <summary>
    /// Gets whether running under WSL2 (Windows Subsystem for Linux 2).
    /// </summary>
    public static bool IsWsl2 => _isWsl2.Value;

    /// <summary>
    /// Gets whether native CUDA is available (not virtualized through WSL2).
    /// Native CUDA supports persistent kernels and system-scope atomics.
    /// </summary>
    public static bool IsNativeCuda => _isNativeCuda.Value;

    /// <summary>
    /// Gets whether running on native Linux (bare metal or VM, not WSL).
    /// </summary>
    public static bool IsNativeLinux => OperatingSystem.IsLinux() && !IsWsl2;

    /// <summary>
    /// Gets whether persistent kernel mode is supported.
    /// Persistent kernels require native Linux - WSL2's GPU-PV does not support system-scope atomics.
    /// </summary>
    public static bool IsPersistentKernelSupported => IsCudaAvailable && IsNativeLinux;

    /// <summary>
    /// Gets the detected CUDA version string, or null if not available.
    /// </summary>
    public static string? CudaVersion => _cudaVersion.Value;

    /// <summary>
    /// Gets detected GPU information, or null if no GPU detected.
    /// </summary>
    public static GpuInfo? GpuInfo => _gpuInfo.Value;

    /// <summary>
    /// Gets the reason why CUDA is not available, for diagnostic messages.
    /// </summary>
    public static string GetCudaUnavailableReason()
    {
        if (IsCudaAvailable)
            return "CUDA is available";

        if (!OperatingSystem.IsLinux() && !OperatingSystem.IsWindows())
            return $"CUDA is only supported on Linux and Windows. Current OS: {RuntimeInformation.OSDescription}";

        if (OperatingSystem.IsWindows())
            return "CUDA runtime not detected on Windows. Ensure NVIDIA drivers and CUDA toolkit are installed.";

        // Linux
        if (!File.Exists("/usr/lib/wsl/lib/libcuda.so") && !File.Exists("/usr/local/cuda/lib64/libcudart.so"))
            return "CUDA runtime libraries not found at expected paths.";

        return "nvidia-smi not found or returned error. Check GPU drivers are installed.";
    }

    /// <summary>
    /// Gets the reason why persistent kernels are not supported.
    /// </summary>
    public static string GetPersistentKernelUnavailableReason()
    {
        if (IsPersistentKernelSupported)
            return "Persistent kernels are supported";

        if (!IsCudaAvailable)
            return $"CUDA not available: {GetCudaUnavailableReason()}";

        if (IsWsl2)
            return "WSL2's GPU-PV does not support system-scope atomics required for persistent kernels. " +
                   "Use native Linux for persistent kernel mode.";

        if (OperatingSystem.IsWindows())
            return "Persistent kernels require Linux. Windows detected.";

        return "Unknown reason - persistent kernels require native Linux with CUDA.";
    }

    private static bool DetectCuda()
    {
        if (OperatingSystem.IsLinux())
        {
            // Check for WSL2 CUDA library path first
            if (File.Exists("/usr/lib/wsl/lib/libcuda.so"))
                return true;

            // Check for native CUDA installation
            if (File.Exists("/usr/local/cuda/lib64/libcudart.so"))
                return true;

            // Check via nvidia-smi
            return CheckNvidiaSmi();
        }

        if (OperatingSystem.IsWindows())
        {
            // On Windows, check for nvidia-smi
            return CheckNvidiaSmi();
        }

        return false;
    }

    private static bool DetectWsl2()
    {
        if (!OperatingSystem.IsLinux())
            return false;

        try
        {
            // WSL2 has "microsoft" in the kernel version
            var osRelease = File.ReadAllText("/proc/version");
            return osRelease.Contains("microsoft", StringComparison.OrdinalIgnoreCase) ||
                   osRelease.Contains("WSL", StringComparison.OrdinalIgnoreCase);
        }
        catch
        {
            return false;
        }
    }

    private static bool DetectNativeCuda()
    {
        return IsCudaAvailable && IsNativeLinux;
    }

    private static string? DetectCudaVersion()
    {
        if (!IsCudaAvailable)
            return null;

        try
        {
            using var process = Process.Start(new ProcessStartInfo
            {
                FileName = "nvcc",
                Arguments = "--version",
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true
            });

            if (process == null)
                return null;

            var output = process.StandardOutput.ReadToEnd();
            process.WaitForExit(5000);

            // Parse "release X.Y" from output
            var releaseIndex = output.IndexOf("release ", StringComparison.OrdinalIgnoreCase);
            if (releaseIndex >= 0)
            {
                var versionStart = releaseIndex + "release ".Length;
                var versionEnd = output.IndexOf(',', versionStart);
                if (versionEnd > versionStart)
                    return output[versionStart..versionEnd].Trim();
            }

            return output.Contains("CUDA") ? "Unknown version" : null;
        }
        catch
        {
            return null;
        }
    }

    private static GpuInfo? DetectGpuInfo()
    {
        if (!CheckNvidiaSmi())
            return null;

        try
        {
            using var process = Process.Start(new ProcessStartInfo
            {
                FileName = "nvidia-smi",
                Arguments = "--query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader,nounits",
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true
            });

            if (process == null)
                return null;

            var output = process.StandardOutput.ReadToEnd();
            process.WaitForExit(5000);

            if (process.ExitCode != 0 || string.IsNullOrWhiteSpace(output))
                return null;

            var parts = output.Split(',', StringSplitOptions.TrimEntries);
            if (parts.Length >= 4)
            {
                return new GpuInfo
                {
                    Name = parts[0],
                    MemoryMB = int.TryParse(parts[1], out var mem) ? mem : 0,
                    DriverVersion = parts[2],
                    ComputeCapability = parts[3]
                };
            }

            return new GpuInfo { Name = parts[0] };
        }
        catch
        {
            return null;
        }
    }

    private static bool CheckNvidiaSmi()
    {
        try
        {
            using var process = Process.Start(new ProcessStartInfo
            {
                FileName = "nvidia-smi",
                Arguments = "-L",
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true
            });

            if (process == null)
                return false;

            process.WaitForExit(5000);
            return process.ExitCode == 0;
        }
        catch
        {
            return false;
        }
    }
}

/// <summary>
/// Information about a detected GPU.
/// </summary>
public sealed class GpuInfo
{
    /// <summary>
    /// GPU model name (e.g., "NVIDIA GeForce RTX 4070").
    /// </summary>
    public required string Name { get; init; }

    /// <summary>
    /// Total GPU memory in megabytes.
    /// </summary>
    public int MemoryMB { get; init; }

    /// <summary>
    /// NVIDIA driver version.
    /// </summary>
    public string? DriverVersion { get; init; }

    /// <summary>
    /// CUDA compute capability (e.g., "8.9").
    /// </summary>
    public string? ComputeCapability { get; init; }
}

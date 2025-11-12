using System;
using System.Threading.Tasks;
using FluentAssertions;
using Xunit;

namespace Orleans.GpuBridge.Hardware.Tests;

/// <summary>
/// Hardware-dependent tests that require an RTX GPU with CUDA support.
/// These tests will be skipped if no compatible GPU is found.
/// Note: Tests are currently simplified due to DotCompute API availability.
/// </summary>
public class CudaBackendTests
{
    /// <summary>
    /// Verifies that CUDA runtime libraries are available on the system.
    /// </summary>
    [SkippableFact]
    public void CudaRuntime_ShouldBeAvailable()
    {
        // Check if CUDA runtime DLL/SO exists
        var isLinux = System.Runtime.InteropServices.RuntimeInformation.IsOSPlatform(System.Runtime.InteropServices.OSPlatform.Linux);
        var isWindows = System.Runtime.InteropServices.RuntimeInformation.IsOSPlatform(System.Runtime.InteropServices.OSPlatform.Windows);

        if (isLinux)
        {
            var cudaLibPath = "/usr/local/cuda/lib64/libcudart.so";
            var cudaAltPath = "/usr/lib/x86_64-linux-gnu/libcudart.so";
            var hasCuda = System.IO.File.Exists(cudaLibPath) || System.IO.File.Exists(cudaAltPath);
            Skip.IfNot(hasCuda, "CUDA runtime library not found on system");
        }
        else if (isWindows)
        {
            Skip.If(true, "Windows CUDA detection not yet implemented");
        }
        else
        {
            Skip.If(true, "CUDA only supported on Linux and Windows");
        }
    }

    /// <summary>
    /// Verifies that nvidia-smi utility is available and can detect GPUs.
    /// </summary>
    [SkippableFact]
    public async Task NvidiaSmi_ShouldDetectGpu()
    {
        try
        {
            var startInfo = new System.Diagnostics.ProcessStartInfo
            {
                FileName = "nvidia-smi",
                Arguments = "--query-gpu=name,memory.total --format=csv,noheader",
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false
            };

            using var process = System.Diagnostics.Process.Start(startInfo);
            Skip.If(process == null, "nvidia-smi not found");

            var output = await process!.StandardOutput.ReadToEndAsync();
            await process.WaitForExitAsync();

            Skip.If(process.ExitCode != 0, "nvidia-smi failed to execute");

            output.Should().NotBeEmpty("nvidia-smi should return GPU information");
            output.Should().Contain("RTX", "Expected RTX GPU");
        }
        catch (System.ComponentModel.Win32Exception)
        {
            throw new Xunit.SkipException("nvidia-smi not found on system");
        }
    }

    /// <summary>
    /// Verifies that CUDA toolkit is installed and nvcc compiler is available.
    /// </summary>
    [SkippableFact]
    public async Task Nvcc_ShouldBeAvailable()
    {
        try
        {
            var startInfo = new System.Diagnostics.ProcessStartInfo
            {
                FileName = "nvcc",
                Arguments = "--version",
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false
            };

            using var process = System.Diagnostics.Process.Start(startInfo);
            Skip.If(process == null, "nvcc not found");

            var output = await process!.StandardOutput.ReadToEndAsync();
            await process.WaitForExitAsync();

            Skip.If(process.ExitCode != 0, "nvcc failed to execute");

            output.Should().Contain("cuda", "nvcc should report CUDA version");
        }
        catch (System.ComponentModel.Win32Exception)
        {
            throw new Xunit.SkipException("nvcc not found on system");
        }
    }

    /// <summary>
    /// Placeholder test for future DotCompute integration.
    /// Once DotCompute CUDA backend is fully integrated, this will test actual GPU acceleration.
    /// </summary>
    [Fact(Skip = "DotCompute CUDA backend integration pending")]
    public void DotComputeCuda_Integration_Pending()
    {
        // This test is a placeholder for future DotCompute integration
        // Once DotCompute.Backends.CUDA is fully integrated with Orleans.GpuBridge,
        // implement actual GPU kernel compilation and execution tests here
    }
}

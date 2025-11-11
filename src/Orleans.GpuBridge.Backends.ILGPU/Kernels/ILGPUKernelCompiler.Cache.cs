using System;
using Microsoft.Extensions.Logging;
using System.Diagnostics.CodeAnalysis;
using System.Threading;
using System.Threading.Tasks;
using Orleans.GpuBridge.Abstractions.Models;
using Orleans.GpuBridge.Abstractions.Models.Compilation;

namespace Orleans.GpuBridge.Backends.ILGPU.Kernels;

/// <summary>
/// ILGPU kernel compiler - Cache management and diagnostics
/// </summary>
internal sealed partial class ILGPUKernelCompiler
{
    #region Diagnostics

    public Task<CompilationDiagnostics> GetDiagnosticsAsync(
        [NotNull] CompiledKernel kernel,
        CancellationToken cancellationToken = default)
    {
        if (kernel == null)
            throw new ArgumentNullException(nameof(kernel));

        try
        {
            var diagnostics = new CompilationDiagnostics(
                IntermediateCode: "ILGPU IL intermediate code not exposed",
                AssemblyCode: "ILGPU assembly code not exposed",
                OptimizationReport: "ILGPU optimization details not available",
                CompilationTime: TimeSpan.FromMilliseconds(
                    kernel.Metadata.ExtendedMetadata?.TryGetValue("compilation_time_ms", out var timeObj) == true
                        ? Convert.ToDouble(timeObj)
                        : 0),
                CompiledCodeSize: kernel.CompiledCode?.Length ?? 0,
                AdditionalInfo: kernel.Metadata.ExtendedMetadata);

            return Task.FromResult(diagnostics);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to get diagnostics for kernel: {KernelId}", kernel.KernelId);
            throw;
        }
    }

    #endregion

    #region Cache Management

    public void ClearCache()
    {
        _logger.LogInformation("Clearing ILGPU kernel compilation cache");

        try
        {
            // Dispose cached kernels
            foreach (var kernel in _compilationCache.Values)
            {
                try
                {
                    kernel.Dispose();
                }
                catch (Exception ex)
                {
                    _logger.LogWarning(ex, "Error disposing cached kernel: {KernelId}", kernel.KernelId);
                }
            }

            _compilationCache.Clear();
            _ilgpuKernelCache.Clear();

            _logger.LogInformation("ILGPU kernel compilation cache cleared");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error clearing ILGPU kernel compilation cache");
            throw;
        }
    }

    #endregion
}

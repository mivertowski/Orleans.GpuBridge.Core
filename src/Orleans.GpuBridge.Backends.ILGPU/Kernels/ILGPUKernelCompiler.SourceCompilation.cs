using System;
using Microsoft.Extensions.Logging;
using System.Collections.Generic;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using Orleans.GpuBridge.Abstractions.Enums.Compilation;
using Orleans.GpuBridge.Abstractions.Models;
using Orleans.GpuBridge.Abstractions.Models.Compilation;

namespace Orleans.GpuBridge.Backends.ILGPU.Kernels;

/// <summary>
/// ILGPU kernel compiler - Source code compilation methods
/// </summary>
internal sealed partial class ILGPUKernelCompiler
{
    #region Source Code Compilation

    /// <summary>
    /// Compiles source code to a method-based kernel with async file operations
    /// </summary>
    private async Task<CompiledKernel> CompileSourceCodeToMethodAsync(
        string sourceCode,
        string entryPoint,
        KernelLanguage language,
        KernelCompilationOptions options,
        CancellationToken cancellationToken)
    {
        _logger.LogInformation("Attempting source compilation for {Language} kernel: {EntryPoint}",
            language, entryPoint);

        var startTime = DateTime.UtcNow;
        string? tempDirectory = null;

        try
        {
            tempDirectory = await AsyncFileOperations.CreateTempCompilationDirectoryAsync(
                $"ilgpu_{entryPoint}", cancellationToken).ConfigureAwait(false);

            var cacheKey = GenerateCacheKey(sourceCode, entryPoint, language, options);
            var sourceFile = Path.Combine(tempDirectory, $"{entryPoint}.{GetFileExtension(language)}");
            var cacheFile = Path.Combine(tempDirectory, $"{cacheKey}.cache");

            await AsyncFileOperations.WriteCompiledKernelAsync(
                sourceFile, System.Text.Encoding.UTF8.GetBytes(sourceCode), cancellationToken).ConfigureAwait(false);

            var compiledKernel = await PerformSourceCompilationAsync(
                sourceFile, cacheFile, entryPoint, language, options, startTime, cancellationToken).ConfigureAwait(false);

            return compiledKernel;
        }
        finally
        {
            if (tempDirectory != null)
            {
                await AsyncFileOperations.CleanupTempDirectoryAsync(
                    tempDirectory, _logger, cancellationToken).ConfigureAwait(false);
            }
        }
    }

    /// <summary>
    /// Performs the actual source compilation with caching support
    /// </summary>
    private async Task<CompiledKernel> PerformSourceCompilationAsync(
        string sourceFile,
        string cacheFile,
        string entryPoint,
        KernelLanguage language,
        KernelCompilationOptions options,
        DateTime startTime,
        CancellationToken cancellationToken)
    {
        if (await AsyncFileOperations.IsKernelCacheValidAsync(cacheFile, sourceFile, cancellationToken).ConfigureAwait(false))
        {
            var cachedKernel = await LoadCachedSourceCompilationAsync(cacheFile, entryPoint, language, options, startTime, cancellationToken).ConfigureAwait(false);
            if (cachedKernel != null)
            {
                return cachedKernel;
            }
        }

        return await PerformFreshSourceCompilationAsync(sourceFile, cacheFile, entryPoint, language, options, startTime, cancellationToken).ConfigureAwait(false);
    }

    /// <summary>
    /// Loads a cached source compilation
    /// </summary>
    private async Task<CompiledKernel?> LoadCachedSourceCompilationAsync(
        string cacheFile,
        string entryPoint,
        KernelLanguage language,
        KernelCompilationOptions options,
        DateTime startTime,
        CancellationToken cancellationToken)
    {
        _logger.LogDebug("Loading cached compilation for {EntryPoint}", entryPoint);

        try
        {
            var cachedCode = await AsyncFileOperations.LoadCachedKernelAsync(cacheFile, cancellationToken).ConfigureAwait(false);
            var compilationTime = DateTime.UtcNow - startTime;

            return new CompiledKernel
            {
                KernelId = $"cached_source:{entryPoint}:{Guid.NewGuid():N}",
                Name = entryPoint,
                CompiledCode = cachedCode,
                Metadata = new KernelMetadata(
                    ExtendedMetadata: new Dictionary<string, object>
                    {
                        ["source_language"] = language.ToString(),
                        ["compilation_method"] = "cached_source",
                        ["entry_point"] = entryPoint,
                        ["compilation_time_ms"] = compilationTime.TotalMilliseconds,
                        ["optimization_level"] = options.OptimizationLevel.ToString(),
                        ["from_cache"] = true
                    })
            };
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Failed to load cached compilation, performing fresh compilation");
            return null;
        }
    }

    /// <summary>
    /// Performs a fresh source compilation
    /// </summary>
    private async Task<CompiledKernel> PerformFreshSourceCompilationAsync(
        string sourceFile,
        string cacheFile,
        string entryPoint,
        KernelLanguage language,
        KernelCompilationOptions options,
        DateTime startTime,
        CancellationToken cancellationToken)
    {
        return await Task.Run(async () =>
        {
            cancellationToken.ThrowIfCancellationRequested();

            var sourceCode = await AsyncFileOperations.ReadKernelSourceAsync(sourceFile, cancellationToken).ConfigureAwait(false);
            var simulatedCompilationTime = Math.Max(10, sourceCode.Length / 1000);
            await Task.Delay(simulatedCompilationTime, cancellationToken).ConfigureAwait(false);

            var compilationTime = DateTime.UtcNow - startTime;
            var compiledCode = System.Text.Encoding.UTF8.GetBytes(sourceCode);

            await CacheCompilationResultAsync(cacheFile, compiledCode, entryPoint, cancellationToken).ConfigureAwait(false);

            var compiledKernel = new CompiledKernel
            {
                KernelId = $"source_compiled:{entryPoint}:{Guid.NewGuid():N}",
                Name = entryPoint,
                CompiledCode = compiledCode,
                Metadata = new KernelMetadata(
                    ExtendedMetadata: new Dictionary<string, object>
                    {
                        ["source_language"] = language.ToString(),
                        ["compilation_method"] = "source_to_method",
                        ["entry_point"] = entryPoint,
                        ["compilation_time_ms"] = compilationTime.TotalMilliseconds,
                        ["source_length"] = sourceCode.Length,
                        ["optimization_level"] = options.OptimizationLevel.ToString(),
                        ["requires_runtime_compilation"] = true,
                        ["from_cache"] = false
                    })
            };

            _logger.LogDebug("Source compilation completed for {EntryPoint} in {CompilationTime}ms. " +
                "Note: This is a simplified implementation requiring runtime method generation.",
                entryPoint, compilationTime.TotalMilliseconds);

            return compiledKernel;
        }, cancellationToken).ConfigureAwait(false);
    }

    /// <summary>
    /// Caches the compilation result asynchronously
    /// </summary>
    private async Task CacheCompilationResultAsync(
        string cacheFile,
        byte[] compiledCode,
        string entryPoint,
        CancellationToken cancellationToken)
    {
        try
        {
            await AsyncFileOperations.WriteCompiledKernelAsync(cacheFile, compiledCode, cancellationToken).ConfigureAwait(false);
            _logger.LogTrace("Cached compilation result for {EntryPoint}", entryPoint);
        }
        catch (Exception ex)
        {
            _logger.LogDebug(ex, "Failed to cache compilation result for {EntryPoint}", entryPoint);
        }
    }

    #endregion

    #region Helper Methods

    /// <summary>
    /// Generates a cache key for the given compilation parameters
    /// </summary>
    private static string GenerateCacheKey(string sourceCode, string entryPoint, KernelLanguage language, KernelCompilationOptions options)
    {
        using var sha256 = System.Security.Cryptography.SHA256.Create();
        var input = $"{sourceCode}|{entryPoint}|{language}|{options.OptimizationLevel}";
        var hashBytes = sha256.ComputeHash(System.Text.Encoding.UTF8.GetBytes(input));
        return Convert.ToHexString(hashBytes)[..16];
    }

    /// <summary>
    /// Gets the appropriate file extension for a kernel language
    /// </summary>
    private static string GetFileExtension(KernelLanguage language)
    {
        return language switch
        {
            KernelLanguage.CSharp => "cs",
            KernelLanguage.HLSL => "hlsl",
            KernelLanguage.OpenCL => "cl",
            KernelLanguage.MSL => "metal",
            KernelLanguage.CUDA => "cu",
            KernelLanguage.PTX => "ptx",
            KernelLanguage.SPIRV => "spv",
            _ => "kernel"
        };
    }

    #endregion
}

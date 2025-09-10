using System;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;

namespace Orleans.GpuBridge.Backends.ILGPU.Kernels;

/// <summary>
/// Provides async file operations for kernel compilation with proper resource management
/// </summary>
internal static class AsyncFileOperations
{
    /// <summary>
    /// Asynchronously reads kernel source code from a file
    /// </summary>
    public static async Task<string> ReadKernelSourceAsync(
        string filePath, 
        CancellationToken cancellationToken = default)
    {
        if (string.IsNullOrEmpty(filePath))
            throw new ArgumentException("File path cannot be null or empty", nameof(filePath));

        try
        {
            using var fileStream = new FileStream(
                filePath, 
                FileMode.Open, 
                FileAccess.Read, 
                FileShare.Read, 
                bufferSize: 4096, 
                useAsync: true);
                
            using var reader = new StreamReader(fileStream);
            
            return await reader.ReadToEndAsync().ConfigureAwait(false);
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException($"Failed to read kernel source from {filePath}: {ex.Message}", ex);
        }
    }

    /// <summary>
    /// Asynchronously writes compiled kernel binary to a file
    /// </summary>
    public static async Task WriteCompiledKernelAsync(
        string filePath, 
        byte[] compiledCode, 
        CancellationToken cancellationToken = default)
    {
        if (string.IsNullOrEmpty(filePath))
            throw new ArgumentException("File path cannot be null or empty", nameof(filePath));

        if (compiledCode == null)
            throw new ArgumentNullException(nameof(compiledCode));

        try
        {
            // Ensure directory exists
            var directory = Path.GetDirectoryName(filePath);
            if (!string.IsNullOrEmpty(directory) && !Directory.Exists(directory))
            {
                Directory.CreateDirectory(directory);
            }

            using var fileStream = new FileStream(
                filePath, 
                FileMode.Create, 
                FileAccess.Write, 
                FileShare.None, 
                bufferSize: 4096, 
                useAsync: true);
                
            await fileStream.WriteAsync(compiledCode, cancellationToken).ConfigureAwait(false);
            await fileStream.FlushAsync(cancellationToken).ConfigureAwait(false);
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException($"Failed to write compiled kernel to {filePath}: {ex.Message}", ex);
        }
    }

    /// <summary>
    /// Asynchronously checks if a kernel cache file exists and is valid
    /// </summary>
    public static async Task<bool> IsKernelCacheValidAsync(
        string cacheFilePath, 
        string sourceFilePath, 
        CancellationToken cancellationToken = default)
    {
        try
        {
            if (!File.Exists(cacheFilePath))
                return false;

            if (!File.Exists(sourceFilePath))
                return false;

            // Check modification times asynchronously
            var cacheInfo = await Task.Run(() => new FileInfo(cacheFilePath), cancellationToken).ConfigureAwait(false);
            var sourceInfo = await Task.Run(() => new FileInfo(sourceFilePath), cancellationToken).ConfigureAwait(false);

            // Cache is valid if it's newer than the source
            return cacheInfo.LastWriteTimeUtc >= sourceInfo.LastWriteTimeUtc;
        }
        catch
        {
            // If any error occurs, assume cache is invalid
            return false;
        }
    }

    /// <summary>
    /// Asynchronously loads a cached compiled kernel
    /// </summary>
    public static async Task<byte[]> LoadCachedKernelAsync(
        string cacheFilePath, 
        CancellationToken cancellationToken = default)
    {
        if (string.IsNullOrEmpty(cacheFilePath))
            throw new ArgumentException("Cache file path cannot be null or empty", nameof(cacheFilePath));

        try
        {
            using var fileStream = new FileStream(
                cacheFilePath, 
                FileMode.Open, 
                FileAccess.Read, 
                FileShare.Read, 
                bufferSize: 4096, 
                useAsync: true);

            var buffer = new byte[fileStream.Length];
            await fileStream.ReadExactlyAsync(buffer, cancellationToken).ConfigureAwait(false);
            
            return buffer;
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException($"Failed to load cached kernel from {cacheFilePath}: {ex.Message}", ex);
        }
    }

    /// <summary>
    /// Asynchronously creates a temporary directory for compilation
    /// </summary>
    public static async Task<string> CreateTempCompilationDirectoryAsync(
        string baseName = "ilgpu_compilation",
        CancellationToken cancellationToken = default)
    {
        return await Task.Run(() =>
        {
            cancellationToken.ThrowIfCancellationRequested();
            
            var tempPath = Path.Combine(Path.GetTempPath(), $"{baseName}_{Guid.NewGuid():N}");
            Directory.CreateDirectory(tempPath);
            
            return tempPath;
        }, cancellationToken).ConfigureAwait(false);
    }

    /// <summary>
    /// Asynchronously cleans up a temporary compilation directory
    /// </summary>
    public static async Task CleanupTempDirectoryAsync(
        string tempDirectory, 
        ILogger? logger = null,
        CancellationToken cancellationToken = default)
    {
        if (string.IsNullOrEmpty(tempDirectory) || !Directory.Exists(tempDirectory))
            return;

        try
        {
            await Task.Run(() =>
            {
                cancellationToken.ThrowIfCancellationRequested();
                Directory.Delete(tempDirectory, recursive: true);
            }, cancellationToken).ConfigureAwait(false);
            
            logger?.LogDebug("Cleaned up temporary compilation directory: {TempDirectory}", tempDirectory);
        }
        catch (Exception ex)
        {
            logger?.LogWarning(ex, "Failed to cleanup temporary compilation directory: {TempDirectory}", tempDirectory);
        }
    }

    /// <summary>
    /// Asynchronously monitors a directory for kernel source file changes
    /// </summary>
    public static async Task<FileSystemWatcher> CreateKernelSourceWatcherAsync(
        string directory,
        string filter = "*.cs",
        CancellationToken cancellationToken = default)
    {
        return await Task.Run(() =>
        {
            cancellationToken.ThrowIfCancellationRequested();
            
            var watcher = new FileSystemWatcher(directory, filter)
            {
                NotifyFilter = NotifyFilters.LastWrite | NotifyFilters.FileName,
                EnableRaisingEvents = true
            };

            return watcher;
        }, cancellationToken).ConfigureAwait(false);
    }
}
using System;
using System.Reflection;
using System.Threading;
using System.Threading.Tasks;

namespace Orleans.GpuBridge.Abstractions.Providers;

/// <summary>
/// Interface for kernel compilation in GPU backends
/// </summary>
public interface IKernelCompiler
{
    /// <summary>
    /// Compiles a kernel from C# method
    /// </summary>
    Task<CompiledKernel> CompileFromMethodAsync(
        MethodInfo method,
        KernelCompilationOptions options,
        CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Compiles a kernel from source code
    /// </summary>
    Task<CompiledKernel> CompileFromSourceAsync(
        string sourceCode,
        string entryPoint,
        KernelLanguage language,
        KernelCompilationOptions options,
        CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Compiles a kernel from assembly IL
    /// </summary>
    Task<CompiledKernel> CompileFromAssemblyAsync(
        Assembly assembly,
        string typeName,
        string methodName,
        KernelCompilationOptions options,
        CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Validates if a method can be compiled as a kernel
    /// </summary>
    Task<KernelValidationResult> ValidateMethodAsync(
        MethodInfo method,
        CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Gets compilation diagnostics for debugging
    /// </summary>
    Task<CompilationDiagnostics> GetDiagnosticsAsync(
        CompiledKernel kernel,
        CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Clears the compilation cache
    /// </summary>
    void ClearCache();
}

/// <summary>
/// Represents a compiled kernel
/// </summary>
public sealed class CompiledKernel : IDisposable
{
    public string KernelId { get; init; } = string.Empty;
    public string Name { get; init; } = string.Empty;
    public byte[] CompiledCode { get; init; } = Array.Empty<byte>();
    public KernelMetadata Metadata { get; init; } = new();
    public IntPtr NativeHandle { get; init; }
    public bool IsDisposed { get; private set; }
    
    public void Dispose()
    {
        if (!IsDisposed)
        {
            // Cleanup native resources
            IsDisposed = true;
            GC.SuppressFinalize(this);
        }
    }
}

/// <summary>
/// Kernel compilation options
/// </summary>
public sealed record KernelCompilationOptions(
    OptimizationLevel OptimizationLevel = OptimizationLevel.O2,
    bool EnableDebugInfo = false,
    bool EnableProfiling = false,
    bool EnableFastMath = true,
    int MaxRegisterCount = 0,
    int MinBlockSize = 0,
    string? TargetArchitecture = null,
    IReadOnlyDictionary<string, object>? CustomOptions = null);

/// <summary>
/// Optimization levels for kernel compilation
/// </summary>
public enum OptimizationLevel
{
    /// <summary>No optimization</summary>
    O0,
    /// <summary>Basic optimization</summary>
    O1,
    /// <summary>Standard optimization</summary>
    O2,
    /// <summary>Aggressive optimization</summary>
    O3
}

/// <summary>
/// Supported kernel source languages
/// </summary>
public enum KernelLanguage
{
    CSharp,
    CUDA,
    OpenCL,
    HLSL,
    MSL,
    SPIRV,
    PTX
}

/// <summary>
/// Kernel metadata
/// </summary>
public sealed record KernelMetadata(
    int RequiredSharedMemory = 0,
    int RequiredRegisters = 0,
    int MaxThreadsPerBlock = 256,
    int PreferredBlockSize = 0,
    bool UsesAtomics = false,
    bool UsesSharedMemory = false,
    bool UsesDynamicParallelism = false,
    IReadOnlyDictionary<string, object>? ExtendedMetadata = null);

/// <summary>
/// Result of kernel validation
/// </summary>
public sealed record KernelValidationResult(
    bool IsValid,
    string? ErrorMessage = null,
    IReadOnlyList<string>? Warnings = null,
    IReadOnlyList<string>? UnsupportedFeatures = null);

/// <summary>
/// Compilation diagnostics for debugging
/// </summary>
public sealed record CompilationDiagnostics(
    string? IntermediateCode = null,
    string? AssemblyCode = null,
    string? OptimizationReport = null,
    TimeSpan CompilationTime = default,
    long CompiledCodeSize = 0,
    IReadOnlyDictionary<string, object>? AdditionalInfo = null);
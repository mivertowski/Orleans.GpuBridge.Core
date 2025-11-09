using System;
using System.Diagnostics.CodeAnalysis;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Orleans.GpuBridge.Abstractions.Enums.Compilation;
using Orleans.GpuBridge.Abstractions.Models.Compilation;
using Orleans.GpuBridge.Abstractions.Providers;

namespace Orleans.GpuBridge.Runtime.Providers;

/// <summary>
/// CPU kernel compiler for fallback provider
/// </summary>
internal sealed class CpuKernelCompiler : IKernelCompiler
{
    private readonly ILogger<CpuKernelCompiler> _logger;

    public CpuKernelCompiler(ILogger<CpuKernelCompiler> logger)
    {
        _logger = logger;
    }

    [RequiresUnreferencedCode("Uses method validation which may not work with trimming.")]
    public Task<CompiledKernel> CompileFromMethodAsync(
        [NotNull] System.Reflection.MethodInfo method,
        [NotNull] KernelCompilationOptions options,
        CancellationToken cancellationToken = default)
    {
        return Task.FromResult(new CompiledKernel
        {
            KernelId = method.Name,
            Name = method.Name,
            CompiledCode = Array.Empty<byte>(),
            Metadata = new KernelMetadata()
        });
    }

    public Task<CompiledKernel> CompileFromSourceAsync(
        [NotNull] string sourceCode,
        [NotNull] string entryPoint,
        KernelLanguage language,
        [NotNull] KernelCompilationOptions options,
        CancellationToken cancellationToken = default)
    {
        return Task.FromResult(new CompiledKernel
        {
            KernelId = entryPoint,
            Name = entryPoint,
            CompiledCode = System.Text.Encoding.UTF8.GetBytes(sourceCode),
            Metadata = new KernelMetadata()
        });
    }

    [RequiresUnreferencedCode("Uses reflection to find types and methods which may be trimmed.")]
    public Task<CompiledKernel> CompileFromAssemblyAsync(
        [NotNull] System.Reflection.Assembly assembly,
        [NotNull] string typeName,
        [NotNull] string methodName,
        [NotNull] KernelCompilationOptions options,
        CancellationToken cancellationToken = default)
    {
        return Task.FromResult(new CompiledKernel
        {
            KernelId = $"{typeName}.{methodName}",
            Name = methodName,
            CompiledCode = Array.Empty<byte>(),
            Metadata = new KernelMetadata()
        });
    }

    [RequiresUnreferencedCode("Uses method body analysis which may not work with trimming.")]
    public Task<KernelValidationResult> ValidateMethodAsync(
        [NotNull] System.Reflection.MethodInfo method,
        CancellationToken cancellationToken = default)
    {
        return Task.FromResult(new KernelValidationResult(IsValid: true));
    }

    public Task<CompilationDiagnostics> GetDiagnosticsAsync(
        [NotNull] CompiledKernel kernel,
        CancellationToken cancellationToken = default)
    {
        return Task.FromResult(new CompilationDiagnostics());
    }

    public void ClearCache() { }

    public void Dispose() { }
}

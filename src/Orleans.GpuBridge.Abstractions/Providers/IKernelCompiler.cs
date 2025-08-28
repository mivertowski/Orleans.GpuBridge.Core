using System.Reflection;
using System.Threading;
using System.Threading.Tasks;
using Orleans.GpuBridge.Abstractions.Enums.Compilation;
using Orleans.GpuBridge.Abstractions.Models;
using Orleans.GpuBridge.Abstractions.Models.Compilation;

namespace Orleans.GpuBridge.Abstractions.Providers;

/// <summary>
/// Defines the contract for GPU kernel compilation services.
/// </summary>
/// <remarks>
/// This interface abstracts the kernel compilation process across different GPU backends,
/// providing a unified API for compiling C# methods, source code, or assembly IL into
/// GPU-executable kernels. Implementations handle backend-specific compilation details
/// while maintaining consistent behavior across different GPU architectures and vendors.
/// 
/// <para>
/// The compiler interface supports multiple compilation paths:
/// - Direct C# method compilation with reflection
/// - Source code compilation from various GPU languages
/// - Assembly IL compilation for advanced scenarios
/// </para>
/// 
/// <para>
/// All compilation operations are asynchronous and support cancellation to handle
/// potentially long-running compilation processes, especially when aggressive
/// optimizations are enabled.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Compile a C# method to a GPU kernel
/// var compiler = serviceProvider.GetRequiredService&lt;IKernelCompiler&gt;();
/// var options = new KernelCompilationOptions(OptimizationLevel.O2);
/// var kernel = await compiler.CompileFromMethodAsync(methodInfo, options);
/// 
/// // Compile from CUDA source code
/// var cudaKernel = await compiler.CompileFromSourceAsync(
///     cudaCode, "vectorAdd", KernelLanguage.CUDA, options);
/// 
/// // Validate before compilation
/// var validation = await compiler.ValidateMethodAsync(methodInfo);
/// if (validation.IsValid)
/// {
///     var kernel = await compiler.CompileFromMethodAsync(methodInfo, options);
/// }
/// </code>
/// </example>
public interface IKernelCompiler
{
    /// <summary>
    /// Compiles a GPU kernel from a C# method using reflection.
    /// </summary>
    /// <param name="method">
    /// The <see cref="MethodInfo"/> representing the C# method to compile.
    /// The method should be static and follow GPU kernel conventions.
    /// </param>
    /// <param name="options">
    /// Compilation options that control optimization level, debug information,
    /// and other compilation behaviors.
    /// </param>
    /// <param name="cancellationToken">
    /// A cancellation token that can be used to cancel the compilation operation.
    /// </param>
    /// <returns>
    /// A task that represents the asynchronous compilation operation.
    /// The task result contains the compiled kernel if successful.
    /// </returns>
    /// <exception cref="ArgumentNullException">
    /// Thrown when <paramref name="method"/> or <paramref name="options"/> is null.
    /// </exception>
    /// <exception cref="InvalidOperationException">
    /// Thrown when the method cannot be compiled as a GPU kernel due to
    /// unsupported constructs or incompatible signature.
    /// </exception>
    /// <exception cref="OperationCanceledException">
    /// Thrown when the compilation is cancelled via <paramref name="cancellationToken"/>.
    /// </exception>
    /// <remarks>
    /// This method analyzes the provided C# method and generates equivalent GPU code.
    /// Not all C# constructs are supported in GPU kernels - use <see cref="ValidateMethodAsync"/>
    /// to check compatibility before compilation.
    /// </remarks>
    Task<CompiledKernel> CompileFromMethodAsync(
        MethodInfo method,
        KernelCompilationOptions options,
        CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Compiles a GPU kernel from source code in a specified language.
    /// </summary>
    /// <param name="sourceCode">
    /// The source code of the kernel in the specified language.
    /// </param>
    /// <param name="entryPoint">
    /// The name of the entry point function within the source code.
    /// </param>
    /// <param name="language">
    /// The programming language of the source code.
    /// </param>
    /// <param name="options">
    /// Compilation options that control optimization level, debug information,
    /// and other compilation behaviors.
    /// </param>
    /// <param name="cancellationToken">
    /// A cancellation token that can be used to cancel the compilation operation.
    /// </param>
    /// <returns>
    /// A task that represents the asynchronous compilation operation.
    /// The task result contains the compiled kernel if successful.
    /// </returns>
    /// <exception cref="ArgumentNullException">
    /// Thrown when <paramref name="sourceCode"/>, <paramref name="entryPoint"/>,
    /// or <paramref name="options"/> is null.
    /// </exception>
    /// <exception cref="ArgumentException">
    /// Thrown when <paramref name="sourceCode"/> or <paramref name="entryPoint"/> is empty.
    /// </exception>
    /// <exception cref="NotSupportedException">
    /// Thrown when the specified <paramref name="language"/> is not supported
    /// by the current backend.
    /// </exception>
    /// <exception cref="InvalidOperationException">
    /// Thrown when the source code contains compilation errors or the entry point
    /// cannot be found.
    /// </exception>
    /// <exception cref="OperationCanceledException">
    /// Thrown when the compilation is cancelled via <paramref name="cancellationToken"/>.
    /// </exception>
    /// <remarks>
    /// This method provides flexibility to compile kernels written in native GPU
    /// languages like CUDA C, OpenCL C, or other supported languages. The availability
    /// of specific languages depends on the backend implementation.
    /// </remarks>
    Task<CompiledKernel> CompileFromSourceAsync(
        string sourceCode,
        string entryPoint,
        KernelLanguage language,
        KernelCompilationOptions options,
        CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Compiles a GPU kernel from .NET assembly intermediate language (IL).
    /// </summary>
    /// <param name="assembly">
    /// The assembly containing the method to compile.
    /// </param>
    /// <param name="typeName">
    /// The full name of the type containing the kernel method.
    /// </param>
    /// <param name="methodName">
    /// The name of the method to compile as a kernel.
    /// </param>
    /// <param name="options">
    /// Compilation options that control optimization level, debug information,
    /// and other compilation behaviors.
    /// </param>
    /// <param name="cancellationToken">
    /// A cancellation token that can be used to cancel the compilation operation.
    /// </param>
    /// <returns>
    /// A task that represents the asynchronous compilation operation.
    /// The task result contains the compiled kernel if successful.
    /// </returns>
    /// <exception cref="ArgumentNullException">
    /// Thrown when <paramref name="assembly"/>, <paramref name="typeName"/>,
    /// <paramref name="methodName"/>, or <paramref name="options"/> is null.
    /// </exception>
    /// <exception cref="ArgumentException">
    /// Thrown when <paramref name="typeName"/> or <paramref name="methodName"/> is empty.
    /// </exception>
    /// <exception cref="InvalidOperationException">
    /// Thrown when the specified type or method cannot be found in the assembly,
    /// or the method cannot be compiled as a GPU kernel.
    /// </exception>
    /// <exception cref="OperationCanceledException">
    /// Thrown when the compilation is cancelled via <paramref name="cancellationToken"/>.
    /// </exception>
    /// <remarks>
    /// This method enables compilation of kernels from pre-compiled assemblies,
    /// which is useful for dynamic kernel loading scenarios or when working with
    /// assemblies loaded at runtime. The method will be located using reflection
    /// and compiled to GPU code.
    /// </remarks>
    Task<CompiledKernel> CompileFromAssemblyAsync(
        Assembly assembly,
        string typeName,
        string methodName,
        KernelCompilationOptions options,
        CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Validates whether a C# method can be successfully compiled as a GPU kernel.
    /// </summary>
    /// <param name="method">
    /// The <see cref="MethodInfo"/> representing the C# method to validate.
    /// </param>
    /// <param name="cancellationToken">
    /// A cancellation token that can be used to cancel the validation operation.
    /// </param>
    /// <returns>
    /// A task that represents the asynchronous validation operation.
    /// The task result contains validation results including any errors or warnings.
    /// </returns>
    /// <exception cref="ArgumentNullException">
    /// Thrown when <paramref name="method"/> is null.
    /// </exception>
    /// <exception cref="OperationCanceledException">
    /// Thrown when the validation is cancelled via <paramref name="cancellationToken"/>.
    /// </exception>
    /// <remarks>
    /// This method performs static analysis of the C# method to determine if it
    /// can be successfully compiled to GPU code. It checks for unsupported language
    /// constructs, invalid memory access patterns, and other compatibility issues.
    /// 
    /// <para>
    /// Use this method before attempting compilation to provide better error messages
    /// and avoid unnecessary compilation attempts for incompatible methods.
    /// </para>
    /// </remarks>
    Task<KernelValidationResult> ValidateMethodAsync(
        MethodInfo method,
        CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Retrieves detailed compilation diagnostics for a compiled kernel.
    /// </summary>
    /// <param name="kernel">
    /// The compiled kernel for which to retrieve diagnostics.
    /// </param>
    /// <param name="cancellationToken">
    /// A cancellation token that can be used to cancel the diagnostic operation.
    /// </param>
    /// <returns>
    /// A task that represents the asynchronous diagnostic operation.
    /// The task result contains detailed compilation diagnostics.
    /// </returns>
    /// <exception cref="ArgumentNullException">
    /// Thrown when <paramref name="kernel"/> is null.
    /// </exception>
    /// <exception cref="ObjectDisposedException">
    /// Thrown when <paramref name="kernel"/> has been disposed.
    /// </exception>
    /// <exception cref="OperationCanceledException">
    /// Thrown when the operation is cancelled via <paramref name="cancellationToken"/>.
    /// </exception>
    /// <remarks>
    /// Diagnostics provide detailed information about the compilation process,
    /// including intermediate representations, optimization reports, and performance
    /// characteristics. This information is valuable for debugging and optimization.
    /// 
    /// <para>
    /// The availability of specific diagnostic information depends on the backend
    /// and compilation options used when the kernel was originally compiled.
    /// </para>
    /// </remarks>
    Task<CompilationDiagnostics> GetDiagnosticsAsync(
        CompiledKernel kernel,
        CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Clears the internal compilation cache to free memory and force recompilation.
    /// </summary>
    /// <remarks>
    /// The compiler may cache compiled kernels to improve performance of repeated
    /// compilations. This method clears all cached data, which may be useful for:
    /// - Freeing memory in long-running applications
    /// - Forcing recompilation with updated options
    /// - Debugging compilation-related issues
    /// 
    /// <para>
    /// After calling this method, subsequent compilation requests will not benefit
    /// from cached results until the cache is repopulated.
    /// </para>
    /// </remarks>
    void ClearCache();
}
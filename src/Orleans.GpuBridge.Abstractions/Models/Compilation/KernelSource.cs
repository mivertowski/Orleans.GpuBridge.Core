using Orleans.GpuBridge.Abstractions.Enums.Compilation;

namespace Orleans.GpuBridge.Abstractions.Models.Compilation;

/// <summary>
/// Represents GPU kernel source code with metadata required for compilation.
/// </summary>
/// <param name="Name">
/// The name of the kernel. This serves as an identifier and is typically
/// used as the kernel function name in compiled code. Must be a valid
/// identifier for the target language.
/// </param>
/// <param name="SourceCode">
/// The complete source code of the kernel in the specified language.
/// This should contain all necessary code including the kernel function
/// and any helper functions or data structures.
/// </param>
/// <param name="Language">
/// The programming language used in the source code.
/// This determines which compiler backend will be used for compilation.
/// </param>
/// <param name="EntryPoint">
/// The name of the entry point function within the source code.
/// This is the function that will be called when the kernel is executed.
/// If null, the <paramref name="Name"/> will be used as the entry point.
/// </param>
/// <param name="IncludePaths">
/// Optional list of include paths for resolving header files and dependencies.
/// These paths will be passed to the compiler for resolving #include directives
/// or similar language-specific include mechanisms. Default is null.
/// </param>
/// <param name="Dependencies">
/// Optional list of additional source files or libraries that this kernel
/// depends on. These will be compiled together with the main source code.
/// Default is null.
/// </param>
/// <remarks>
/// This record encapsulates all information needed to compile a kernel from
/// source code. Different GPU backends may have varying requirements for
/// source code format and structure.
/// 
/// <para>
/// The source code should follow the conventions of the specified language
/// and be compatible with GPU execution constraints such as:
/// - Limited recursion depth
/// - Restricted function calls
/// - Memory access patterns
/// - Synchronization primitives
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // CUDA kernel source
/// var cudaSource = new KernelSource(
///     Name: "vectorAdd",
///     SourceCode: @"
///         __global__ void vectorAdd(float* a, float* b, float* result, int n) {
///             int idx = blockIdx.x * blockDim.x + threadIdx.x;
///             if (idx &lt; n) {
///                 result[idx] = a[idx] + b[idx];
///             }
///         }",
///     Language: KernelLanguage.CUDA,
///     EntryPoint: "vectorAdd");
/// 
/// // OpenCL kernel source
/// var openCLSource = new KernelSource(
///     Name: "matrixMultiply", 
///     SourceCode: @"
///         __kernel void matrixMultiply(__global float* A, 
///                                     __global float* B, 
///                                     __global float* C, 
///                                     int N) {
///             int row = get_global_id(0);
///             int col = get_global_id(1);
///             float sum = 0.0f;
///             for (int k = 0; k &lt; N; k++) {
///                 sum += A[row * N + k] * B[k * N + col];
///             }
///             C[row * N + col] = sum;
///         }",
///     Language: KernelLanguage.OpenCL,
///     EntryPoint: "matrixMultiply");
/// 
/// // C# kernel source with dependencies
/// var csharpSource = new KernelSource(
///     Name: "processArray",
///     SourceCode: "/* C# kernel code */",
///     Language: KernelLanguage.CSharp,
///     EntryPoint: "ProcessArrayKernel",
///     IncludePaths: new[] { "/usr/include/gpu" },
///     Dependencies: new[] { "helper.cs", "math_utils.cs" });
/// </code>
/// </example>
public sealed record KernelSource(
    string Name,
    string SourceCode,
    KernelLanguage Language,
    string? EntryPoint = null,
    IReadOnlyList<string>? IncludePaths = null,
    IReadOnlyList<string>? Dependencies = null)
{
    /// <summary>
    /// Gets the effective entry point name.
    /// Returns <see cref="EntryPoint"/> if specified, otherwise returns <see cref="Name"/>.
    /// </summary>
    public string EffectiveEntryPoint => EntryPoint ?? Name;

    /// <summary>
    /// Gets a hash code that uniquely identifies this kernel source.
    /// The hash is based on source code content, language, and entry point.
    /// </summary>
    /// <returns>
    /// A hash code that can be used for caching and comparison purposes.
    /// </returns>
    public override int GetHashCode()
    {
        var hash = new HashCode();
        hash.Add(Name);
        hash.Add(SourceCode);
        hash.Add(Language);
        hash.Add(EffectiveEntryPoint);

        if (IncludePaths != null)
        {
            foreach (var path in IncludePaths)
                hash.Add(path);
        }

        if (Dependencies != null)
        {
            foreach (var dependency in Dependencies)
                hash.Add(dependency);
        }

        return hash.ToHashCode();
    }
}
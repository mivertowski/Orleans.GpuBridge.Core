using System;
using System.Collections.Generic;

namespace Orleans.GpuBridge.Abstractions.Models.Compilation;

/// <summary>
/// Provides detailed diagnostic information about kernel compilation for debugging,
/// optimization analysis, and performance tuning.
/// </summary>
/// <param name="IntermediateCode">
/// The intermediate representation (IR) code generated during compilation.
/// This may be LLVM IR, SPIR-V, or another intermediate format depending
/// on the compiler backend. Useful for understanding compiler transformations
/// and optimizations. May be <c>null</c> if not available or not requested.
/// </param>
/// <param name="AssemblyCode">
/// The final assembly code generated for the target GPU architecture.
/// This is the human-readable representation of the compiled binary code,
/// such as PTX for NVIDIA GPUs or GCN assembly for AMD GPUs. Valuable for
/// performance analysis and low-level optimization. May be <c>null</c> if
/// not available or not supported by the backend.
/// </param>
/// <param name="OptimizationReport">
/// A detailed report of optimizations applied during compilation.
/// Includes information about loop unrolling, vectorization, memory coalescing,
/// register allocation, and other compiler optimizations. Helps understand
/// performance characteristics and identify optimization opportunities.
/// May be <c>null</c> if not available or not requested.
/// </param>
/// <param name="CompilationTime">
/// The total time taken to compile the kernel from source to final binary.
/// Includes parsing, optimization, and code generation phases. Useful for
/// performance analysis of the compilation pipeline and identifying compilation
/// bottlenecks. Default is <see cref="TimeSpan.Zero"/>.
/// </param>
/// <param name="CompiledCodeSize">
/// The size in bytes of the final compiled binary code.
/// Larger kernels may have longer load times and higher memory usage.
/// This metric helps assess the impact of optimizations on code size.
/// Default is 0.
/// </param>
/// <param name="AdditionalInfo">
/// Backend-specific additional diagnostic information that doesn't fit
/// into the standard diagnostic categories. May include profiling data,
/// resource usage statistics, or vendor-specific metrics. Keys should be
/// descriptive names, values can be any serializable diagnostic data.
/// Default is <c>null</c>.
/// </param>
/// <remarks>
/// Compilation diagnostics provide valuable insights for:
/// - Performance optimization and tuning
/// - Understanding compiler behavior and optimizations  
/// - Debugging compilation issues and unexpected behavior
/// - Comparing different compilation strategies
/// - Educational purposes and learning GPU programming
/// 
/// <para>
/// The availability of specific diagnostic information depends on the
/// compiler backend and compilation options. Not all backends support
/// all diagnostic features, and some information may only be available
/// when debug information is enabled.
/// </para>
/// 
/// <para>
/// For production use, diagnostic information is typically only gathered
/// when specifically requested, as it can significantly increase compilation
/// time and memory usage.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Basic diagnostics with timing information
/// var basicDiagnostics = new CompilationDiagnostics(
///     CompilationTime: TimeSpan.FromMilliseconds(150),
///     CompiledCodeSize: 2048);
/// 
/// // Detailed diagnostics with all information
/// var detailedDiagnostics = new CompilationDiagnostics(
///     IntermediateCode: "define void @kernel(...) { ... }",
///     AssemblyCode: "ld.global.f32 %f1, [%rd1+0];\nmul.f32 %f2, %f1, %f1;",
///     OptimizationReport: "Applied loop unrolling (factor 4), vectorized memory accesses",
///     CompilationTime: TimeSpan.FromMilliseconds(350),
///     CompiledCodeSize: 1536,
///     AdditionalInfo: new Dictionary&lt;string, object&gt;
///     {
///         ["RegistersPerThread"] = 24,
///         ["OccupancyPercent"] = 75.0,
///         ["InstructionCount"] = 128
///     });
/// 
/// // Diagnostics for performance analysis
/// var performanceDiagnostics = new CompilationDiagnostics(
///     OptimizationReport: "Memory coalescing: 95% efficient, Branch divergence: minimal",
///     CompilationTime: TimeSpan.FromMilliseconds(200),
///     CompiledCodeSize: 1024,
///     AdditionalInfo: new Dictionary&lt;string, object&gt;
///     {
///         ["EstimatedPerformance"] = "High",
///         ["MemoryBandwidthUtilization"] = 0.85
///     });
/// </code>
/// </example>
public sealed record CompilationDiagnostics(
    string? IntermediateCode = null,
    string? AssemblyCode = null,
    string? OptimizationReport = null,
    TimeSpan CompilationTime = default,
    long CompiledCodeSize = 0,
    IReadOnlyDictionary<string, object>? AdditionalInfo = null)
{
    /// <summary>
    /// Gets a value indicating whether this diagnostic instance contains intermediate code information.
    /// </summary>
    /// <value>
    /// <c>true</c> if intermediate code is available; otherwise, <c>false</c>.
    /// </value>
    public bool HasIntermediateCode => !string.IsNullOrEmpty(IntermediateCode);

    /// <summary>
    /// Gets a value indicating whether this diagnostic instance contains assembly code information.
    /// </summary>
    /// <value>
    /// <c>true</c> if assembly code is available; otherwise, <c>false</c>.
    /// </value>
    public bool HasAssemblyCode => !string.IsNullOrEmpty(AssemblyCode);

    /// <summary>
    /// Gets a value indicating whether this diagnostic instance contains an optimization report.
    /// </summary>
    /// <value>
    /// <c>true</c> if an optimization report is available; otherwise, <c>false</c>.
    /// </value>
    public bool HasOptimizationReport => !string.IsNullOrEmpty(OptimizationReport);

    /// <summary>
    /// Gets a value indicating whether this diagnostic instance contains additional backend-specific information.
    /// </summary>
    /// <value>
    /// <c>true</c> if additional information is available; otherwise, <c>false</c>.
    /// </value>
    public bool HasAdditionalInfo => AdditionalInfo?.Count > 0;
}
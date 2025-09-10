using System.Collections.Generic;
using Orleans.GpuBridge.Abstractions.Enums.Compilation;

namespace Orleans.GpuBridge.Abstractions.Models.Compilation;

/// <summary>
/// Represents compilation options for GPU kernel compilation.
/// </summary>
/// <param name="OptimizationLevel">
/// The optimization level to apply during compilation. Higher levels produce
/// more optimized code but may increase compilation time. Default is <see cref="Orleans.GpuBridge.Abstractions.Enums.Compilation.OptimizationLevel.O2"/>.
/// </param>
/// <param name="EnableDebugInfo">
/// Indicates whether to include debug information in the compiled kernel.
/// When enabled, allows for better debugging and profiling capabilities but
/// increases binary size. Default is <c>false</c>.
/// </param>
/// <param name="EnableProfiling">
/// Indicates whether to enable profiling support in the compiled kernel.
/// When enabled, the kernel can provide detailed execution metrics and timing
/// information. May impact performance. Default is <c>false</c>.
/// </param>
/// <param name="EnableFastMath">
/// Indicates whether to enable fast math optimizations that may sacrifice
/// numerical precision for improved performance. Useful for applications
/// where approximate results are acceptable. Default is <c>true</c>.
/// </param>
/// <param name="MaxRegisterCount">
/// The maximum number of registers the kernel is allowed to use.
/// A value of 0 indicates no explicit limit, allowing the compiler to
/// use as many registers as needed. Non-zero values can improve occupancy
/// by limiting register usage per thread. Default is 0.
/// </param>
/// <param name="MinBlockSize">
/// The minimum block size (number of threads per block) to optimize for.
/// A value of 0 indicates no minimum, allowing the compiler to choose
/// optimal block sizes. Specifying a minimum helps optimize for specific
/// execution patterns. Default is 0.
/// </param>
/// <param name="TargetArchitecture">
/// The target GPU architecture to compile for (e.g., "sm_75", "gfx906").
/// A null value indicates compilation for the default or detected architecture.
/// Specifying an architecture enables architecture-specific optimizations.
/// Default is <c>null</c>.
/// </param>
/// <param name="Defines">
/// Preprocessor definitions to pass to the compiler as key-value pairs.
/// These definitions are equivalent to #define directives and can be used
/// to conditionally compile code sections. Keys are definition names,
/// values are definition values (empty string for flag-only definitions).
/// Default is <c>null</c>.
/// </param>
/// <param name="CustomOptions">
/// Additional custom compilation options specific to the backend compiler.
/// This dictionary allows passing backend-specific flags and options that
/// are not covered by the standard options. Keys should be option names,
/// values should be option values. Default is <c>null</c>.
/// </param>
/// <param name="TargetDevice">
/// The specific target device for compilation optimization.
/// When specified, the compiler can optimize for device-specific features
/// and characteristics. A null value uses default device selection.
/// Default is <c>null</c>.
/// </param>
/// <remarks>
/// This record provides comprehensive control over kernel compilation behavior.
/// Different GPU backends may interpret these options differently, and not all
/// options may be supported by every backend. Unsupported options are typically
/// ignored with a warning.
/// 
/// <para>
/// For optimal performance, consider the following guidelines:
/// - Use <see cref="Orleans.GpuBridge.Abstractions.Enums.Compilation.OptimizationLevel.O2"/> or higher for production kernels
/// - Enable debug info only during development
/// - Set MaxRegisterCount when optimizing for high occupancy
/// - Specify TargetArchitecture for deployment-specific optimizations
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Basic optimization options
/// var options = new KernelCompilationOptions(
///     OptimizationLevel: OptimizationLevel.O3,
///     EnableFastMath: true);
/// 
/// // Debug-enabled options
/// var debugOptions = new KernelCompilationOptions(
///     OptimizationLevel: OptimizationLevel.O0,
///     EnableDebugInfo: true,
///     EnableProfiling: true);
/// 
/// // Architecture-specific options
/// var targetedOptions = new KernelCompilationOptions(
///     TargetArchitecture: "sm_80",
///     MaxRegisterCount: 64,
///     MinBlockSize: 128);
/// </code>
/// </example>
public sealed record KernelCompilationOptions(
    OptimizationLevel OptimizationLevel = OptimizationLevel.O2,
    bool EnableDebugInfo = false,
    bool EnableProfiling = false,
    bool EnableFastMath = true,
    int MaxRegisterCount = 0,
    int MinBlockSize = 0,
    string? TargetArchitecture = null,
    IReadOnlyDictionary<string, string>? Defines = null,
    IReadOnlyDictionary<string, object>? CustomOptions = null,
    object? TargetDevice = null);
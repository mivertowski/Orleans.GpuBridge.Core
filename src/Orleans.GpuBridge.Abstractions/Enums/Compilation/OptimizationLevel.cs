namespace Orleans.GpuBridge.Abstractions.Enums.Compilation;

/// <summary>
/// Defines the optimization levels available for GPU kernel compilation.
/// </summary>
/// <remarks>
/// Optimization levels control the trade-off between compilation time and
/// runtime performance. Higher optimization levels generally produce faster
/// code but take longer to compile and may make debugging more difficult.
/// 
/// <para>
/// The specific optimizations applied at each level are backend-dependent,
/// but generally follow these principles:
/// - O0: Minimal optimization, best for debugging
/// - O1: Basic optimizations with reasonable compilation time
/// - O2: Standard optimizations suitable for most production use
/// - O3: Aggressive optimizations, best runtime performance
/// </para>
/// 
/// <para>
/// Choose the optimization level based on your use case:
/// - Development and debugging: Use O0 or O1
/// - Production with balanced performance: Use O2
/// - Performance-critical applications: Use O3
/// </para>
/// </remarks>
public enum OptimizationLevel
{
    /// <summary>
    /// No optimization applied during compilation.
    /// </summary>
    /// <remarks>
    /// Disables most compiler optimizations to preserve the original code
    /// structure as much as possible. This level provides:
    /// - Fastest compilation time
    /// - Best debugging experience with accurate line-by-line correspondence
    /// - Predictable code generation
    /// - Largest binary size and slowest runtime performance
    /// 
    /// Recommended for development, debugging, and educational purposes.
    /// </remarks>
    /// <example>
    /// <code>
    /// var debugOptions = new KernelCompilationOptions(
    ///     OptimizationLevel: OptimizationLevel.O0,
    ///     EnableDebugInfo: true);
    /// </code>
    /// </example>
    O0,

    /// <summary>
    /// Basic optimization level with fundamental optimizations applied.
    /// </summary>
    /// <remarks>
    /// Applies essential optimizations that provide significant performance
    /// improvements with minimal impact on compilation time. Typical optimizations include:
    /// - Dead code elimination
    /// - Constant folding and propagation
    /// - Basic instruction scheduling
    /// - Simple loop optimizations
    /// 
    /// This level maintains good debuggability while providing reasonable performance.
    /// Suitable for development builds where some optimization is desired.
    /// </remarks>
    /// <example>
    /// <code>
    /// var devOptions = new KernelCompilationOptions(
    ///     OptimizationLevel: OptimizationLevel.O1);
    /// </code>
    /// </example>
    O1,

    /// <summary>
    /// Standard optimization level suitable for most production scenarios.
    /// </summary>
    /// <remarks>
    /// Applies comprehensive optimizations that balance compilation time,
    /// binary size, and runtime performance. This is the recommended level
    /// for most production deployments. Optimizations typically include:
    /// - All O1 optimizations
    /// - Advanced loop optimizations (unrolling, vectorization)
    /// - Function inlining
    /// - Register allocation optimization
    /// - Memory access pattern optimization
    /// - Control flow optimizations
    /// 
    /// Provides excellent performance while maintaining reasonable compilation
    /// times and binary sizes.
    /// </remarks>
    /// <example>
    /// <code>
    /// var productionOptions = new KernelCompilationOptions(
    ///     OptimizationLevel: OptimizationLevel.O2,
    ///     EnableFastMath: true);
    /// </code>
    /// </example>
    O2,

    /// <summary>
    /// Aggressive optimization level for maximum runtime performance.
    /// </summary>
    /// <remarks>
    /// Applies the most aggressive optimizations available, prioritizing
    /// runtime performance over compilation time and binary size. May include:
    /// - All O2 optimizations
    /// - Aggressive function inlining
    /// - Cross-procedural optimizations
    /// - Advanced loop transformations
    /// - Speculative optimizations
    /// - Architecture-specific micro-optimizations
    /// 
    /// This level can significantly increase compilation time and binary size
    /// but provides the best runtime performance. Recommended for performance-critical
    /// kernels where maximum speed is essential.
    /// 
    /// <para>
    /// Note: Some aggressive optimizations may make debugging difficult
    /// and could potentially affect numerical stability in edge cases.
    /// </para>
    /// </remarks>
    /// <example>
    /// <code>
    /// var maxPerformanceOptions = new KernelCompilationOptions(
    ///     OptimizationLevel: OptimizationLevel.O3,
    ///     EnableFastMath: true,
    ///     TargetArchitecture: "sm_80");
    /// </code>
    /// </example>
    O3
}
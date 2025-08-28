using Orleans.GpuBridge.Abstractions.Enums;
using Orleans.GpuBridge.Abstractions.Enums.Compilation;

namespace Orleans.GpuBridge.Backends.DotCompute.Configuration;

/// <summary>
/// Kernel language compilation configuration settings for the DotCompute backend.
/// </summary>
/// <remarks>
/// This class provides comprehensive configuration for kernel programming language
/// support, compilation preferences, and language-specific settings in the DotCompute
/// backend. The backend supports multiple kernel languages and can automatically
/// translate between them when needed.
/// 
/// <para>
/// Language selection affects:
/// - Compilation performance and kernel optimization
/// - Platform compatibility and portability
/// - Development experience and debugging capabilities
/// - Runtime performance characteristics
/// - Feature availability and language-specific optimizations
/// </para>
/// 
/// <para>
/// The DotCompute backend provides intelligent language selection based on platform
/// capabilities, performance characteristics, and developer preferences. When a
/// preferred language is not available, the backend can automatically translate
/// to a supported language or use a fallback language.
/// </para>
/// 
/// <para>
/// The default configuration provides optimal language selection for each platform
/// while enabling automatic translation for maximum compatibility.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // CUDA-optimized configuration
/// var cudaSettings = new KernelLanguageSettings();
/// cudaSettings.PreferredLanguages.Clear();
/// cudaSettings.PreferredLanguages[GpuBackend.Cuda] = KernelLanguage.CUDA;
/// cudaSettings.PreferredLanguages[GpuBackend.OpenCL] = KernelLanguage.OpenCL;
/// cudaSettings.EnableLanguageTranslation = true;
/// 
/// // Cross-platform configuration
/// var crossPlatformSettings = new KernelLanguageSettings();
/// crossPlatformSettings.PreferredLanguages.Clear();
/// crossPlatformSettings.PreferredLanguages[GpuBackend.OpenCL] = KernelLanguage.OpenCL;
/// crossPlatformSettings.PreferredLanguages[GpuBackend.Vulkan] = KernelLanguage.SPIRV;
/// cudaSettings.EnableLanguageTranslation = true;
/// 
/// // C#-first configuration for .NET developers
/// var csharpSettings = new KernelLanguageSettings();
/// foreach (var platform in Enum.GetValues&lt;GpuBackend&gt;())
/// {
///     csharpSettings.PreferredLanguages[platform] = KernelLanguage.CSharp;
/// }
/// csharpSettings.EnableLanguageTranslation = true;
/// </code>
/// </example>
public class KernelLanguageSettings
{
    /// <summary>
    /// Gets or sets the preferred kernel language for each compute platform.
    /// </summary>
    /// <value>
    /// A dictionary mapping <see cref="GpuBackend"/> values to <see cref="KernelLanguage"/> preferences.
    /// Default includes optimal language selection for each platform.
    /// </value>
    /// <remarks>
    /// This dictionary defines the preferred source language for kernel compilation on each
    /// platform. The DotCompute backend will attempt to use the specified language when
    /// compiling kernels for the corresponding platform.
    /// 
    /// <para>
    /// Default language mappings are optimized for performance and platform capabilities:
    /// - <see cref="GpuBackend.Cuda"/>: <see cref="KernelLanguage.CUDA"/> for native performance
    /// - <see cref="GpuBackend.OpenCL"/>: <see cref="KernelLanguage.OpenCL"/> for compatibility
    /// - <see cref="GpuBackend.DirectCompute"/>: <see cref="KernelLanguage.HLSL"/> for DirectX integration
    /// - <see cref="GpuBackend.Metal"/>: <see cref="KernelLanguage.MSL"/> for Apple optimization
    /// - <see cref="GpuBackend.Vulkan"/>: <see cref="KernelLanguage.HLSL"/> with SPIR-V compilation
    /// </para>
    /// 
    /// <para>
    /// Language selection considerations:
    /// - Performance: Native platform languages typically offer best performance
    /// - Portability: Cross-platform languages (OpenCL, C#) improve portability
    /// - Development experience: C# provides familiar syntax and tooling
    /// - Feature support: Some languages expose more platform-specific features
    /// - Ecosystem: Available libraries, tools, and community support
    /// </para>
    /// 
    /// <para>
    /// If a preferred language is not supported on a platform, the backend will:
    /// 1. Attempt automatic translation if <see cref="EnableLanguageTranslation"/> is enabled
    /// 2. Fall back to a supported language for the platform
    /// 3. Use C# as the ultimate fallback with translation
    /// </para>
    /// </remarks>
    /// <example>
    /// <code>
    /// // Performance-oriented configuration
    /// languageSettings.PreferredLanguages[GpuBackend.Cuda] = KernelLanguage.CUDA;
    /// languageSettings.PreferredLanguages[GpuBackend.OpenCL] = KernelLanguage.OpenCL;
    /// languageSettings.PreferredLanguages[GpuBackend.Metal] = KernelLanguage.MSL;
    /// 
    /// // Developer-friendly configuration
    /// foreach (var backend in Enum.GetValues&lt;GpuBackend&gt;())
    /// {
    ///     languageSettings.PreferredLanguages[backend] = KernelLanguage.CSharp;
    /// }
    /// 
    /// // Portability-focused configuration
    /// languageSettings.PreferredLanguages[GpuBackend.Cuda] = KernelLanguage.OpenCL;
    /// languageSettings.PreferredLanguages[GpuBackend.OpenCL] = KernelLanguage.OpenCL;
    /// languageSettings.PreferredLanguages[GpuBackend.Vulkan] = KernelLanguage.SPIRV;
    /// </code>
    /// </example>
    public Dictionary<GpuBackend, KernelLanguage> PreferredLanguages { get; set; } = new()
    {
        { GpuBackend.Cuda, KernelLanguage.CUDA },
        { GpuBackend.OpenCL, KernelLanguage.OpenCL },
        { GpuBackend.DirectCompute, KernelLanguage.HLSL },
        { GpuBackend.Metal, KernelLanguage.MSL },
        { GpuBackend.Vulkan, KernelLanguage.HLSL }
    };

    /// <summary>
    /// Gets or sets whether to enable automatic language translation between kernel languages.
    /// </summary>
    /// <value>
    /// <c>true</c> to enable automatic language translation; otherwise, <c>false</c>. Default is <c>true</c>.
    /// </value>
    /// <remarks>
    /// Automatic language translation allows the DotCompute backend to automatically
    /// convert kernel source code between different programming languages when the
    /// preferred language is not available or supported on a target platform.
    /// 
    /// <para>
    /// Translation capabilities:
    /// - C# to platform-specific languages (CUDA, OpenCL, HLSL, MSL)
    /// - OpenCL to other C-based languages with platform adaptations
    /// - HLSL to SPIR-V for Vulkan compatibility
    /// - Basic cross-translation between similar C-based languages
    /// - Automatic API and syntax adaptation for platform differences
    /// </para>
    /// 
    /// <para>
    /// Translation benefits:
    /// - Improved platform compatibility and portability
    /// - Reduced need for multiple language implementations
    /// - Automatic optimization for target platform capabilities
    /// - Simplified development workflow for multi-platform deployment
    /// - Fallback support when preferred languages are unavailable
    /// </para>
    /// 
    /// <para>
    /// Translation limitations:
    /// - May not preserve all language-specific optimizations
    /// - Some advanced language features may not translate perfectly
    /// - Potential performance overhead from translated code
    /// - Debugging information may be less precise in translated code
    /// - Complex language constructs may require manual implementation
    /// </para>
    /// 
    /// <para>
    /// Translation is performed at compilation time and cached to minimize overhead.
    /// The system uses sophisticated analysis to ensure translated code maintains
    /// semantic correctness while optimizing for the target platform.
    /// </para>
    /// </remarks>
    public bool EnableLanguageTranslation { get; set; } = true;

    /// <summary>
    /// Gets or sets custom preprocessor definitions for kernel compilation.
    /// </summary>
    /// <value>
    /// A dictionary mapping preprocessor macro names to their values. 
    /// Empty values represent macro definitions without values.
    /// </value>
    /// <remarks>
    /// Preprocessor definitions allow you to control conditional compilation and
    /// provide compile-time constants to kernel source code. These definitions
    /// are applied during kernel compilation and can affect code generation,
    /// optimization, and feature selection.
    /// 
    /// <para>
    /// Common use cases for preprocessor definitions:
    /// - Feature flags to enable/disable specific functionality
    /// - Platform-specific code paths and optimizations
    /// - Compile-time constants for algorithm parameters
    /// - Debug flags for development and troubleshooting
    /// - Version-specific compatibility macros
    /// </para>
    /// 
    /// <para>
    /// Definition formats:
    /// - Simple macro: <c>{"DEBUG", ""}</c> defines <c>#define DEBUG</c>
    /// - Valued macro: <c>{"BLOCK_SIZE", "256"}</c> defines <c>#define BLOCK_SIZE 256</c>
    /// - Complex value: <c>{"PI", "3.14159f"}</c> defines <c>#define PI 3.14159f</c>
    /// </para>
    /// 
    /// <para>
    /// Platform considerations:
    /// - CUDA: Uses nvcc preprocessor with C++ extensions
    /// - OpenCL: Uses standard C preprocessor with OpenCL extensions
    /// - HLSL: Uses HLSL compiler preprocessor with shader-specific features
    /// - MSL: Uses Metal compiler preprocessor with C++ extensions
    /// - C#: Definitions may be converted to constants in translated code
    /// </para>
    /// 
    /// <para>
    /// The preprocessor definitions are applied to all kernel compilations.
    /// For platform-specific definitions, consider using conditional logic
    /// within the kernel code or platform-specific build configurations.
    /// </para>
    /// </remarks>
    /// <example>
    /// <code>
    /// // Performance and debugging definitions
    /// languageSettings.PreprocessorDefines["BLOCK_SIZE"] = "256";
    /// languageSettings.PreprocessorDefines["ENABLE_BOUNDS_CHECK"] = "";  // Simple flag
    /// languageSettings.PreprocessorDefines["MAX_ITERATIONS"] = "1000";
    /// languageSettings.PreprocessorDefines["PI"] = "3.14159265359f";
    /// 
    /// // Platform-specific optimizations
    /// languageSettings.PreprocessorDefines["USE_FAST_MATH"] = "";
    /// languageSettings.PreprocessorDefines["UNROLL_LOOPS"] = "4";
    /// languageSettings.PreprocessorDefines["CACHE_LINE_SIZE"] = "128";
    /// 
    /// // Feature flags
    /// languageSettings.PreprocessorDefines["ENABLE_DOUBLE_PRECISION"] = "";
    /// languageSettings.PreprocessorDefines["USE_TEXTURE_MEMORY"] = "";
    /// languageSettings.PreprocessorDefines["DEBUG_VERBOSE"] = "";
    /// </code>
    /// </example>
    public Dictionary<string, string> PreprocessorDefines { get; set; } = new();

    /// <summary>
    /// Gets or sets the list of include directories for kernel compilation.
    /// </summary>
    /// <value>
    /// A list of directory paths to search for header files and includes.
    /// Paths can be absolute or relative to the application working directory.
    /// </value>
    /// <remarks>
    /// Include directories specify the search paths for header files, libraries,
    /// and other resources referenced by kernel source code. This enables modular
    /// kernel development with shared code, libraries, and platform-specific headers.
    /// 
    /// <para>
    /// Include directory use cases:
    /// - Shared kernel utilities and common functions
    /// - Platform-specific header files and extensions
    /// - Third-party GPU libraries and frameworks
    /// - Algorithm implementations and mathematical functions
    /// - Custom data structures and type definitions
    /// </para>
    /// 
    /// <para>
    /// Path resolution:
    /// - Absolute paths: Used directly for include searches
    /// - Relative paths: Resolved relative to application working directory
    /// - Environment variables: Expanded if supported by platform
    /// - Platform conventions: May follow platform-specific search patterns
    /// </para>
    /// 
    /// <para>
    /// Platform-specific behavior:
    /// - CUDA: Uses nvcc include search with standard CUDA headers
    /// - OpenCL: Uses compiler-specific include paths and OpenCL headers
    /// - HLSL: Uses HLSL compiler include search with DirectX headers
    /// - MSL: Uses Metal compiler include search with Metal framework headers
    /// - C#: May be converted to using statements in translated code
    /// </para>
    /// 
    /// <para>
    /// Include directories are searched in the order specified. Standard system
    /// and platform-specific include directories are automatically included
    /// by the compiler and do not need to be specified explicitly.
    /// </para>
    /// 
    /// <para>
    /// Security consideration: Include directories should only reference trusted
    /// locations to prevent inclusion of malicious or unintended code.
    /// </para>
    /// </remarks>
    /// <example>
    /// <code>
    /// // Common kernel utilities
    /// languageSettings.IncludeDirectories.Add("/opt/app/kernels/common");
    /// languageSettings.IncludeDirectories.Add("./kernels/shared");
    /// 
    /// // Platform-specific libraries
    /// languageSettings.IncludeDirectories.Add("/usr/local/cuda/include");
    /// languageSettings.IncludeDirectories.Add("/opt/intel/opencl/include");
    /// 
    /// // Application-specific headers
    /// languageSettings.IncludeDirectories.Add("../shared/gpu-headers");
    /// languageSettings.IncludeDirectories.Add("./resources/kernels");
    /// 
    /// // Third-party libraries
    /// languageSettings.IncludeDirectories.Add("/opt/nvidia/math-libs/include");
    /// languageSettings.IncludeDirectories.Add("./third-party/thrust/include");
    /// </code>
    /// </example>
    public List<string> IncludeDirectories { get; set; } = new();
}
namespace Orleans.GpuBridge.Abstractions.Enums.Compilation;

/// <summary>
/// Defines the supported source languages for GPU kernel compilation.
/// </summary>
/// <remarks>
/// This enumeration represents the various programming languages and
/// intermediate representations that can be compiled into GPU kernels.
/// Different GPU backends may support different subsets of these languages.
/// 
/// <para>
/// The availability of specific languages depends on:
/// - The GPU backend implementation (CUDA, OpenCL, etc.)
/// - Runtime environment capabilities
/// - Driver and compiler toolchain support
/// </para>
/// 
/// <para>
/// When selecting a kernel language, consider:
/// - Target GPU architecture compatibility
/// - Performance characteristics and optimization support
/// - Development tools and debugging capabilities
/// - Portability requirements across different GPU vendors
/// </para>
/// </remarks>
public enum KernelLanguage
{
    /// <summary>
    /// C# source code compiled to GPU kernels.
    /// </summary>
    /// <remarks>
    /// Represents C# code that is transpiled or compiled to GPU-executable
    /// code. This allows .NET developers to write GPU kernels using familiar
    /// C# syntax and constructs, with automatic translation to the target
    /// GPU language.
    /// 
    /// <para>
    /// Advantages:
    /// - Familiar syntax for .NET developers
    /// - Type safety and IntelliSense support
    /// - Integration with .NET tooling and debugging
    /// - Automatic memory management translation
    /// </para>
    /// 
    /// <para>
    /// Limitations:
    /// - Not all C# constructs are supported on GPU
    /// - May have performance overhead from translation
    /// - Limited to subset of .NET API surface
    /// </para>
    /// </remarks>
    /// <example>
    /// <code>
    /// // Example C# kernel method
    /// [GpuKernel]
    /// public static float VectorAdd(float[] a, float[] b, int index)
    /// {
    ///     return a[index] + b[index];
    /// }
    /// </code>
    /// </example>
    CSharp,

    /// <summary>
    /// CUDA C/C++ source code for NVIDIA GPUs.
    /// </summary>
    /// <remarks>
    /// CUDA (Compute Unified Device Architecture) is NVIDIA's parallel
    /// computing platform and programming model. CUDA kernels are written
    /// in an extended C/C++ language with special syntax for GPU programming.
    /// 
    /// <para>
    /// Advantages:
    /// - Native performance on NVIDIA GPUs
    /// - Extensive optimization support
    /// - Rich ecosystem of libraries and tools
    /// - Fine-grained control over GPU resources
    /// </para>
    /// 
    /// <para>
    /// Limitations:
    /// - NVIDIA GPU exclusive
    /// - Requires CUDA SDK and compatible drivers
    /// - Platform-specific code
    /// </para>
    /// </remarks>
    /// <example>
    /// <code>
    /// // Example CUDA kernel
    /// __global__ void vectorAdd(float* a, float* b, float* c, int n)
    /// {
    ///     int i = blockIdx.x * blockDim.x + threadIdx.x;
    ///     if (i &lt; n) c[i] = a[i] + b[i];
    /// }
    /// </code>
    /// </example>
    CUDA,

    /// <summary>
    /// OpenCL C source code for cross-platform GPU computing.
    /// </summary>
    /// <remarks>
    /// OpenCL (Open Computing Language) is an open standard for parallel
    /// computing across heterogeneous platforms including CPUs, GPUs, and
    /// other accelerators. OpenCL C is based on C99 with extensions for
    /// parallel programming.
    /// 
    /// <para>
    /// Advantages:
    /// - Cross-platform compatibility (NVIDIA, AMD, Intel)
    /// - Industry standard with broad support
    /// - Works on CPUs and other accelerators
    /// - Portable across vendors
    /// </para>
    /// 
    /// <para>
    /// Limitations:
    /// - More verbose than vendor-specific solutions
    /// - May not expose all hardware-specific features
    /// - Can have performance variations across implementations
    /// </para>
    /// </remarks>
    /// <example>
    /// <code>
    /// // Example OpenCL kernel
    /// __kernel void vector_add(__global float* a, __global float* b, __global float* c)
    /// {
    ///     int gid = get_global_id(0);
    ///     c[gid] = a[gid] + b[gid];
    /// }
    /// </code>
    /// </example>
    OpenCL,

    /// <summary>
    /// HLSL (High-Level Shading Language) source code.
    /// </summary>
    /// <remarks>
    /// HLSL is Microsoft's shading language used primarily for graphics
    /// programming but increasingly used for general-purpose compute shaders.
    /// Commonly used with DirectX and D3D12 compute pipelines.
    /// 
    /// <para>
    /// Advantages:
    /// - Well-integrated with DirectX ecosystem
    /// - Good tooling support in Visual Studio
    /// - Suitable for graphics and compute workloads
    /// - Strong Windows platform integration
    /// </para>
    /// 
    /// <para>
    /// Limitations:
    /// - Primarily Windows/DirectX focused
    /// - Limited cross-platform support
    /// - Graphics-oriented language design
    /// </para>
    /// </remarks>
    /// <example>
    /// <code>
    /// // Example HLSL compute shader
    /// [numthreads(256, 1, 1)]
    /// void CSMain(uint3 id : SV_DispatchThreadID)
    /// {
    ///     outputBuffer[id.x] = inputBufferA[id.x] + inputBufferB[id.x];
    /// }
    /// </code>
    /// </example>
    HLSL,

    /// <summary>
    /// MSL (Metal Shading Language) source code for Apple platforms.
    /// </summary>
    /// <remarks>
    /// MSL is Apple's shading language used for GPU programming on iOS,
    /// macOS, and other Apple platforms. Based on C++14 with GPU-specific
    /// extensions and optimizations.
    /// 
    /// <para>
    /// Advantages:
    /// - Optimized for Apple hardware (M-series, A-series chips)
    /// - Excellent integration with Apple development tools
    /// - High performance on Apple platforms
    /// - Modern C++-based syntax
    /// </para>
    /// 
    /// <para>
    /// Limitations:
    /// - Apple platforms exclusive
    /// - Limited cross-platform portability
    /// - Requires Metal framework support
    /// </para>
    /// </remarks>
    /// <example>
    /// <code>
    /// // Example MSL compute kernel
    /// kernel void vector_add(device float* a [[buffer(0)]],
    ///                       device float* b [[buffer(1)]],
    ///                       device float* c [[buffer(2)]],
    ///                       uint index [[thread_position_in_grid]])
    /// {
    ///     c[index] = a[index] + b[index];
    /// }
    /// </code>
    /// </example>
    MSL,

    /// <summary>
    /// SPIR-V (Standard Portable Intermediate Representation) bytecode.
    /// </summary>
    /// <remarks>
    /// SPIR-V is a binary intermediate language for graphics and compute
    /// kernels. It serves as a portable compilation target that can be
    /// consumed by various GPU drivers and runtimes.
    /// 
    /// <para>
    /// Advantages:
    /// - Vendor-neutral intermediate representation
    /// - Portable across different GPU APIs (Vulkan, OpenCL)
    /// - Enables offline compilation and caching
    /// - Compact binary format
    /// </para>
    /// 
    /// <para>
    /// Limitations:
    /// - Binary format, not human-readable
    /// - Requires specialized tools for generation/inspection
    /// - Abstraction may limit access to vendor-specific features
    /// </para>
    /// </remarks>
    /// <example>
    /// <code>
    /// // SPIR-V is binary format, typically generated from higher-level languages
    /// // Example shows conceptual representation, actual format is binary
    /// OpCapability Shader
    /// OpMemoryModel Logical GLSL450
    /// OpEntryPoint GLCompute %main "main"
    /// </code>
    /// </example>
    SPIRV,

    /// <summary>
    /// PTX (Parallel Thread Execution) assembly code for NVIDIA GPUs.
    /// </summary>
    /// <remarks>
    /// PTX is NVIDIA's low-level parallel thread execution instruction set
    /// architecture. It serves as a portable assembly language that is
    /// compiled to native GPU code at runtime or installation time.
    /// 
    /// <para>
    /// Advantages:
    /// - Low-level control over GPU execution
    /// - Portable across NVIDIA GPU generations
    /// - Enables hand-optimized critical code paths
    /// - Direct access to hardware features
    /// </para>
    /// 
    /// <para>
    /// Limitations:
    /// - NVIDIA GPU exclusive
    /// - Assembly-level programming complexity
    /// - Requires deep GPU architecture knowledge
    /// - Platform-specific instruction set
    /// </para>
    /// </remarks>
    /// <example>
    /// <code>
    /// // Example PTX code
    /// .version 7.0
    /// .target sm_75
    /// .visible .entry vector_add(.param .u64 ptr_a, .param .u64 ptr_b, .param .u64 ptr_c)
    /// {
    ///     ld.param.u64    %rd1, [ptr_a];
    ///     ld.param.u64    %rd2, [ptr_b];
    ///     ld.global.f32   %f1, [%rd1];
    ///     ld.global.f32   %f2, [%rd2];
    ///     add.f32         %f3, %f1, %f2;
    ///     st.global.f32   [%rd3], %f3;
    ///     ret;
    /// }
    /// </code>
    /// </example>
    PTX
}
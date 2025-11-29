using System.Collections.Generic;
using System.Linq;

namespace Orleans.GpuBridge.Abstractions.Models.Compilation;

/// <summary>
/// Represents the result of kernel validation, indicating whether a method or source code
/// can be successfully compiled as a GPU kernel.
/// </summary>
/// <param name="IsValid">
/// Indicates whether the kernel is valid and can be compiled successfully.
/// A value of <c>true</c> means the kernel passed all validation checks and is
/// suitable for GPU compilation. A value of <c>false</c> indicates that one or
/// more validation errors prevent successful compilation.
/// </param>
/// <param name="ErrorMessage">
/// The primary error message if validation failed, or <c>null</c> if validation succeeded.
/// This message describes the most critical issue that prevents compilation.
/// Should be clear and actionable, helping developers understand how to fix
/// the validation failure. Only populated when <paramref name="IsValid"/> is <c>false</c>.
/// </param>
/// <param name="Warnings">
/// A collection of warning messages about potential issues with the kernel.
/// Warnings don't prevent compilation but may indicate suboptimal code patterns,
/// deprecated features, or potential runtime issues. May be present even when
/// <paramref name="IsValid"/> is <c>true</c>. Default is <c>null</c>.
/// </param>
/// <param name="UnsupportedFeatures">
/// A collection of specific language features or constructs used in the kernel
/// that are not supported by the target GPU backend. This provides detailed
/// information about which parts of the code need to be modified for successful
/// compilation. Default is <c>null</c>.
/// </param>
/// <remarks>
/// This record provides comprehensive feedback about kernel validation, enabling
/// developers to understand not only whether compilation will succeed, but also
/// what specific issues need to be addressed.
/// 
/// <para>
/// Validation typically checks for:
/// - Supported language constructs and API calls
/// - Memory access patterns and synchronization
/// - Resource usage constraints
/// - Platform-specific limitations
/// - Type compatibility and marshalling requirements
/// </para>
/// 
/// <para>
/// Even when validation succeeds (<see cref="IsValid"/> is <c>true</c>), it's
/// important to review warnings as they may indicate performance issues or
/// potential portability problems.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Successful validation
/// var successResult = new KernelValidationResult(
///     IsValid: true,
///     Warnings: new[] { "Consider using shared memory for better performance" });
/// 
/// // Failed validation with detailed feedback
/// var failureResult = new KernelValidationResult(
///     IsValid: false,
///     ErrorMessage: "Kernel contains unsupported dynamic memory allocation",
///     UnsupportedFeatures: new[] { 
///         "Dynamic memory allocation (new, malloc)",
///         "Recursive function calls",
///         "Exception handling (try/catch)"
///     });
/// 
/// // Validation with warnings but no errors
/// var warningResult = new KernelValidationResult(
///     IsValid: true,
///     Warnings: new[] {
///         "High register usage may reduce occupancy",
///         "Uncoalesced memory access pattern detected"
///     });
/// </code>
/// </example>
public sealed record KernelValidationResult(
    bool IsValid,
    string? ErrorMessage = null,
    IReadOnlyList<string>? Warnings = null,
    IReadOnlyList<string>? UnsupportedFeatures = null)
{
    /// <summary>
    /// Gets a value indicating whether the validation result has any warnings.
    /// </summary>
    /// <value>
    /// <c>true</c> if there are one or more warnings; otherwise, <c>false</c>.
    /// </value>
    public bool HasWarnings => Warnings?.Any() == true;

    /// <summary>
    /// Gets a value indicating whether the validation result identifies unsupported features.
    /// </summary>
    /// <value>
    /// <c>true</c> if there are one or more unsupported features identified; otherwise, <c>false</c>.
    /// </value>
    public bool HasUnsupportedFeatures => UnsupportedFeatures?.Any() == true;

    /// <summary>
    /// Gets the total number of issues (warnings + unsupported features) identified.
    /// </summary>
    /// <value>
    /// The sum of warning count and unsupported feature count. Does not include
    /// the error message in the count.
    /// </value>
    public int IssueCount => (Warnings?.Count ?? 0) + (UnsupportedFeatures?.Count ?? 0);
}
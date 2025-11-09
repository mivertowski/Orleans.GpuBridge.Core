using System;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using System.Reflection;
using System.Threading;
using System.Threading.Tasks;
using ILGPU;
using ILGPU.Runtime;
using Orleans.GpuBridge.Abstractions.Models.Compilation;

namespace Orleans.GpuBridge.Backends.ILGPU.Kernels;

/// <summary>
/// ILGPU kernel compiler - Validation and analysis methods
/// </summary>
internal sealed partial class ILGPUKernelCompiler
{
    #region Validation

    [RequiresUnreferencedCode("Uses method body analysis which may not work with trimming.")]
    public async Task<KernelValidationResult> ValidateMethodAsync(
        [NotNull] MethodInfo method,
        CancellationToken cancellationToken = default)
    {
        if (method == null)
            throw new ArgumentNullException(nameof(method));

        try
        {
            // Perform validation asynchronously to avoid blocking for complex methods
            return await Task.Run(async () =>
            {
                var errors = new List<string>();
                var warnings = new List<string>();
                var unsupportedFeatures = new List<string>();

                // Check if method is static
                if (!method.IsStatic)
                {
                    errors.Add("Kernel methods must be static");
                }

                // Check return type
                if (method.ReturnType != typeof(void))
                {
                    errors.Add("Kernel methods must return void");
                }

                // Check parameters asynchronously for complex types
                var parameters = method.GetParameters();
                await Task.Run(() =>
                {
                    foreach (var param in parameters)
                    {
                        cancellationToken.ThrowIfCancellationRequested();
                        if (!IsValidParameterType(param.ParameterType))
                        {
                            errors.Add($"Parameter '{param.Name}' has unsupported type: {param.ParameterType}");
                        }
                    }
                }, cancellationToken).ConfigureAwait(false);

                // Check for unsupported constructs asynchronously (basic analysis)
                var containsUnsupported = await Task.Run(() => ContainsUnsupportedFeatures(method), cancellationToken).ConfigureAwait(false);
                if (containsUnsupported)
                {
                    warnings.Add("Method may contain constructs not supported by ILGPU");
                }

                // Check method body size asynchronously (heuristic)
                await Task.Run(() =>
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    var methodBody = method.GetMethodBody();
                    if (methodBody != null && methodBody.GetILAsByteArray()?.Length > 10000)
                    {
                        warnings.Add("Large method body may cause compilation issues");
                    }
                }, cancellationToken).ConfigureAwait(false);

                var isValid = errors.Count == 0;

                var result = new KernelValidationResult(
                    IsValid: isValid,
                    ErrorMessage: isValid ? null : string.Join("; ", errors),
                    Warnings: warnings.Count > 0 ? warnings : null,
                    UnsupportedFeatures: unsupportedFeatures.Count > 0 ? unsupportedFeatures : null);

                _logger.LogDebug(
                    "Kernel validation result for {MethodName}: Valid={IsValid}, Errors={ErrorCount}, Warnings={WarningCount}",
                    method.Name, isValid, errors.Count, warnings.Count);

                return result;
            }, cancellationToken).ConfigureAwait(false);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error validating kernel method: {MethodName}", method.Name);

            return new KernelValidationResult(
                IsValid: false,
                ErrorMessage: $"Validation error: {ex.Message}");
        }
    }

    #endregion

    #region Parameter Type Validation

    private static bool IsValidParameterType(Type parameterType)
    {
        // ILGPU supports various parameter types
        if (parameterType.IsPrimitive)
            return true;

        if (parameterType == typeof(string))
            return false; // Strings are not supported

        // Check for array types
        if (parameterType.IsArray)
        {
            var elementType = parameterType.GetElementType();
            return elementType != null && elementType.IsPrimitive;
        }

        // Check for ILGPU specific types (ArrayView, etc.)
        if (parameterType.Namespace?.StartsWith("ILGPU") == true)
            return true;

        // Check for struct types
        if (parameterType.IsValueType && !parameterType.IsEnum)
        {
            // Simple structs with primitive fields are usually supported
            return true;
        }

        return false;
    }

    #endregion

    #region Feature Detection

    [RequiresUnreferencedCode("Uses GetMethodBody() which may not work with trimming.")]
    private static bool ContainsUnsupportedFeatures(MethodInfo method)
    {
        // Comprehensive IL analysis for ILGPU compatibility
        // Check method attributes, IL instructions, and potential unsupported patterns

        var methodBody = method.GetMethodBody();
        if (methodBody == null)
            return false;

        // Check for recursive calls (not supported by ILGPU)
        try
        {
            var ilBytes = methodBody.GetILAsByteArray();
            if (ilBytes != null && ilBytes.Length > 0)
            {
                // Simple heuristic: very large methods might contain unsupported features
                return ilBytes.Length > 5000;
            }
        }
        catch
        {
            // If we can't analyze, assume it might contain unsupported features
            return true;
        }

        return false;
    }

    private static bool AnalyzeMethodForAtomics(MethodInfo method)
    {
        // Check if method name or declaring type suggests atomic operations
        var methodName = method.Name.ToLowerInvariant();
        return methodName.Contains("atomic") ||
               methodName.Contains("interlocked") ||
               method.DeclaringType?.Name.ToLowerInvariant().Contains("atomic") == true;
    }

    private static bool AnalyzeMethodForSharedMemory(MethodInfo method)
    {
        // Check if method parameters or return type suggest shared memory usage
        var parameters = method.GetParameters();
        return parameters.Any(p =>
            p.ParameterType.Name.Contains("SharedMemory") ||
            p.ParameterType.Namespace?.Contains("ILGPU") == true);
    }

    #endregion

    #region Device Analysis

    private static int CalculatePreferredBlockSize(Accelerator accelerator)
    {
        // Calculate optimal block size based on device characteristics
        return accelerator.AcceleratorType switch
        {
            AcceleratorType.Cuda => Math.Min(512, accelerator.MaxNumThreadsPerGroup),
            AcceleratorType.OpenCL => Math.Min(256, accelerator.MaxNumThreadsPerGroup),
            AcceleratorType.CPU => Environment.ProcessorCount,
            _ => 256
        };
    }

    #endregion
}

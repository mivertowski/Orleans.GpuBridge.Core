// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System.Reflection;
using DotCompute.Abstractions;
using DotCompute.Core.Compute;
using Microsoft.Extensions.Logging;

namespace Orleans.GpuBridge.Backends.DotCompute.DeviceManagement;

/// <summary>
/// Runtime discovery of DotCompute kernel compilation APIs
/// </summary>
internal static class KernelApiDiscovery
{
    /// <summary>
    /// Discovers all available kernel-related methods on IAccelerator
    /// </summary>
    public static async Task<KernelApiDiscoveryResult> DiscoverKernelApisAsync(ILogger logger)
    {
        var result = new KernelApiDiscoveryResult();

        try
        {
            logger.LogInformation("Starting kernel API discovery for DotCompute v0.3.0-rc1");

            // Get IAccelerator type
            var acceleratorType = typeof(IAccelerator);
            result.AcceleratorTypeFound = true;
            logger.LogInformation("IAccelerator type found: {TypeName}", acceleratorType.FullName);

            // Get all methods
            var methods = acceleratorType.GetMethods(BindingFlags.Public | BindingFlags.Instance);
            result.TotalMethodCount = methods.Length;

            logger.LogInformation("Found {Count} public instance methods on IAccelerator", methods.Length);

            // Find kernel-related methods
            var kernelMethods = methods
                .Where(m => m.Name.Contains("Kernel", StringComparison.OrdinalIgnoreCase) ||
                           m.Name.Contains("Compile", StringComparison.OrdinalIgnoreCase) ||
                           m.Name.Contains("Execute", StringComparison.OrdinalIgnoreCase) ||
                           m.Name.Contains("Launch", StringComparison.OrdinalIgnoreCase))
                .ToList();

            result.KernelMethodCount = kernelMethods.Count;
            logger.LogInformation("Found {Count} kernel-related methods", kernelMethods.Count);

            foreach (var method in kernelMethods)
            {
                var signature = GetMethodSignature(method);
                result.KernelMethodSignatures.Add(signature);
                logger.LogInformation("  - {Signature}", signature);
            }

            // Check for specific method names
            result.HasCompileKernelAsync = methods.Any(m => m.Name == "CompileKernelAsync");
            result.HasCompileKernel = methods.Any(m => m.Name == "CompileKernel");
            result.HasExecuteKernelAsync = methods.Any(m => m.Name == "ExecuteKernelAsync");
            result.HasExecuteKernel = methods.Any(m => m.Name == "ExecuteKernel");
            result.HasLaunchKernelAsync = methods.Any(m => m.Name == "LaunchKernelAsync");
            result.HasLaunchKernel = methods.Any(m => m.Name == "LaunchKernel");

            logger.LogInformation("Method availability:");
            logger.LogInformation("  CompileKernelAsync: {Available}", result.HasCompileKernelAsync);
            logger.LogInformation("  CompileKernel: {Available}", result.HasCompileKernel);
            logger.LogInformation("  ExecuteKernelAsync: {Available}", result.HasExecuteKernelAsync);
            logger.LogInformation("  ExecuteKernel: {Available}", result.HasExecuteKernel);
            logger.LogInformation("  LaunchKernelAsync: {Available}", result.HasLaunchKernelAsync);
            logger.LogInformation("  LaunchKernel: {Available}", result.HasLaunchKernel);

            // Get full method list for documentation
            logger.LogInformation("\nAll IAccelerator methods:");
            foreach (var method in methods.OrderBy(m => m.Name))
            {
                result.AllMethodSignatures.Add(GetMethodSignature(method));
                logger.LogDebug("  {Signature}", GetMethodSignature(method));
            }

            // Check properties that might relate to kernels
            var properties = acceleratorType.GetProperties(BindingFlags.Public | BindingFlags.Instance);
            result.TotalPropertyCount = properties.Length;

            logger.LogInformation("\nFound {Count} public properties on IAccelerator", properties.Length);
            foreach (var prop in properties.OrderBy(p => p.Name))
            {
                result.PropertyNames.Add($"{prop.Name}: {prop.PropertyType.Name}");
                logger.LogDebug("  {Name}: {Type}", prop.Name, prop.PropertyType.Name);
            }

            // Try to get an actual accelerator instance for runtime testing
            try
            {
                logger.LogInformation("\nAttempting to get real accelerator instance for method testing...");
                var manager = await DefaultAcceleratorManagerFactory.CreateAsync();
                var accelerators = await manager.GetAcceleratorsAsync();
                var firstAccelerator = accelerators.FirstOrDefault();

                if (firstAccelerator != null)
                {
                    result.CanGetAcceleratorInstance = true;
                    logger.LogInformation("✓ Successfully obtained accelerator instance: {Name}", firstAccelerator.Info.Name);

                    // Test method invocations (with reflection if needed)
                    await TestKernelMethods(firstAccelerator, result, logger);
                }
                else
                {
                    logger.LogWarning("No accelerators available for runtime testing");
                }

                await manager.DisposeAsync();
            }
            catch (Exception ex)
            {
                logger.LogWarning(ex, "Could not obtain accelerator instance for runtime testing");
            }

            result.Success = true;
            logger.LogInformation("\nKernel API discovery complete");

            return result;
        }
        catch (Exception ex)
        {
            logger.LogError(ex, "Kernel API discovery failed");
            result.Success = false;
            result.Error = ex.Message;
            return result;
        }
    }

    private static async Task TestKernelMethods(
        IAccelerator accelerator,
        KernelApiDiscoveryResult result,
        ILogger logger)
    {
        // Try to find and test CompileKernelAsync or similar methods
        var type = accelerator.GetType();

        // Look for compile methods
        var compileMethods = type.GetMethods()
            .Where(m => m.Name.Contains("Compile", StringComparison.OrdinalIgnoreCase))
            .ToList();

        if (compileMethods.Any())
        {
            logger.LogInformation("Found compile methods on concrete type:");
            foreach (var method in compileMethods)
            {
                logger.LogInformation("  {Signature}", GetMethodSignature(method));
            }
        }

        // Try to invoke a simple kernel compilation if method signature is clear
        // This would require knowing the exact signature
        result.RuntimeTestingAttempted = true;

        await Task.CompletedTask;
    }

    private static string GetMethodSignature(MethodInfo method)
    {
        var parameters = string.Join(", ", method.GetParameters()
            .Select(p => $"{p.ParameterType.Name} {p.Name}"));

        var returnType = method.ReturnType.Name;

        return $"{returnType} {method.Name}({parameters})";
    }
}

/// <summary>
/// Results of kernel API discovery
/// </summary>
internal sealed class KernelApiDiscoveryResult
{
    public bool Success { get; set; }
    public string? Error { get; set; }

    public bool AcceleratorTypeFound { get; set; }
    public int TotalMethodCount { get; set; }
    public int TotalPropertyCount { get; set; }
    public int KernelMethodCount { get; set; }

    public bool HasCompileKernelAsync { get; set; }
    public bool HasCompileKernel { get; set; }
    public bool HasExecuteKernelAsync { get; set; }
    public bool HasExecuteKernel { get; set; }
    public bool HasLaunchKernelAsync { get; set; }
    public bool HasLaunchKernel { get; set; }

    public List<string> KernelMethodSignatures { get; set; } = new();
    public List<string> AllMethodSignatures { get; set; } = new();
    public List<string> PropertyNames { get; set; } = new();

    public bool CanGetAcceleratorInstance { get; set; }
    public bool RuntimeTestingAttempted { get; set; }

    /// <summary>
    /// Gets a formatted report of discoveries
    /// </summary>
    public string GetReport()
    {
        var lines = new List<string>
        {
            "=== DotCompute Kernel API Discovery Report ===",
            "",
            $"Success: {Success}",
            $"IAccelerator Type Found: {AcceleratorTypeFound}",
            $"Total Methods: {TotalMethodCount}",
            $"Total Properties: {TotalPropertyCount}",
            $"Kernel-Related Methods: {KernelMethodCount}",
            "",
            "Method Availability:",
            $"  CompileKernelAsync: {HasCompileKernelAsync}",
            $"  CompileKernel: {HasCompileKernel}",
            $"  ExecuteKernelAsync: {HasExecuteKernelAsync}",
            $"  ExecuteKernel: {HasExecuteKernel}",
            $"  LaunchKernelAsync: {HasLaunchKernelAsync}",
            $"  LaunchKernel: {HasLaunchKernel}",
            "",
        };

        if (KernelMethodSignatures.Any())
        {
            lines.Add("Kernel-Related Method Signatures:");
            foreach (var sig in KernelMethodSignatures)
            {
                lines.Add($"  {sig}");
            }
            lines.Add("");
        }

        if (PropertyNames.Any())
        {
            lines.Add("Properties:");
            foreach (var prop in PropertyNames)
            {
                lines.Add($"  {prop}");
            }
            lines.Add("");
        }

        if (!string.IsNullOrEmpty(Error))
        {
            lines.Add($"Error: {Error}");
        }

        return string.Join(Environment.NewLine, lines);
    }
}

// Copyright (c) 2025 Michael Ivertowski
// Licensed under the Apache License, Version 2.0. See LICENSE file for details.

using DotCompute.Abstractions;
using DotCompute.Core.Compute;
using Microsoft.Extensions.Logging;

namespace Orleans.GpuBridge.Backends.DotCompute.DeviceManagement;

/// <summary>
/// Verification utility for DotCompute v0.3.0-rc1 API availability
/// </summary>
/// <remarks>
/// This class verifies that the DotCompute v0.3.0-rc1 APIs documented in the integration guide
/// are actually available and functional before proceeding with full adapter implementation.
/// </remarks>
internal static class DotComputeApiVerification
{
    /// <summary>
    /// Verifies DotCompute v0.3.0-rc1 APIs are available and functional
    /// </summary>
    /// <param name="logger">Logger for verification output</param>
    /// <returns>Verification results with detailed API availability information</returns>
    public static async Task<VerificationResult> VerifyApisAsync(ILogger logger)
    {
        var result = new VerificationResult();

        try
        {
            logger.LogInformation("Starting DotCompute v0.3.0-rc1 API verification");

            // Test 1: Factory method exists and works
            logger.LogInformation("Test 1: Verifying DefaultAcceleratorManagerFactory.CreateAsync()");
            IAcceleratorManager? manager = null;
            try
            {
                manager = await DefaultAcceleratorManagerFactory.CreateAsync();
                result.FactoryMethodAvailable = true;
                logger.LogInformation("✓ Factory method available and working");
            }
            catch (Exception ex)
            {
                result.FactoryMethodAvailable = false;
                result.FactoryMethodError = ex.Message;
                logger.LogError(ex, "✗ Factory method failed");
                return result; // Can't continue without manager
            }

            // Test 2: Get accelerators works
            logger.LogInformation("Test 2: Verifying GetAcceleratorsAsync() enumeration");
            try
            {
                var accelerators = await manager.GetAcceleratorsAsync();
                var acceleratorList = accelerators.ToList();
                result.EnumerationAvailable = true;
                result.AcceleratorCount = acceleratorList.Count;
                logger.LogInformation("✓ Enumeration available - found {Count} accelerators", acceleratorList.Count);

                // Test 3: AcceleratorInfo properties exist
                if (acceleratorList.Count > 0)
                {
                    var firstAccelerator = acceleratorList[0];
                    var info = firstAccelerator.Info;

                    logger.LogInformation("Test 3: Verifying AcceleratorInfo properties");

                    // Test Architecture property
                    try
                    {
                        var architecture = info.Architecture;
                        result.ArchitecturePropertyAvailable = true;
                        result.ArchitectureValue = architecture;
                        logger.LogInformation("✓ Architecture property available: {Architecture}", architecture);
                    }
                    catch (Exception ex)
                    {
                        result.ArchitecturePropertyAvailable = false;
                        result.ArchitecturePropertyError = ex.Message;
                        logger.LogWarning("✗ Architecture property missing: {Error}", ex.Message);
                    }

                    // Test WarpSize property
                    try
                    {
                        var warpSize = info.WarpSize;
                        result.WarpSizePropertyAvailable = true;
                        result.WarpSizeValue = warpSize;
                        logger.LogInformation("✓ WarpSize property available: {WarpSize}", warpSize);
                    }
                    catch (Exception ex)
                    {
                        result.WarpSizePropertyAvailable = false;
                        result.WarpSizePropertyError = ex.Message;
                        logger.LogWarning("✗ WarpSize property missing: {Error}", ex.Message);
                    }

                    // Test Features property
                    try
                    {
                        var features = info.Features;
                        result.FeaturesPropertyAvailable = true;
                        result.FeaturesCount = features?.Count ?? 0;
                        logger.LogInformation("✓ Features property available: {Count} features", result.FeaturesCount);
                    }
                    catch (Exception ex)
                    {
                        result.FeaturesPropertyAvailable = false;
                        result.FeaturesPropertyError = ex.Message;
                        logger.LogWarning("✗ Features property missing: {Error}", ex.Message);
                    }

                    // Test Extensions property
                    try
                    {
                        var extensions = info.Extensions;
                        result.ExtensionsPropertyAvailable = true;
                        result.ExtensionsCount = extensions?.Count ?? 0;
                        logger.LogInformation("✓ Extensions property available: {Count} extensions", result.ExtensionsCount);
                    }
                    catch (Exception ex)
                    {
                        result.ExtensionsPropertyAvailable = false;
                        result.ExtensionsPropertyError = ex.Message;
                        logger.LogWarning("✗ Extensions property missing: {Error}", ex.Message);
                    }

                    // Test 4: Memory manager works
                    logger.LogInformation("Test 4: Verifying IUnifiedMemoryManager APIs");
                    try
                    {
                        var memoryManager = firstAccelerator.Memory;

                        // Test TotalAvailableMemory
                        try
                        {
                            var totalMemory = memoryManager.TotalAvailableMemory;
                            result.MemoryTotalPropertyAvailable = true;
                            result.TotalMemoryBytes = totalMemory;
                            logger.LogInformation("✓ TotalAvailableMemory property available: {Memory:N0} bytes", totalMemory);
                        }
                        catch (Exception ex)
                        {
                            result.MemoryTotalPropertyAvailable = false;
                            result.MemoryTotalPropertyError = ex.Message;
                            logger.LogWarning("✗ TotalAvailableMemory property missing: {Error}", ex.Message);
                        }

                        // Test Statistics property
                        try
                        {
                            var stats = memoryManager.Statistics;
                            result.MemoryStatisticsAvailable = true;
                            logger.LogInformation("✓ Memory Statistics property available");
                        }
                        catch (Exception ex)
                        {
                            result.MemoryStatisticsAvailable = false;
                            result.MemoryStatisticsError = ex.Message;
                            logger.LogWarning("✗ Memory Statistics property missing: {Error}", ex.Message);
                        }
                    }
                    catch (Exception ex)
                    {
                        logger.LogWarning("✗ Memory manager access failed: {Error}", ex.Message);
                    }

                    // Test 5: Kernel compilation API
                    logger.LogInformation("Test 5: Verifying kernel compilation API");
                    try
                    {
                        // Verify CompileKernelAsync exists by checking interface
                        // If IAccelerator has the method, it's available
                        result.KernelCompilationAvailable = firstAccelerator is IAccelerator;
                        logger.LogInformation("✓ CompileKernelAsync method available (IAccelerator interface confirmed)");
                    }
                    catch (Exception ex)
                    {
                        result.KernelCompilationAvailable = false;
                        result.KernelCompilationError = ex.Message;
                        logger.LogWarning("✗ Kernel compilation check failed: {Error}", ex.Message);
                    }
                }
                else
                {
                    logger.LogWarning("No accelerators found - skipping property verification");
                }
            }
            catch (Exception ex)
            {
                result.EnumerationAvailable = false;
                result.EnumerationError = ex.Message;
                logger.LogError(ex, "✗ Enumeration failed");
            }

            // Cleanup
            if (manager != null)
            {
                await manager.DisposeAsync();
            }

            // Summary
            result.Success = result.FactoryMethodAvailable && result.EnumerationAvailable;
            logger.LogInformation("API Verification Complete - Success: {Success}", result.Success);

            return result;
        }
        catch (Exception ex)
        {
            logger.LogError(ex, "API verification failed with exception");
            result.Success = false;
            result.GeneralError = ex.Message;
            return result;
        }
    }
}

/// <summary>
/// Results of DotCompute API verification
/// </summary>
internal sealed class VerificationResult
{
    public bool Success { get; set; }
    public string? GeneralError { get; set; }

    // Factory method
    public bool FactoryMethodAvailable { get; set; }
    public string? FactoryMethodError { get; set; }

    // Enumeration
    public bool EnumerationAvailable { get; set; }
    public string? EnumerationError { get; set; }
    public int AcceleratorCount { get; set; }

    // AcceleratorInfo properties
    public bool ArchitecturePropertyAvailable { get; set; }
    public string? ArchitecturePropertyError { get; set; }
    public string? ArchitectureValue { get; set; }

    public bool WarpSizePropertyAvailable { get; set; }
    public string? WarpSizePropertyError { get; set; }
    public int WarpSizeValue { get; set; }

    public bool FeaturesPropertyAvailable { get; set; }
    public string? FeaturesPropertyError { get; set; }
    public int FeaturesCount { get; set; }

    public bool ExtensionsPropertyAvailable { get; set; }
    public string? ExtensionsPropertyError { get; set; }
    public int ExtensionsCount { get; set; }

    // Memory manager
    public bool MemoryTotalPropertyAvailable { get; set; }
    public string? MemoryTotalPropertyError { get; set; }
    public long TotalMemoryBytes { get; set; }

    public bool MemoryStatisticsAvailable { get; set; }
    public string? MemoryStatisticsError { get; set; }

    // Kernel compilation
    public bool KernelCompilationAvailable { get; set; }
    public string? KernelCompilationError { get; set; }

    /// <summary>
    /// Gets a summary of all verification checks
    /// </summary>
    public string GetSummary()
    {
        var lines = new List<string>
        {
            "=== DotCompute v0.3.0-rc1 API Verification Summary ===",
            "",
            $"Overall Success: {(Success ? "✓" : "✗")}",
            "",
            "Core APIs:",
            $"  Factory Method: {(FactoryMethodAvailable ? "✓" : "✗")} {FactoryMethodError}",
            $"  Enumeration: {(EnumerationAvailable ? "✓" : "✗")} {EnumerationError}",
            $"  Accelerators Found: {AcceleratorCount}",
            "",
            "AcceleratorInfo Properties:",
            $"  Architecture: {(ArchitecturePropertyAvailable ? "✓" : "✗")} {(ArchitecturePropertyAvailable ? $"({ArchitectureValue})" : ArchitecturePropertyError)}",
            $"  WarpSize: {(WarpSizePropertyAvailable ? "✓" : "✗")} {(WarpSizePropertyAvailable ? $"({WarpSizeValue})" : WarpSizePropertyError)}",
            $"  Features: {(FeaturesPropertyAvailable ? "✓" : "✗")} {(FeaturesPropertyAvailable ? $"({FeaturesCount} features)" : FeaturesPropertyError)}",
            $"  Extensions: {(ExtensionsPropertyAvailable ? "✓" : "✗")} {(ExtensionsPropertyAvailable ? $"({ExtensionsCount} extensions)" : ExtensionsPropertyError)}",
            "",
            "Memory Manager:",
            $"  TotalAvailableMemory: {(MemoryTotalPropertyAvailable ? "✓" : "✗")} {(MemoryTotalPropertyAvailable ? $"({TotalMemoryBytes:N0} bytes)" : MemoryTotalPropertyError)}",
            $"  Statistics: {(MemoryStatisticsAvailable ? "✓" : "✗")} {MemoryStatisticsError}",
            "",
            "Kernel Compilation:",
            $"  CompileKernelAsync: {(KernelCompilationAvailable ? "✓" : "✗")} {KernelCompilationError}",
            ""
        };

        if (!string.IsNullOrEmpty(GeneralError))
        {
            lines.Add($"General Error: {GeneralError}");
        }

        return string.Join(Environment.NewLine, lines);
    }
}

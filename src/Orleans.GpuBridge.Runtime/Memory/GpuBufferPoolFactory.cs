// Copyright (c) 2025 Michael Ivertowski
// Licensed under the Apache License, Version 2.0. See LICENSE file for details.

using System;
using Microsoft.Extensions.Logging;
using DotCompute.Backends.CUDA;
using DotCompute.Backends.CUDA.Memory;

namespace Orleans.GpuBridge.Runtime.Memory;

/// <summary>
/// Factory for creating GPU buffer pools with proper CUDA initialization.
/// </summary>
/// <remarks>
/// <para>
/// This factory handles the complexity of initializing CUDA context and memory manager.
/// It provides three creation modes:
/// <list type="bullet">
/// <item><description><b>GPU Mode</b>: Full GPU unified memory (requires CUDA)</description></item>
/// <item><description><b>CPU Fallback Mode</b>: CPU memory with GPU transfer support</description></item>
/// <item><description><b>Auto Mode</b>: GPU if available, otherwise CPU fallback</description></item>
/// </list>
/// </para>
/// </remarks>
public static class GpuBufferPoolFactory
{
    /// <summary>
    /// Creates a GPU buffer pool with automatic CUDA detection.
    /// </summary>
    /// <param name="logger">Logger instance.</param>
    /// <param name="deviceId">CUDA device ID (default: 0 for first GPU).</param>
    /// <returns>GPU buffer pool configured for GPU or CPU mode.</returns>
    /// <remarks>
    /// Attempts to initialize CUDA. If successful, uses GPU unified memory.
    /// If CUDA initialization fails, falls back to CPU memory with GPU transfer support.
    /// </remarks>
    public static GpuBufferPool CreateAuto(ILogger<GpuBufferPool> logger, int deviceId = 0)
    {
        try
        {
            return CreateGpuMode(logger, deviceId);
        }
        catch (Exception ex)
        {
            logger.LogWarning(
                ex,
                "CUDA initialization failed, falling back to CPU memory mode");

            return CreateCpuFallbackMode(logger);
        }
    }

    /// <summary>
    /// Creates a GPU buffer pool with GPU unified memory (requires CUDA).
    /// </summary>
    /// <param name="logger">Logger instance.</param>
    /// <param name="deviceId">CUDA device ID (default: 0 for first GPU).</param>
    /// <returns>GPU buffer pool configured for GPU unified memory.</returns>
    /// <exception cref="InvalidOperationException">CUDA initialization failed.</exception>
    public static GpuBufferPool CreateGpuMode(ILogger<GpuBufferPool> logger, int deviceId = 0)
    {
        // Create CUDA context
        var context = new CudaContext(deviceId);

        logger.LogInformation(
            "Created CUDA context for device {DeviceId}",
            deviceId);

        // Create CUDA memory manager
        var memoryManager = new CudaMemoryManager(context, logger);

        logger.LogInformation(
            "Created CUDA memory manager (total={TotalMemory}MB, max={MaxAllocation}MB)",
            memoryManager.TotalMemory / (1024 * 1024),
            memoryManager.MaxAllocationSize / (1024 * 1024));

        // Create buffer pool with GPU support
        return new GpuBufferPool(logger, context, memoryManager);
    }

    /// <summary>
    /// Creates a GPU buffer pool with CPU memory fallback (no CUDA required).
    /// </summary>
    /// <param name="logger">Logger instance.</param>
    /// <returns>GPU buffer pool configured for CPU memory with GPU transfer support.</returns>
    /// <remarks>
    /// CPU memory mode still supports GPU transfers via DotCompute's copy operations.
    /// Useful for development, testing, or environments without CUDA.
    /// </remarks>
    public static GpuBufferPool CreateCpuFallbackMode(ILogger<GpuBufferPool> logger)
    {
        logger.LogInformation(
            "Creating GPU buffer pool in CPU fallback mode (no CUDA initialization)");

        return new GpuBufferPool(logger);
    }
}

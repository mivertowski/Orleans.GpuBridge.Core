using System;
using Microsoft.Extensions.Logging;
using Orleans.GpuBridge.Abstractions;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using Orleans.GpuBridge.Abstractions.Providers.Execution.Parameters;
using Orleans.GpuBridge.Abstractions.Providers.Memory.Interfaces;

namespace Orleans.GpuBridge.Grains.Batch;

public sealed partial class GpuBatchGrainEnhanced<TIn, TOut>
{
    #region Memory Management

    private async Task<(IDeviceMemory inputMemory, IDeviceMemory outputMemory, TimeSpan allocTime)>
        AllocateGpuMemoryAsync(IReadOnlyList<TIn> batch)
    {
        var stopwatch = Stopwatch.StartNew();

        try
        {
            var inputSize = Marshal.SizeOf<TIn>() * batch.Count;
            var outputSize = Marshal.SizeOf<TOut>() * batch.Count;

            // Create memory allocation options
            var allocOptions = new Abstractions.Providers.Memory.Options.MemoryAllocationOptions(
                Type: Abstractions.Providers.Memory.Enums.MemoryType.Device,
                ZeroInitialize: false,
                PreferredDevice: _primaryDevice);

            // Allocate input buffer
            var inputMemory = await _memoryAllocator!.AllocateAsync(
                inputSize,
                allocOptions,
                CancellationToken.None).ConfigureAwait(false);

            // Allocate output buffer
            var outputMemory = await _memoryAllocator.AllocateAsync(
                outputSize,
                allocOptions,
                CancellationToken.None).ConfigureAwait(false);

            // Copy input data to GPU
            var inputBytes = MemoryMarshal.AsBytes(batch.ToArray().AsSpan()).ToArray();
            var pinnedInputBytes = GCHandle.Alloc(inputBytes, GCHandleType.Pinned);
            try
            {
                await inputMemory.CopyFromHostAsync(
                    pinnedInputBytes.AddrOfPinnedObject(),
                    0,
                    inputSize,
                    CancellationToken.None).ConfigureAwait(false);
            }
            finally
            {
                pinnedInputBytes.Free();
            }

            stopwatch.Stop();

            _logger.LogDebug(
                "Allocated GPU memory: Input={InputMB}MB, Output={OutputMB}MB in {Time}ms",
                inputSize / (1024 * 1024.0),
                outputSize / (1024 * 1024.0),
                stopwatch.ElapsedMilliseconds);

            return (inputMemory, outputMemory, stopwatch.Elapsed);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to allocate GPU memory");
            throw;
        }
    }

    private async Task<(List<TOut> results, TimeSpan readTime)> ReadResultsFromGpuAsync(
        IDeviceMemory outputMemory,
        int count)
    {
        var stopwatch = Stopwatch.StartNew();

        try
        {
            var outputSize = Marshal.SizeOf<TOut>() * count;
            var outputBuffer = new byte[outputSize];

            // Read results from GPU
            var pinnedOutputBuffer = GCHandle.Alloc(outputBuffer, GCHandleType.Pinned);
            try
            {
                await outputMemory.CopyToHostAsync(
                    pinnedOutputBuffer.AddrOfPinnedObject(),
                    0,
                    outputSize,
                    CancellationToken.None).ConfigureAwait(false);
            }
            finally
            {
                pinnedOutputBuffer.Free();
            }

            // Convert bytes to TOut array
            var results = new List<TOut>(count);
            var outputSpan = MemoryMarshal.Cast<byte, TOut>(outputBuffer);

            for (int i = 0; i < count; i++)
            {
                results.Add(outputSpan[i]);
            }

            stopwatch.Stop();

            _logger.LogDebug(
                "Read {Count} results from GPU in {Time}ms",
                count,
                stopwatch.ElapsedMilliseconds);

            return (results, stopwatch.Elapsed);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to read results from GPU");
            throw;
        }
    }

    private Task FreeGpuMemoryAsync(IDeviceMemory inputMemory, IDeviceMemory outputMemory)
    {
        try
        {
            inputMemory?.Dispose();
            outputMemory?.Dispose();
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Error freeing GPU memory (may already be freed)");
        }

        return Task.CompletedTask;
    }

    private long CalculateTotalMemorySize(IReadOnlyList<TIn> batch)
    {
        var inputSize = Marshal.SizeOf<TIn>() * batch.Count;
        var outputSize = Marshal.SizeOf<TOut>() * batch.Count;
        return inputSize + outputSize;
    }

    #endregion

    #region Kernel Execution Parameters

    private KernelExecutionParameters PrepareExecutionParameters(
        IDeviceMemory inputMemory,
        IDeviceMemory outputMemory,
        int count)
    {
        var memoryArgs = new Dictionary<string, IDeviceMemory>
        {
            ["input"] = inputMemory,
            ["output"] = outputMemory
        };

        var scalarArgs = new Dictionary<string, object>
        {
            ["count"] = count
        };

        var globalWorkSize = new[] { count };
        var localWorkSize = new[] { Math.Min(256, count) }; // 256 threads per block

        return new KernelExecutionParameters
        {
            GlobalWorkSize = globalWorkSize,
            LocalWorkSize = localWorkSize,
            MemoryArguments = memoryArgs,
            ScalarArguments = scalarArgs
        };
    }

    #endregion
}

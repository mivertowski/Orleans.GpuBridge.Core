using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Orleans;
using Orleans.GpuBridge.Abstractions;
using Orleans.GpuBridge.Grains.Enums;
using Orleans.GpuBridge.Grains.Interfaces;
using Orleans.GpuBridge.Grains.Models;
using Orleans.GpuBridge.Runtime.Extensions;
using Orleans.Hosting;
using Orleans.TestingHost;

namespace Orleans.GpuBridge.Sample.VectorAddition;

/// <summary>
/// Sample application demonstrating Orleans.GpuBridge.Core Ring Kernel API.
/// This sample showcases GPU memory management, DMA transfers, and resident memory operations.
/// </summary>
internal sealed class Program
{
    private const int VectorSize = 1024;
    private const string GrainKey = "vector-resident-grain";

    private static async Task<int> Main()
    {
        Console.WriteLine("═══════════════════════════════════════════════════════════════");
        Console.WriteLine("  Orleans.GpuBridge.Core - Ring Kernel API Sample");
        Console.WriteLine("  Demonstrating GPU Resident Memory & DMA Operations");
        Console.WriteLine("═══════════════════════════════════════════════════════════════");
        Console.WriteLine();

        TestCluster? cluster = null;

        try
        {
            // Step 1: Initialize Orleans cluster with GPU Bridge
            Console.WriteLine("▶ Step 1: Initializing Orleans cluster with GPU Bridge...");
            cluster = await InitializeOrleansClusterAsync();
            Console.WriteLine("✓ Orleans cluster initialized successfully");
            Console.WriteLine();

            // Step 2: Get GPU Resident Grain
            Console.WriteLine("▶ Step 2: Obtaining GPU Resident Grain...");
            var grainFactory = cluster.GrainFactory;
            var residentGrain = grainFactory.GetGrain<IGpuResidentGrain<float>>(GrainKey);
            Console.WriteLine($"✓ Acquired grain with key: {GrainKey}");
            Console.WriteLine();

            // Step 3: Allocate GPU memory using Ring Kernel API
            Console.WriteLine("▶ Step 3: Allocating GPU memory...");
            var sizeInBytes = VectorSize * sizeof(float);
            var inputHandle = await AllocateGpuMemoryAsync(residentGrain, sizeInBytes, "Input Buffer");
            var outputHandle = await AllocateGpuMemoryAsync(residentGrain, sizeInBytes, "Output Buffer");
            Console.WriteLine();

            // Step 4: Write test data to GPU (DMA Transfer)
            Console.WriteLine("▶ Step 4: Writing test data to GPU via DMA...");
            var testData = GenerateTestData(VectorSize);
            await WriteDmaDataAsync(residentGrain, inputHandle, testData);
            Console.WriteLine();

            // Step 5: Read data back from GPU (DMA Transfer)
            Console.WriteLine("▶ Step 5: Reading data back from GPU via DMA...");
            var readData = await ReadDmaDataAsync(residentGrain, inputHandle, VectorSize);
            await ValidateDataTransferAsync(testData, readData);
            Console.WriteLine();

            // Step 6: Display memory metrics
            Console.WriteLine("▶ Step 6: Retrieving GPU memory metrics...");
            await DisplayMemoryMetricsAsync(residentGrain);
            Console.WriteLine();

            // Step 7: Demonstrate kernel execution (if supported)
            Console.WriteLine("▶ Step 7: Demonstrating kernel execution...");
            await DemonstrateKernelExecutionAsync(residentGrain, inputHandle, outputHandle);
            Console.WriteLine();

            // Step 8: Release GPU memory
            Console.WriteLine("▶ Step 8: Releasing GPU memory allocations...");
            await ReleaseGpuMemoryAsync(residentGrain, inputHandle, "Input Buffer");
            await ReleaseGpuMemoryAsync(residentGrain, outputHandle, "Output Buffer");
            Console.WriteLine();

            // Step 9: Final cleanup
            Console.WriteLine("▶ Step 9: Performing final cleanup...");
            await residentGrain.ClearAsync();
            Console.WriteLine("✓ All resident memory cleared");
            Console.WriteLine();

            Console.WriteLine("═══════════════════════════════════════════════════════════════");
            Console.WriteLine("  Sample completed successfully!");
            Console.WriteLine("═══════════════════════════════════════════════════════════════");

            return 0;
        }
        catch (Exception ex)
        {
            Console.WriteLine();
            Console.WriteLine("═══════════════════════════════════════════════════════════════");
            Console.WriteLine("  ✗ ERROR OCCURRED");
            Console.WriteLine("═══════════════════════════════════════════════════════════════");
            Console.WriteLine($"Type: {ex.GetType().Name}");
            Console.WriteLine($"Message: {ex.Message}");
            Console.WriteLine($"Stack Trace:\n{ex.StackTrace}");
            Console.WriteLine("═══════════════════════════════════════════════════════════════");

            return 1;
        }
        finally
        {
            // Ensure cluster cleanup
            if (cluster is not null)
            {
                Console.WriteLine("\nShutting down Orleans cluster...");
                await cluster.StopAllSilosAsync();
                cluster.Dispose();
                Console.WriteLine("✓ Cluster shutdown complete");
            }
        }
    }

    /// <summary>
    /// Initializes an Orleans TestCluster with GPU Bridge configuration.
    /// Uses TestingHost for simplicity in sample applications.
    /// </summary>
    private static async Task<TestCluster> InitializeOrleansClusterAsync()
    {
        var builder = new TestClusterBuilder();

        builder.AddSiloBuilderConfigurator<SiloConfigurator>();

        var cluster = builder.Build();
        await cluster.DeployAsync();

        return cluster;
    }

    /// <summary>
    /// Configures the Orleans silo with GPU Bridge services.
    /// </summary>
    private sealed class SiloConfigurator : ISiloConfigurator
    {
        public void Configure(ISiloBuilder siloBuilder)
        {
            siloBuilder.ConfigureServices(services =>
            {
                // Add GPU Bridge with configuration
                services.AddGpuBridge(options =>
                {
                    options.PreferGpu = true;
                });
            });

            // Configure logging for better diagnostics
            siloBuilder.ConfigureLogging(logging =>
            {
                logging.SetMinimumLevel(LogLevel.Warning);
                logging.AddConsole();
            });
        }
    }

    /// <summary>
    /// Allocates GPU memory using the Ring Kernel API.
    /// </summary>
    private static async Task<GpuMemoryHandle> AllocateGpuMemoryAsync(
        IGpuResidentGrain<float> grain,
        long sizeBytes,
        string description)
    {
        try
        {
            var handle = await grain.AllocateAsync(
                sizeBytes,
                GpuMemoryType.Default);

            Console.WriteLine($"  ✓ Allocated {description}:");
            Console.WriteLine($"    - Handle ID: {handle.Id}");
            Console.WriteLine($"    - Size: {FormatBytes(sizeBytes)}");
            Console.WriteLine($"    - Type: {handle.Type}");
            Console.WriteLine($"    - Device: GPU {handle.DeviceIndex}");
            Console.WriteLine($"    - Timestamp: {handle.AllocatedAt:yyyy-MM-dd HH:mm:ss.fff} UTC");

            return handle;
        }
        catch (OutOfMemoryException ex)
        {
            Console.WriteLine($"  ✗ GPU out of memory for {description}");
            throw new InvalidOperationException(
                $"Insufficient GPU memory to allocate {FormatBytes(sizeBytes)} for {description}", ex);
        }
        catch (InvalidOperationException ex)
        {
            Console.WriteLine($"  ℹ No GPU available - using CPU fallback for {description}");
            // In production, you might want to handle fallback differently
            throw new InvalidOperationException(
                $"GPU allocation failed for {description}. CPU fallback active.", ex);
        }
    }

    /// <summary>
    /// Generates test data for demonstration purposes.
    /// </summary>
    private static float[] GenerateTestData(int size)
    {
        var data = new float[size];
        var random = new Random(42); // Fixed seed for reproducibility

        for (var i = 0; i < size; i++)
        {
            data[i] = (float)random.NextDouble() * 100.0f;
        }

        Console.WriteLine($"  ✓ Generated {size} test values");
        Console.WriteLine($"    - Range: 0.0 to 100.0");
        Console.WriteLine($"    - Sample values: [{data[0]:F2}, {data[1]:F2}, {data[2]:F2}, ...]");

        return data;
    }

    /// <summary>
    /// Writes data to GPU memory via DMA transfer.
    /// This demonstrates the WriteAsync API for host-to-device transfers.
    /// </summary>
    private static async Task WriteDmaDataAsync(
        IGpuResidentGrain<float> grain,
        GpuMemoryHandle handle,
        float[] data)
    {
        var startTime = DateTime.UtcNow;

        try
        {
            await grain.WriteAsync(handle, data, offset: 0);

            var elapsed = DateTime.UtcNow - startTime;
            var bandwidth = CalculateBandwidth(data.Length * sizeof(float), elapsed);

            Console.WriteLine($"  ✓ DMA write completed:");
            Console.WriteLine($"    - Elements transferred: {data.Length:N0}");
            Console.WriteLine($"    - Data size: {FormatBytes(data.Length * sizeof(float))}");
            Console.WriteLine($"    - Transfer time: {elapsed.TotalMilliseconds:F3} ms");
            Console.WriteLine($"    - Bandwidth: {bandwidth:F2} GB/s");
        }
        catch (ArgumentOutOfRangeException ex)
        {
            Console.WriteLine($"  ✗ Write operation exceeds allocated memory bounds");
            throw new InvalidOperationException("DMA write failed - out of bounds access", ex);
        }
        catch (ArgumentException ex)
        {
            Console.WriteLine($"  ✗ Invalid memory handle provided");
            throw new InvalidOperationException("DMA write failed - memory handle not found", ex);
        }
    }

    /// <summary>
    /// Reads data from GPU memory via DMA transfer.
    /// This demonstrates the ReadAsync API for device-to-host transfers.
    /// </summary>
    private static async Task<float[]> ReadDmaDataAsync(
        IGpuResidentGrain<float> grain,
        GpuMemoryHandle handle,
        int elementCount)
    {
        var startTime = DateTime.UtcNow;

        try
        {
            var data = await grain.ReadAsync<float>(handle, elementCount, offset: 0);

            var elapsed = DateTime.UtcNow - startTime;
            var bandwidth = CalculateBandwidth(data.Length * sizeof(float), elapsed);

            Console.WriteLine($"  ✓ DMA read completed:");
            Console.WriteLine($"    - Elements transferred: {data.Length:N0}");
            Console.WriteLine($"    - Data size: {FormatBytes(data.Length * sizeof(float))}");
            Console.WriteLine($"    - Transfer time: {elapsed.TotalMilliseconds:F3} ms");
            Console.WriteLine($"    - Bandwidth: {bandwidth:F2} GB/s");
            Console.WriteLine($"    - Sample values: [{data[0]:F2}, {data[1]:F2}, {data[2]:F2}, ...]");

            return data;
        }
        catch (ArgumentOutOfRangeException ex)
        {
            Console.WriteLine($"  ✗ Read operation exceeds allocated memory bounds");
            throw new InvalidOperationException("DMA read failed - out of bounds access", ex);
        }
        catch (ArgumentException ex)
        {
            Console.WriteLine($"  ✗ Invalid memory handle provided");
            throw new InvalidOperationException("DMA read failed - memory handle not found", ex);
        }
    }

    /// <summary>
    /// Validates data transfer by comparing source and destination arrays.
    /// </summary>
    private static Task ValidateDataTransferAsync(float[] original, float[] transferred)
    {
        if (original.Length != transferred.Length)
        {
            throw new InvalidOperationException(
                $"Data length mismatch: expected {original.Length}, got {transferred.Length}");
        }

        var mismatches = 0;
        const float tolerance = 1e-6f;

        for (var i = 0; i < original.Length; i++)
        {
            if (Math.Abs(original[i] - transferred[i]) > tolerance)
            {
                mismatches++;
            }
        }

        if (mismatches > 0)
        {
            throw new InvalidOperationException(
                $"Data validation failed: {mismatches} elements differ");
        }

        Console.WriteLine($"  ✓ Data validation successful:");
        Console.WriteLine($"    - All {original.Length:N0} elements match");
        Console.WriteLine($"    - Tolerance: ±{tolerance:E2}");

        return Task.CompletedTask;
    }

    /// <summary>
    /// Displays comprehensive GPU memory metrics using GetMemoryInfoAsync.
    /// </summary>
    private static async Task DisplayMemoryMetricsAsync(IGpuResidentGrain<float> grain)
    {
        try
        {
            var memoryInfo = await grain.GetMemoryInfoAsync();

            Console.WriteLine($"  ✓ GPU Memory Statistics:");
            Console.WriteLine($"    ┌─────────────────────────────────────────────────");
            Console.WriteLine($"    │ Device: {memoryInfo.DeviceName} (GPU {memoryInfo.DeviceIndex})");
            Console.WriteLine($"    │ Timestamp: {memoryInfo.Timestamp:yyyy-MM-dd HH:mm:ss.fff} UTC");
            Console.WriteLine($"    ├─────────────────────────────────────────────────");
            Console.WriteLine($"    │ Total Memory:      {FormatBytes(memoryInfo.TotalMemoryBytes),15}");
            Console.WriteLine($"    │ Allocated:         {FormatBytes(memoryInfo.AllocatedMemoryBytes),15}");
            Console.WriteLine($"    │ Free:              {FormatBytes(memoryInfo.FreeMemoryBytes),15}");
            Console.WriteLine($"    │ Reserved:          {FormatBytes(memoryInfo.ReservedMemoryBytes),15}");
            Console.WriteLine($"    ├─────────────────────────────────────────────────");
            Console.WriteLine($"    │ Buffer Memory:     {FormatBytes(memoryInfo.BufferMemoryBytes),15}");
            Console.WriteLine($"    │ Kernel Memory:     {FormatBytes(memoryInfo.PersistentKernelMemoryBytes),15}");
            Console.WriteLine($"    │ Texture Memory:    {FormatBytes(memoryInfo.TextureMemoryBytes),15}");
            Console.WriteLine($"    ├─────────────────────────────────────────────────");
            Console.WriteLine($"    │ Utilization:       {memoryInfo.UtilizationPercentage,14:F2}%");
            Console.WriteLine($"    │ Fragmentation:     {memoryInfo.FragmentationPercentage,14:F2}%");
            Console.WriteLine($"    └─────────────────────────────────────────────────");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"  ⚠ Could not retrieve memory metrics: {ex.Message}");
        }
    }

    /// <summary>
    /// Demonstrates kernel execution using resident memory buffers.
    /// Note: This requires a kernel to be registered in the system.
    /// </summary>
    private static Task DemonstrateKernelExecutionAsync(
        IGpuResidentGrain<float> grain,
        GpuMemoryHandle inputHandle,
        GpuMemoryHandle outputHandle)
    {
        try
        {
            // Note: In a real scenario, you would have a kernel registered
            // For this sample, we demonstrate the API usage pattern
            var kernelId = KernelId.Parse("vector-add-kernel");
            var parameters = new GpuComputeParams(
                WorkGroupSize: 256,
                WorkGroups: 0, // Auto-calculate
                Constants: new Dictionary<string, object>
                {
                    ["scale_factor"] = 2.0f
                });

            Console.WriteLine($"  ℹ Kernel execution demonstration:");
            Console.WriteLine($"    - Kernel ID: {kernelId}");
            Console.WriteLine($"    - Input Handle: {inputHandle.Id}");
            Console.WriteLine($"    - Output Handle: {outputHandle.Id}");
            Console.WriteLine($"    - Work Group Size: {parameters.WorkGroupSize}");
            Console.WriteLine($"    - Work Groups: {(parameters.WorkGroups == 0 ? "Auto" : parameters.WorkGroups.ToString())}");

            // In this sample, we note that kernel execution would happen here
            // Actual kernel registration and execution is beyond this API demo scope
            Console.WriteLine($"  ℹ Note: Actual kernel execution requires registered kernels");
            Console.WriteLine($"    See Orleans.GpuBridge documentation for kernel registration");

            // Uncomment this line when you have kernels registered:
            // var result = await grain.ComputeAsync(kernelId, inputHandle, outputHandle, parameters);
            // Console.WriteLine($"  ✓ Kernel executed in {result.ExecutionTime.TotalMilliseconds:F3} ms");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"  ℹ Kernel execution skipped: {ex.Message}");
        }

        return Task.CompletedTask;
    }

    /// <summary>
    /// Releases GPU memory allocation.
    /// </summary>
    private static async Task ReleaseGpuMemoryAsync(
        IGpuResidentGrain<float> grain,
        GpuMemoryHandle handle,
        string description)
    {
        try
        {
            await grain.ReleaseAsync(handle);
            Console.WriteLine($"  ✓ Released {description} (Handle: {handle.Id})");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"  ⚠ Failed to release {description}: {ex.Message}");
        }
    }

    /// <summary>
    /// Formats byte count into human-readable format.
    /// </summary>
    private static string FormatBytes(long bytes)
    {
        string[] suffixes = ["B", "KB", "MB", "GB", "TB"];
        var index = 0;
        var value = (double)bytes;

        while (value >= 1024 && index < suffixes.Length - 1)
        {
            value /= 1024;
            index++;
        }

        return $"{value:F2} {suffixes[index]}";
    }

    /// <summary>
    /// Calculates data transfer bandwidth in GB/s.
    /// </summary>
    private static double CalculateBandwidth(long bytes, TimeSpan elapsed)
    {
        if (elapsed.TotalSeconds == 0)
            return 0;

        var gigabytes = bytes / (1024.0 * 1024.0 * 1024.0);
        return gigabytes / elapsed.TotalSeconds;
    }
}

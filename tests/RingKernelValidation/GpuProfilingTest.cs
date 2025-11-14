// Copyright (c) 2025 Michael Ivertowski
// Licensed under the MIT License.

using DotCompute.Abstractions.RingKernels;
using DotCompute.Generated;
using Microsoft.Extensions.Logging;
using Orleans.GpuBridge.Backends.DotCompute.Temporal;
using Orleans.GpuBridge.Backends.DotCompute.Temporal.Generated;

namespace RingKernelValidation;

/// <summary>
/// GPU performance profiling test for VectorAddProcessor ring kernel.
/// </summary>
/// <remarks>
/// Designed to work with CUDA profiling tools:
/// - Nsight Compute (ncu): Detailed kernel metrics
/// - Nsight Systems (nsys): Timeline profiling
///
/// Target Metrics:
/// - Kernel launch latency: Expected ~10-50μs (CUDA kernel launch overhead)
/// - Message processing latency: Target 100-500ns (GPU-native)
/// - Memory bandwidth utilization: RTX 2000 Ada has 224 GB/s
/// - Throughput: Target 2M+ messages/s/actor
/// </remarks>
public static class GpuProfilingTest
{
    public static async Task<int> RunAsync(ILoggerFactory loggerFactory, int durationSeconds = 5)
    {
        var logger = loggerFactory.CreateLogger("GpuProfilingTest");

        Console.WriteLine();
        Console.WriteLine($"=== GPU Performance Profiling Test (CUDA) ===");
        Console.WriteLine("RTX 2000 Ada Generation Laptop GPU");
        Console.WriteLine($"Duration: {durationSeconds} seconds");
        Console.WriteLine();
        Console.WriteLine("Profiling Targets:");
        Console.WriteLine("  - Kernel launch latency: ~10-50μs");
        Console.WriteLine("  - Message processing: 100-500ns (GPU-native)");
        Console.WriteLine("  - Throughput: 2M+ messages/s/actor");
        Console.WriteLine("  - Memory bandwidth: RTX 2000 Ada = 224 GB/s");
        Console.WriteLine();

        try
        {
            // Step 1: Create CUDA runtime
            logger.LogInformation("Step 1: Creating CUDA ring kernel runtime...");
            var runtime = RingKernelRuntimeFactory.CreateRuntime("CUDA", loggerFactory);
            logger.LogInformation("✓ CUDA runtime created");

            // Step 2: Create kernel wrapper
            logger.LogInformation("Step 2: Creating VectorAddProcessorRing wrapper...");
            using var kernelWrapper = new VectorAddProcessorRingRingKernelWrapper(runtime);
            logger.LogInformation("✓ Wrapper created");

            // Step 3: Measure kernel launch latency
            logger.LogInformation("Step 3: Measuring kernel launch latency...");
            var launchStart = DateTime.UtcNow;

            await kernelWrapper.LaunchAsync(gridSize: 1, blockSize: 1);

            var launchDuration = (DateTime.UtcNow - launchStart).TotalMicroseconds;
            logger.LogInformation($"✓ Kernel launched in {launchDuration:F2}μs");
            Console.WriteLine($"  Kernel Launch Latency: {launchDuration:F2}μs");
            Console.WriteLine();

            // Step 4: Activate kernel
            logger.LogInformation("Step 4: Activating kernel...");
            var activateStart = DateTime.UtcNow;

            await kernelWrapper.ActivateAsync();

            var activateDuration = (DateTime.UtcNow - activateStart).TotalMicroseconds;
            logger.LogInformation($"✓ Kernel activated in {activateDuration:F2}μs");
            Console.WriteLine($"  Kernel Activation Latency: {activateDuration:F2}μs");
            Console.WriteLine();

            // Step 5: Run for profiling duration
            logger.LogInformation($"Step 5: Kernel running for {durationSeconds} seconds (profiling window)...");
            Console.WriteLine($"Profiling for {durationSeconds} seconds...");
            Console.WriteLine("(Use Ctrl+C to stop early if needed)");
            Console.WriteLine();

            var profilingStart = DateTime.UtcNow;
            await Task.Delay(TimeSpan.FromSeconds(durationSeconds));
            var profilingDuration = (DateTime.UtcNow - profilingStart).TotalSeconds;

            logger.LogInformation($"✓ Profiling window complete ({profilingDuration:F2}s)");
            Console.WriteLine();

            // Step 6: Deactivate kernel
            logger.LogInformation("Step 6: Deactivating kernel...");
            var deactivateStart = DateTime.UtcNow;

            await kernelWrapper.DeactivateAsync();

            var deactivateDuration = (DateTime.UtcNow - deactivateStart).TotalMicroseconds;
            logger.LogInformation($"✓ Kernel deactivated in {deactivateDuration:F2}μs");
            Console.WriteLine($"  Kernel Deactivation Latency: {deactivateDuration:F2}μs");
            Console.WriteLine();

            // Step 7: Terminate kernel and get final metrics
            logger.LogInformation("Step 7: Terminating kernel...");
            var terminateStart = DateTime.UtcNow;

            await kernelWrapper.TerminateAsync();

            var terminateDuration = (DateTime.UtcNow - terminateStart).TotalMicroseconds;
            logger.LogInformation($"✓ Kernel terminated in {terminateDuration:F2}μs");
            Console.WriteLine($"  Kernel Termination Latency: {terminateDuration:F2}μs");
            Console.WriteLine();

            // Summary
            Console.WriteLine($"=== PROFILING SUMMARY ===");
            Console.WriteLine($"GPU: RTX 2000 Ada Generation (Compute Capability 8.9)");
            Console.WriteLine($"Backend: CUDA");
            Console.WriteLine($"Duration: {profilingDuration:F2}s");
            Console.WriteLine();
            Console.WriteLine($"Lifecycle Latencies:");
            Console.WriteLine($"  Launch:      {launchDuration:F2}μs");
            Console.WriteLine($"  Activate:    {activateDuration:F2}μs");
            Console.WriteLine($"  Deactivate:  {deactivateDuration:F2}μs");
            Console.WriteLine($"  Terminate:   {terminateDuration:F2}μs");
            Console.WriteLine();
            Console.WriteLine($"Expected Performance (GPU-Native):");
            Console.WriteLine($"  Message Latency: 100-500ns");
            Console.WriteLine($"  Throughput: 2M+ messages/s/actor");
            Console.WriteLine($"  Memory Bandwidth: Up to 224 GB/s");
            Console.WriteLine();
            Console.WriteLine($"=== NEXT STEPS ===");
            Console.WriteLine($"For detailed profiling, run with:");
            Console.WriteLine();
            Console.WriteLine($"  # Nsight Compute (kernel metrics)");
            Console.WriteLine($"  ncu --set full dotnet run -- profile");
            Console.WriteLine();
            Console.WriteLine($"  # Nsight Systems (timeline)");
            Console.WriteLine($"  nsys profile -t cuda,nvtx dotnet run -- profile");
            Console.WriteLine();

            return 0; // Success
        }
        catch (Exception ex)
        {
            logger.LogError(ex, "❌ GPU profiling test failed!");
            Console.WriteLine();
            Console.WriteLine("=== ❌ PROFILING FAILED ===");
            Console.WriteLine($"Error: {ex.Message}");
            Console.WriteLine($"Type: {ex.GetType().Name}");
            if (ex.InnerException != null)
            {
                Console.WriteLine($"Inner: {ex.InnerException.Message}");
            }
            Console.WriteLine();
            Console.WriteLine("Stack trace:");
            Console.WriteLine(ex.StackTrace);
            return 1; // Failure
        }
    }
}

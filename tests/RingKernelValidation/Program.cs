// Copyright (c) 2025 Michael Ivertowski
// Licensed under the MIT License.

using DotCompute.Generated;
using Microsoft.Extensions.Logging;
using Orleans.GpuBridge.Backends.DotCompute.Temporal.Generated;

namespace RingKernelValidation;

/// <summary>
/// Simple validation test for VectorAddProcessorRing ring kernel.
/// </summary>
/// <remarks>
/// Tests the complete lifecycle:
/// 1. Create runtime (CPU backend)
/// 2. Instantiate ring kernel wrapper
/// 3. Launch kernel on GPU
/// 4. Activate message processing
/// 5. (Future: Send test message)
/// 6. Deactivate and terminate
/// 7. Clean up resources
/// </remarks>
class Program
{
    static async Task<int> Main(string[] args)
    {
        Console.WriteLine("╔════════════════════════════════════════════════════════════════╗");
        Console.WriteLine("║         Ring Kernel Validation Test Suite                     ║");
        Console.WriteLine("║         Orleans.GpuBridge.Core - GPU-Native Actors             ║");
        Console.WriteLine("╚════════════════════════════════════════════════════════════════╝");
        Console.WriteLine();

        // Setup logging
        using var loggerFactory = LoggerFactory.Create(builder =>
        {
            builder.SetMinimumLevel(LogLevel.Information);
            builder.AddConsole();
        });

        var logger = loggerFactory.CreateLogger<Program>();

        // Determine which test to run
        var runCuda = args.Length > 0 && args[0].Equals("cuda", StringComparison.OrdinalIgnoreCase);
        var runAll = args.Length > 0 && args[0].Equals("all", StringComparison.OrdinalIgnoreCase);
        var runMessage = args.Length > 0 && args[0].Equals("message", StringComparison.OrdinalIgnoreCase);
        var runMessageCuda = args.Length > 0 && args[0].Equals("message-cuda", StringComparison.OrdinalIgnoreCase);
        var runProfile = args.Length > 0 && args[0].Equals("profile", StringComparison.OrdinalIgnoreCase);

        // GPU profiling test
        if (runProfile)
        {
            int duration = 5; // Default 5 seconds
            if (args.Length > 1 && int.TryParse(args[1], out var parsedDuration))
            {
                duration = parsedDuration;
            }
            var profileResult = await GpuProfilingTest.RunAsync(loggerFactory, duration);
            return profileResult;
        }

        // Message passing tests - RE-ENABLED (DotCompute SDK updated with message queue support)
        if (runMessage || runMessageCuda)
        {
            var backend = runMessageCuda ? "CUDA" : "CPU";
            var messageResult = await MessagePassingTest.RunAsync(loggerFactory, backend);
            return messageResult;
        }

        if (runCuda || runAll)
        {
            // Run CUDA test
            var cudaResult = await CudaTest.RunAsync(loggerFactory);

            if (!runAll)
            {
                return cudaResult;
            }

            if (cudaResult == 0)
            {
                Console.WriteLine("✓ CUDA test passed");
            }
            else if (cudaResult == 2)
            {
                Console.WriteLine("⚠ CUDA test skipped (no GPU)");
            }
            else
            {
                Console.WriteLine("✗ CUDA test failed");
                return cudaResult;
            }

            Console.WriteLine();
        }

        // Run CPU test (always, or if 'all' specified)
        Console.WriteLine("=== Ring Kernel Validation Test ===");
        Console.WriteLine("Testing: VectorAddProcessorRing (CPU backend)");
        Console.WriteLine();

        try
        {
            // Step 1: Create CPU ring kernel runtime
            logger.LogInformation("Step 1: Creating CPU ring kernel runtime...");
            var runtime = RingKernelRuntimeFactory.CreateRuntime("CPU", loggerFactory);
            logger.LogInformation("✓ Runtime created successfully");
            Console.WriteLine();

            // Step 2: Create ring kernel wrapper
            logger.LogInformation("Step 2: Creating VectorAddProcessorRing wrapper...");
            using var kernelWrapper = new VectorAddProcessorRingRingKernelWrapper(runtime);
            logger.LogInformation("✓ Wrapper created successfully");
            Console.WriteLine();

            // Step 3: Launch kernel
            logger.LogInformation("Step 3: Launching ring kernel (gridSize=1, blockSize=1)...");
            await kernelWrapper.LaunchAsync(gridSize: 1, blockSize: 1);
            logger.LogInformation("✓ Kernel launched successfully");
            Console.WriteLine();

            // Step 4: Activate kernel (start message processing loop)
            logger.LogInformation("Step 4: Activating kernel (start infinite dispatch loop)...");
            await kernelWrapper.ActivateAsync();
            logger.LogInformation("✓ Kernel activated - now processing messages!");
            Console.WriteLine();

            // Step 5: Let kernel run for a bit
            logger.LogInformation("Step 5: Kernel running for 2 seconds...");
            await Task.Delay(TimeSpan.FromSeconds(2));
            logger.LogInformation("✓ Kernel still alive after 2 seconds");
            Console.WriteLine();

            // Step 6: Deactivate kernel (pause message processing)
            logger.LogInformation("Step 6: Deactivating kernel (pause dispatch loop)...");
            await kernelWrapper.DeactivateAsync();
            logger.LogInformation("✓ Kernel deactivated successfully");
            Console.WriteLine();

            // Step 7: Terminate kernel (stop and cleanup)
            logger.LogInformation("Step 7: Terminating kernel...");
            await kernelWrapper.TerminateAsync();
            logger.LogInformation("✓ Kernel terminated successfully");
            Console.WriteLine();

            // Success!
            Console.WriteLine("=== ✓ ALL TESTS PASSED ===");
            Console.WriteLine();
            Console.WriteLine("Ring Kernel Lifecycle Validated:");
            Console.WriteLine("  1. Runtime creation");
            Console.WriteLine("  2. Wrapper instantiation");
            Console.WriteLine("  3. Kernel launch");
            Console.WriteLine("  4. Activation (infinite loop started)");
            Console.WriteLine("  5. Kernel execution (2s)");
            Console.WriteLine("  6. Deactivation (loop paused)");
            Console.WriteLine("  7. Termination (cleanup)");
            Console.WriteLine();
            Console.WriteLine("Available Tests:");
            Console.WriteLine("  dotnet run                      # CPU lifecycle test (this test)");
            Console.WriteLine("  dotnet run -- cuda              # CUDA/GPU lifecycle test");
            Console.WriteLine("  dotnet run -- profile [secs]    # GPU profiling test (default: 5s)");
            Console.WriteLine("  dotnet run -- message           # Message passing test (CPU)");
            Console.WriteLine("  dotnet run -- message-cuda      # Message passing test (CUDA/GPU)");
            Console.WriteLine("  dotnet run -- all               # Run all lifecycle tests");
            Console.WriteLine();
            Console.WriteLine("GPU Profiling Commands:");
            Console.WriteLine("  ncu --set full dotnet run -- profile        # Nsight Compute (detailed metrics)");
            Console.WriteLine("  nsys profile -t cuda dotnet run -- profile  # Nsight Systems (timeline)");
            Console.WriteLine();

            return 0; // Success
        }
        catch (Exception ex)
        {
            logger.LogError(ex, "❌ Ring kernel validation failed!");
            Console.WriteLine();
            Console.WriteLine($"=== ❌ TEST FAILED ===");
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

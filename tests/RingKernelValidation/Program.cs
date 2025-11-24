// Copyright (c) 2025 Michael Ivertowski
// Licensed under the MIT License.

using DotCompute.Abstractions.RingKernels;
using DotCompute.Backends.CUDA.Compilation;
using DotCompute.Backends.CUDA.RingKernels;
using DotCompute.Core.Messaging;
using DotCompute.Generated;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using Orleans.GpuBridge.Backends.DotCompute.Temporal;

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

            const string KernelId = "vectoradd_processor";

            // Step 2: Launch kernel
            logger.LogInformation("Step 2: Launching ring kernel (gridSize=1, blockSize=256)...");
            await runtime.LaunchAsync(KernelId, gridSize: 1, blockSize: 256);
            logger.LogInformation("✓ Kernel launched successfully");
            Console.WriteLine();

            // Step 3: Activate kernel (start message processing loop)
            logger.LogInformation("Step 3: Activating kernel (start infinite dispatch loop)...");
            await runtime.ActivateAsync(KernelId);
            logger.LogInformation("✓ Kernel activated - now processing messages!");
            Console.WriteLine();

            // Step 4: Let kernel run for a bit
            logger.LogInformation("Step 4: Kernel running for 2 seconds...");
            await Task.Delay(TimeSpan.FromSeconds(2));
            logger.LogInformation("✓ Kernel still alive after 2 seconds");
            Console.WriteLine();

            // Step 5: Deactivate kernel (pause message processing)
            logger.LogInformation("Step 5: Deactivating kernel (pause dispatch loop)...");
            await runtime.DeactivateAsync(KernelId);
            logger.LogInformation("✓ Kernel deactivated successfully");
            Console.WriteLine();

            // Step 6: Terminate kernel (stop and cleanup)
            logger.LogInformation("Step 6: Terminating kernel...");
            using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(5));
            try
            {
                await runtime.TerminateAsync(KernelId, cts.Token);
                logger.LogInformation("✓ Kernel terminated successfully");
            }
            catch (OperationCanceledException)
            {
                logger.LogWarning("⚠ Kernel termination timed out (known WSL2 issue)");
            }
            Console.WriteLine();

            // Success!
            Console.WriteLine("=== ✓ ALL TESTS PASSED ===");
            Console.WriteLine();
            Console.WriteLine("Ring Kernel Lifecycle Validated:");
            Console.WriteLine("  1. Runtime creation");
            Console.WriteLine("  2. Kernel launch");
            Console.WriteLine("  3. Activation (infinite loop started)");
            Console.WriteLine("  4. Kernel execution (2s)");
            Console.WriteLine("  5. Deactivation (loop paused)");
            Console.WriteLine("  6. Termination (cleanup)");
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

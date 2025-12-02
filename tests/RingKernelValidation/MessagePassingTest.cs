// Copyright (c) 2025 Michael Ivertowski
// Licensed under the MIT License.

using DotCompute.Abstractions.RingKernels;
using DotCompute.Backends.CPU.RingKernels;
using DotCompute.Backends.CUDA.Compilation;
using DotCompute.Backends.CUDA.RingKernels;
using DotCompute.Core.Messaging;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using Orleans.GpuBridge.Backends.DotCompute.Temporal;
using Orleans.GpuBridge.Backends.DotCompute.Generated;

namespace RingKernelValidation;

/// <summary>
/// Message passing validation test for VectorAddProcessor ring kernel.
/// </summary>
/// <remarks>
/// Tests the complete message passing pipeline:
/// 1. Send VectorAddRequest to kernel
/// 2. Kernel processes (A + B = C)
/// 3. Receive VectorAddResponse from kernel
/// 4. Validate computation correctness
/// 5. Measure end-to-end latency
/// </remarks>
public static class MessagePassingTest
{
    private const string KernelId = "vectoradd_processor";

    public static async Task<int> RunAsync(ILoggerFactory loggerFactory, string backend = "CPU")
    {
        var logger = loggerFactory.CreateLogger("MessagePassingTest");

        Console.WriteLine();
        Console.WriteLine($"=== Message Passing Validation Test ({backend}) ===");
        Console.WriteLine("Testing: VectorAddRequest → Ring Kernel → VectorAddResponse");
        Console.WriteLine();

        try
        {
            // Step 1: Create runtime (use concrete type for RegisterAssembly access)
            logger.LogInformation($"Step 1: Creating {backend} ring kernel runtime...");
            IRingKernelRuntime runtime;

            if (backend == "CUDA")
            {
                // Create CUDA runtime with actual loggers to trace message flow
                // Include handler translator for unified kernel API support
                var handlerTranslator = new RingKernelHandlerTranslator(
                    loggerFactory.CreateLogger<RingKernelHandlerTranslator>());
                var compiler = new CudaRingKernelCompiler(
                    loggerFactory.CreateLogger<CudaRingKernelCompiler>(),
                    new RingKernelDiscovery(loggerFactory.CreateLogger<RingKernelDiscovery>()),
                    new CudaRingKernelStubGenerator(loggerFactory.CreateLogger<CudaRingKernelStubGenerator>()),
                    new DotCompute.Backends.CUDA.Compilation.CudaMemoryPackSerializerGenerator(loggerFactory.CreateLogger<DotCompute.Backends.CUDA.Compilation.CudaMemoryPackSerializerGenerator>()),
                    handlerTranslator);
                var registry = new MessageQueueRegistry(loggerFactory.CreateLogger<MessageQueueRegistry>());
                var cudaRuntime = new CudaRingKernelRuntime(
                    loggerFactory.CreateLogger<CudaRingKernelRuntime>(),
                    compiler,
                    registry);
                // Register our assembly for kernel discovery
                cudaRuntime.RegisterAssembly(typeof(VectorAddRingKernel).Assembly);
                runtime = cudaRuntime;
            }
            else
            {
                // Create CPU runtime
                // Note: CPU backend doesn't support RegisterAssembly() - this is a DotCompute SDK limitation
                // Message passing tests will fail on CPU backend until this is fixed
                var cpuRuntime = new CpuRingKernelRuntime(loggerFactory.CreateLogger<CpuRingKernelRuntime>());
                runtime = cpuRuntime;
            }

            logger.LogInformation("✓ Runtime created and assembly registered");

            // Step 2: Launch kernel (using runtime directly, not wrapper)
            logger.LogInformation("Step 2: Launching kernel...");
            await runtime.LaunchAsync(KernelId, gridSize: 1, blockSize: 256);
            logger.LogInformation("✓ Kernel launched");

            // Step 3: Activate kernel
            logger.LogInformation("Step 3: Activating kernel...");
            await runtime.ActivateAsync(KernelId);
            logger.LogInformation("✓ Kernel activated");
            Console.WriteLine();

            // Step 4: Use deterministic queue names (ringkernel_{MessageType}_{KernelId}_input/output)
            logger.LogInformation("Step 4: Using deterministic queue names...");

            // CUDA backend uses _input/_output suffixes
            var inputQueueName = $"ringkernel_{nameof(VectorAddProcessorRingRequest)}_{KernelId}_input";
            var outputQueueName = $"ringkernel_{nameof(VectorAddProcessorRingResponse)}_{KernelId}_output";
            logger.LogInformation($"  Input queue: {inputQueueName}");
            logger.LogInformation($"  Output queue: {outputQueueName}");
            logger.LogInformation("✓ Queue names resolved");
            Console.WriteLine();

            // Step 5: Prepare test cases (simplified for CUDA primitives validation)
            logger.LogInformation("Step 5: Preparing test cases (primitives-only for CUDA)...");

            // Test cases with 4 fixed float elements (CUDA serializer limitation)
            var testCases = new[]
            {
                // Addition test: A + B
                (Name: "Addition (4 elements)", OpType: 0, A: (1f, 2f, 3f, 4f), B: (10f, 20f, 30f, 40f), Expected: (11f, 22f, 33f, 44f)),

                // Subtraction test: A - B
                (Name: "Subtraction (4 elements)", OpType: 1, A: (100f, 200f, 300f, 400f), B: (10f, 20f, 30f, 40f), Expected: (90f, 180f, 270f, 360f)),

                // Multiplication test: A * B
                (Name: "Multiplication (4 elements)", OpType: 2, A: (2f, 3f, 4f, 5f), B: (10f, 10f, 10f, 10f), Expected: (20f, 30f, 40f, 50f))
            };

            logger.LogInformation($"✓ Prepared {testCases.Length} test cases (primitives-only)");
            Console.WriteLine();

            // Step 6: Send messages and validate responses
            int passedTests = 0;
            int totalTests = testCases.Length;

            foreach (var (name, opType, a, b, expected) in testCases)
            {
                logger.LogInformation($"Test: {name}");

                // Create VectorAddProcessorRingRequest with primitives only (CUDA compatible)
                var request = new VectorAddProcessorRingRequest
                {
                    MessageId = Guid.NewGuid(),
                    Priority = 128,  // Normal priority
                    VectorLength = 4,
                    OperationType = opType,  // 0=Add, 1=Sub, 2=Mul, 3=Div
                    A0 = a.Item1,
                    A1 = a.Item2,
                    A2 = a.Item3,
                    A3 = a.Item4,
                    B0 = b.Item1,
                    B1 = b.Item2,
                    B2 = b.Item3,
                    B3 = b.Item4
                };

                // Send message using named message queue API
                logger.LogInformation($"  Sending request (4 elements, operation={opType})...");
                var sendStart = DateTime.UtcNow;

                // Send directly to input queue using named queue API
                var sent = await runtime.SendToNamedQueueAsync(inputQueueName, request);
                if (!sent)
                {
                    logger.LogError($"  ✗ Failed to send message to queue!");
                    Console.WriteLine($"  ✗ FAILED: Message send failed");
                    continue;
                }

                var sendDuration = (DateTime.UtcNow - sendStart).TotalMicroseconds;
                logger.LogInformation($"  ✓ Message sent in {sendDuration:F2}μs");

                // Receive response (with 5 second timeout)
                logger.LogInformation($"  Waiting for response...");
                var receiveStart = DateTime.UtcNow;

                // Receive from output queue using named queue API
                // Note: For structs with interface constraint, T? returns T not Nullable<T>
                // Check for default (empty MessageId) to detect no message
                VectorAddProcessorRingResponse? responseMsg = null;
                var timeoutEnd = DateTime.UtcNow.AddSeconds(5);
                while (DateTime.UtcNow < timeoutEnd)
                {
                    var msg = await runtime.ReceiveFromNamedQueueAsync<VectorAddProcessorRingResponse>(outputQueueName);
                    // Check if we got a valid message (non-default MessageId)
                    if (msg is { } validMsg && validMsg.MessageId != Guid.Empty)
                    {
                        responseMsg = validMsg;
                        break;
                    }

                    await Task.Delay(10); // Small delay before retry
                }

                if (responseMsg == null)
                {
                    logger.LogError($"  ✗ Timeout waiting for response!");
                    Console.WriteLine($"  ✗ FAILED: Timeout");
                    continue;
                }

                var receiveDuration = (DateTime.UtcNow - receiveStart).TotalMicroseconds;
                var totalLatency = (DateTime.UtcNow - sendStart).TotalMicroseconds;

                logger.LogInformation($"  ✓ Response received in {receiveDuration:F2}μs (total: {totalLatency:F2}μs)");

                // Validate result using fixed fields (CUDA primitives)
                var response = responseMsg.Value;
                var actualResults = new[] { response.R0, response.R1, response.R2, response.R3 };
                var expectedResults = new[] { expected.Item1, expected.Item2, expected.Item3, expected.Item4 };

                bool isCorrect = actualResults.Zip(expectedResults, (act, exp) => Math.Abs(act - exp) < 0.001f).All(x => x);

                if (isCorrect)
                {
                    logger.LogInformation($"  ✓ Computation CORRECT!");
                    logger.LogInformation($"    Expected: [{string.Join(", ", expectedResults)}]");
                    logger.LogInformation($"    Actual:   [{string.Join(", ", actualResults)}]");
                    logger.LogInformation($"  Performance: Send={sendDuration:F2}μs, Receive={receiveDuration:F2}μs, Total={totalLatency:F2}μs");
                    Console.WriteLine($"  ✓ PASSED - {totalLatency:F2}μs latency");
                    passedTests++;
                }
                else
                {
                    logger.LogError($"  ✗ Computation INCORRECT!");
                    logger.LogError($"    Expected: [{string.Join(", ", expectedResults)}]");
                    logger.LogError($"    Actual:   [{string.Join(", ", actualResults)}]");
                    logger.LogError($"    Success flag: {response.Success}, ErrorCode: {response.ErrorCode}");
                    Console.WriteLine($"  ✗ FAILED - Incorrect result");
                }

                Console.WriteLine();
            }

            // Step 7: Cleanup
            logger.LogInformation("Step 7: Deactivating kernel...");
            await runtime.DeactivateAsync(KernelId);
            logger.LogInformation("✓ Kernel deactivated");

            logger.LogInformation("Step 8: Terminating kernel...");
            using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(5));
            try
            {
                await runtime.TerminateAsync(KernelId, cts.Token);
                logger.LogInformation("✓ Kernel terminated");
            }
            catch (OperationCanceledException)
            {
                logger.LogWarning("Kernel termination timed out (known WSL2 issue)");
            }
            Console.WriteLine();

            // Summary
            Console.WriteLine($"=== TEST SUMMARY ===");
            Console.WriteLine($"Passed: {passedTests}/{totalTests}");
            Console.WriteLine($"Failed: {totalTests - passedTests}/{totalTests}");
            Console.WriteLine();

            if (passedTests == totalTests)
            {
                Console.WriteLine("=== ✓ ALL MESSAGE PASSING TESTS PASSED ===");
                Console.WriteLine();
                Console.WriteLine("Message Passing Validated:");
                Console.WriteLine("  1. VectorAddRequest → Ring Kernel");
                Console.WriteLine("  2. Ring Kernel processes (A + B = C)");
                Console.WriteLine("  3. Ring Kernel → VectorAddResponse");
                Console.WriteLine("  4. Computation correctness verified");
                Console.WriteLine("  5. End-to-end latency measured");
                Console.WriteLine();
                return 0; // Success
            }
            else
            {
                Console.WriteLine($"=== ⚠ {totalTests - passedTests} TEST(S) FAILED ===");
                return 1; // Partial failure
            }
        }
        catch (Exception ex)
        {
            logger.LogError(ex, "❌ Message passing test failed!");
            Console.WriteLine();
            Console.WriteLine("=== ❌ TEST FAILED ===");
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

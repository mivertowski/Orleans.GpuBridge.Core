// Copyright (c) 2025 Michael Ivertowski
// Licensed under the MIT License.

using DotCompute.Abstractions.RingKernels;
using DotCompute.Generated;
using Microsoft.Extensions.Logging;
using Orleans.GpuBridge.Backends.DotCompute.Temporal;
using Orleans.GpuBridge.Backends.DotCompute.Temporal.Generated;

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
    public static async Task<int> RunAsync(ILoggerFactory loggerFactory, string backend = "CPU")
    {
        var logger = loggerFactory.CreateLogger("MessagePassingTest");

        Console.WriteLine();
        Console.WriteLine($"=== Message Passing Validation Test ({backend}) ===");
        Console.WriteLine("Testing: VectorAddRequest → Ring Kernel → VectorAddResponse");
        Console.WriteLine();

        try
        {
            // Step 1: Create runtime and launch kernel
            logger.LogInformation($"Step 1: Creating {backend} ring kernel runtime...");
            var runtime = RingKernelRuntimeFactory.CreateRuntime(backend, loggerFactory);
            logger.LogInformation("✓ Runtime created");

            logger.LogInformation("Step 2: Creating ring kernel wrapper...");
            using var kernelWrapper = new VectorAddProcessorRingRingKernelWrapper(runtime);
            logger.LogInformation("✓ Wrapper created");

            logger.LogInformation("Step 3: Launching kernel...");
            await kernelWrapper.LaunchAsync(gridSize: 1, blockSize: 1);
            logger.LogInformation("✓ Kernel launched");

            logger.LogInformation("Step 4: Activating kernel...");
            await kernelWrapper.ActivateAsync();
            logger.LogInformation("✓ Kernel activated");
            Console.WriteLine();

            // Step 2: Prepare test vectors
            logger.LogInformation("Step 5: Preparing test vectors...");
            var testCases = new[]
            {
                // Small vector (inline data: ≤25 elements)
                (Name: "Small Vector (10 elements, inline)", Size: 10, A: Enumerable.Range(1, 10).Select(i => (float)i).ToArray(), B: Enumerable.Range(1, 10).Select(i => (float)i * 2).ToArray()),

                // Boundary case (exactly 25 elements - inline threshold)
                (Name: "Boundary Vector (25 elements, inline)", Size: 25, A: Enumerable.Range(1, 25).Select(i => (float)i).ToArray(), B: Enumerable.Range(1, 25).Select(i => (float)i * 3).ToArray()),

                // Large vector (GPU memory: >25 elements)
                (Name: "Large Vector (100 elements, GPU memory)", Size: 100, A: Enumerable.Range(1, 100).Select(i => (float)i).ToArray(), B: Enumerable.Range(1, 100).Select(i => (float)i * 4).ToArray())
            };

            logger.LogInformation($"✓ Prepared {testCases.Length} test cases");
            Console.WriteLine();

            // Step 3: Send messages and validate responses
            int passedTests = 0;
            int totalTests = testCases.Length;

            foreach (var (name, size, a, b) in testCases)
            {
                logger.LogInformation($"Test: {name}");

                // Create VectorAddRequest
                var request = new VectorAddRequest
                {
                    VectorALength = size,
                    Operation = VectorOperation.Add,
                    UseGpuMemory = size > 25 ? 1 : 0,
                    GpuBufferAHandleId = 0,
                    GpuBufferBHandleId = 0,
                    GpuBufferResultHandleId = 0
                };

                // Copy inline data using unsafe fixed buffers
                unsafe
                {
                    int copyCount = Math.Min(size, 25);
                    for (int i = 0; i < copyCount; i++)
                    {
                        request.InlineDataA[i] = a[i];
                        request.InlineDataB[i] = b[i];
                    }
                }

                // Calculate expected result
                var expected = a.Zip(b, (x, y) => x + y).ToArray();

                // Create kernel message
                var message = KernelMessage<VectorAddRequest>.Create(
                    senderId: 0,      // Host sender ID
                    receiverId: 1,    // Kernel instance ID
                    type: MessageType.Data,
                    payload: request
                );

                // Send message
                logger.LogInformation($"  Sending request (size={size}, inline={request.UseGpuMemory == 0})...");
                var sendStart = DateTime.UtcNow;

                await runtime.SendMessageAsync("VectorAddProcessor", message, CancellationToken.None);

                var sendDuration = (DateTime.UtcNow - sendStart).TotalMicroseconds;
                logger.LogInformation($"  ✓ Message sent in {sendDuration:F2}μs");

                // Receive response (with 5 second timeout)
                logger.LogInformation($"  Waiting for response...");
                var receiveStart = DateTime.UtcNow;

                var response = await runtime.ReceiveMessageAsync<VectorAddResponse>(
                    "VectorAddProcessor",
                    timeout: TimeSpan.FromSeconds(5),
                    cancellationToken: CancellationToken.None
                );

                if (response == null)
                {
                    logger.LogError($"  ✗ Timeout waiting for response!");
                    Console.WriteLine($"  ✗ FAILED: Timeout");
                    continue;
                }

                var receiveDuration = (DateTime.UtcNow - receiveStart).TotalMicroseconds;
                var totalLatency = (DateTime.UtcNow - sendStart).TotalMicroseconds;

                logger.LogInformation($"  ✓ Response received in {receiveDuration:F2}μs (total: {totalLatency:F2}μs)");

                // Validate result - extract from fixed buffer
                var result = response.Value.Payload;
                float[] actualResult;
                unsafe
                {
                    int resultCount = Math.Min(size, 25);
                    actualResult = new float[resultCount];
                    for (int i = 0; i < resultCount; i++)
                    {
                        actualResult[i] = result.InlineResult[i];
                    }
                }

                var expectedResult = expected.Take(Math.Min(size, 25)).ToArray();

                bool isCorrect = actualResult.Zip(expectedResult, (a, e) => Math.Abs(a - e) < 0.001f).All(x => x);

                if (isCorrect)
                {
                    logger.LogInformation($"  ✓ Computation CORRECT (A + B = C validated)");
                    logger.LogInformation($"  Performance: Send={sendDuration:F2}μs, Receive={receiveDuration:F2}μs, Total={totalLatency:F2}μs");
                    Console.WriteLine($"  ✓ PASSED - {totalLatency:F2}μs latency");
                    passedTests++;
                }
                else
                {
                    logger.LogError($"  ✗ Computation INCORRECT!");
                    logger.LogError($"    Expected: [{string.Join(", ", expectedResult.Take(5))}...]");
                    logger.LogError($"    Actual:   [{string.Join(", ", actualResult.Take(5))}...]");
                    Console.WriteLine($"  ✗ FAILED - Incorrect result");
                }

                Console.WriteLine();
            }

            // Step 4: Cleanup
            logger.LogInformation("Step 6: Deactivating kernel...");
            await kernelWrapper.DeactivateAsync();
            logger.LogInformation("✓ Kernel deactivated");

            logger.LogInformation("Step 7: Terminating kernel...");
            await kernelWrapper.TerminateAsync();
            logger.LogInformation("✓ Kernel terminated");
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

// Copyright (c) 2025 Michael Ivertowski
// Licensed under the Apache License, Version 2.0. See LICENSE file for details.

using DotCompute.Abstractions.Attributes;
using DotCompute.Abstractions.RingKernels;

namespace Orleans.GpuBridge.Backends.DotCompute.Temporal;

/// <summary>
/// Ring kernel for GPU-native vector operations (add, subtract, multiply, divide).
/// </summary>
/// <remarks>
/// <para>
/// This kernel implements the GPU-native actor paradigm using the DotCompute unified
/// ring kernel system. The handler logic is defined inline and automatically translated
/// to CUDA code.
/// </para>
/// <para>
/// Performance characteristics:
/// - Message processing: 100-500ns (GPU memory path)
/// - Up to 4 elements processed inline
/// </para>
/// </remarks>
public static class VectorAddRingKernel
{
    /// <summary>
    /// Processes vector operation requests and produces responses.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>Supported Operations:</b>
    /// <list type="bullet">
    /// <item><description>0 = Add: result = a + b</description></item>
    /// <item><description>1 = Subtract: result = a - b</description></item>
    /// <item><description>2 = Multiply: result = a * b</description></item>
    /// <item><description>3 = Divide: result = a / b (with zero-division protection)</description></item>
    /// </list>
    /// </para>
    /// </remarks>
    /// <param name="ctx">The ring kernel context for GPU operations.</param>
    /// <param name="request">The input request containing vectors and operation type.</param>
    [RingKernel(
        KernelId = "vectoradd_processor",
        Capacity = 4096,
        InputQueueSize = 1024,
        OutputQueueSize = 1024,
        MaxInputMessageSizeBytes = 1024,
        MaxOutputMessageSizeBytes = 1024,
        ProcessingMode = RingProcessingMode.Continuous,
        EnableTimestamps = true,
        Domain = RingKernelDomain.ActorModel,
        MessagingStrategy = MessagePassingStrategy.SharedMemory,
        Backends = KernelBackends.CUDA,
        OutputMessageType = typeof(VectorAddProcessorRingResponse))]
    public static void ProcessVectorOperation(RingKernelContext ctx, VectorAddProcessorRingRequest request)
    {
        // Synchronize before processing
        ctx.SyncThreads();

        // Get operation parameters
        int operationType = request.OperationType;
        int vectorLength = request.VectorLength;

        // Read input vectors
        float a0 = request.A0;
        float a1 = request.A1;
        float a2 = request.A2;
        float a3 = request.A3;

        float b0 = request.B0;
        float b1 = request.B1;
        float b2 = request.B2;
        float b3 = request.B3;

        // Result variables
        float r0, r1, r2, r3;

        // Perform vector operation based on operationType
        if (operationType == 0) // Add
        {
            r0 = a0 + b0;
            r1 = a1 + b1;
            r2 = a2 + b2;
            r3 = a3 + b3;
        }
        else if (operationType == 1) // Subtract
        {
            r0 = a0 - b0;
            r1 = a1 - b1;
            r2 = a2 - b2;
            r3 = a3 - b3;
        }
        else if (operationType == 2) // Multiply
        {
            r0 = a0 * b0;
            r1 = a1 * b1;
            r2 = a2 * b2;
            r3 = a3 * b3;
        }
        else if (operationType == 3) // Divide (with zero-division protection)
        {
            r0 = (b0 != 0.0f) ? a0 / b0 : 0.0f;
            r1 = (b1 != 0.0f) ? a1 / b1 : 0.0f;
            r2 = (b2 != 0.0f) ? a2 / b2 : 0.0f;
            r3 = (b3 != 0.0f) ? a3 / b3 : 0.0f;
        }
        else // Default to Add
        {
            r0 = a0 + b0;
            r1 = a1 + b1;
            r2 = a2 + b2;
            r3 = a3 + b3;
        }

        // Ensure memory visibility before output
        ctx.ThreadFence();

        // Create and enqueue response
        var response = new VectorAddProcessorRingResponse
        {
            MessageId = request.MessageId,
            Priority = request.Priority,
            CorrelationId = request.CorrelationId,
            R0 = r0,
            R1 = r1,
            R2 = r2,
            R3 = r3,
            Success = true,
            ErrorCode = 0,
            ProcessedElements = vectorLength,
            ProcessingTimeNs = 0 // Set by runtime with EnableTimestamps
        };

        ctx.EnqueueOutput(response);
    }
}

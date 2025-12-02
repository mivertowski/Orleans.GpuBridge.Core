// Orleans.GpuBridge - GPU-native distributed computing for Microsoft Orleans
// Copyright (c) 2025 Michael Ivertowski

using Orleans.GpuBridge.Abstractions.RingKernels;
using Orleans.GpuBridge.Backends.DotCompute.Temporal;

namespace Orleans.GpuBridge.Backends.DotCompute.RingKernels.CpuFallbackHandlers;

/// <summary>
/// CPU fallback handler for vector addition ring kernel operations.
/// </summary>
/// <remarks>
/// <para>
/// This handler mirrors the GPU ring kernel logic for vector operations,
/// providing equivalent functionality when GPU execution is unavailable.
/// </para>
/// <para>
/// Supported operations:
/// <list type="bullet">
/// <item><description>0 = Add: result[i] = a[i] + b[i]</description></item>
/// <item><description>1 = Subtract: result[i] = a[i] - b[i]</description></item>
/// <item><description>2 = Multiply: result[i] = a[i] * b[i]</description></item>
/// <item><description>3 = Divide: result[i] = a[i] / b[i]</description></item>
/// </list>
/// </para>
/// </remarks>
[CpuFallbackHandler("vectoradd_processor", 0)]
public sealed class VectorAddCpuHandler
    : IStatelessCpuFallbackHandler<VectorAddProcessorRingRequest, VectorAddProcessorRingResponse>
{
    /// <inheritdoc/>
    public string KernelId => "vectoradd_processor";

    /// <inheritdoc/>
    public int HandlerId => 0;

    /// <inheritdoc/>
    public string Description => "CPU fallback for vector arithmetic operations (add, subtract, multiply, divide)";

    /// <inheritdoc/>
    public VectorAddProcessorRingResponse Execute(VectorAddProcessorRingRequest request)
    {
        var startTicks = System.Diagnostics.Stopwatch.GetTimestamp();

        float r0, r1, r2, r3;
        var vectorLength = Math.Min(request.VectorLength, 4);

        switch (request.OperationType)
        {
            case 0: // Add
                r0 = request.A0 + request.B0;
                r1 = request.A1 + request.B1;
                r2 = request.A2 + request.B2;
                r3 = request.A3 + request.B3;
                break;

            case 1: // Subtract
                r0 = request.A0 - request.B0;
                r1 = request.A1 - request.B1;
                r2 = request.A2 - request.B2;
                r3 = request.A3 - request.B3;
                break;

            case 2: // Multiply
                r0 = request.A0 * request.B0;
                r1 = request.A1 * request.B1;
                r2 = request.A2 * request.B2;
                r3 = request.A3 * request.B3;
                break;

            case 3: // Divide (with zero-division protection)
                r0 = request.B0 != 0.0f ? request.A0 / request.B0 : 0.0f;
                r1 = request.B1 != 0.0f ? request.A1 / request.B1 : 0.0f;
                r2 = request.B2 != 0.0f ? request.A2 / request.B2 : 0.0f;
                r3 = request.B3 != 0.0f ? request.A3 / request.B3 : 0.0f;
                break;

            default: // Default to Add
                r0 = request.A0 + request.B0;
                r1 = request.A1 + request.B1;
                r2 = request.A2 + request.B2;
                r3 = request.A3 + request.B3;
                break;
        }

        var endTicks = System.Diagnostics.Stopwatch.GetTimestamp();
        var elapsedNs = (endTicks - startTicks) * 1_000_000_000L / System.Diagnostics.Stopwatch.Frequency;

        return new VectorAddProcessorRingResponse
        {
            MessageId = request.MessageId,
            Priority = request.Priority,
            CorrelationId = request.CorrelationId,
            Success = true,
            ErrorCode = 0,
            ProcessedElements = vectorLength,
            ProcessingTimeNs = elapsedNs,
            R0 = r0,
            R1 = r1,
            R2 = r2,
            R3 = r3
        };
    }
}

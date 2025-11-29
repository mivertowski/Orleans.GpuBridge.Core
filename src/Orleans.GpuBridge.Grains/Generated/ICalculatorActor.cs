// Orleans.GpuBridge - GPU-native distributed computing for Microsoft Orleans
// Copyright (c) 2025 Michael Ivertowski
//
// This file demonstrates the USER-WRITTEN interface that triggers code generation.
// The source generator produces corresponding messages, grain, and kernel code.

using System;
using System.Threading.Tasks;
using Orleans;

namespace Orleans.GpuBridge.Grains.Generated;

#region Sample Attributes (Would be source-generated in real usage)

/// <summary>
/// Marks a method as a GPU-accelerated handler.
/// </summary>
/// <remarks>
/// In production, this attribute would be provided by Orleans.GpuBridge.Generators.
/// This is a sample definition for demonstration purposes.
/// </remarks>
[AttributeUsage(AttributeTargets.Method, AllowMultiple = false)]
public sealed class GpuHandlerAttribute : Attribute
{
    /// <summary>
    /// Gets or sets the handler execution mode.
    /// </summary>
    public GpuHandlerMode Mode { get; set; } = GpuHandlerMode.RequestResponse;

    /// <summary>
    /// Gets or sets the maximum payload size in bytes.
    /// </summary>
    public int MaxPayloadSize { get; set; } = 256;

    /// <summary>
    /// Gets or sets the message queue depth.
    /// </summary>
    public int QueueDepth { get; set; } = 1024;
}

/// <summary>
/// GPU handler execution modes.
/// </summary>
public enum GpuHandlerMode
{
    /// <summary>
    /// Standard request-response pattern. Waits for result.
    /// </summary>
    RequestResponse,

    /// <summary>
    /// Fire-and-forget pattern. Returns immediately after queuing.
    /// </summary>
    FireAndForget
}

#endregion

/// <summary>
/// GPU-native calculator actor demonstrating source-generated ring kernel integration.
/// </summary>
/// <remarks>
/// <para>
/// This interface is decorated with <see cref="Orleans.GpuBridge.Runtime.RingKernels.GpuNativeActorAttribute"/> to trigger
/// the GpuNativeActorGenerator source generator, which produces:
/// </para>
/// <list type="bullet">
/// <item><description>CalculatorActorMessages.g.cs - Blittable message structs</description></item>
/// <item><description>CalculatorActorGrain.g.cs - Orleans grain implementation</description></item>
/// <item><description>CalculatorActorKernels.g.cs - DotCompute ring kernel code</description></item>
/// </list>
/// <para>
/// <strong>Generated Message Flow:</strong>
/// <code>
/// AddAsync(3, 5) → AddRequest{A=3, B=5} → GPU Kernel → AddResponse{Result=8} → 8
/// </code>
/// </para>
/// </remarks>
public interface ICalculatorActor : IGrainWithIntegerKey
{
    /// <summary>
    /// Adds two integers on the GPU.
    /// </summary>
    /// <param name="a">First operand</param>
    /// <param name="b">Second operand</param>
    /// <returns>Sum of a and b</returns>
    [GpuHandler]
    Task<int> AddAsync(int a, int b);

    /// <summary>
    /// Subtracts b from a on the GPU.
    /// </summary>
    /// <param name="a">First operand</param>
    /// <param name="b">Second operand</param>
    /// <returns>Difference (a - b)</returns>
    [GpuHandler]
    Task<int> SubtractAsync(int a, int b);

    /// <summary>
    /// Multiplies two integers on the GPU.
    /// </summary>
    /// <param name="a">First operand</param>
    /// <param name="b">Second operand</param>
    /// <returns>Product of a and b</returns>
    [GpuHandler(Mode = GpuHandlerMode.FireAndForget)]
    Task MultiplyAsync(int a, int b);

    /// <summary>
    /// Computes factorial of n on the GPU.
    /// </summary>
    /// <param name="n">Input value (must be non-negative)</param>
    /// <returns>Factorial of n (n!)</returns>
    [GpuHandler(MaxPayloadSize = 128, QueueDepth = 512)]
    Task<long> FactorialAsync(int n);

    /// <summary>
    /// Gets the current accumulated value.
    /// </summary>
    /// <returns>Current accumulator value</returns>
    [GpuHandler]
    Task<long> GetAccumulatorAsync();

    /// <summary>
    /// Resets the accumulator to zero.
    /// </summary>
    [GpuHandler(Mode = GpuHandlerMode.FireAndForget)]
    Task ResetAsync();
}

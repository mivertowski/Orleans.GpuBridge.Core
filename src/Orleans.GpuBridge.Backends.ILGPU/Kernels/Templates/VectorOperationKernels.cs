using System;
using ILGPU;
using ILGPU.Runtime;

namespace Orleans.GpuBridge.Backends.ILGPU.Kernels.Templates;

/// <summary>
/// ILGPU kernel templates for vector operations with actual GPU execution
/// </summary>
public static class VectorOperationKernels
{
    /// <summary>
    /// Vector addition kernel: C[i] = A[i] + B[i]
    /// </summary>
    public static void VectorAdd(
        Index1D index,
        ArrayView<float> a,
        ArrayView<float> b,
        ArrayView<float> c)
    {
        c[index] = a[index] + b[index];
    }

    /// <summary>
    /// Vector subtraction kernel: C[i] = A[i] - B[i]
    /// </summary>
    public static void VectorSubtract(
        Index1D index,
        ArrayView<float> a,
        ArrayView<float> b,
        ArrayView<float> c)
    {
        c[index] = a[index] - b[index];
    }

    /// <summary>
    /// Vector multiplication kernel: C[i] = A[i] * B[i]
    /// </summary>
    public static void VectorMultiply(
        Index1D index,
        ArrayView<float> a,
        ArrayView<float> b,
        ArrayView<float> c)
    {
        c[index] = a[index] * b[index];
    }

    /// <summary>
    /// Vector division kernel: C[i] = A[i] / B[i]
    /// </summary>
    public static void VectorDivide(
        Index1D index,
        ArrayView<float> a,
        ArrayView<float> b,
        ArrayView<float> c)
    {
        var denominator = b[index];
        c[index] = denominator != 0.0f ? a[index] / denominator : 0.0f;
    }

    /// <summary>
    /// Scalar multiplication kernel: B[i] = A[i] * scalar
    /// </summary>
    public static void ScalarMultiply(
        Index1D index,
        ArrayView<float> input,
        ArrayView<float> output,
        float scalar)
    {
        output[index] = input[index] * scalar;
    }

    /// <summary>
    /// Scalar addition kernel: B[i] = A[i] + scalar
    /// </summary>
    public static void ScalarAdd(
        Index1D index,
        ArrayView<float> input,
        ArrayView<float> output,
        float scalar)
    {
        output[index] = input[index] + scalar;
    }

    /// <summary>
    /// Fused multiply-add kernel: D[i] = A[i] * B[i] + C[i]
    /// </summary>
    public static void FusedMultiplyAdd(
        Index1D index,
        ArrayView<float> a,
        ArrayView<float> b,
        ArrayView<float> c,
        ArrayView<float> d)
    {
        d[index] = a[index] * b[index] + c[index];
    }

    /// <summary>
    /// Dot product kernel (partial sums)
    /// </summary>
    public static void DotProductPartial(
        Index1D index,
        ArrayView<float> a,
        ArrayView<float> b,
        ArrayView<float> partialSums)
    {
        partialSums[index] = a[index] * b[index];
    }

    /// <summary>
    /// Vector norm calculation (L2 norm partial)
    /// </summary>
    public static void VectorNormPartial(
        Index1D index,
        ArrayView<float> input,
        ArrayView<float> partialSums)
    {
        var value = input[index];
        partialSums[index] = value * value;
    }

    /// <summary>
    /// Clamp values kernel: output[i] = clamp(input[i], min, max)
    /// </summary>
    public static void Clamp(
        Index1D index,
        ArrayView<float> input,
        ArrayView<float> output,
        float minValue,
        float maxValue)
    {
        var value = input[index];
        output[index] = XMath.Max(minValue, XMath.Min(maxValue, value));
    }

    /// <summary>
    /// Absolute value kernel: output[i] = abs(input[i])
    /// </summary>
    public static void Abs(
        Index1D index,
        ArrayView<float> input,
        ArrayView<float> output)
    {
        output[index] = XMath.Abs(input[index]);
    }

    /// <summary>
    /// Power kernel: output[i] = pow(input[i], exponent)
    /// </summary>
    public static void Power(
        Index1D index,
        ArrayView<float> input,
        ArrayView<float> output,
        float exponent)
    {
        output[index] = XMath.Pow(input[index], exponent);
    }

    /// <summary>
    /// Square root kernel: output[i] = sqrt(input[i])
    /// </summary>
    public static void Sqrt(
        Index1D index,
        ArrayView<float> input,
        ArrayView<float> output)
    {
        output[index] = XMath.Sqrt(XMath.Max(0.0f, input[index]));
    }

    /// <summary>
    /// Exponential kernel: output[i] = exp(input[i])
    /// </summary>
    public static void Exp(
        Index1D index,
        ArrayView<float> input,
        ArrayView<float> output)
    {
        output[index] = XMath.Exp(input[index]);
    }

    /// <summary>
    /// Natural logarithm kernel: output[i] = log(input[i])
    /// </summary>
    public static void Log(
        Index1D index,
        ArrayView<float> input,
        ArrayView<float> output)
    {
        var value = input[index];
        output[index] = value > 0.0f ? XMath.Log(value) : float.NegativeInfinity;
    }

    /// <summary>
    /// Sine kernel: output[i] = sin(input[i])
    /// </summary>
    public static void Sin(
        Index1D index,
        ArrayView<float> input,
        ArrayView<float> output)
    {
        output[index] = XMath.Sin(input[index]);
    }

    /// <summary>
    /// Cosine kernel: output[i] = cos(input[i])
    /// </summary>
    public static void Cos(
        Index1D index,
        ArrayView<float> input,
        ArrayView<float> output)
    {
        output[index] = XMath.Cos(input[index]);
    }

    /// <summary>
    /// SAXPY operation: Y[i] = alpha * X[i] + Y[i]
    /// </summary>
    public static void SAXPY(
        Index1D index,
        ArrayView<float> x,
        ArrayView<float> y,
        float alpha)
    {
        y[index] = alpha * x[index] + y[index];
    }

    /// <summary>
    /// Copy kernel: output[i] = input[i]
    /// </summary>
    public static void Copy(
        Index1D index,
        ArrayView<float> input,
        ArrayView<float> output)
    {
        output[index] = input[index];
    }

    /// <summary>
    /// Fill kernel: output[i] = value
    /// </summary>
    public static void Fill(
        Index1D index,
        ArrayView<float> output,
        float value)
    {
        output[index] = value;
    }
}
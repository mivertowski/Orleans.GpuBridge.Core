using System;
using System.Runtime.CompilerServices;
using Orleans.GpuBridge.Abstractions;
using Orleans.GpuBridge.Backends.DotCompute.Attributes;

namespace Orleans.GpuBridge.Backends.DotCompute.Kernels;

/// <summary>
/// Sample GPU kernels for common operations
/// </summary>
public static class SampleKernels
{
    /// <summary>
    /// Vector addition kernel
    /// </summary>
    [Kernel("vector/add", PreferredWorkGroupSize = 256)]
    public static void VectorAdd(
        ReadOnlySpan<float> a,
        ReadOnlySpan<float> b,
        Span<float> result,
        int size)
    {
        // This will be compiled to GPU code
        // CPU fallback implementation:
        for (int i = 0; i < size; i++)
        {
            result[i] = a[i] + b[i];
        }
    }

    /// <summary>
    /// Vector multiplication kernel
    /// </summary>
    [Kernel("vector/multiply", PreferredWorkGroupSize = 256)]
    public static void VectorMultiply(
        ReadOnlySpan<float> a,
        ReadOnlySpan<float> b,
        Span<float> result,
        int size)
    {
        for (int i = 0; i < size; i++)
        {
            result[i] = a[i] * b[i];
        }
    }

    /// <summary>
    /// Matrix multiplication kernel
    /// </summary>
    [Kernel("matrix/multiply", PreferredWorkGroupSize = 16, RequiresSharedMemory = true)]
    public static void MatrixMultiply(
        ReadOnlySpan<float> a,
        ReadOnlySpan<float> b,
        Span<float> c,
        int m,
        int n,
        int k)
    {
        // Naive matrix multiplication for CPU fallback
        // GPU version would use shared memory and tiling
        for (int row = 0; row < m; row++)
        {
            for (int col = 0; col < n; col++)
            {
                float sum = 0.0f;
                for (int i = 0; i < k; i++)
                {
                    sum += a[row * k + i] * b[i * n + col];
                }
                c[row * n + col] = sum;
            }
        }
    }

    /// <summary>
    /// Reduction kernel (sum)
    /// </summary>
    [Kernel("reduce/sum", PreferredWorkGroupSize = 128)]
    public static void ReduceSum(
        ReadOnlySpan<float> input,
        Span<float> output,
        int size)
    {
        float sum = 0.0f;
        for (int i = 0; i < size; i++)
        {
            sum += input[i];
        }
        output[0] = sum;
    }

    /// <summary>
    /// Convolution kernel
    /// </summary>
    [Kernel("conv/2d", PreferredWorkGroupSize = 32)]
    public static void Convolution2D(
        ReadOnlySpan<float> input,
        ReadOnlySpan<float> kernel,
        Span<float> output,
        int inputWidth,
        int inputHeight,
        int kernelWidth,
        int kernelHeight,
        int outputWidth,
        int outputHeight)
    {
        // Simple 2D convolution for CPU fallback
        for (int y = 0; y < outputHeight; y++)
        {
            for (int x = 0; x < outputWidth; x++)
            {
                float sum = 0.0f;

                for (int ky = 0; ky < kernelHeight; ky++)
                {
                    for (int kx = 0; kx < kernelWidth; kx++)
                    {
                        int inputY = y + ky;
                        int inputX = x + kx;

                        if (inputY < inputHeight && inputX < inputWidth)
                        {
                            sum += input[inputY * inputWidth + inputX] *
                                   kernel[ky * kernelWidth + kx];
                        }
                    }
                }

                output[y * outputWidth + x] = sum;
            }
        }
    }

    /// <summary>
    /// Softmax kernel
    /// </summary>
    [Kernel("activation/softmax", PreferredWorkGroupSize = 256)]
    public static void Softmax(
        ReadOnlySpan<float> input,
        Span<float> output,
        int size)
    {
        // Find max for numerical stability
        float max = float.MinValue;
        for (int i = 0; i < size; i++)
        {
            if (input[i] > max) max = input[i];
        }

        // Compute exp and sum
        float sum = 0.0f;
        for (int i = 0; i < size; i++)
        {
            output[i] = MathF.Exp(input[i] - max);
            sum += output[i];
        }

        // Normalize
        for (int i = 0; i < size; i++)
        {
            output[i] /= sum;
        }
    }

    /// <summary>
    /// ReLU activation kernel
    /// </summary>
    [Kernel("activation/relu", PreferredWorkGroupSize = 512)]
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void ReLU(
        ReadOnlySpan<float> input,
        Span<float> output,
        int size)
    {
        for (int i = 0; i < size; i++)
        {
            output[i] = MathF.Max(0.0f, input[i]);
        }
    }

    /// <summary>
    /// Batch normalization kernel
    /// </summary>
    [Kernel("norm/batch", PreferredWorkGroupSize = 128)]
    public static void BatchNormalization(
        ReadOnlySpan<float> input,
        ReadOnlySpan<float> mean,
        ReadOnlySpan<float> variance,
        ReadOnlySpan<float> gamma,
        ReadOnlySpan<float> beta,
        Span<float> output,
        int batchSize,
        int channels,
        int spatialSize,
        float epsilon = 1e-5f)
    {
        int totalSize = batchSize * channels * spatialSize;

        for (int idx = 0; idx < totalSize; idx++)
        {
            int c = (idx / spatialSize) % channels;

            float normalized = (input[idx] - mean[c]) /
                              MathF.Sqrt(variance[c] + epsilon);
            output[idx] = gamma[c] * normalized + beta[c];
        }
    }
}

/// <summary>
/// Image processing kernels
/// </summary>
public static class ImageKernels
{
    /// <summary>
    /// Grayscale conversion kernel
    /// </summary>
    [Kernel("image/grayscale", PreferredWorkGroupSize = 256)]
    public static void RgbToGrayscale(
        ReadOnlySpan<byte> rgb,
        Span<byte> gray,
        int pixelCount)
    {
        for (int i = 0; i < pixelCount; i++)
        {
            int idx = i * 3;
            byte r = rgb[idx];
            byte g = rgb[idx + 1];
            byte b = rgb[idx + 2];

            // Standard grayscale conversion
            gray[i] = (byte)(0.299f * r + 0.587f * g + 0.114f * b);
        }
    }

    /// <summary>
    /// Gaussian blur kernel
    /// </summary>
    [Kernel("image/blur", PreferredWorkGroupSize = 16)]
    public static void GaussianBlur(
        ReadOnlySpan<float> input,
        Span<float> output,
        int width,
        int height,
        float sigma = 1.0f)
    {
        // Simple 3x3 Gaussian blur
        float[,] kernel = {
            { 0.0625f, 0.125f, 0.0625f },
            { 0.125f,  0.25f,  0.125f },
            { 0.0625f, 0.125f, 0.0625f }
        };

        for (int y = 1; y < height - 1; y++)
        {
            for (int x = 1; x < width - 1; x++)
            {
                float sum = 0.0f;

                for (int ky = -1; ky <= 1; ky++)
                {
                    for (int kx = -1; kx <= 1; kx++)
                    {
                        int idx = (y + ky) * width + (x + kx);
                        sum += input[idx] * kernel[ky + 1, kx + 1];
                    }
                }

                output[y * width + x] = sum;
            }
        }
    }

    /// <summary>
    /// Edge detection kernel (Sobel)
    /// </summary>
    [Kernel("image/edges", PreferredWorkGroupSize = 16)]
    public static void SobelEdgeDetection(
        ReadOnlySpan<float> input,
        Span<float> output,
        int width,
        int height)
    {
        // Sobel operators
        int[,] sobelX = {
            { -1, 0, 1 },
            { -2, 0, 2 },
            { -1, 0, 1 }
        };

        int[,] sobelY = {
            { -1, -2, -1 },
            {  0,  0,  0 },
            {  1,  2,  1 }
        };

        for (int y = 1; y < height - 1; y++)
        {
            for (int x = 1; x < width - 1; x++)
            {
                float gx = 0.0f;
                float gy = 0.0f;

                for (int ky = -1; ky <= 1; ky++)
                {
                    for (int kx = -1; kx <= 1; kx++)
                    {
                        int idx = (y + ky) * width + (x + kx);
                        gx += input[idx] * sobelX[ky + 1, kx + 1];
                        gy += input[idx] * sobelY[ky + 1, kx + 1];
                    }
                }

                output[y * width + x] = MathF.Sqrt(gx * gx + gy * gy);
            }
        }
    }
}
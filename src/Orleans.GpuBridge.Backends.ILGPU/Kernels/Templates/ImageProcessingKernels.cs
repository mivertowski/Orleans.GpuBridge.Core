using System;
using ILGPU;
using ILGPU.Runtime;

namespace Orleans.GpuBridge.Backends.ILGPU.Kernels.Templates;

/// <summary>
/// ILGPU kernel templates for image processing operations with actual GPU execution
/// </summary>
public static class ImageProcessingKernels
{
    /// <summary>
    /// Grayscale conversion kernel (RGB to Gray)
    /// </summary>
    public static void RGBToGrayscale(
        Index2D index,
        ArrayView<byte> rgbInput,
        ArrayView<byte> grayOutput,
        int width,
        int height)
    {
        var x = index.X;
        var y = index.Y;
        
        if (x >= width || y >= height)
            return;
        
        var pixelIndex = y * width + x;
        var rgbIndex = pixelIndex * 3;
        
        var r = rgbInput[rgbIndex];
        var g = rgbInput[rgbIndex + 1];
        var b = rgbInput[rgbIndex + 2];
        
        // Standard luminance formula
        var gray = (byte)(0.299f * r + 0.587f * g + 0.114f * b);
        grayOutput[pixelIndex] = gray;
    }

    /// <summary>
    /// Gaussian blur kernel (3x3)
    /// </summary>
    public static void GaussianBlur3x3(
        Index2D index,
        ArrayView<float> input,
        ArrayView<float> output,
        int width,
        int height)
    {
        var x = index.X;
        var y = index.Y;
        
        if (x >= width || y >= height)
            return;
        
        // Gaussian kernel weights (normalized)
        float[,] kernel = {
            { 1/16f, 2/16f, 1/16f },
            { 2/16f, 4/16f, 2/16f },
            { 1/16f, 2/16f, 1/16f }
        };
        
        float sum = 0.0f;
        
        for (int dy = -1; dy <= 1; dy++)
        {
            for (int dx = -1; dx <= 1; dx++)
            {
                var nx = XMath.Clamp(x + dx, 0, width - 1);
                var ny = XMath.Clamp(y + dy, 0, height - 1);
                
                sum += input[ny * width + nx] * kernel[dy + 1, dx + 1];
            }
        }
        
        output[y * width + x] = sum;
    }

    /// <summary>
    /// Sobel edge detection kernel (gradient magnitude)
    /// </summary>
    public static void SobelEdgeDetection(
        Index2D index,
        ArrayView<float> input,
        ArrayView<float> output,
        int width,
        int height)
    {
        var x = index.X;
        var y = index.Y;
        
        if (x >= width || y >= height)
            return;
        
        // Sobel X kernel
        float[,] sobelX = {
            { -1, 0, 1 },
            { -2, 0, 2 },
            { -1, 0, 1 }
        };
        
        // Sobel Y kernel
        float[,] sobelY = {
            { -1, -2, -1 },
            {  0,  0,  0 },
            {  1,  2,  1 }
        };
        
        float gx = 0.0f;
        float gy = 0.0f;
        
        for (int dy = -1; dy <= 1; dy++)
        {
            for (int dx = -1; dx <= 1; dx++)
            {
                var nx = XMath.Clamp(x + dx, 0, width - 1);
                var ny = XMath.Clamp(y + dy, 0, height - 1);
                
                var pixel = input[ny * width + nx];
                gx += pixel * sobelX[dy + 1, dx + 1];
                gy += pixel * sobelY[dy + 1, dx + 1];
            }
        }
        
        // Gradient magnitude
        output[y * width + x] = XMath.Sqrt(gx * gx + gy * gy);
    }

    /// <summary>
    /// Image thresholding kernel (binary threshold)
    /// </summary>
    public static void Threshold(
        Index2D index,
        ArrayView<float> input,
        ArrayView<float> output,
        float threshold,
        int width,
        int height)
    {
        var x = index.X;
        var y = index.Y;
        
        if (x >= width || y >= height)
            return;
        
        var pixelIndex = y * width + x;
        output[pixelIndex] = input[pixelIndex] > threshold ? 1.0f : 0.0f;
    }

    /// <summary>
    /// Histogram equalization kernel (compute cumulative distribution)
    /// </summary>
    public static void HistogramEqualization(
        Index1D index,
        ArrayView<byte> input,
        ArrayView<byte> output,
        ArrayView<int> histogram,
        ArrayView<float> cdf,
        int numPixels)
    {
        if (index >= input.Length)
            return;
        
        var pixelValue = input[index];
        var equalizedValue = (byte)(cdf[pixelValue] * 255.0f);
        output[index] = equalizedValue;
    }

    /// <summary>
    /// Box blur kernel (average filter)
    /// </summary>
    public static void BoxBlur(
        Index2D index,
        ArrayView<float> input,
        ArrayView<float> output,
        int width,
        int height,
        int kernelRadius)
    {
        var x = index.X;
        var y = index.Y;
        
        if (x >= width || y >= height)
            return;
        
        float sum = 0.0f;
        int count = 0;
        
        for (int dy = -kernelRadius; dy <= kernelRadius; dy++)
        {
            for (int dx = -kernelRadius; dx <= kernelRadius; dx++)
            {
                var nx = x + dx;
                var ny = y + dy;
                
                if (nx >= 0 && nx < width && ny >= 0 && ny < height)
                {
                    sum += input[ny * width + nx];
                    count++;
                }
            }
        }
        
        output[y * width + x] = sum / count;
    }

    /// <summary>
    /// Image rotation kernel (bilinear interpolation)
    /// </summary>
    public static void RotateImage(
        Index2D index,
        ArrayView<float> input,
        ArrayView<float> output,
        int width,
        int height,
        float angle)
    {
        var x = index.X;
        var y = index.Y;
        
        if (x >= width || y >= height)
            return;
        
        // Center coordinates
        float cx = width / 2.0f;
        float cy = height / 2.0f;
        
        // Translate to center
        float tx = x - cx;
        float ty = y - cy;
        
        // Rotate
        float cosAngle = XMath.Cos(-angle);
        float sinAngle = XMath.Sin(-angle);
        
        float srcX = tx * cosAngle - ty * sinAngle + cx;
        float srcY = tx * sinAngle + ty * cosAngle + cy;
        
        // Bilinear interpolation
        if (srcX >= 0 && srcX < width - 1 && srcY >= 0 && srcY < height - 1)
        {
            int x0 = (int)srcX;
            int y0 = (int)srcY;
            int x1 = x0 + 1;
            int y1 = y0 + 1;
            
            float dx = srcX - x0;
            float dy = srcY - y0;
            
            float v00 = input[y0 * width + x0];
            float v10 = input[y0 * width + x1];
            float v01 = input[y1 * width + x0];
            float v11 = input[y1 * width + x1];
            
            float value = v00 * (1 - dx) * (1 - dy) +
                         v10 * dx * (1 - dy) +
                         v01 * (1 - dx) * dy +
                         v11 * dx * dy;
            
            output[y * width + x] = value;
        }
        else
        {
            output[y * width + x] = 0.0f;
        }
    }

    /// <summary>
    /// Image scaling kernel (nearest neighbor)
    /// </summary>
    public static void ScaleImageNearestNeighbor(
        Index2D index,
        ArrayView<float> input,
        ArrayView<float> output,
        int srcWidth,
        int srcHeight,
        int dstWidth,
        int dstHeight)
    {
        var x = index.X;
        var y = index.Y;
        
        if (x >= dstWidth || y >= dstHeight)
            return;
        
        float scaleX = (float)srcWidth / dstWidth;
        float scaleY = (float)srcHeight / dstHeight;
        
        int srcX = (int)(x * scaleX);
        int srcY = (int)(y * scaleY);
        
        srcX = XMath.Clamp(srcX, 0, srcWidth - 1);
        srcY = XMath.Clamp(srcY, 0, srcHeight - 1);
        
        output[y * dstWidth + x] = input[srcY * srcWidth + srcX];
    }

    /// <summary>
    /// Median filter kernel (3x3) - simplified version
    /// </summary>
    public static void MedianFilter3x3(
        Index2D index,
        ArrayView<float> input,
        ArrayView<float> output,
        int width,
        int height)
    {
        var x = index.X;
        var y = index.Y;
        
        if (x >= width || y >= height)
            return;
        
        // Collect 3x3 neighborhood
        float[] values = new float[9];
        int count = 0;
        
        for (int dy = -1; dy <= 1; dy++)
        {
            for (int dx = -1; dx <= 1; dx++)
            {
                var nx = XMath.Clamp(x + dx, 0, width - 1);
                var ny = XMath.Clamp(y + dy, 0, height - 1);
                
                values[count++] = input[ny * width + nx];
            }
        }
        
        // Simple bubble sort for 9 elements
        for (int i = 0; i < 9; i++)
        {
            for (int j = i + 1; j < 9; j++)
            {
                if (values[j] < values[i])
                {
                    var temp = values[i];
                    values[i] = values[j];
                    values[j] = temp;
                }
            }
        }
        
        // Median is the middle element
        output[y * width + x] = values[4];
    }

    /// <summary>
    /// Brightness adjustment kernel
    /// </summary>
    public static void AdjustBrightness(
        Index1D index,
        ArrayView<float> input,
        ArrayView<float> output,
        float brightness)
    {
        if (index >= input.Length)
            return;
        
        output[index] = XMath.Clamp(input[index] + brightness, 0.0f, 1.0f);
    }

    /// <summary>
    /// Contrast adjustment kernel
    /// </summary>
    public static void AdjustContrast(
        Index1D index,
        ArrayView<float> input,
        ArrayView<float> output,
        float contrast)
    {
        if (index >= input.Length)
            return;
        
        var value = input[index];
        var adjusted = 0.5f + contrast * (value - 0.5f);
        output[index] = XMath.Clamp(adjusted, 0.0f, 1.0f);
    }

    /// <summary>
    /// Gamma correction kernel
    /// </summary>
    public static void GammaCorrection(
        Index1D index,
        ArrayView<float> input,
        ArrayView<float> output,
        float gamma)
    {
        if (index >= input.Length)
            return;
        
        output[index] = XMath.Pow(input[index], 1.0f / gamma);
    }

    /// <summary>
    /// Image convolution kernel (generic)
    /// </summary>
    public static void Convolution(
        Index2D index,
        ArrayView<float> input,
        ArrayView<float> output,
        ArrayView<float> kernel,
        int width,
        int height,
        int kernelSize)
    {
        var x = index.X;
        var y = index.Y;
        
        if (x >= width || y >= height)
            return;
        
        int halfKernel = kernelSize / 2;
        float sum = 0.0f;
        
        for (int ky = 0; ky < kernelSize; ky++)
        {
            for (int kx = 0; kx < kernelSize; kx++)
            {
                var nx = XMath.Clamp(x + kx - halfKernel, 0, width - 1);
                var ny = XMath.Clamp(y + ky - halfKernel, 0, height - 1);
                
                sum += input[ny * width + nx] * kernel[ky * kernelSize + kx];
            }
        }
        
        output[y * width + x] = sum;
    }
}
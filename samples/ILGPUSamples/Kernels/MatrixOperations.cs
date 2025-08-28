// Copyright (c) 2025 Michael Ivertowski. All Rights Reserved.
// Licensed under the Apache License, Version 2.0.

using System;
using ILGPU;
using ILGPU.Runtime;

namespace Orleans.GpuBridge.Samples.ILGPU.Kernels;

/// <summary>
/// Sample ILGPU kernels for matrix operations
/// </summary>
public static class MatrixOperations
{
    /// <summary>
    /// Matrix multiplication: C = A * B
    /// Using tiled/blocked approach for better cache usage
    /// </summary>
    public static void MatrixMultiplyTiled(
        Index2D index,
        ArrayView2D<float, Stride2D.DenseX> a,
        ArrayView2D<float, Stride2D.DenseX> b,
        ArrayView2D<float, Stride2D.DenseX> c,
        int tileSize)
    {
        var globalRow = index.Y;
        var globalCol = index.X;
        
        if (globalRow >= c.Extent.Y || globalCol >= c.Extent.X)
            return;
        
        var sum = 0.0f;
        var numTiles = (a.Extent.X + tileSize - 1) / tileSize;
        
        for (int t = 0; t < numTiles; t++)
        {
            // Load tile boundaries
            var tileStart = t * tileSize;
            var tileEnd = IntrinsicMath.Min(tileStart + tileSize, a.Extent.X);
            
            // Compute partial sum for this tile
            for (int k = tileStart; k < tileEnd; k++)
            {
                sum += a[globalRow, k] * b[k, globalCol];
            }
        }
        
        c[globalRow, globalCol] = sum;
    }
    
    /// <summary>
    /// Simple matrix multiplication without tiling
    /// </summary>
    public static void MatrixMultiplySimple(
        Index2D index,
        ArrayView2D<float, Stride2D.DenseX> a,
        ArrayView2D<float, Stride2D.DenseX> b,
        ArrayView2D<float, Stride2D.DenseX> c)
    {
        var row = index.Y;
        var col = index.X;
        
        if (row >= c.Extent.Y || col >= c.Extent.X)
            return;
        
        var sum = 0.0f;
        for (int k = 0; k < a.Extent.X; k++)
        {
            sum += a[row, k] * b[k, col];
        }
        
        c[row, col] = sum;
    }
    
    /// <summary>
    /// Matrix transpose: B = A^T
    /// </summary>
    public static void MatrixTranspose(
        Index2D index,
        ArrayView2D<float, Stride2D.DenseX> input,
        ArrayView2D<float, Stride2D.DenseX> output)
    {
        var row = index.Y;
        var col = index.X;
        
        if (row < input.Extent.Y && col < input.Extent.X)
        {
            output[col, row] = input[row, col];
        }
    }
    
    /// <summary>
    /// Matrix addition: C = A + B
    /// </summary>
    public static void MatrixAdd(
        Index2D index,
        ArrayView2D<float, Stride2D.DenseX> a,
        ArrayView2D<float, Stride2D.DenseX> b,
        ArrayView2D<float, Stride2D.DenseX> result)
    {
        var row = index.Y;
        var col = index.X;
        
        if (row < result.Extent.Y && col < result.Extent.X)
        {
            result[row, col] = a[row, col] + b[row, col];
        }
    }
    
    /// <summary>
    /// Matrix scalar multiplication: B = scalar * A
    /// </summary>
    public static void MatrixScalarMultiply(
        Index2D index,
        ArrayView2D<float, Stride2D.DenseX> matrix,
        float scalar,
        ArrayView2D<float, Stride2D.DenseX> result)
    {
        var row = index.Y;
        var col = index.X;
        
        if (row < result.Extent.Y && col < result.Extent.X)
        {
            result[row, col] = matrix[row, col] * scalar;
        }
    }
    
    /// <summary>
    /// Computes row-wise sum of a matrix
    /// </summary>
    public static void MatrixRowSum(
        Index1D index,
        ArrayView2D<float, Stride2D.DenseX> matrix,
        ArrayView<float> rowSums)
    {
        var row = index.X;
        
        if (row >= matrix.Extent.Y)
            return;
        
        var sum = 0.0f;
        for (int col = 0; col < matrix.Extent.X; col++)
        {
            sum += matrix[row, col];
        }
        
        rowSums[row] = sum;
    }
    
    /// <summary>
    /// Computes column-wise sum of a matrix
    /// </summary>
    public static void MatrixColumnSum(
        Index1D index,
        ArrayView2D<float, Stride2D.DenseX> matrix,
        ArrayView<float> colSums)
    {
        var col = index.X;
        
        if (col >= matrix.Extent.X)
            return;
        
        var sum = 0.0f;
        for (int row = 0; row < matrix.Extent.Y; row++)
        {
            sum += matrix[row, col];
        }
        
        colSums[col] = sum;
    }
    
    /// <summary>
    /// Applies softmax to each row of a matrix
    /// This is a simplified version - full implementation would need two passes
    /// </summary>
    public static void MatrixSoftmaxRow(
        Index2D index,
        ArrayView2D<float, Stride2D.DenseX> input,
        ArrayView2D<float, Stride2D.DenseX> output,
        ArrayView<float> rowMaxValues,
        ArrayView<float> rowSumExp)
    {
        var row = index.Y;
        var col = index.X;
        
        if (row >= input.Extent.Y || col >= input.Extent.X)
            return;
        
        // Assume max and sum have been precomputed
        var maxVal = rowMaxValues[row];
        var sumExp = rowSumExp[row];
        
        // Compute softmax
        var expVal = IntrinsicMath.Exp(input[row, col] - maxVal);
        output[row, col] = expVal / sumExp;
    }
    
    /// <summary>
    /// Applies ReLU activation to matrix elements
    /// </summary>
    public static void MatrixReLU(
        Index2D index,
        ArrayView2D<float, Stride2D.DenseX> input,
        ArrayView2D<float, Stride2D.DenseX> output)
    {
        var row = index.Y;
        var col = index.X;
        
        if (row < input.Extent.Y && col < input.Extent.X)
        {
            output[row, col] = IntrinsicMath.Max(0.0f, input[row, col]);
        }
    }
    
    /// <summary>
    /// Computes element-wise matrix multiplication (Hadamard product)
    /// </summary>
    public static void MatrixElementwiseMultiply(
        Index2D index,
        ArrayView2D<float, Stride2D.DenseX> a,
        ArrayView2D<float, Stride2D.DenseX> b,
        ArrayView2D<float, Stride2D.DenseX> result)
    {
        var row = index.Y;
        var col = index.X;
        
        if (row < result.Extent.Y && col < result.Extent.X)
        {
            result[row, col] = a[row, col] * b[row, col];
        }
    }
}
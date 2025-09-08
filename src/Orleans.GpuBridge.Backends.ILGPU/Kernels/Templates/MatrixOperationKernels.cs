using System;
using ILGPU;
using ILGPU.Runtime;

namespace Orleans.GpuBridge.Backends.ILGPU.Kernels.Templates;

/// <summary>
/// ILGPU kernel templates for matrix operations with actual GPU execution
/// </summary>
public static class MatrixOperationKernels
{
    /// <summary>
    /// Matrix multiplication kernel: C = A * B
    /// Uses tiled algorithm for improved cache performance
    /// </summary>
    public static void MatrixMultiply(
        Index2D index,
        ArrayView2D<float, Stride2D.DenseX> a,
        ArrayView2D<float, Stride2D.DenseX> b,
        ArrayView2D<float, Stride2D.DenseX> c,
        SpecializedValue<int> tileSize)
    {
        var globalRow = index.Y;
        var globalCol = index.X;
        
        if (globalRow >= c.IntExtent.Y || globalCol >= c.IntExtent.X)
            return;
        
        var tileRow = Group.IdxY;
        var tileCol = Group.IdxX;
        
        var sharedA = SharedMemory.Allocate2D<float, Stride2D.DenseX>(
            new Index2D(tileSize, tileSize),
            new Stride2D.DenseX(tileSize));
        var sharedB = SharedMemory.Allocate2D<float, Stride2D.DenseX>(
            new Index2D(tileSize, tileSize),
            new Stride2D.DenseX(tileSize));
        
        float sum = 0.0f;
        var numTiles = (a.IntExtent.X + tileSize - 1) / tileSize;
        
        for (int tile = 0; tile < numTiles; tile++)
        {
            // Load tiles into shared memory
            var aCol = tile * tileSize + tileCol;
            var bRow = tile * tileSize + tileRow;
            
            sharedA[tileRow, tileCol] = (globalRow < a.IntExtent.Y && aCol < a.IntExtent.X) 
                ? a[globalRow, aCol] : 0.0f;
            sharedB[tileRow, tileCol] = (bRow < b.IntExtent.Y && globalCol < b.IntExtent.X) 
                ? b[bRow, globalCol] : 0.0f;
            
            Group.Barrier();
            
            // Compute partial dot product
            for (int k = 0; k < tileSize; k++)
            {
                sum += sharedA[tileRow, k] * sharedB[k, tileCol];
            }
            
            Group.Barrier();
        }
        
        c[globalRow, globalCol] = sum;
    }

    /// <summary>
    /// Matrix transpose kernel: B = A^T
    /// </summary>
    public static void MatrixTranspose(
        Index2D index,
        ArrayView2D<float, Stride2D.DenseX> input,
        ArrayView2D<float, Stride2D.DenseX> output)
    {
        var row = index.Y;
        var col = index.X;
        
        if (row < input.IntExtent.Y && col < input.IntExtent.X)
        {
            output[col, row] = input[row, col];
        }
    }

    /// <summary>
    /// Matrix addition kernel: C = A + B
    /// </summary>
    public static void MatrixAdd(
        Index2D index,
        ArrayView2D<float, Stride2D.DenseX> a,
        ArrayView2D<float, Stride2D.DenseX> b,
        ArrayView2D<float, Stride2D.DenseX> c)
    {
        var row = index.Y;
        var col = index.X;
        
        if (row < a.IntExtent.Y && col < a.IntExtent.X)
        {
            c[row, col] = a[row, col] + b[row, col];
        }
    }

    /// <summary>
    /// Matrix subtraction kernel: C = A - B
    /// </summary>
    public static void MatrixSubtract(
        Index2D index,
        ArrayView2D<float, Stride2D.DenseX> a,
        ArrayView2D<float, Stride2D.DenseX> b,
        ArrayView2D<float, Stride2D.DenseX> c)
    {
        var row = index.Y;
        var col = index.X;
        
        if (row < a.IntExtent.Y && col < a.IntExtent.X)
        {
            c[row, col] = a[row, col] - b[row, col];
        }
    }

    /// <summary>
    /// Matrix scalar multiplication kernel: B = alpha * A
    /// </summary>
    public static void MatrixScalarMultiply(
        Index2D index,
        ArrayView2D<float, Stride2D.DenseX> input,
        ArrayView2D<float, Stride2D.DenseX> output,
        float scalar)
    {
        var row = index.Y;
        var col = index.X;
        
        if (row < input.IntExtent.Y && col < input.IntExtent.X)
        {
            output[row, col] = input[row, col] * scalar;
        }
    }

    /// <summary>
    /// Element-wise matrix multiplication kernel: C = A .* B
    /// </summary>
    public static void MatrixElementwiseMultiply(
        Index2D index,
        ArrayView2D<float, Stride2D.DenseX> a,
        ArrayView2D<float, Stride2D.DenseX> b,
        ArrayView2D<float, Stride2D.DenseX> c)
    {
        var row = index.Y;
        var col = index.X;
        
        if (row < a.IntExtent.Y && col < a.IntExtent.X)
        {
            c[row, col] = a[row, col] * b[row, col];
        }
    }

    /// <summary>
    /// Matrix-vector multiplication kernel: y = A * x
    /// </summary>
    public static void MatrixVectorMultiply(
        Index1D index,
        ArrayView2D<float, Stride2D.DenseX> matrix,
        ArrayView<float> vector,
        ArrayView<float> result)
    {
        var row = index;
        
        if (row >= matrix.IntExtent.Y)
            return;
        
        float sum = 0.0f;
        for (int col = 0; col < matrix.IntExtent.X; col++)
        {
            sum += matrix[row, col] * vector[col];
        }
        
        result[row] = sum;
    }

    /// <summary>
    /// Outer product kernel: C = x * y^T
    /// </summary>
    public static void OuterProduct(
        Index2D index,
        ArrayView<float> x,
        ArrayView<float> y,
        ArrayView2D<float, Stride2D.DenseX> result)
    {
        var row = index.Y;
        var col = index.X;
        
        if (row < x.Length && col < y.Length)
        {
            result[row, col] = x[row] * y[col];
        }
    }

    /// <summary>
    /// Row sum reduction kernel: rowSums[i] = sum(A[i, :])
    /// </summary>
    public static void RowSum(
        Index1D index,
        ArrayView2D<float, Stride2D.DenseX> matrix,
        ArrayView<float> rowSums)
    {
        var row = index;
        
        if (row >= matrix.IntExtent.Y)
            return;
        
        float sum = 0.0f;
        for (int col = 0; col < matrix.IntExtent.X; col++)
        {
            sum += matrix[row, col];
        }
        
        rowSums[row] = sum;
    }

    /// <summary>
    /// Column sum reduction kernel: colSums[j] = sum(A[:, j])
    /// </summary>
    public static void ColumnSum(
        Index1D index,
        ArrayView2D<float, Stride2D.DenseX> matrix,
        ArrayView<float> colSums)
    {
        var col = index;
        
        if (col >= matrix.IntExtent.X)
            return;
        
        float sum = 0.0f;
        for (int row = 0; row < matrix.IntExtent.Y; row++)
        {
            sum += matrix[row, col];
        }
        
        colSums[col] = sum;
    }

    /// <summary>
    /// Diagonal extraction kernel: diag = diagonal(A)
    /// </summary>
    public static void ExtractDiagonal(
        Index1D index,
        ArrayView2D<float, Stride2D.DenseX> matrix,
        ArrayView<float> diagonal)
    {
        var idx = index;
        
        if (idx < matrix.IntExtent.Y && idx < matrix.IntExtent.X)
        {
            diagonal[idx] = matrix[idx, idx];
        }
    }

    /// <summary>
    /// Set diagonal kernel: Sets diagonal elements of a matrix
    /// </summary>
    public static void SetDiagonal(
        Index1D index,
        ArrayView2D<float, Stride2D.DenseX> matrix,
        ArrayView<float> diagonal)
    {
        var idx = index;
        
        if (idx < matrix.IntExtent.Y && idx < matrix.IntExtent.X)
        {
            matrix[idx, idx] = diagonal[idx];
        }
    }

    /// <summary>
    /// Matrix normalization kernel: Normalizes each row to unit length
    /// </summary>
    public static void NormalizeRows(
        Index1D index,
        ArrayView2D<float, Stride2D.DenseX> matrix)
    {
        var row = index;
        
        if (row >= matrix.IntExtent.Y)
            return;
        
        // Compute row norm
        float norm = 0.0f;
        for (int col = 0; col < matrix.IntExtent.X; col++)
        {
            var value = matrix[row, col];
            norm += value * value;
        }
        norm = XMath.Sqrt(norm);
        
        // Normalize row
        if (norm > 0.0f)
        {
            for (int col = 0; col < matrix.IntExtent.X; col++)
            {
                matrix[row, col] /= norm;
            }
        }
    }

    /// <summary>
    /// Apply activation function to matrix (ReLU)
    /// </summary>
    public static void ApplyReLU(
        Index2D index,
        ArrayView2D<float, Stride2D.DenseX> input,
        ArrayView2D<float, Stride2D.DenseX> output)
    {
        var row = index.Y;
        var col = index.X;
        
        if (row < input.IntExtent.Y && col < input.IntExtent.X)
        {
            output[row, col] = XMath.Max(0.0f, input[row, col]);
        }
    }

    /// <summary>
    /// Apply activation function to matrix (Sigmoid)
    /// </summary>
    public static void ApplySigmoid(
        Index2D index,
        ArrayView2D<float, Stride2D.DenseX> input,
        ArrayView2D<float, Stride2D.DenseX> output)
    {
        var row = index.Y;
        var col = index.X;
        
        if (row < input.IntExtent.Y && col < input.IntExtent.X)
        {
            var value = input[row, col];
            output[row, col] = 1.0f / (1.0f + XMath.Exp(-value));
        }
    }

    /// <summary>
    /// Apply activation function to matrix (Tanh)
    /// </summary>
    public static void ApplyTanh(
        Index2D index,
        ArrayView2D<float, Stride2D.DenseX> input,
        ArrayView2D<float, Stride2D.DenseX> output)
    {
        var row = index.Y;
        var col = index.X;
        
        if (row < input.IntExtent.Y && col < input.IntExtent.X)
        {
            output[row, col] = XMath.Tanh(input[row, col]);
        }
    }
}
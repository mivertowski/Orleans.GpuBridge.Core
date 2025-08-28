// Copyright (c) 2025 Michael Ivertowski. All Rights Reserved.
// Licensed under the Apache License, Version 2.0.

using System;
using ILGPU;
using ILGPU.Runtime;

namespace Orleans.GpuBridge.Samples.ILGPU.Kernels;

/// <summary>
/// Sample ILGPU kernels for vector operations
/// </summary>
public static class VectorOperations
{
    /// <summary>
    /// Adds two vectors element-wise: c[i] = a[i] + b[i]
    /// </summary>
    public static void VectorAdd(
        Index1D index,
        ArrayView<float> a,
        ArrayView<float> b,
        ArrayView<float> result)
    {
        result[index] = a[index] + b[index];
    }
    
    /// <summary>
    /// Multiplies vector by scalar: result[i] = a[i] * scalar
    /// </summary>
    public static void VectorScalarMultiply(
        Index1D index,
        ArrayView<float> a,
        float scalar,
        ArrayView<float> result)
    {
        result[index] = a[index] * scalar;
    }
    
    /// <summary>
    /// Computes dot product of two vectors using shared memory reduction
    /// </summary>
    public static void VectorDotProduct(
        Index1D index,
        ArrayView<float> a,
        ArrayView<float> b,
        ArrayView<float> partialSums)
    {
        var groupId = Grid.IdxX;
        var groupSize = Group.DimX;
        var localId = Group.IdxX;
        
        // Allocate shared memory
        var sharedMemory = SharedMemory.Allocate<float>(groupSize);
        
        // Compute local product
        var localSum = 0.0f;
        for (var i = index.X; i < a.Length; i += Grid.DimX)
        {
            localSum += a[i] * b[i];
        }
        
        // Store in shared memory
        sharedMemory[localId] = localSum;
        Group.Barrier();
        
        // Reduction in shared memory
        for (var stride = groupSize / 2; stride > 0; stride >>= 1)
        {
            if (localId < stride)
            {
                sharedMemory[localId] += sharedMemory[localId + stride];
            }
            Group.Barrier();
        }
        
        // Write result
        if (localId == 0)
        {
            partialSums[groupId] = sharedMemory[0];
        }
    }
    
    /// <summary>
    /// Normalizes a vector: result[i] = a[i] / magnitude
    /// </summary>
    public static void VectorNormalize(
        Index1D index,
        ArrayView<float> a,
        ArrayView<float> result,
        ArrayView<float> magnitude)
    {
        // First pass would compute magnitude, then normalize
        // This simplified version assumes magnitude is precomputed
        var mag = magnitude[0];
        if (mag > 0)
        {
            result[index] = a[index] / mag;
        }
        else
        {
            result[index] = 0;
        }
    }
    
    /// <summary>
    /// Computes element-wise maximum: result[i] = max(a[i], b[i])
    /// </summary>
    public static void VectorMaximum(
        Index1D index,
        ArrayView<float> a,
        ArrayView<float> b,
        ArrayView<float> result)
    {
        result[index] = IntrinsicMath.Max(a[index], b[index]);
    }
    
    /// <summary>
    /// Applies a function to each element: result[i] = sin(a[i])
    /// </summary>
    public static void VectorSine(
        Index1D index,
        ArrayView<float> a,
        ArrayView<float> result)
    {
        result[index] = IntrinsicMath.Sin(a[index]);
    }
    
    /// <summary>
    /// Computes cumulative sum (prefix sum) - simplified version
    /// </summary>
    public static void VectorPrefixSum(
        Index1D index,
        ArrayView<float> input,
        ArrayView<float> output,
        int stride)
    {
        if (index >= stride && index < input.Length)
        {
            output[index] = input[index] + input[index - stride];
        }
        else if (index < input.Length)
        {
            output[index] = input[index];
        }
    }
    
    /// <summary>
    /// Filters vector elements based on a threshold
    /// </summary>
    public static void VectorThreshold(
        Index1D index,
        ArrayView<float> input,
        ArrayView<float> output,
        float threshold,
        float replaceValue)
    {
        output[index] = input[index] > threshold ? input[index] : replaceValue;
    }
}
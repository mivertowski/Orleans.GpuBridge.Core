using System;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Algorithms;

namespace Orleans.GpuBridge.Backends.ILGPU.Kernels.Templates;

/// <summary>
/// ILGPU kernel templates for reduction operations with actual GPU execution
/// </summary>
public static class ReductionKernels
{
    /// <summary>
    /// Sum reduction kernel - computes sum of all elements
    /// </summary>
    public static void SumReduction(
        Index1D index,
        ArrayView<float> input,
        ArrayView<float> output,
        SpecializedValue<int> groupSize)
    {
        var sharedMemory = SharedMemory.Allocate<float>(groupSize);
        var localIndex = Group.IdxX;
        var globalIndex = Grid.GlobalIndex.X;
        
        // Load data into shared memory
        sharedMemory[localIndex] = globalIndex < input.Length ? input[globalIndex] : 0.0f;
        Group.Barrier();
        
        // Perform reduction in shared memory
        for (int stride = groupSize / 2; stride > 0; stride >>= 1)
        {
            if (localIndex < stride)
            {
                sharedMemory[localIndex] += sharedMemory[localIndex + stride];
            }
            Group.Barrier();
        }
        
        // Write result for this block
        if (localIndex == 0)
        {
            Atomic.Add(ref output[0], sharedMemory[0]);
        }
    }

    /// <summary>
    /// Maximum reduction kernel - finds maximum element
    /// </summary>
    public static void MaxReduction(
        Index1D index,
        ArrayView<float> input,
        ArrayView<float> output,
        SpecializedValue<int> groupSize)
    {
        var sharedMemory = SharedMemory.Allocate<float>(groupSize);
        var localIndex = Group.IdxX;
        var globalIndex = Grid.GlobalIndex.X;
        
        // Load data into shared memory
        sharedMemory[localIndex] = globalIndex < input.Length ? input[globalIndex] : float.MinValue;
        Group.Barrier();
        
        // Perform reduction in shared memory
        for (int stride = groupSize / 2; stride > 0; stride >>= 1)
        {
            if (localIndex < stride)
            {
                sharedMemory[localIndex] = XMath.Max(sharedMemory[localIndex], sharedMemory[localIndex + stride]);
            }
            Group.Barrier();
        }
        
        // Write result for this block
        if (localIndex == 0)
        {
            Atomic.Max(ref output[0], sharedMemory[0]);
        }
    }

    /// <summary>
    /// Minimum reduction kernel - finds minimum element
    /// </summary>
    public static void MinReduction(
        Index1D index,
        ArrayView<float> input,
        ArrayView<float> output,
        SpecializedValue<int> groupSize)
    {
        var sharedMemory = SharedMemory.Allocate<float>(groupSize);
        var localIndex = Group.IdxX;
        var globalIndex = Grid.GlobalIndex.X;
        
        // Load data into shared memory
        sharedMemory[localIndex] = globalIndex < input.Length ? input[globalIndex] : float.MaxValue;
        Group.Barrier();
        
        // Perform reduction in shared memory
        for (int stride = groupSize / 2; stride > 0; stride >>= 1)
        {
            if (localIndex < stride)
            {
                sharedMemory[localIndex] = XMath.Min(sharedMemory[localIndex], sharedMemory[localIndex + stride]);
            }
            Group.Barrier();
        }
        
        // Write result for this block
        if (localIndex == 0)
        {
            Atomic.Min(ref output[0], sharedMemory[0]);
        }
    }

    /// <summary>
    /// Product reduction kernel - computes product of all elements
    /// </summary>
    public static void ProductReduction(
        Index1D index,
        ArrayView<float> input,
        ArrayView<float> output,
        SpecializedValue<int> groupSize)
    {
        var sharedMemory = SharedMemory.Allocate<float>(groupSize);
        var localIndex = Group.IdxX;
        var globalIndex = Grid.GlobalIndex.X;
        
        // Load data into shared memory
        sharedMemory[localIndex] = globalIndex < input.Length ? input[globalIndex] : 1.0f;
        Group.Barrier();
        
        // Perform reduction in shared memory
        for (int stride = groupSize / 2; stride > 0; stride >>= 1)
        {
            if (localIndex < stride)
            {
                sharedMemory[localIndex] *= sharedMemory[localIndex + stride];
            }
            Group.Barrier();
        }
        
        // Write result for this block (using atomic CAS for multiplication)
        if (localIndex == 0)
        {
            float oldValue, newValue;
            do
            {
                oldValue = output[0];
                newValue = oldValue * sharedMemory[0];
            } while (Atomic.CompareExchange(ref output[0], newValue, oldValue) != oldValue);
        }
    }

    /// <summary>
    /// Count non-zero elements reduction
    /// </summary>
    public static void CountNonZero(
        Index1D index,
        ArrayView<float> input,
        ArrayView<int> output,
        SpecializedValue<int> groupSize)
    {
        var sharedMemory = SharedMemory.Allocate<int>(groupSize);
        var localIndex = Group.IdxX;
        var globalIndex = Grid.GlobalIndex.X;
        
        // Count non-zero elements
        sharedMemory[localIndex] = (globalIndex < input.Length && input[globalIndex] != 0.0f) ? 1 : 0;
        Group.Barrier();
        
        // Perform reduction in shared memory
        for (int stride = groupSize / 2; stride > 0; stride >>= 1)
        {
            if (localIndex < stride)
            {
                sharedMemory[localIndex] += sharedMemory[localIndex + stride];
            }
            Group.Barrier();
        }
        
        // Write result for this block
        if (localIndex == 0)
        {
            Atomic.Add(ref output[0], sharedMemory[0]);
        }
    }

    /// <summary>
    /// Average reduction kernel - computes mean of all elements
    /// </summary>
    public static void AverageReduction(
        Index1D index,
        ArrayView<float> input,
        ArrayView<float> sumOutput,
        ArrayView<int> countOutput,
        SpecializedValue<int> groupSize)
    {
        var sharedSum = SharedMemory.Allocate<float>(groupSize);
        var sharedCount = SharedMemory.Allocate<int>(groupSize);
        var localIndex = Group.IdxX;
        var globalIndex = Grid.GlobalIndex.X;
        
        // Load data into shared memory
        if (globalIndex < input.Length)
        {
            sharedSum[localIndex] = input[globalIndex];
            sharedCount[localIndex] = 1;
        }
        else
        {
            sharedSum[localIndex] = 0.0f;
            sharedCount[localIndex] = 0;
        }
        Group.Barrier();
        
        // Perform reduction in shared memory
        for (int stride = groupSize / 2; stride > 0; stride >>= 1)
        {
            if (localIndex < stride)
            {
                sharedSum[localIndex] += sharedSum[localIndex + stride];
                sharedCount[localIndex] += sharedCount[localIndex + stride];
            }
            Group.Barrier();
        }
        
        // Write result for this block
        if (localIndex == 0)
        {
            Atomic.Add(ref sumOutput[0], sharedSum[0]);
            Atomic.Add(ref countOutput[0], sharedCount[0]);
        }
    }

    /// <summary>
    /// Standard deviation reduction kernel (first pass - compute mean)
    /// </summary>
    public static void StdDevPass1(
        Index1D index,
        ArrayView<float> input,
        ArrayView<float> sumOutput,
        ArrayView<int> countOutput,
        SpecializedValue<int> groupSize)
    {
        AverageReduction(index, input, sumOutput, countOutput, groupSize);
    }

    /// <summary>
    /// Standard deviation reduction kernel (second pass - compute variance)
    /// </summary>
    public static void StdDevPass2(
        Index1D index,
        ArrayView<float> input,
        ArrayView<float> varianceOutput,
        float mean,
        SpecializedValue<int> groupSize)
    {
        var sharedMemory = SharedMemory.Allocate<float>(groupSize);
        var localIndex = Group.IdxX;
        var globalIndex = Grid.GlobalIndex.X;
        
        // Compute squared differences
        if (globalIndex < input.Length)
        {
            var diff = input[globalIndex] - mean;
            sharedMemory[localIndex] = diff * diff;
        }
        else
        {
            sharedMemory[localIndex] = 0.0f;
        }
        Group.Barrier();
        
        // Perform reduction in shared memory
        for (int stride = groupSize / 2; stride > 0; stride >>= 1)
        {
            if (localIndex < stride)
            {
                sharedMemory[localIndex] += sharedMemory[localIndex + stride];
            }
            Group.Barrier();
        }
        
        // Write result for this block
        if (localIndex == 0)
        {
            Atomic.Add(ref varianceOutput[0], sharedMemory[0]);
        }
    }

    /// <summary>
    /// ArgMax reduction - finds index of maximum element
    /// </summary>
    public static void ArgMaxReduction(
        Index1D index,
        ArrayView<float> input,
        ArrayView<float> maxValue,
        ArrayView<int> maxIndex,
        SpecializedValue<int> groupSize)
    {
        var sharedValues = SharedMemory.Allocate<float>(groupSize);
        var sharedIndices = SharedMemory.Allocate<int>(groupSize);
        var localIndex = Group.IdxX;
        var globalIndex = Grid.GlobalIndex.X;
        
        // Load data into shared memory
        if (globalIndex < input.Length)
        {
            sharedValues[localIndex] = input[globalIndex];
            sharedIndices[localIndex] = globalIndex;
        }
        else
        {
            sharedValues[localIndex] = float.MinValue;
            sharedIndices[localIndex] = -1;
        }
        Group.Barrier();
        
        // Perform reduction in shared memory
        for (int stride = groupSize / 2; stride > 0; stride >>= 1)
        {
            if (localIndex < stride)
            {
                if (sharedValues[localIndex + stride] > sharedValues[localIndex])
                {
                    sharedValues[localIndex] = sharedValues[localIndex + stride];
                    sharedIndices[localIndex] = sharedIndices[localIndex + stride];
                }
            }
            Group.Barrier();
        }
        
        // Write result for this block
        if (localIndex == 0)
        {
            // Atomically update global maximum
            float oldMax, newMax;
            int oldIdx, newIdx;
            do
            {
                oldMax = maxValue[0];
                oldIdx = maxIndex[0];
                
                if (sharedValues[0] > oldMax)
                {
                    newMax = sharedValues[0];
                    newIdx = sharedIndices[0];
                }
                else
                {
                    newMax = oldMax;
                    newIdx = oldIdx;
                }
            } while (Atomic.CompareExchange(ref maxValue[0], newMax, oldMax) != oldMax ||
                     Atomic.CompareExchange(ref maxIndex[0], newIdx, oldIdx) != oldIdx);
        }
    }

    /// <summary>
    /// Histogram computation kernel
    /// </summary>
    public static void Histogram(
        Index1D index,
        ArrayView<float> input,
        ArrayView<int> histogram,
        float minValue,
        float maxValue,
        int numBins)
    {
        if (index >= input.Length)
            return;
        
        var value = input[index];
        
        // Compute bin index
        if (value >= minValue && value <= maxValue)
        {
            var normalizedValue = (value - minValue) / (maxValue - minValue);
            var binIndex = (int)(normalizedValue * (numBins - 1));
            binIndex = XMath.Clamp(binIndex, 0, numBins - 1);
            
            // Atomic increment
            Atomic.Add(ref histogram[binIndex], 1);
        }
    }
}
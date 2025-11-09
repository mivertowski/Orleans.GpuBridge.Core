using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Runtime.Intrinsics.Arm;
using System.Threading.Tasks;

namespace Orleans.GpuBridge.Performance;

/// <summary>
/// SIMD-optimized operation implementations for VectorizedKernelExecutor
/// </summary>
public sealed partial class VectorizedKernelExecutor
{
    #region Vector Addition Operations

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private unsafe void ExecuteVectorAdd(float[] a, float[] b, float[] result)
    {
        var length = a.Length;

        if (_hasAvx512 && length >= Vector512<float>.Count)
        {
            ExecuteVectorAddAvx512(a, b, result);
        }
        else if (_hasAvx2 && length >= Vector256<float>.Count)
        {
            ExecuteVectorAddAvx2(a, b, result);
        }
        else if (_hasNeon && length >= Vector128<float>.Count)
        {
            ExecuteVectorAddNeon(a, b, result);
        }
        else
        {
            ExecuteVectorAddGeneric(a, b, result);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private unsafe void ExecuteVectorAddAvx512(float[] a, float[] b, float[] result)
    {
        var vectorSize = Vector512<float>.Count;
        var vectorCount = a.Length / vectorSize;

        fixed (float* pA = a, pB = b, pResult = result)
        {
            for (int i = 0; i < vectorCount; i++)
            {
                var offset = i * vectorSize;
                var vecA = Avx512F.LoadVector512(pA + offset);
                var vecB = Avx512F.LoadVector512(pB + offset);
                var vecResult = Avx512F.Add(vecA, vecB);
                Avx512F.Store(pResult + offset, vecResult);
            }

            // Handle remainder
            var remaining = a.Length % vectorSize;
            if (remaining > 0)
            {
                var startIdx = vectorCount * vectorSize;
                for (int i = 0; i < remaining; i++)
                {
                    result[startIdx + i] = a[startIdx + i] + b[startIdx + i];
                }
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private unsafe void ExecuteVectorAddAvx2(float[] a, float[] b, float[] result)
    {
        var vectorSize = Vector256<float>.Count;
        var vectorCount = a.Length / vectorSize;

        fixed (float* pA = a, pB = b, pResult = result)
        {
            for (int i = 0; i < vectorCount; i++)
            {
                var offset = i * vectorSize;
                var vecA = Avx.LoadVector256(pA + offset);
                var vecB = Avx.LoadVector256(pB + offset);
                var vecResult = Avx.Add(vecA, vecB);
                Avx.Store(pResult + offset, vecResult);
            }
        }

        // Handle remainder with scalar operations
        HandleVectorRemainder(a, b, result, vectorCount * vectorSize);
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private unsafe void ExecuteVectorAddNeon(float[] a, float[] b, float[] result)
    {
        var vectorSize = Vector128<float>.Count;
        var vectorCount = a.Length / vectorSize;

        fixed (float* pA = a, pB = b, pResult = result)
        {
            for (int i = 0; i < vectorCount; i++)
            {
                var offset = i * vectorSize;
                var vecA = AdvSimd.LoadVector128(pA + offset);
                var vecB = AdvSimd.LoadVector128(pB + offset);
                var vecResult = AdvSimd.Add(vecA, vecB);
                AdvSimd.Store(pResult + offset, vecResult);
            }
        }

        HandleVectorRemainder(a, b, result, vectorCount * vectorSize);
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private void ExecuteVectorAddGeneric(float[] a, float[] b, float[] result)
    {
        var vectorSize = Vector<float>.Count;
        var vectorCount = a.Length / vectorSize;

        for (int i = 0; i < vectorCount; i++)
        {
            var offset = i * vectorSize;
            var vecA = new Vector<float>(a, offset);
            var vecB = new Vector<float>(b, offset);
            var vecResult = vecA + vecB;
            vecResult.CopyTo(result, offset);
        }

        HandleVectorRemainder(a, b, result, vectorCount * vectorSize);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void HandleVectorRemainder(float[] a, float[] b, float[] result, int startIndex)
    {
        for (int i = startIndex; i < a.Length; i++)
        {
            result[i] = a[i] + b[i];
        }
    }

    #endregion

    #region Fused Multiply-Add Operations

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private unsafe void ExecuteFusedMultiplyAdd(float[] a, float[] b, float[] c, float[] result)
    {
        var length = a.Length;

        if (_hasFma && _hasAvx2 && length >= Vector256<float>.Count)
        {
            ExecuteFmaAvx2(a, b, c, result);
        }
        else if (_hasNeon && length >= Vector128<float>.Count)
        {
            ExecuteFmaNeon(a, b, c, result);
        }
        else
        {
            ExecuteFmaGeneric(a, b, c, result);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private unsafe void ExecuteFmaAvx2(float[] a, float[] b, float[] c, float[] result)
    {
        var vectorSize = Vector256<float>.Count;
        var vectorCount = a.Length / vectorSize;

        fixed (float* pA = a, pB = b, pC = c, pResult = result)
        {
            for (int i = 0; i < vectorCount; i++)
            {
                var offset = i * vectorSize;
                var vecA = Avx.LoadVector256(pA + offset);
                var vecB = Avx.LoadVector256(pB + offset);
                var vecC = Avx.LoadVector256(pC + offset);
                var vecResult = Fma.MultiplyAdd(vecA, vecB, vecC);
                Avx.Store(pResult + offset, vecResult);
            }
        }

        // Handle remainder
        var startIdx = vectorCount * vectorSize;
        for (int i = startIdx; i < a.Length; i++)
        {
            result[i] = a[i] * b[i] + c[i];
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private unsafe void ExecuteFmaNeon(float[] a, float[] b, float[] c, float[] result)
    {
        var vectorSize = Vector128<float>.Count;
        var vectorCount = a.Length / vectorSize;

        fixed (float* pA = a, pB = b, pC = c, pResult = result)
        {
            for (int i = 0; i < vectorCount; i++)
            {
                var offset = i * vectorSize;
                var vecA = AdvSimd.LoadVector128(pA + offset);
                var vecB = AdvSimd.LoadVector128(pB + offset);
                var vecC = AdvSimd.LoadVector128(pC + offset);
                var vecResult = AdvSimd.FusedMultiplyAdd(vecC, vecA, vecB);
                AdvSimd.Store(pResult + offset, vecResult);
            }
        }

        // Handle remainder
        var startIdx = vectorCount * vectorSize;
        for (int i = startIdx; i < a.Length; i++)
        {
            result[i] = a[i] * b[i] + c[i];
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private void ExecuteFmaGeneric(float[] a, float[] b, float[] c, float[] result)
    {
        for (int i = 0; i < a.Length; i++)
        {
            result[i] = Math.FusedMultiplyAdd(a[i], b[i], c[i]);
        }
    }

    #endregion

    #region Matrix Multiplication Operations

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private void ExecuteMatrixMultiply(float[] a, float[] b, float[] result, int m, int n, int k)
    {
        // Cache-optimal blocked matrix multiplication
        const int blockSize = 64; // Optimized for L1 cache

        Parallel.For(0, (m + blockSize - 1) / blockSize, i =>
        {
            var iStart = i * blockSize;
            var iEnd = Math.Min(iStart + blockSize, m);

            for (int j = 0; j < n; j += blockSize)
            {
                var jEnd = Math.Min(j + blockSize, n);

                for (int kBlock = 0; kBlock < k; kBlock += blockSize)
                {
                    var kEnd = Math.Min(kBlock + blockSize, k);

                    ExecuteMatrixBlock(a, b, result, iStart, iEnd, j, jEnd, kBlock, kEnd, m, n, k);
                }
            }
        });
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private void ExecuteMatrixBlock(float[] a, float[] b, float[] result,
        int iStart, int iEnd, int jStart, int jEnd, int kStart, int kEnd,
        int m, int n, int k)
    {
        for (int i = iStart; i < iEnd; i++)
        {
            for (int j = jStart; j < jEnd; j++)
            {
                var sum = 0.0f;
                var resultIndex = i * n + j;

                // Vectorized inner loop
                var kVectorized = (kEnd - kStart) / Vector<float>.Count * Vector<float>.Count;
                var sumVector = Vector<float>.Zero;

                int kIdx = kStart;
                for (; kIdx < kStart + kVectorized; kIdx += Vector<float>.Count)
                {
                    var aVector = new Vector<float>(a, i * k + kIdx);
                    var bVector = LoadBVector(b, kIdx, j, n, Vector<float>.Count);
                    sumVector += aVector * bVector;
                }

                // Sum vector components
                for (int v = 0; v < Vector<float>.Count; v++)
                {
                    sum += sumVector[v];
                }

                // Handle remaining elements
                for (; kIdx < kEnd; kIdx++)
                {
                    sum += a[i * k + kIdx] * b[kIdx * n + j];
                }

                result[resultIndex] += sum;
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private Vector<float> LoadBVector(float[] b, int kStart, int j, int n, int count)
    {
        // Load non-contiguous B matrix elements into vector
        var values = new float[count];
        for (int i = 0; i < count; i++)
        {
            values[i] = b[(kStart + i) * n + j];
        }
        return new Vector<float>(values);
    }

    #endregion

    #region Reduction Operations

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private void ExecuteReduction(float[] input, float[] result, ReductionOperation operation)
    {
        var length = input.Length;
        if (length == 0)
        {
            result[0] = 0;
            return;
        }

        switch (operation)
        {
            case ReductionOperation.Sum:
                result[0] = VectorizedSum(input);
                break;
            case ReductionOperation.Max:
                result[0] = VectorizedMax(input);
                break;
            case ReductionOperation.Min:
                result[0] = VectorizedMin(input);
                break;
            default:
                throw new ArgumentException($"Unsupported reduction operation: {operation}");
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private float VectorizedSum(float[] input)
    {
        var vectorSize = Vector<float>.Count;
        var vectorCount = input.Length / vectorSize;
        var sumVector = Vector<float>.Zero;

        // Vectorized sum
        for (int i = 0; i < vectorCount; i++)
        {
            var vec = new Vector<float>(input, i * vectorSize);
            sumVector += vec;
        }

        // Sum vector components
        float sum = 0;
        for (int i = 0; i < vectorSize; i++)
        {
            sum += sumVector[i];
        }

        // Add remainder
        for (int i = vectorCount * vectorSize; i < input.Length; i++)
        {
            sum += input[i];
        }

        return sum;
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private float VectorizedMax(float[] input)
    {
        var vectorSize = Vector<float>.Count;
        var vectorCount = input.Length / vectorSize;
        var maxVector = new Vector<float>(float.MinValue);

        // Vectorized max
        for (int i = 0; i < vectorCount; i++)
        {
            var vec = new Vector<float>(input, i * vectorSize);
            maxVector = Vector.Max(maxVector, vec);
        }

        // Find max in vector
        float max = float.MinValue;
        for (int i = 0; i < vectorSize; i++)
        {
            max = Math.Max(max, maxVector[i]);
        }

        // Check remainder
        for (int i = vectorCount * vectorSize; i < input.Length; i++)
        {
            max = Math.Max(max, input[i]);
        }

        return max;
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private float VectorizedMin(float[] input)
    {
        var vectorSize = Vector<float>.Count;
        var vectorCount = input.Length / vectorSize;
        var minVector = new Vector<float>(float.MaxValue);

        // Vectorized min
        for (int i = 0; i < vectorCount; i++)
        {
            var vec = new Vector<float>(input, i * vectorSize);
            minVector = Vector.Min(minVector, vec);
        }

        // Find min in vector
        float min = float.MaxValue;
        for (int i = 0; i < vectorSize; i++)
        {
            min = Math.Min(min, minVector[i]);
        }

        // Check remainder
        for (int i = vectorCount * vectorSize; i < input.Length; i++)
        {
            min = Math.Min(min, input[i]);
        }

        return min;
    }

    #endregion
}

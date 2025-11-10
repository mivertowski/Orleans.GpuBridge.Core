using System;
using System.Collections.Generic;
using System.Linq;

namespace Orleans.GpuBridge.Tests.RC2.TestingFramework;

/// <summary>
/// Test data generation utilities for GPU Bridge testing.
/// Provides builders for common test data patterns including arrays, edge cases,
/// and random data generation with reproducible seeds.
/// </summary>
/// <remarks>
/// All builders support fluent API patterns and produce deterministic results
/// when seed values are specified. This ensures test reproducibility across runs.
/// </remarks>
public static class TestDataBuilders
{
    /// <summary>
    /// Default random seed for reproducible test data generation.
    /// </summary>
    public const int DefaultSeed = 42;

    /// <summary>
    /// Creates a builder for float array test data.
    /// </summary>
    /// <param name="size">Initial size of the array.</param>
    /// <returns>A new float array builder.</returns>
    public static FloatArrayBuilder FloatArray(int size) => new(size);

    /// <summary>
    /// Creates a builder for int array test data.
    /// </summary>
    /// <param name="size">Initial size of the array.</param>
    /// <returns>A new int array builder.</returns>
    public static IntArrayBuilder IntArray(int size) => new(size);

    /// <summary>
    /// Creates a builder for double array test data.
    /// </summary>
    /// <param name="size">Initial size of the array.</param>
    /// <returns>A new double array builder.</returns>
    public static DoubleArrayBuilder DoubleArray(int size) => new(size);

    /// <summary>
    /// Creates a builder for generic array test data.
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="size">Initial size of the array.</param>
    /// <param name="generator">Function to generate individual elements.</param>
    /// <returns>A new generic array builder.</returns>
    public static GenericArrayBuilder<T> Array<T>(int size, Func<int, T> generator) => new(size, generator);

    /// <summary>
    /// Gets common edge case sizes for array testing.
    /// </summary>
    public static class CommonSizes
    {
        /// <summary>Empty array</summary>
        public const int Empty = 0;

        /// <summary>Single element</summary>
        public const int Single = 1;

        /// <summary>Small array (16 elements)</summary>
        public const int Small = 16;

        /// <summary>Medium array (1024 elements)</summary>
        public const int Medium = 1024;

        /// <summary>Large array (1 million elements)</summary>
        public const int Large = 1_000_000;

        /// <summary>Huge array (10 million elements)</summary>
        public const int Huge = 10_000_000;

        /// <summary>Power of 2: 256 elements</summary>
        public const int PowerOfTwo256 = 256;

        /// <summary>Power of 2: 1024 elements</summary>
        public const int PowerOfTwo1024 = 1024;

        /// <summary>Power of 2: 4096 elements</summary>
        public const int PowerOfTwo4096 = 4096;
    }

    /// <summary>
    /// Builder for float array test data.
    /// </summary>
    public sealed class FloatArrayBuilder
    {
        private readonly int _size;
        private Func<int, float>? _generator;
        private int _seed = DefaultSeed;

        internal FloatArrayBuilder(int size)
        {
            _size = size;
        }

        /// <summary>
        /// Generates random float values between 0 and 1.
        /// </summary>
        public FloatArrayBuilder WithRandomValues()
        {
            _generator = _ => (float)new Random(_seed).NextDouble();
            return this;
        }

        /// <summary>
        /// Generates sequential float values starting from 0.
        /// </summary>
        public FloatArrayBuilder WithSequentialValues()
        {
            _generator = i => i;
            return this;
        }

        /// <summary>
        /// Generates constant float value for all elements.
        /// </summary>
        public FloatArrayBuilder WithConstantValue(float value)
        {
            _generator = _ => value;
            return this;
        }

        /// <summary>
        /// Generates random float values within specified range.
        /// </summary>
        public FloatArrayBuilder WithRandomRange(float min, float max)
        {
            _generator = i =>
            {
                var random = new Random(_seed + i);
                return min + (float)random.NextDouble() * (max - min);
            };
            return this;
        }

        /// <summary>
        /// Uses custom generator function.
        /// </summary>
        public FloatArrayBuilder WithGenerator(Func<int, float> generator)
        {
            _generator = generator;
            return this;
        }

        /// <summary>
        /// Sets the random seed for reproducible generation.
        /// </summary>
        public FloatArrayBuilder WithSeed(int seed)
        {
            _seed = seed;
            return this;
        }

        /// <summary>
        /// Builds the float array.
        /// </summary>
        public float[] Build()
        {
            var generator = _generator ?? (_ => 0f);
            return Enumerable.Range(0, _size).Select(generator).ToArray();
        }
    }

    /// <summary>
    /// Builder for int array test data.
    /// </summary>
    public sealed class IntArrayBuilder
    {
        private readonly int _size;
        private Func<int, int>? _generator;
        private int _seed = DefaultSeed;

        internal IntArrayBuilder(int size)
        {
            _size = size;
        }

        /// <summary>
        /// Generates random int values.
        /// </summary>
        public IntArrayBuilder WithRandomValues()
        {
            _generator = i => new Random(_seed + i).Next();
            return this;
        }

        /// <summary>
        /// Generates sequential int values starting from 0.
        /// </summary>
        public IntArrayBuilder WithSequentialValues()
        {
            _generator = i => i;
            return this;
        }

        /// <summary>
        /// Generates constant int value for all elements.
        /// </summary>
        public IntArrayBuilder WithConstantValue(int value)
        {
            _generator = _ => value;
            return this;
        }

        /// <summary>
        /// Generates random int values within specified range.
        /// </summary>
        public IntArrayBuilder WithRandomRange(int min, int max)
        {
            _generator = i => new Random(_seed + i).Next(min, max);
            return this;
        }

        /// <summary>
        /// Uses custom generator function.
        /// </summary>
        public IntArrayBuilder WithGenerator(Func<int, int> generator)
        {
            _generator = generator;
            return this;
        }

        /// <summary>
        /// Sets the random seed for reproducible generation.
        /// </summary>
        public IntArrayBuilder WithSeed(int seed)
        {
            _seed = seed;
            return this;
        }

        /// <summary>
        /// Builds the int array.
        /// </summary>
        public int[] Build()
        {
            var generator = _generator ?? (_ => 0);
            return Enumerable.Range(0, _size).Select(generator).ToArray();
        }
    }

    /// <summary>
    /// Builder for double array test data.
    /// </summary>
    public sealed class DoubleArrayBuilder
    {
        private readonly int _size;
        private Func<int, double>? _generator;
        private int _seed = DefaultSeed;

        internal DoubleArrayBuilder(int size)
        {
            _size = size;
        }

        /// <summary>
        /// Generates random double values between 0 and 1.
        /// </summary>
        public DoubleArrayBuilder WithRandomValues()
        {
            _generator = i => new Random(_seed + i).NextDouble();
            return this;
        }

        /// <summary>
        /// Generates sequential double values starting from 0.
        /// </summary>
        public DoubleArrayBuilder WithSequentialValues()
        {
            _generator = i => i;
            return this;
        }

        /// <summary>
        /// Generates constant double value for all elements.
        /// </summary>
        public DoubleArrayBuilder WithConstantValue(double value)
        {
            _generator = _ => value;
            return this;
        }

        /// <summary>
        /// Generates random double values within specified range.
        /// </summary>
        public DoubleArrayBuilder WithRandomRange(double min, double max)
        {
            _generator = i =>
            {
                var random = new Random(_seed + i);
                return min + random.NextDouble() * (max - min);
            };
            return this;
        }

        /// <summary>
        /// Uses custom generator function.
        /// </summary>
        public DoubleArrayBuilder WithGenerator(Func<int, double> generator)
        {
            _generator = generator;
            return this;
        }

        /// <summary>
        /// Sets the random seed for reproducible generation.
        /// </summary>
        public DoubleArrayBuilder WithSeed(int seed)
        {
            _seed = seed;
            return this;
        }

        /// <summary>
        /// Builds the double array.
        /// </summary>
        public double[] Build()
        {
            var generator = _generator ?? (_ => 0.0);
            return Enumerable.Range(0, _size).Select(generator).ToArray();
        }
    }

    /// <summary>
    /// Builder for generic array test data.
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    public sealed class GenericArrayBuilder<T>
    {
        private readonly int _size;
        private readonly Func<int, T> _generator;

        internal GenericArrayBuilder(int size, Func<int, T> generator)
        {
            _size = size;
            _generator = generator;
        }

        /// <summary>
        /// Builds the array.
        /// </summary>
        public T[] Build()
        {
            return Enumerable.Range(0, _size).Select(_generator).ToArray();
        }

        /// <summary>
        /// Builds the array as a list.
        /// </summary>
        public List<T> BuildAsList()
        {
            return Enumerable.Range(0, _size).Select(_generator).ToList();
        }
    }

    /// <summary>
    /// Edge case data generators for testing boundary conditions.
    /// </summary>
    public static class EdgeCases
    {
        /// <summary>
        /// Creates an empty float array.
        /// </summary>
        public static float[] EmptyFloatArray() => System.Array.Empty<float>();

        /// <summary>
        /// Creates an empty int array.
        /// </summary>
        public static int[] EmptyIntArray() => System.Array.Empty<int>();

        /// <summary>
        /// Creates a single-element float array.
        /// </summary>
        public static float[] SingleFloatElement(float value = 1.0f) => new[] { value };

        /// <summary>
        /// Creates a single-element int array.
        /// </summary>
        public static int[] SingleIntElement(int value = 1) => new[] { value };

        /// <summary>
        /// Creates a float array with all zeros.
        /// </summary>
        public static float[] AllZeros(int size) => new float[size];

        /// <summary>
        /// Creates a float array with all ones.
        /// </summary>
        public static float[] AllOnes(int size) => Enumerable.Repeat(1.0f, size).ToArray();

        /// <summary>
        /// Creates a float array with maximum float values.
        /// </summary>
        public static float[] MaxFloatValues(int size) => Enumerable.Repeat(float.MaxValue, size).ToArray();

        /// <summary>
        /// Creates a float array with minimum float values.
        /// </summary>
        public static float[] MinFloatValues(int size) => Enumerable.Repeat(float.MinValue, size).ToArray();

        /// <summary>
        /// Creates a float array with NaN values.
        /// </summary>
        public static float[] NaNValues(int size) => Enumerable.Repeat(float.NaN, size).ToArray();

        /// <summary>
        /// Creates a float array with infinity values.
        /// </summary>
        public static float[] InfinityValues(int size) => Enumerable.Repeat(float.PositiveInfinity, size).ToArray();

        /// <summary>
        /// Creates an int array with maximum int values.
        /// </summary>
        public static int[] MaxIntValues(int size) => Enumerable.Repeat(int.MaxValue, size).ToArray();

        /// <summary>
        /// Creates an int array with minimum int values.
        /// </summary>
        public static int[] MinIntValues(int size) => Enumerable.Repeat(int.MinValue, size).ToArray();

        /// <summary>
        /// Creates a float array with alternating positive and negative values.
        /// </summary>
        public static float[] AlternatingSign(int size) =>
            Enumerable.Range(0, size).Select(i => i % 2 == 0 ? 1.0f : -1.0f).ToArray();

        /// <summary>
        /// Creates a sparse array (mostly zeros with few non-zero values).
        /// </summary>
        public static float[] SparseArray(int size, int nonZeroCount, int seed = DefaultSeed)
        {
            var array = new float[size];
            var random = new Random(seed);
            for (int i = 0; i < nonZeroCount && i < size; i++)
            {
                var index = random.Next(size);
                array[index] = (float)random.NextDouble();
            }
            return array;
        }
    }

    /// <summary>
    /// Matrix test data generators.
    /// </summary>
    public static class Matrices
    {
        /// <summary>
        /// Creates a 2D float array (matrix) with sequential values.
        /// </summary>
        public static float[,] SequentialMatrix(int rows, int cols)
        {
            var matrix = new float[rows, cols];
            var value = 0f;
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    matrix[i, j] = value++;
                }
            }
            return matrix;
        }

        /// <summary>
        /// Creates an identity matrix.
        /// </summary>
        public static float[,] IdentityMatrix(int size)
        {
            var matrix = new float[size, size];
            for (int i = 0; i < size; i++)
            {
                matrix[i, i] = 1.0f;
            }
            return matrix;
        }

        /// <summary>
        /// Creates a random matrix with specified seed.
        /// </summary>
        public static float[,] RandomMatrix(int rows, int cols, int seed = DefaultSeed)
        {
            var matrix = new float[rows, cols];
            var random = new Random(seed);
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    matrix[i, j] = (float)random.NextDouble();
                }
            }
            return matrix;
        }

        /// <summary>
        /// Flattens a 2D matrix to 1D array (row-major order).
        /// </summary>
        public static float[] Flatten(float[,] matrix)
        {
            var rows = matrix.GetLength(0);
            var cols = matrix.GetLength(1);
            var result = new float[rows * cols];
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    result[i * cols + j] = matrix[i, j];
                }
            }
            return result;
        }
    }
}

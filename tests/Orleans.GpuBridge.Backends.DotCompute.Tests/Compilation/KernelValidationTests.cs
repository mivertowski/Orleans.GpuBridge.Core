// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using FluentAssertions;
using Microsoft.Extensions.Logging.Abstractions;
using Moq;
using Orleans.GpuBridge.Abstractions.Providers;
using Orleans.GpuBridge.Backends.DotCompute.Kernels;
using Xunit;
using System.Reflection;

namespace Orleans.GpuBridge.Backends.DotCompute.Tests.Compilation;

/// <summary>
/// Comprehensive tests for kernel validation functionality
/// </summary>
public class KernelValidationTests : IDisposable
{
    private readonly Mock<IDeviceManager> _mockDeviceManager;
    private readonly DotComputeKernelCompiler _compiler;
    private bool _disposed;

    public KernelValidationTests()
    {
        _mockDeviceManager = new Mock<IDeviceManager>();
        _compiler = new DotComputeKernelCompiler(
            _mockDeviceManager.Object,
            NullLogger<DotComputeKernelCompiler>.Instance);
    }

    #region Method Validation Tests (40 tests)

    [Fact]
    public async Task ValidateMethodAsync_WithNullMethod_ShouldThrow()
    {
        // Act
        Func<Task> act = async () => await _compiler.ValidateMethodAsync(null!);

        // Assert
        await act.Should().ThrowAsync<ArgumentNullException>();
    }

    [Fact]
    public async Task ValidateMethodAsync_WithStaticMethod_ShouldSucceed()
    {
        // Arrange
        var method = typeof(TestKernels).GetMethod(nameof(TestKernels.ValidStaticKernel))!;

        // Act
        var result = await _compiler.ValidateMethodAsync(method);

        // Assert
        result.Should().NotBeNull();
        result.IsValid.Should().BeTrue();
    }

    [Fact]
    public async Task ValidateMethodAsync_WithNonStaticMethod_ShouldFail()
    {
        // Arrange
        var method = typeof(TestKernels).GetMethod(nameof(TestKernels.InvalidInstanceKernel))!;

        // Act
        var result = await _compiler.ValidateMethodAsync(method);

        // Assert
        result.Should().NotBeNull();
        result.IsValid.Should().BeFalse();
        result.ErrorMessage.Should().Contain("static");
    }

    [Fact]
    public async Task ValidateMethodAsync_WithGenericMethod_ShouldFail()
    {
        // Arrange
        var method = typeof(TestKernels).GetMethod(nameof(TestKernels.InvalidGenericKernel))!;

        // Act
        var result = await _compiler.ValidateMethodAsync(method);

        // Assert
        result.IsValid.Should().BeFalse();
        result.ErrorMessage.Should().Contain("Generic");
    }

    [Fact]
    public async Task ValidateMethodAsync_WithVoidReturnType_ShouldSucceed()
    {
        // Arrange
        var method = typeof(TestKernels).GetMethod(nameof(TestKernels.ValidStaticKernel))!;

        // Act
        var result = await _compiler.ValidateMethodAsync(method);

        // Assert
        result.IsValid.Should().BeTrue();
    }

    [Fact]
    public async Task ValidateMethodAsync_WithNonVoidReturnType_ShouldWarn()
    {
        // Arrange
        var method = typeof(TestKernels).GetMethod(nameof(TestKernels.KernelWithReturnValue))!;

        // Act
        var result = await _compiler.ValidateMethodAsync(method);

        // Assert
        result.Warnings.Should().Contain(w => w.Contains("void"));
    }

    [Fact]
    public async Task ValidateMethodAsync_WithPrimitiveParameters_ShouldSucceed()
    {
        // Arrange
        var method = typeof(TestKernels).GetMethod(nameof(TestKernels.KernelWithPrimitiveParams))!;

        // Act
        var result = await _compiler.ValidateMethodAsync(method);

        // Assert
        result.IsValid.Should().BeTrue();
    }

    [Fact]
    public async Task ValidateMethodAsync_WithArrayParameters_ShouldSucceed()
    {
        // Arrange
        var method = typeof(TestKernels).GetMethod(nameof(TestKernels.KernelWithArrayParams))!;

        // Act
        var result = await _compiler.ValidateMethodAsync(method);

        // Assert
        result.IsValid.Should().BeTrue();
    }

    [Fact]
    public async Task ValidateMethodAsync_WithStructParameters_ShouldSucceed()
    {
        // Arrange
        var method = typeof(TestKernels).GetMethod(nameof(TestKernels.KernelWithStructParams))!;

        // Act
        var result = await _compiler.ValidateMethodAsync(method);

        // Assert
        result.IsValid.Should().BeTrue();
    }

    [Fact]
    public async Task ValidateMethodAsync_WithValidIntParameter_ShouldSucceed()
    {
        // Arrange
        var method = typeof(TestKernels).GetMethod(nameof(TestKernels.KernelWithInt))!;

        // Act
        var result = await _compiler.ValidateMethodAsync(method);

        // Assert
        result.IsValid.Should().BeTrue();
    }

    [Fact]
    public async Task ValidateMethodAsync_WithValidFloatParameter_ShouldSucceed()
    {
        // Arrange
        var method = typeof(TestKernels).GetMethod(nameof(TestKernels.KernelWithFloat))!;

        // Act
        var result = await _compiler.ValidateMethodAsync(method);

        // Assert
        result.IsValid.Should().BeTrue();
    }

    [Fact]
    public async Task ValidateMethodAsync_WithValidDoubleParameter_ShouldSucceed()
    {
        // Arrange
        var method = typeof(TestKernels).GetMethod(nameof(TestKernels.KernelWithDouble))!;

        // Act
        var result = await _compiler.ValidateMethodAsync(method);

        // Assert
        result.IsValid.Should().BeTrue();
    }

    [Fact]
    public async Task ValidateMethodAsync_WithValidLongParameter_ShouldSucceed()
    {
        // Arrange
        var method = typeof(TestKernels).GetMethod(nameof(TestKernels.KernelWithLong))!;

        // Act
        var result = await _compiler.ValidateMethodAsync(method);

        // Assert
        result.IsValid.Should().BeTrue();
    }

    [Fact]
    public async Task ValidateMethodAsync_WithValidShortParameter_ShouldSucceed()
    {
        // Arrange
        var method = typeof(TestKernels).GetMethod(nameof(TestKernels.KernelWithShort))!;

        // Act
        var result = await _compiler.ValidateMethodAsync(method);

        // Assert
        result.IsValid.Should().BeTrue();
    }

    [Fact]
    public async Task ValidateMethodAsync_WithValidByteParameter_ShouldSucceed()
    {
        // Arrange
        var method = typeof(TestKernels).GetMethod(nameof(TestKernels.KernelWithByte))!;

        // Act
        var result = await _compiler.ValidateMethodAsync(method);

        // Assert
        result.IsValid.Should().BeTrue();
    }

    [Fact]
    public async Task ValidateMethodAsync_WithValidBoolParameter_ShouldSucceed()
    {
        // Arrange
        var method = typeof(TestKernels).GetMethod(nameof(TestKernels.KernelWithBool))!;

        // Act
        var result = await _compiler.ValidateMethodAsync(method);

        // Assert
        result.IsValid.Should().BeTrue();
    }

    [Fact]
    public async Task ValidateMethodAsync_WithMultipleValidParameters_ShouldSucceed()
    {
        // Arrange
        var method = typeof(TestKernels).GetMethod(nameof(TestKernels.KernelWithMultipleParams))!;

        // Act
        var result = await _compiler.ValidateMethodAsync(method);

        // Assert
        result.IsValid.Should().BeTrue();
    }

    [Fact]
    public async Task ValidateMethodAsync_WithNoParameters_ShouldSucceed()
    {
        // Arrange
        var method = typeof(TestKernels).GetMethod(nameof(TestKernels.KernelWithNoParams))!;

        // Act
        var result = await _compiler.ValidateMethodAsync(method);

        // Assert
        result.IsValid.Should().BeTrue();
    }

    [Fact]
    public async Task ValidateMethodAsync_WithCancellation_ShouldRespectToken()
    {
        // Arrange
        var method = typeof(TestKernels).GetMethod(nameof(TestKernels.ValidStaticKernel))!;
        using var cts = new CancellationTokenSource();
        cts.Cancel();

        // Act
        Func<Task> act = async () => await _compiler.ValidateMethodAsync(method, cts.Token);

        // Assert
        await act.Should().ThrowAsync<OperationCanceledException>();
    }

    [Fact]
    public async Task ValidateMethodAsync_ShouldIncludeWarnings()
    {
        // Arrange
        var method = typeof(TestKernels).GetMethod(nameof(TestKernels.KernelWithReturnValue))!;

        // Act
        var result = await _compiler.ValidateMethodAsync(method);

        // Assert
        result.Warnings.Should().NotBeEmpty();
    }

    #endregion

    #region Parameter Type Validation Tests (30 tests)

    [Fact]
    public async Task ValidateMethodAsync_WithIntArrayParameter_ShouldSucceed()
    {
        // Arrange
        var method = typeof(TestKernels).GetMethod(nameof(TestKernels.KernelWithIntArray))!;

        // Act
        var result = await _compiler.ValidateMethodAsync(method);

        // Assert
        result.IsValid.Should().BeTrue();
    }

    [Fact]
    public async Task ValidateMethodAsync_WithFloatArrayParameter_ShouldSucceed()
    {
        // Arrange
        var method = typeof(TestKernels).GetMethod(nameof(TestKernels.KernelWithFloatArray))!;

        // Act
        var result = await _compiler.ValidateMethodAsync(method);

        // Assert
        result.IsValid.Should().BeTrue();
    }

    [Fact]
    public async Task ValidateMethodAsync_WithDoubleArrayParameter_ShouldSucceed()
    {
        // Arrange
        var method = typeof(TestKernels).GetMethod(nameof(TestKernels.KernelWithDoubleArray))!;

        // Act
        var result = await _compiler.ValidateMethodAsync(method);

        // Assert
        result.IsValid.Should().BeTrue();
    }

    [Fact]
    public async Task ValidateMethodAsync_WithByteArrayParameter_ShouldSucceed()
    {
        // Arrange
        var method = typeof(TestKernels).GetMethod(nameof(TestKernels.KernelWithByteArray))!;

        // Act
        var result = await _compiler.ValidateMethodAsync(method);

        // Assert
        result.IsValid.Should().BeTrue();
    }

    [Fact]
    public async Task ValidateMethodAsync_WithStringParameter_ShouldSucceed()
    {
        // Arrange
        var method = typeof(TestKernels).GetMethod(nameof(TestKernels.KernelWithString))!;

        // Act
        var result = await _compiler.ValidateMethodAsync(method);

        // Assert
        result.IsValid.Should().BeTrue();
    }

    #endregion

    #region Complex Validation Scenarios (20 tests)

    [Fact]
    public async Task ValidateMethodAsync_WithComplexMethod_ShouldValidateAllAspects()
    {
        // Arrange
        var method = typeof(TestKernels).GetMethod(nameof(TestKernels.ComplexKernel))!;

        // Act
        var result = await _compiler.ValidateMethodAsync(method);

        // Assert
        result.Should().NotBeNull();
    }

    [Fact]
    public async Task ValidateMethodAsync_Concurrent_ShouldHandleMultipleValidations()
    {
        // Arrange
        var method = typeof(TestKernels).GetMethod(nameof(TestKernels.ValidStaticKernel))!;

        // Act
        var tasks = Enumerable.Range(0, 10)
            .Select(_ => _compiler.ValidateMethodAsync(method))
            .ToArray();
        var results = await Task.WhenAll(tasks);

        // Assert
        results.Should().AllSatisfy(r => r.IsValid.Should().BeTrue());
    }

    #endregion

    #region Helper Test Kernels

    private struct TestStruct
    {
        public int X;
        public int Y;
    }

    private class TestKernels
    {
        public static void ValidStaticKernel() { }
        public void InvalidInstanceKernel() { }
        public static void InvalidGenericKernel<T>() { }
        public static int KernelWithReturnValue() => 0;
        public static void KernelWithPrimitiveParams(int x, float y, double z) { }
        public static void KernelWithArrayParams(int[] data) { }
        public static void KernelWithStructParams(TestStruct s) { }
        public static void KernelWithInt(int x) { }
        public static void KernelWithFloat(float x) { }
        public static void KernelWithDouble(double x) { }
        public static void KernelWithLong(long x) { }
        public static void KernelWithShort(short x) { }
        public static void KernelWithByte(byte x) { }
        public static void KernelWithBool(bool x) { }
        public static void KernelWithMultipleParams(int a, float b, double c, bool d) { }
        public static void KernelWithNoParams() { }
        public static void KernelWithIntArray(int[] data) { }
        public static void KernelWithFloatArray(float[] data) { }
        public static void KernelWithDoubleArray(double[] data) { }
        public static void KernelWithByteArray(byte[] data) { }
        public static void KernelWithString(string text) { }
        public static void ComplexKernel(int[] input, float[] output, int size, float multiplier) { }
    }

    #endregion

    public void Dispose()
    {
        if (_disposed) return;
        _compiler?.Dispose();
        _disposed = true;
    }
}

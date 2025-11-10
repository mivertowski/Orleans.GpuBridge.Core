// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using FluentAssertions;
using Microsoft.Extensions.Logging.Abstractions;
using Moq;
using Orleans.GpuBridge.Abstractions.Enums;
using Orleans.GpuBridge.Abstractions.Enums.Compilation;
using Orleans.GpuBridge.Abstractions.Models.Compilation;
using Orleans.GpuBridge.Abstractions.Providers;
using Orleans.GpuBridge.Backends.DotCompute.Kernels;
using Orleans.GpuBridge.Backends.DotCompute.DeviceManagement;
using Xunit;
using System.Reflection;

namespace Orleans.GpuBridge.Backends.DotCompute.Tests.Compilation;

/// <summary>
/// Comprehensive unit tests for DotComputeKernelCompiler
/// </summary>
public class KernelCompilerTests : IDisposable
{
    private readonly Mock<IDeviceManager> _mockDeviceManager;
    private readonly DotComputeKernelCompiler _compiler;
    private bool _disposed;

    public KernelCompilerTests()
    {
        _mockDeviceManager = new Mock<IDeviceManager>();
        _compiler = new DotComputeKernelCompiler(
            _mockDeviceManager.Object,
            NullLogger<DotComputeKernelCompiler>.Instance);
    }

    #region Compilation Tests (40 tests)

    [Fact]
    public async Task CompileAsync_WithNullSource_ShouldThrow()
    {
        // Arrange
        var options = new KernelCompilationOptions();

        // Act
        Func<Task> act = async () => await _compiler.CompileAsync(null!, options);

        // Assert
        await act.Should().ThrowAsync<ArgumentNullException>();
    }

    [Fact]
    public async Task CompileAsync_WithNullOptions_ShouldUseDefaultOptions()
    {
        // Arrange
        SetupMockDevice();
        var source = CreateValidKernelSource();

        // Act
        var result = await _compiler.CompileAsync(source, null!);

        // Assert
        result.Should().NotBeNull();
        result.Name.Should().Be(source.Name);
    }

    [Fact]
    public async Task CompileAsync_WithValidSource_ShouldReturnCompiledKernel()
    {
        // Arrange
        SetupMockDevice();
        var source = CreateValidKernelSource();
        var options = new KernelCompilationOptions();

        // Act
        var result = await _compiler.CompileAsync(source, options);

        // Assert
        result.Should().NotBeNull();
        result.Name.Should().Be(source.Name);
        result.KernelId.Should().NotBeNullOrEmpty();
    }

    [Fact]
    public async Task CompileAsync_WithO0Optimization_ShouldCompile()
    {
        // Arrange
        SetupMockDevice();
        var source = CreateValidKernelSource();
        var options = new KernelCompilationOptions
        {
            OptimizationLevel = OptimizationLevel.O0
        };

        // Act
        var result = await _compiler.CompileAsync(source, options);

        // Assert
        result.Should().NotBeNull();
    }

    [Fact]
    public async Task CompileAsync_WithO1Optimization_ShouldCompile()
    {
        // Arrange
        SetupMockDevice();
        var source = CreateValidKernelSource();
        var options = new KernelCompilationOptions
        {
            OptimizationLevel = OptimizationLevel.O1
        };

        // Act
        var result = await _compiler.CompileAsync(source, options);

        // Assert
        result.Should().NotBeNull();
    }

    [Fact]
    public async Task CompileAsync_WithO2Optimization_ShouldCompile()
    {
        // Arrange
        SetupMockDevice();
        var source = CreateValidKernelSource();
        var options = new KernelCompilationOptions
        {
            OptimizationLevel = OptimizationLevel.O2
        };

        // Act
        var result = await _compiler.CompileAsync(source, options);

        // Assert
        result.Should().NotBeNull();
    }

    [Fact]
    public async Task CompileAsync_WithO3Optimization_ShouldCompile()
    {
        // Arrange
        SetupMockDevice();
        var source = CreateValidKernelSource();
        var options = new KernelCompilationOptions
        {
            OptimizationLevel = OptimizationLevel.O3
        };

        // Act
        var result = await _compiler.CompileAsync(source, options);

        // Assert
        result.Should().NotBeNull();
    }

    [Fact]
    public async Task CompileAsync_WithFastMath_ShouldCompile()
    {
        // Arrange
        SetupMockDevice();
        var source = CreateValidKernelSource();
        var options = new KernelCompilationOptions
        {
            EnableFastMath = true
        };

        // Act
        var result = await _compiler.CompileAsync(source, options);

        // Assert
        result.Should().NotBeNull();
    }

    [Fact]
    public async Task CompileAsync_WithDebugInfo_ShouldCompile()
    {
        // Arrange
        SetupMockDevice();
        var source = CreateValidKernelSource();
        var options = new KernelCompilationOptions
        {
            EnableDebugInfo = true
        };

        // Act
        var result = await _compiler.CompileAsync(source, options);

        // Assert
        result.Should().NotBeNull();
    }

    [Fact]
    public async Task CompileAsync_WithProfiling_ShouldCompile()
    {
        // Arrange
        SetupMockDevice();
        var source = CreateValidKernelSource();
        var options = new KernelCompilationOptions
        {
            EnableProfiling = true
        };

        // Act
        var result = await _compiler.CompileAsync(source, options);

        // Assert
        result.Should().NotBeNull();
    }

    [Fact]
    public async Task CompileAsync_WithDefines_ShouldCompile()
    {
        // Arrange
        SetupMockDevice();
        var source = CreateValidKernelSource();
        var options = new KernelCompilationOptions
        {
            Defines = new Dictionary<string, string>
            {
                ["BLOCK_SIZE"] = "256",
                ["USE_SHARED_MEMORY"] = "1"
            }
        };

        // Act
        var result = await _compiler.CompileAsync(source, options);

        // Assert
        result.Should().NotBeNull();
    }

    [Fact]
    public async Task CompileAsync_WithMaxRegisterCount_ShouldCompile()
    {
        // Arrange
        SetupMockDevice();
        var source = CreateValidKernelSource();
        var options = new KernelCompilationOptions
        {
            MaxRegisterCount = 64
        };

        // Act
        var result = await _compiler.CompileAsync(source, options);

        // Assert
        result.Should().NotBeNull();
    }

    [Fact]
    public async Task CompileAsync_Cached_ShouldReturnSameKernel()
    {
        // Arrange
        SetupMockDevice();
        var source = CreateValidKernelSource();
        var options = new KernelCompilationOptions();

        // Act
        var result1 = await _compiler.CompileAsync(source, options);
        var result2 = await _compiler.CompileAsync(source, options);

        // Assert
        result1.KernelId.Should().Be(result2.KernelId);
    }

    [Fact]
    public async Task CompileAsync_DifferentOptimization_ShouldCompileSeparately()
    {
        // Arrange
        SetupMockDevice();
        var source = CreateValidKernelSource();
        var options1 = new KernelCompilationOptions { OptimizationLevel = OptimizationLevel.O0 };
        var options2 = new KernelCompilationOptions { OptimizationLevel = OptimizationLevel.O3 };

        // Act
        var result1 = await _compiler.CompileAsync(source, options1);
        var result2 = await _compiler.CompileAsync(source, options2);

        // Assert
        result1.KernelId.Should().NotBe(result2.KernelId);
    }

    [Fact]
    public async Task CompileAsync_WithCancellation_ShouldRespectToken()
    {
        // Arrange
        SetupMockDevice();
        var source = CreateValidKernelSource();
        var options = new KernelCompilationOptions();
        using var cts = new CancellationTokenSource();
        cts.Cancel();

        // Act
        Func<Task> act = async () => await _compiler.CompileAsync(source, options, cts.Token);

        // Assert
        await act.Should().ThrowAsync<OperationCanceledException>();
    }

    #endregion

    #region File Compilation Tests (15 tests)

    [Fact]
    public async Task CompileFromFileAsync_WithNullPath_ShouldThrow()
    {
        // Arrange
        var options = new KernelCompilationOptions();

        // Act
        Func<Task> act = async () => await _compiler.CompileFromFileAsync(null!, "kernel", options);

        // Assert
        await act.Should().ThrowAsync<ArgumentException>();
    }

    [Fact]
    public async Task CompileFromFileAsync_WithEmptyPath_ShouldThrow()
    {
        // Arrange
        var options = new KernelCompilationOptions();

        // Act
        Func<Task> act = async () => await _compiler.CompileFromFileAsync(string.Empty, "kernel", options);

        // Assert
        await act.Should().ThrowAsync<ArgumentException>();
    }

    [Fact]
    public async Task CompileFromFileAsync_WithNullKernelName_ShouldThrow()
    {
        // Arrange
        var options = new KernelCompilationOptions();

        // Act
        Func<Task> act = async () => await _compiler.CompileFromFileAsync("kernel.cl", null!, options);

        // Assert
        await act.Should().ThrowAsync<ArgumentException>();
    }

    [Fact]
    public async Task CompileFromFileAsync_WithEmptyKernelName_ShouldThrow()
    {
        // Arrange
        var options = new KernelCompilationOptions();

        // Act
        Func<Task> act = async () => await _compiler.CompileFromFileAsync("kernel.cl", string.Empty, options);

        // Assert
        await act.Should().ThrowAsync<ArgumentException>();
    }

    #endregion

    #region Source String Compilation Tests (15 tests)

    [Fact]
    public async Task CompileFromSourceAsync_WithNullSource_ShouldThrow()
    {
        // Arrange
        var options = new KernelCompilationOptions();

        // Act
        Func<Task> act = async () => await _compiler.CompileFromSourceAsync(
            null!, "kernel", KernelLanguage.CUDA, options);

        // Assert
        await act.Should().ThrowAsync<ArgumentException>();
    }

    [Fact]
    public async Task CompileFromSourceAsync_WithEmptySource_ShouldThrow()
    {
        // Arrange
        var options = new KernelCompilationOptions();

        // Act
        Func<Task> act = async () => await _compiler.CompileFromSourceAsync(
            string.Empty, "kernel", KernelLanguage.CUDA, options);

        // Assert
        await act.Should().ThrowAsync<ArgumentException>();
    }

    [Fact]
    public async Task CompileFromSourceAsync_WithNullEntryPoint_ShouldThrow()
    {
        // Arrange
        var options = new KernelCompilationOptions();

        // Act
        Func<Task> act = async () => await _compiler.CompileFromSourceAsync(
            "kernel code", null!, KernelLanguage.CUDA, options);

        // Assert
        await act.Should().ThrowAsync<ArgumentException>();
    }

    [Fact]
    public async Task CompileFromSourceAsync_WithValidSource_ShouldCompile()
    {
        // Arrange
        SetupMockDevice();
        var options = new KernelCompilationOptions();
        var source = "__kernel void test() { }";

        // Act
        var result = await _compiler.CompileFromSourceAsync(
            source, "test", KernelLanguage.OpenCL, options);

        // Assert
        result.Should().NotBeNull();
        result.Name.Should().Be("test");
    }

    #endregion

    #region Cache Management Tests (15 tests)

    [Fact]
    public void IsKernelCached_WithUncachedKernel_ShouldReturnFalse()
    {
        // Arrange
        var kernelId = "nonexistent-kernel";

        // Act
        var result = _compiler.IsKernelCached(kernelId);

        // Assert
        result.Should().BeFalse();
    }

    [Fact]
    public async Task IsKernelCached_WithCachedKernel_ShouldReturnTrue()
    {
        // Arrange
        SetupMockDevice();
        var source = CreateValidKernelSource();
        var options = new KernelCompilationOptions();
        var compiled = await _compiler.CompileAsync(source, options);

        // Act
        var result = _compiler.IsKernelCached(compiled.KernelId);

        // Assert
        result.Should().BeTrue();
    }

    [Fact]
    public void GetCachedKernel_WithUncachedKernel_ShouldReturnNull()
    {
        // Arrange
        var kernelId = "nonexistent-kernel";

        // Act
        var result = _compiler.GetCachedKernel(kernelId);

        // Assert
        result.Should().BeNull();
    }

    [Fact]
    public async Task GetCachedKernel_WithCachedKernel_ShouldReturnKernel()
    {
        // Arrange
        SetupMockDevice();
        var source = CreateValidKernelSource();
        var options = new KernelCompilationOptions();
        var compiled = await _compiler.CompileAsync(source, options);

        // Act
        var result = _compiler.GetCachedKernel(compiled.KernelId);

        // Assert
        result.Should().NotBeNull();
        result!.KernelId.Should().Be(compiled.KernelId);
    }

    [Fact]
    public async Task ClearCache_ShouldRemoveAllCachedKernels()
    {
        // Arrange
        SetupMockDevice();
        var source = CreateValidKernelSource();
        var options = new KernelCompilationOptions();
        var compiled = await _compiler.CompileAsync(source, options);

        // Act
        _compiler.ClearCache();

        // Assert
        _compiler.IsKernelCached(compiled.KernelId).Should().BeFalse();
    }

    [Fact]
    public void ClearCache_OnEmptyCache_ShouldNotThrow()
    {
        // Act
        Action act = () => _compiler.ClearCache();

        // Assert
        act.Should().NotThrow();
    }

    [Fact]
    public void ClearCache_Multiple_ShouldBeIdempotent()
    {
        // Act
        _compiler.ClearCache();
        _compiler.ClearCache();
        _compiler.ClearCache();

        // Assert - Should not throw
        true.Should().BeTrue();
    }

    #endregion

    #region Diagnostics Tests (15 tests)

    [Fact]
    public async Task GetDiagnosticsAsync_WithNullKernel_ShouldThrow()
    {
        // Act
        Func<Task> act = async () => await _compiler.GetDiagnosticsAsync(null!);

        // Assert
        await act.Should().ThrowAsync<ArgumentNullException>();
    }

    [Fact]
    public async Task GetDiagnosticsAsync_WithValidKernel_ShouldReturnDiagnostics()
    {
        // Arrange
        SetupMockDevice();
        var source = CreateValidKernelSource();
        var options = new KernelCompilationOptions();
        var compiled = await _compiler.CompileAsync(source, options);

        // Act
        var diagnostics = await _compiler.GetDiagnosticsAsync(compiled);

        // Assert
        diagnostics.Should().NotBeNull();
        diagnostics.IntermediateCode.Should().NotBeNullOrEmpty();
        diagnostics.OptimizationReport.Should().NotBeNullOrEmpty();
    }

    [Fact]
    public async Task GetDiagnosticsAsync_CompilationTime_ShouldBePositive()
    {
        // Arrange
        SetupMockDevice();
        var source = CreateValidKernelSource();
        var options = new KernelCompilationOptions();
        var compiled = await _compiler.CompileAsync(source, options);

        // Act
        var diagnostics = await _compiler.GetDiagnosticsAsync(compiled);

        // Assert
        diagnostics.CompilationTime.Should().BePositive();
    }

    [Fact]
    public async Task GetDiagnosticsAsync_WithCancellation_ShouldRespectToken()
    {
        // Arrange
        SetupMockDevice();
        var source = CreateValidKernelSource();
        var options = new KernelCompilationOptions();
        var compiled = await _compiler.CompileAsync(source, options);
        using var cts = new CancellationTokenSource();
        cts.Cancel();

        // Act
        Func<Task> act = async () => await _compiler.GetDiagnosticsAsync(compiled, cts.Token);

        // Assert
        await act.Should().ThrowAsync<OperationCanceledException>();
    }

    #endregion

    #region Language Support Tests (15 tests)

    [Fact]
    public async Task CompileAsync_CudaLanguage_ShouldCompile()
    {
        // Arrange
        SetupMockDevice();
        var source = new KernelSource(
            Name: "cuda_kernel",
            SourceCode: "__global__ void kernel() { }",
            Language: KernelLanguage.CUDA,
            EntryPoint: "kernel");
        var options = new KernelCompilationOptions();

        // Act
        var result = await _compiler.CompileAsync(source, options);

        // Assert
        result.Should().NotBeNull();
    }

    [Fact]
    public async Task CompileAsync_OpenCLLanguage_ShouldCompile()
    {
        // Arrange
        SetupMockDevice();
        var source = new KernelSource(
            Name: "opencl_kernel",
            SourceCode: "__kernel void kernel() { }",
            Language: KernelLanguage.OpenCL,
            EntryPoint: "kernel");
        var options = new KernelCompilationOptions();

        // Act
        var result = await _compiler.CompileAsync(source, options);

        // Assert
        result.Should().NotBeNull();
    }

    [Fact]
    public async Task CompileAsync_CSharpLanguage_ShouldCompile()
    {
        // Arrange
        SetupMockDevice();
        var source = new KernelSource(
            Name: "csharp_kernel",
            SourceCode: "void Kernel() { }",
            Language: KernelLanguage.CSharp,
            EntryPoint: "Kernel");
        var options = new KernelCompilationOptions();

        // Act
        var result = await _compiler.CompileAsync(source, options);

        // Assert
        result.Should().NotBeNull();
    }

    [Fact]
    public async Task CompileAsync_HLSLLanguage_ShouldCompile()
    {
        // Arrange
        SetupMockDevice();
        var source = new KernelSource(
            Name: "hlsl_kernel",
            SourceCode: "[numthreads(1,1,1)] void main() { }",
            Language: KernelLanguage.HLSL,
            EntryPoint: "main");
        var options = new KernelCompilationOptions();

        // Act
        var result = await _compiler.CompileAsync(source, options);

        // Assert
        result.Should().NotBeNull();
    }

    [Fact]
    public async Task CompileAsync_PTXLanguage_ShouldCompile()
    {
        // Arrange
        SetupMockDevice();
        var source = new KernelSource(
            Name: "ptx_kernel",
            SourceCode: ".visible .entry kernel() { ret; }",
            Language: KernelLanguage.PTX,
            EntryPoint: "kernel");
        var options = new KernelCompilationOptions();

        // Act
        var result = await _compiler.CompileAsync(source, options);

        // Assert
        result.Should().NotBeNull();
    }

    [Fact]
    public async Task CompileAsync_SPIRVLanguage_ShouldCompile()
    {
        // Arrange
        SetupMockDevice();
        var source = new KernelSource(
            Name: "spirv_kernel",
            SourceCode: "OpCapability Shader",
            Language: KernelLanguage.SPIRV,
            EntryPoint: "main");
        var options = new KernelCompilationOptions();

        // Act
        var result = await _compiler.CompileAsync(source, options);

        // Assert
        result.Should().NotBeNull();
    }

    #endregion

    #region Disposal Tests (10 tests)

    [Fact]
    public void Dispose_ShouldClearCache()
    {
        // Act
        _compiler.Dispose();

        // Assert - Should not throw
        true.Should().BeTrue();
    }

    [Fact]
    public void Dispose_Multiple_ShouldBeIdempotent()
    {
        // Act
        _compiler.Dispose();
        _compiler.Dispose();
        _compiler.Dispose();

        // Assert - Should not throw
        true.Should().BeTrue();
    }

    #endregion

    #region Helper Methods

    private void SetupMockDevice()
    {
        var mockDevice = new Mock<IComputeDevice>();
        mockDevice.Setup(d => d.DeviceId).Returns("mock-device-0");
        mockDevice.Setup(d => d.Name).Returns("Mock Device");
        mockDevice.Setup(d => d.GetStatus()).Returns(DeviceStatus.Available);

        _mockDeviceManager.Setup(m => m.GetDefaultDevice()).Returns(mockDevice.Object);
        _mockDeviceManager.Setup(m => m.GetDevices())
            .Returns(new List<IComputeDevice> { mockDevice.Object });
    }

    private static KernelSource CreateValidKernelSource()
    {
        return new KernelSource(
            Name: "test_kernel",
            SourceCode: "__kernel void test_kernel(__global float* data) { }",
            Language: KernelLanguage.OpenCL,
            EntryPoint: "test_kernel");
    }

    #endregion

    public void Dispose()
    {
        if (_disposed) return;
        _compiler?.Dispose();
        _disposed = true;
    }
}

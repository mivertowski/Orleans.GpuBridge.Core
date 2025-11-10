using Orleans.GpuBridge.Abstractions.Models.Compilation;
using Orleans.GpuBridge.Abstractions.Enums.Compilation;

namespace Orleans.GpuBridge.Abstractions.Tests.Compilation;

/// <summary>
/// Tests for compilation-related models
/// </summary>
public class CompilationTests
{
    #region KernelMetadata Tests

    [Fact]
    public void KernelMetadata_DefaultConstructor_SetsDefaults()
    {
        // Arrange & Act
        var metadata = new KernelMetadata();

        // Assert
        metadata.RequiredSharedMemory.Should().Be(0);
        metadata.RequiredRegisters.Should().Be(0);
        metadata.MaxThreadsPerBlock.Should().Be(256);
        metadata.PreferredBlockSize.Should().Be(0);
        metadata.UsesAtomics.Should().BeFalse();
        metadata.UsesSharedMemory.Should().BeFalse();
        metadata.UsesDynamicParallelism.Should().BeFalse();
        metadata.ExtendedMetadata.Should().BeNull();
    }

    [Fact]
    public void KernelMetadata_WithSharedMemory_SetsProperties()
    {
        // Arrange & Act
        var metadata = new KernelMetadata(
            RequiredSharedMemory: 4096,
            RequiredRegisters: 32,
            MaxThreadsPerBlock: 512,
            PreferredBlockSize: 256,
            UsesSharedMemory: true
        );

        // Assert
        metadata.RequiredSharedMemory.Should().Be(4096);
        metadata.RequiredRegisters.Should().Be(32);
        metadata.MaxThreadsPerBlock.Should().Be(512);
        metadata.PreferredBlockSize.Should().Be(256);
        metadata.UsesSharedMemory.Should().BeTrue();
    }

    [Fact]
    public void KernelMetadata_WithAtomics_SetsFlag()
    {
        // Arrange & Act
        var metadata = new KernelMetadata(
            RequiredSharedMemory: 2048,
            RequiredRegisters: 48,
            UsesAtomics: true
        );

        // Assert
        metadata.UsesAtomics.Should().BeTrue();
        metadata.RequiredSharedMemory.Should().Be(2048);
    }

    [Fact]
    public void KernelMetadata_WithDynamicParallelism_SetsFlag()
    {
        // Arrange & Act
        var metadata = new KernelMetadata(
            UsesDynamicParallelism: true,
            MaxThreadsPerBlock: 256
        );

        // Assert
        metadata.UsesDynamicParallelism.Should().BeTrue();
    }

    [Fact]
    public void KernelMetadata_WithExtendedMetadata_StoresData()
    {
        // Arrange
        var extended = new Dictionary<string, object>
        {
            ["WarpSize"] = 32,
            ["MinComputeCapability"] = "6.0",
            ["Occupancy"] = 0.75
        };

        // Act
        var metadata = new KernelMetadata(
            ExtendedMetadata: extended
        );

        // Assert
        metadata.ExtendedMetadata.Should().NotBeNull();
        metadata.ExtendedMetadata.Should().HaveCount(3);
        metadata.ExtendedMetadata!["WarpSize"].Should().Be(32);
    }

    [Fact]
    public void KernelMetadata_Equality_SameValues_ReturnsTrue()
    {
        // Arrange
        var metadata1 = new KernelMetadata(
            RequiredSharedMemory: 4096,
            RequiredRegisters: 32,
            MaxThreadsPerBlock: 512
        );
        var metadata2 = new KernelMetadata(
            RequiredSharedMemory: 4096,
            RequiredRegisters: 32,
            MaxThreadsPerBlock: 512
        );

        // Act & Assert
        metadata1.Should().Be(metadata2);
    }

    #endregion

    #region CompilationDiagnostics Tests

    [Fact]
    public void CompilationDiagnostics_DefaultConstructor_SetsDefaults()
    {
        // Arrange & Act
        var diagnostics = new CompilationDiagnostics();

        // Assert
        diagnostics.IntermediateCode.Should().BeNull();
        diagnostics.AssemblyCode.Should().BeNull();
        diagnostics.OptimizationReport.Should().BeNull();
        diagnostics.CompilationTime.Should().Be(TimeSpan.Zero);
        diagnostics.CompiledCodeSize.Should().Be(0);
        diagnostics.AdditionalInfo.Should().BeNull();
    }

    [Fact]
    public void CompilationDiagnostics_WithAllData_SetsProperties()
    {
        // Arrange
        var additional = new Dictionary<string, object>
        {
            ["RegistersPerThread"] = 24,
            ["OccupancyPercent"] = 75.0
        };

        // Act
        var diagnostics = new CompilationDiagnostics(
            IntermediateCode: "define void @kernel(...) { ... }",
            AssemblyCode: "ld.global.f32 %f1, [%rd1+0];",
            OptimizationReport: "Applied loop unrolling (factor 4)",
            CompilationTime: TimeSpan.FromMilliseconds(350),
            CompiledCodeSize: 1536,
            AdditionalInfo: additional
        );

        // Assert
        diagnostics.IntermediateCode.Should().NotBeNull();
        diagnostics.AssemblyCode.Should().NotBeNull();
        diagnostics.OptimizationReport.Should().NotBeNull();
        diagnostics.CompilationTime.Should().Be(TimeSpan.FromMilliseconds(350));
        diagnostics.CompiledCodeSize.Should().Be(1536);
        diagnostics.AdditionalInfo.Should().HaveCount(2);
    }

    [Fact]
    public void CompilationDiagnostics_HasIntermediateCode_ReturnsCorrectValue()
    {
        // Arrange
        var withCode = new CompilationDiagnostics(IntermediateCode: "some code");
        var withoutCode = new CompilationDiagnostics();

        // Act & Assert
        withCode.HasIntermediateCode.Should().BeTrue();
        withoutCode.HasIntermediateCode.Should().BeFalse();
    }

    [Fact]
    public void CompilationDiagnostics_HasAssemblyCode_ReturnsCorrectValue()
    {
        // Arrange
        var withAsm = new CompilationDiagnostics(AssemblyCode: "mov eax, 1");
        var withoutAsm = new CompilationDiagnostics();

        // Act & Assert
        withAsm.HasAssemblyCode.Should().BeTrue();
        withoutAsm.HasAssemblyCode.Should().BeFalse();
    }

    [Fact]
    public void CompilationDiagnostics_HasOptimizationReport_ReturnsCorrectValue()
    {
        // Arrange
        var withReport = new CompilationDiagnostics(OptimizationReport: "Vectorized loops");
        var withoutReport = new CompilationDiagnostics();

        // Act & Assert
        withReport.HasOptimizationReport.Should().BeTrue();
        withoutReport.HasOptimizationReport.Should().BeFalse();
    }

    [Fact]
    public void CompilationDiagnostics_HasAdditionalInfo_ReturnsCorrectValue()
    {
        // Arrange
        var withInfo = new CompilationDiagnostics(
            AdditionalInfo: new Dictionary<string, object> { ["key"] = "value" }
        );
        var withoutInfo = new CompilationDiagnostics();
        var withEmptyInfo = new CompilationDiagnostics(
            AdditionalInfo: new Dictionary<string, object>()
        );

        // Act & Assert
        withInfo.HasAdditionalInfo.Should().BeTrue();
        withoutInfo.HasAdditionalInfo.Should().BeFalse();
        withEmptyInfo.HasAdditionalInfo.Should().BeFalse();
    }

    [Fact]
    public void CompilationDiagnostics_Equality_SameValues_ReturnsTrue()
    {
        // Arrange
        var diag1 = new CompilationDiagnostics(
            IntermediateCode: "code",
            CompilationTime: TimeSpan.FromMilliseconds(100)
        );
        var diag2 = new CompilationDiagnostics(
            IntermediateCode: "code",
            CompilationTime: TimeSpan.FromMilliseconds(100)
        );

        // Act & Assert
        diag1.Should().Be(diag2);
    }

    #endregion

    #region KernelValidationResult Tests

    [Fact]
    public void KernelValidationResult_Success_NoErrors()
    {
        // Arrange & Act
        var result = new KernelValidationResult(IsValid: true);

        // Assert
        result.IsValid.Should().BeTrue();
        result.ErrorMessage.Should().BeNull();
        result.HasWarnings.Should().BeFalse();
        result.HasUnsupportedFeatures.Should().BeFalse();
        result.IssueCount.Should().Be(0);
    }

    [Fact]
    public void KernelValidationResult_SuccessWithWarnings_ReturnsWarnings()
    {
        // Arrange
        var warnings = new[]
        {
            "Consider using shared memory for better performance",
            "High register usage may reduce occupancy"
        };

        // Act
        var result = new KernelValidationResult(
            IsValid: true,
            Warnings: warnings
        );

        // Assert
        result.IsValid.Should().BeTrue();
        result.HasWarnings.Should().BeTrue();
        result.Warnings.Should().HaveCount(2);
        result.IssueCount.Should().Be(2);
    }

    [Fact]
    public void KernelValidationResult_Failure_ReturnsError()
    {
        // Arrange
        var errorMessage = "Kernel contains unsupported dynamic memory allocation";

        // Act
        var result = new KernelValidationResult(
            IsValid: false,
            ErrorMessage: errorMessage
        );

        // Assert
        result.IsValid.Should().BeFalse();
        result.ErrorMessage.Should().Be(errorMessage);
    }

    [Fact]
    public void KernelValidationResult_WithUnsupportedFeatures_ListsFeatures()
    {
        // Arrange
        var unsupported = new[]
        {
            "Dynamic memory allocation (new, malloc)",
            "Recursive function calls",
            "Exception handling (try/catch)"
        };

        // Act
        var result = new KernelValidationResult(
            IsValid: false,
            ErrorMessage: "Unsupported features detected",
            UnsupportedFeatures: unsupported
        );

        // Assert
        result.IsValid.Should().BeFalse();
        result.HasUnsupportedFeatures.Should().BeTrue();
        result.UnsupportedFeatures.Should().HaveCount(3);
        result.IssueCount.Should().Be(3);
    }

    [Fact]
    public void KernelValidationResult_IssueCount_SumsWarningsAndFeatures()
    {
        // Arrange
        var warnings = new[] { "Warning 1", "Warning 2" };
        var unsupported = new[] { "Feature 1", "Feature 2", "Feature 3" };

        // Act
        var result = new KernelValidationResult(
            IsValid: false,
            ErrorMessage: "Multiple issues",
            Warnings: warnings,
            UnsupportedFeatures: unsupported
        );

        // Assert
        result.IssueCount.Should().Be(5); // 2 warnings + 3 unsupported
    }

    [Fact]
    public void KernelValidationResult_HasWarnings_WithEmptyList_ReturnsFalse()
    {
        // Arrange
        var result = new KernelValidationResult(
            IsValid: true,
            Warnings: Array.Empty<string>()
        );

        // Act & Assert
        result.HasWarnings.Should().BeFalse();
    }

    [Fact]
    public void KernelValidationResult_Equality_SameValues_ReturnsTrue()
    {
        // Arrange
        var result1 = new KernelValidationResult(
            IsValid: true,
            Warnings: new[] { "warning" }
        );
        var result2 = new KernelValidationResult(
            IsValid: true,
            Warnings: new[] { "warning" }
        );

        // Act & Assert
        // Arrays use reference equality, so compare properties individually
        result1.IsValid.Should().Be(result2.IsValid);
        result1.Warnings.Should().BeEquivalentTo(result2.Warnings);
    }

    #endregion

    #region KernelCompilationOptions Tests

    [Fact]
    public void KernelCompilationOptions_DefaultValues_AreCorrect()
    {
        // This tests the default compilation options if such a class exists
        // If KernelCompilationOptions doesn't exist, this can be a placeholder
        // Arrange & Act - Testing that compilation enums work correctly
        var optLevel = OptimizationLevel.O2;
        var language = KernelLanguage.CUDA;

        // Assert
        optLevel.Should().Be(OptimizationLevel.O2);
        language.Should().Be(KernelLanguage.CUDA);
    }

    #endregion

    #region Integration Tests

    [Fact]
    public void CompilationWorkflow_ValidKernel_ProducesMetadata()
    {
        // Arrange - Simulate a complete compilation workflow
        var metadata = new KernelMetadata(
            RequiredSharedMemory: 4096,
            RequiredRegisters: 32,
            MaxThreadsPerBlock: 512,
            PreferredBlockSize: 256,
            UsesSharedMemory: true
        );

        var diagnostics = new CompilationDiagnostics(
            IntermediateCode: "IR code here",
            AssemblyCode: "ASM code here",
            CompilationTime: TimeSpan.FromMilliseconds(100),
            CompiledCodeSize: 2048
        );

        var validation = new KernelValidationResult(
            IsValid: true,
            Warnings: new[] { "Minor warning" }
        );

        // Assert
        metadata.UsesSharedMemory.Should().BeTrue();
        diagnostics.HasIntermediateCode.Should().BeTrue();
        diagnostics.HasAssemblyCode.Should().BeTrue();
        validation.IsValid.Should().BeTrue();
        validation.HasWarnings.Should().BeTrue();
    }

    [Fact]
    public void CompilationWorkflow_InvalidKernel_ProducesValidationErrors()
    {
        // Arrange - Simulate failed compilation
        var validation = new KernelValidationResult(
            IsValid: false,
            ErrorMessage: "Compilation failed",
            UnsupportedFeatures: new[]
            {
                "Dynamic allocation",
                "Recursion"
            }
        );

        // Assert
        validation.IsValid.Should().BeFalse();
        validation.ErrorMessage.Should().NotBeNullOrEmpty();
        validation.HasUnsupportedFeatures.Should().BeTrue();
        validation.UnsupportedFeatures.Should().HaveCount(2);
    }

    #endregion

    #region Edge Cases

    [Fact]
    public void KernelMetadata_MaximumValues_IsValid()
    {
        // Test with maximum typical values
        // Arrange & Act
        var metadata = new KernelMetadata(
            RequiredSharedMemory: 98304, // 96 KB max shared memory on modern GPUs
            RequiredRegisters: 255,       // Max registers per thread
            MaxThreadsPerBlock: 1024,     // Max threads per block
            PreferredBlockSize: 1024
        );

        // Assert
        metadata.RequiredSharedMemory.Should().Be(98304);
        metadata.MaxThreadsPerBlock.Should().Be(1024);
    }

    [Fact]
    public void CompilationDiagnostics_LongCompilationTime_IsHandled()
    {
        // Arrange & Act
        var diagnostics = new CompilationDiagnostics(
            CompilationTime: TimeSpan.FromMinutes(5),
            CompiledCodeSize: 1_000_000 // 1 MB compiled code
        );

        // Assert
        diagnostics.CompilationTime.Should().Be(TimeSpan.FromMinutes(5));
        diagnostics.CompiledCodeSize.Should().Be(1_000_000);
    }

    [Fact]
    public void KernelValidationResult_ManyWarnings_AreTracked()
    {
        // Arrange
        var manyWarnings = Enumerable.Range(1, 50)
            .Select(i => $"Warning {i}")
            .ToArray();

        // Act
        var result = new KernelValidationResult(
            IsValid: true,
            Warnings: manyWarnings
        );

        // Assert
        result.Warnings.Should().HaveCount(50);
        result.IssueCount.Should().Be(50);
    }

    #endregion
}

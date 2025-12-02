// Orleans.GpuBridge - GPU-native distributed computing for Microsoft Orleans
// Copyright (c) 2025 Michael Ivertowski

using Orleans.GpuBridge.HealthChecks.Exceptions;

namespace Orleans.GpuBridge.HealthChecks.Tests;

/// <summary>
/// Tests for GPU-specific exception classes.
/// </summary>
public class ExceptionTests
{
    [Fact]
    public void GpuOperationException_ShouldSetMessage()
    {
        // Arrange
        var message = "GPU operation failed";

        // Act
        var exception = new GpuOperationException(message);

        // Assert
        exception.Message.Should().Be(message);
    }

    [Fact]
    public void GpuOperationException_ShouldSetInnerException()
    {
        // Arrange
        var message = "GPU operation failed";
        var innerException = new InvalidOperationException("Inner error");

        // Act
        var exception = new GpuOperationException(message, innerException);

        // Assert
        exception.Message.Should().Be(message);
        exception.InnerException.Should().Be(innerException);
    }

    [Fact]
    public void GpuKernelException_ShouldSetKernelName()
    {
        // Arrange
        var kernelName = "vectorAdd";
        var message = "Kernel execution failed";

        // Act
        var exception = new GpuKernelException(kernelName, message);

        // Assert
        exception.KernelName.Should().Be(kernelName);
        exception.Message.Should().Contain(kernelName);
        exception.Message.Should().Contain(message);
    }

    [Fact]
    public void GpuDeviceException_ShouldSetDeviceIndex()
    {
        // Arrange
        var deviceIndex = 2;
        var message = "Device not responding";

        // Act
        var exception = new GpuDeviceException(deviceIndex, message);

        // Assert
        exception.DeviceIndex.Should().Be(deviceIndex);
        exception.Message.Should().Contain(deviceIndex.ToString());
        exception.Message.Should().Contain(message);
    }

    [Fact]
    public void GpuMemoryException_ShouldSetIsTransient()
    {
        // Arrange
        var message = "Out of memory";
        var isTransient = true;

        // Act
        var exception = new GpuMemoryException(message, isTransient);

        // Assert
        exception.Message.Should().Be(message);
        exception.IsTransient.Should().BeTrue();
    }

    [Fact]
    public void GpuMemoryException_ShouldDefaultToTransient()
    {
        // Arrange
        var message = "Memory allocation failed";

        // Act
        var exception = new GpuMemoryException(message);

        // Assert
        exception.IsTransient.Should().BeTrue();
    }

    [Fact]
    public void GpuMemoryException_NonTransient_ShouldBeSetCorrectly()
    {
        // Arrange
        var message = "Invalid memory address";

        // Act
        var exception = new GpuMemoryException(message, isTransient: false);

        // Assert
        exception.IsTransient.Should().BeFalse();
    }

    [Fact]
    public void AllExceptions_ShouldInheritFromGpuOperationException()
    {
        // Arrange & Act
        var kernelException = new GpuKernelException("kernel", "error");
        var deviceException = new GpuDeviceException(0, "error");
        var memoryException = new GpuMemoryException("error");

        // Assert
        kernelException.Should().BeAssignableTo<GpuOperationException>();
        deviceException.Should().BeAssignableTo<GpuOperationException>();
        memoryException.Should().BeAssignableTo<GpuOperationException>();
    }
}

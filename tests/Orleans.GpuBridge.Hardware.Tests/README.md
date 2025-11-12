# Orleans.GpuBridge.Hardware.Tests

Hardware-dependent tests for Orleans.GpuBridge.Core that verify CUDA GPU availability and capabilities.

## Overview

This test suite verifies GPU functionality on actual hardware through system-level checks. All tests use the `SkippableFact` attribute to automatically skip if no compatible GPU is detected.

**Current Status**: These tests perform system-level GPU detection using `nvidia-smi`. Full DotCompute integration tests are marked as pending and will be implemented once DotCompute.Backends.CUDA integration is complete.

## Test Categories

### CudaBackendTests
- **CudaRuntime_ShouldBeAvailable**: Verifies CUDA runtime libraries are installed
- **NvidiaSmi_ShouldDetectGpu**: Tests GPU detection via nvidia-smi utility
- **Nvcc_ShouldBeAvailable**: Verifies CUDA compiler toolkit installation
- **DotComputeCuda_Integration_Pending**: Placeholder for future DotCompute integration

### CudaMemoryTests
- **GpuMemory_ShouldHaveSufficientCapacity**: Verifies GPU has >2GB memory
- **GpuMemory_ShouldHaveFreeMemory**: Checks for available GPU memory
- **GpuMemory_Allocation_Integration_Pending**: Placeholder for future memory tests

### CudaPerformanceTests
- **GpuPerformance_ShouldHaveAcceptableClockSpeed**: Verifies GPU clock speeds
- **GpuPerformance_ShouldHaveModernComputeCapability**: Tests for compute capability >= 3.5
- **GpuPerformance_KernelExecution_Integration_Pending**: Placeholder for performance benchmarks

## Requirements

### Hardware
- NVIDIA RTX GPU (or any CUDA-compatible GPU)
- CUDA Toolkit installed (version 12.0 or later recommended)
- Minimum 2GB GPU memory
- `nvidia-smi` utility available in system PATH

### Software
- .NET 9.0 SDK
- xUnit test framework
- FluentAssertions for test assertions

## Running Tests

### Run all hardware tests
```bash
cd tests/Orleans.GpuBridge.Hardware.Tests
dotnet test
```

### Run specific test class
```bash
dotnet test --filter "FullyQualifiedName~CudaBackendTests"
dotnet test --filter "FullyQualifiedName~CudaMemoryTests"
dotnet test --filter "FullyQualifiedName~CudaPerformanceTests"
```

### Run with detailed output
```bash
dotnet test --verbosity detailed
```

### Skip hardware tests
If you want to run tests but skip hardware-dependent ones:
```bash
# Hardware tests will automatically skip if no GPU is found
dotnet test
```

## Expected Results

### With RTX GPU Available
Tests should pass with the following characteristics:
- **GPU Detected**: nvidia-smi returns valid GPU information
- **Memory**: >2GB total GPU memory
- **Compute Capability**: 7.5+ (Turing) or 8.0+ (Ampere/Ada) for RTX cards
- **Clock Speed**: >300 MHz (typical modern GPUs run at 1-2 GHz)
- **Free Memory**: At least 100MB available

### Without GPU
All tests will be skipped with messages like:
- "nvidia-smi not found on system"
- "CUDA runtime library not found on system"

## Troubleshooting

### Tests Skip Even With GPU Present

1. **Verify CUDA drivers are installed:**
```bash
nvidia-smi
```

2. **Check CUDA Toolkit version:**
```bash
nvcc --version
```

3. **Verify nvidia-smi is in PATH:**
```bash
which nvidia-smi  # Linux/Mac
where nvidia-smi  # Windows
```

### nvidia-smi Not Found

Install NVIDIA drivers:
```bash
# Ubuntu/Debian
sudo apt-get install nvidia-driver-535  # or latest version

# Check installation
nvidia-smi
```

### CUDA Toolkit Not Installed

Download and install from: https://developer.nvidia.com/cuda-downloads

Verify installation:
```bash
nvcc --version
```

## Performance Baselines

These are expected characteristics for various RTX cards:

| GPU Model | Memory | Compute Cap | Typical Clock | CUDA Cores |
|-----------|--------|-------------|---------------|------------|
| RTX 2060  | 6 GB   | 7.5        | 1365-1680 MHz | 1920       |
| RTX 3060  | 12 GB  | 8.6        | 1320-1777 MHz | 3584       |
| RTX 4060  | 8 GB   | 8.9        | 1830-2535 MHz | 3072       |
| RTX 3080  | 10 GB  | 8.6        | 1440-1710 MHz | 8704       |
| RTX 4080  | 16 GB  | 8.9        | 2205-2505 MHz | 9728       |

## Future Development

The following tests are planned for implementation once DotCompute integration is complete:

### Memory Tests
- GPU memory allocation and deallocation
- Host-to-device memory transfers
- Device-to-host memory transfers
- Device-to-device copies
- Memory pressure handling
- Async operations with streams

### Performance Benchmarks
- Memory bandwidth measurements
- Kernel launch overhead (target: <100μs)
- Concurrent kernel execution
- Compute throughput (GFLOPS)
- Matrix multiplication performance
- Vector operations performance

### Integration Tests
- Kernel compilation
- Multiple kernel execution
- Stream creation and synchronization
- Error handling and recovery

## Contributing

When adding new hardware tests:
1. Always use `[SkippableFact]` or `[SkippableTheory]`
2. Provide clear skip messages for missing requirements
3. Document expected performance characteristics
4. Include error handling for system command failures
5. Update this README with new test descriptions

## Related Documentation

- [Orleans.GpuBridge Documentation](../../docs/)
- [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
- [nvidia-smi Documentation](https://developer.nvidia.com/nvidia-system-management-interface)
- [xUnit Documentation](https://xunit.net/)

## License

Copyright © 2025 Orleans.GpuBridge.Core. All rights reserved.

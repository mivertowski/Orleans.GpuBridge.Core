# Orleans.GpuBridge.Core

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![.NET 9.0](https://img.shields.io/badge/.NET-9.0-purple)](https://dotnet.microsoft.com/download/dotnet/9.0)

GPU acceleration bridge for the Orleans distributed computing framework.

## Project Status

**Version**: 0.1.0  
**Stage**: Early Development  
**Production Ready**: No

This project provides GPU compute abstractions for Orleans grains. While the architecture is in place, actual GPU execution is not yet implemented. All operations currently fall back to CPU execution.

## Current Implementation

### Working Components
- Core abstractions and interfaces
- Orleans grain integration
- CPU fallback for all operations
- Basic health monitoring
- Memory management interfaces
- Pipeline execution model

### Not Yet Implemented
- Actual GPU kernel execution
- Hardware acceleration
- Performance optimizations
- Comprehensive testing
- Production hardening

## Architecture

```
src/
├── Orleans.GpuBridge.Abstractions/     # Core interfaces
├── Orleans.GpuBridge.Runtime/          # Runtime orchestration
├── Orleans.GpuBridge.Grains/          # Orleans grain implementations
├── Orleans.GpuBridge.BridgeFX/        # Pipeline API
├── Orleans.GpuBridge.Backends.ILGPU/  # ILGPU backend (partial)
└── Orleans.GpuBridge.Backends.DotCompute/ # DotCompute backend (pending)
```

## Requirements

- .NET 9.0 SDK
- Orleans 9.0+
- Windows, Linux, or macOS (64-bit)

## Building from Source

```bash
git clone https://github.com/mivertowski/GpuBridgeCore.git
cd Orleans.GpuBridge.Core
dotnet build
```

Note: Some DotCompute package references may fail to restore as they are not yet published.

## Basic Usage

```csharp
// Service registration
services.AddGpuBridge(options =>
{
    options.PreferGpu = true;  // Currently ignored - always uses CPU
});

// Grain usage
[GpuAccelerated]
public class ComputeGrain : Grain, IComputeGrain
{
    private readonly IGpuBridge _gpuBridge;
    
    public async Task<float[]> ProcessAsync(float[] data)
    {
        // Executes on CPU regardless of configuration
        var kernel = await _gpuBridge.GetKernelAsync<float[], float[]>("process");
        return await kernel.ExecuteAsync(data);
    }
}
```

## Supported Backends (Planned)

| Backend | Status | Notes |
|---------|--------|-------|
| CPU | ✓ Working | Default fallback |
| CUDA | ✗ Planned | Requires NVIDIA GPU |
| OpenCL | ✗ Planned | Cross-platform |
| DirectCompute | ✗ Planned | Windows only |
| Metal | ✗ Planned | macOS only |
| Vulkan | ✗ Planned | Modern cross-platform |

Currently, all backends route to CPU execution.

## Performance Expectations

**Current Performance**: No GPU acceleration benefits. CPU execution with abstraction overhead.

**Future Goals**: Actual GPU acceleration for suitable workloads, though performance gains will be highly workload-dependent.

## Known Limitations

- No actual GPU execution
- Incomplete backend implementations
- Limited test coverage
- API surface subject to breaking changes
- Not suitable for production use

## Development Roadmap

- [ ] Implement GPU kernel execution
- [ ] Complete ILGPU integration
- [ ] Add DotCompute support
- [ ] Comprehensive testing
- [ ] Performance optimization
- [ ] Production hardening
- [ ] Documentation

## Contributing

Contributions are welcome, particularly for:
- Backend implementations
- Test coverage
- Documentation
- Bug fixes

Please discuss major changes before implementation.

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.

## Author

Michael Ivertowski

## Dependencies

- [Microsoft Orleans](https://github.com/dotnet/orleans) - Distributed actor framework
- [ILGPU](https://github.com/m4rs-mt/ILGPU) - .NET GPU compiler (partial integration)
- [DotCompute](https://github.com/mivertowski/DotCompute) - GPU framework (pending)

## Disclaimer

This is experimental software in early development. It should not be used in production environments. The API is unstable and will change. Performance characteristics do not currently reflect GPU acceleration capabilities.

For production GPU workloads, consider established alternatives.
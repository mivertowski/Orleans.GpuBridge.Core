# Support

Thank you for using Orleans.GpuBridge.Core! This document describes the available support options.

## Getting Help

### Documentation

Before asking for help, please check these resources:

- **[Documentation](https://github.com/mivertowski/Orleans.GpuBridge.Core/tree/main/docs)**: Comprehensive guides and API documentation
- **[README](README.md)**: Quick start guide and overview
- **[CHANGELOG](CHANGELOG.md)**: Recent changes and release notes
- **[Examples](https://github.com/mivertowski/Orleans.GpuBridge.Core/tree/main/examples)**: Sample code and use cases

### GitHub Issues

For **bug reports** and **feature requests**, please use GitHub Issues:

1. **Search First**: Check if your issue has already been reported
2. **Use Templates**: Use the provided issue templates
3. **Be Specific**: Include version numbers, error messages, and reproduction steps

[Open an Issue](https://github.com/mivertowski/Orleans.GpuBridge.Core/issues/new/choose)

### GitHub Discussions

For **questions**, **ideas**, and **community discussion**:

- **Q&A**: Get help from the community
- **Ideas**: Share feature ideas and suggestions
- **Show and Tell**: Share what you've built with Orleans.GpuBridge.Core

[Start a Discussion](https://github.com/mivertowski/Orleans.GpuBridge.Core/discussions)

## Common Questions

### Q: Which GPU backends are supported?

**A**: Currently supported backends:
- **CUDA** (NVIDIA GPUs) - Primary, best tested
- **CPU Fallback** - Always available

Planned backends:
- OpenCL
- Vulkan Compute
- Metal (macOS)

### Q: Does it work on WSL2?

**A**: Yes, with limitations:
- Basic GPU operations work
- Persistent kernel mode does not work (GPU-PV limitation)
- EventDriven mode is available as a workaround
- Native Linux is recommended for production

See [WSL2 GPU Limitations](docs/articles/wsl2-gpu-limitations.md) for details.

### Q: What Orleans versions are supported?

**A**: Orleans 9.0 and later. We recommend using Orleans 9.2.1+.

### Q: Can I use this in production?

**A**: Version 0.1.0 is suitable for:
- Proof of concept deployments
- Development and testing
- Non-critical production workloads

For mission-critical production use, please evaluate thoroughly and consider:
- GPU driver stability
- Proper error handling
- Monitoring and alerting

### Q: How do I report a security issue?

**A**: Please see [SECURITY.md](SECURITY.md) for responsible disclosure guidelines.

## Troubleshooting

### GPU Not Detected

1. Verify GPU drivers are installed:
   ```bash
   # NVIDIA
   nvidia-smi

   # Check CUDA
   nvcc --version
   ```

2. Check Orleans.GpuBridge logs for device discovery errors

3. Ensure you have the correct backend package installed

### Performance Issues

1. Check batch sizes - too small or too large can hurt performance
2. Monitor GPU memory usage with `nvidia-smi`
3. Use the diagnostic tools in `Orleans.GpuBridge.Diagnostics`
4. Consider memory pooling for repeated operations

### Build Errors

1. Ensure .NET 9.0 SDK is installed
2. Check that all NuGet packages are restored
3. For CUDA, ensure CUDA Toolkit is installed
4. Clean and rebuild: `dotnet clean && dotnet build`

## Contributing

Interested in contributing? See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

Orleans.GpuBridge.Core is licensed under the [MIT License](LICENSE).

---

Can't find what you need? [Open an issue](https://github.com/mivertowski/Orleans.GpuBridge.Core/issues/new) and we'll help!

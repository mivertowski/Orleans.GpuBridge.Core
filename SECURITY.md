# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

We take security vulnerabilities seriously. If you discover a security issue in Orleans.GpuBridge.Core, please report it responsibly.

### How to Report

**DO NOT** open a public GitHub issue for security vulnerabilities.

Instead, please report security issues via:

1. **GitHub Security Advisories** (Preferred)
   - Go to the [Security tab](https://github.com/mivertowski/Orleans.GpuBridge.Core/security) of this repository
   - Click "Report a vulnerability"
   - Fill out the private vulnerability report form

2. **Email**
   - Send details to: **security@mivertowski.dev**
   - Use the subject line: `[SECURITY] Orleans.GpuBridge.Core - Brief Description`

### What to Include

Please include as much of the following information as possible:

- **Description**: A clear description of the vulnerability
- **Impact**: Potential impact and severity assessment
- **Affected Versions**: Which versions are affected
- **Reproduction Steps**: Detailed steps to reproduce the issue
- **Proof of Concept**: If available, a minimal code example
- **Suggested Fix**: If you have ideas for remediation

### Response Timeline

- **Acknowledgment**: Within 72 hours of report receipt
- **Initial Assessment**: Within 7 days
- **Resolution Timeline**: Depends on severity
  - Critical: 1-7 days
  - High: 7-14 days
  - Medium: 14-30 days
  - Low: 30-90 days

### What to Expect

1. **Acknowledgment**: We will acknowledge receipt of your report
2. **Investigation**: We will investigate and determine the impact
3. **Updates**: We will keep you informed of our progress
4. **Credit**: We will credit you in the security advisory (unless you prefer anonymity)
5. **Disclosure**: We will coordinate disclosure timing with you

## Security Best Practices

When using Orleans.GpuBridge.Core in production:

### GPU Memory Security

- **Memory Isolation**: GPU memory is shared across processes. Do not store sensitive data in GPU memory without proper encryption
- **Memory Cleanup**: Ensure sensitive data is cleared from GPU buffers after use
- **Access Control**: Restrict access to GPU resources in multi-tenant environments

### Network Security

- **Orleans Security**: Follow [Orleans security best practices](https://learn.microsoft.com/en-us/dotnet/orleans/implementation/security)
- **TLS**: Enable TLS for Orleans silo-to-silo and client-to-silo communication
- **Authentication**: Use Orleans authentication and authorization features

### Configuration Security

- **Secrets Management**: Never commit secrets or API keys to source control
- **Configuration Validation**: Validate all configuration inputs
- **Least Privilege**: Run with minimum required permissions

### Dependency Security

- **Keep Updated**: Regularly update dependencies to patch known vulnerabilities
- **Vulnerability Scanning**: Use tools like `dotnet list package --vulnerable`
- **SBOM**: Generate Software Bill of Materials for production deployments

## Known Security Considerations

### GPU Driver Vulnerabilities

GPU drivers may have security vulnerabilities. Keep your GPU drivers updated:
- NVIDIA: [Security Bulletins](https://www.nvidia.com/en-us/security/)
- AMD: [Security Updates](https://www.amd.com/en/resources/product-security.html)

### WSL2 Considerations

When running in WSL2:
- GPU-PV layer adds attack surface
- Ensure Windows host is patched
- Be aware of shared memory implications

## Security Acknowledgments

We thank the following individuals for responsibly disclosing security issues:

*No security issues have been reported yet.*

---

Thank you for helping keep Orleans.GpuBridge.Core and its users safe!

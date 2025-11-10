# Orleans.GpuBridge.Core Technical Articles

This section contains in-depth technical articles covering the design, implementation, and usage of Orleans.GpuBridge.Core components.

## Temporal Correctness Series

A comprehensive exploration of temporal correctness mechanisms for GPU-native distributed actors.

### Foundational Concepts

1. **[Introduction to Temporal Correctness](temporal/introduction/README.md)**
   - Overview of temporal ordering in distributed systems
   - Challenges in GPU-accelerated distributed computing
   - Requirements for behavioral analytics on temporal graphs

2. **[Hybrid Logical Clocks](temporal/hlc/README.md)**
   - Theory and implementation of HLC
   - Comparison with physical time and Lamport clocks
   - Integration with Orleans grain lifecycle

3. **[Vector Clocks and Causal Ordering](temporal/vector-clocks/README.md)**
   - Causal dependency tracking across actors
   - Detecting concurrent operations and conflicts
   - Message ordering guarantees

### Advanced Topics

4. **[Temporal Pattern Detection](temporal/pattern-detection/README.md)**
   - Sliding window pattern matching
   - Financial fraud detection patterns
   - Real-time behavioral analytics

5. **[Architecture and Design](temporal/architecture/README.md)**
   - System architecture overview
   - Integration with Orleans and GPU kernels
   - Design decisions and trade-offs

6. **[Performance Characteristics](temporal/performance/README.md)**
   - Benchmarking methodology
   - Performance results and analysis
   - Scalability considerations

## GPU-Native Actors Series

A comprehensive guide to building GPU-accelerated distributed applications with Orleans.GpuBridge.Core.

### Foundational Concepts

1. **[Introduction to GPU-Native Actors](gpu-actors/introduction/README.md)**
   - The GPU-Native Actor paradigm
   - Comparison with traditional GPU programming (CUDA, OpenCL)
   - Benefits over CPU-only actor systems
   - Ring kernels and persistent GPU computation

2. **[Use Cases and Applications](gpu-actors/use-cases/README.md)**
   - Financial services (HFT, risk analytics, fraud detection)
   - Scientific computing (molecular dynamics, weather forecasting)
   - Real-time analytics (stream processing, graph analytics)
   - Gaming and simulation (multiplayer servers, digital twins)
   - Production case studies and performance results

3. **[Developer Experience](gpu-actors/developer-experience/README.md)**
   - C# vs C/C++ for GPU programming
   - Advantages over Python-based solutions
   - Enterprise-grade tooling and debugging
   - Team productivity and maintainability
   - Testing and observability

### Practical Guides

4. **[Getting Started Tutorial](gpu-actors/getting-started/README.md)**
   - Installation and setup
   - Creating your first GPU grain
   - Writing CUDA kernels
   - Orleans cluster configuration
   - Testing and debugging
   - Deployment best practices

5. **[Architecture Overview](gpu-actors/architecture/README.md)**
   - System architecture and components
   - Ring kernels and memory architecture
   - Distribution and scalability
   - Fault tolerance and grain lifecycle
   - Performance optimization
   - Security and observability

## Additional Topics

Coming soon:
- GPU Kernel Integration
- DotCompute Backend Architecture
- Placement Strategies for GPU Grains

## Contributing

Technical articles follow academic writing standards:
- Precise technical language
- Citations for external research
- Diagrams using Mermaid or PlantUML
- Code examples in C# 9+
- Performance data with methodology

## License

Documentation is licensed under CC BY 4.0. Code examples follow the repository license.

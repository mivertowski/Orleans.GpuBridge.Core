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

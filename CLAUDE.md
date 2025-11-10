# Claude Code Configuration - SPARC Development Environment

## 🚨 CRITICAL: CONCURRENT EXECUTION & FILE MANAGEMENT

**ABSOLUTE RULES**:
1. ALL operations MUST be concurrent/parallel in a single message
2. **NEVER save working files, text/mds and tests to the root folder**
3. ALWAYS organize files in appropriate subdirectories
4. **USE CLAUDE CODE'S TASK TOOL** for spawning agents concurrently, not just MCP

### ⚡ GOLDEN RULE: "1 MESSAGE = ALL RELATED OPERATIONS"

**MANDATORY PATTERNS:**
- **TodoWrite**: ALWAYS batch ALL todos in ONE call (5-10+ todos minimum)
- **Task tool (Claude Code)**: ALWAYS spawn ALL agents in ONE message with full instructions
- **File operations**: ALWAYS batch ALL reads/writes/edits in ONE message
- **Bash commands**: ALWAYS batch ALL terminal operations in ONE message
- **Memory operations**: ALWAYS batch ALL memory store/retrieve in ONE message

### 🎯 CRITICAL: Claude Code Task Tool for Agent Execution

**Claude Code's Task tool is the PRIMARY way to spawn agents:**
```javascript
// ✅ CORRECT: Use Claude Code's Task tool for parallel agent execution
[Single Message]:
  Task("Research agent", "Analyze requirements and patterns...", "researcher")
  Task("Coder agent", "Implement core features...", "coder")
  Task("Tester agent", "Create comprehensive tests...", "tester")
  Task("Reviewer agent", "Review code quality...", "reviewer")
  Task("Architect agent", "Design system architecture...", "system-architect")
```

**MCP tools are ONLY for coordination setup:**
- `mcp__claude-flow__swarm_init` - Initialize coordination topology
- `mcp__claude-flow__agent_spawn` - Define agent types for coordination
- `mcp__claude-flow__task_orchestrate` - Orchestrate high-level workflows

### 📁 File Organization Rules

**NEVER save to root folder. Use these directories:**
- `/src` - Source code files
- `/tests` - Test files
- `/docs` - Documentation and markdown files
- `/config` - Configuration files
- `/scripts` - Utility scripts
- `/examples` - Example code

## Project Overview

This project uses SPARC (Specification, Pseudocode, Architecture, Refinement, Completion) methodology with Claude-Flow orchestration for systematic Test-Driven Development.

## SPARC Commands

### Core Commands
- `npx claude-flow sparc modes` - List available modes
- `npx claude-flow sparc run <mode> "<task>"` - Execute specific mode
- `npx claude-flow sparc tdd "<feature>"` - Run complete TDD workflow
- `npx claude-flow sparc info <mode>` - Get mode details

### Batchtools Commands
- `npx claude-flow sparc batch <modes> "<task>"` - Parallel execution
- `npx claude-flow sparc pipeline "<task>"` - Full pipeline processing
- `npx claude-flow sparc concurrent <mode> "<tasks-file>"` - Multi-task processing

### Build Commands
- `npm run build` - Build project
- `npm run test` - Run tests
- `npm run lint` - Linting
- `npm run typecheck` - Type checking

## SPARC Workflow Phases

1. **Specification** - Requirements analysis (`sparc run spec-pseudocode`)
2. **Pseudocode** - Algorithm design (`sparc run spec-pseudocode`)
3. **Architecture** - System design (`sparc run architect`)
4. **Refinement** - TDD implementation (`sparc tdd`)
5. **Completion** - Integration (`sparc run integration`)

## Code Style & Best Practices

- **Modular Design**: Files under 500 lines
- **Environment Safety**: Never hardcode secrets
- **Test-First**: Write tests before implementation
- **Clean Architecture**: Separate concerns
- **Documentation**: Keep updated

## 🚀 Available Agents (54 Total)

### Core Development
`coder`, `reviewer`, `tester`, `planner`, `researcher`

### Swarm Coordination
`hierarchical-coordinator`, `mesh-coordinator`, `adaptive-coordinator`, `collective-intelligence-coordinator`, `swarm-memory-manager`

### Consensus & Distributed
`byzantine-coordinator`, `raft-manager`, `gossip-coordinator`, `consensus-builder`, `crdt-synchronizer`, `quorum-manager`, `security-manager`

### Performance & Optimization
`perf-analyzer`, `performance-benchmarker`, `task-orchestrator`, `memory-coordinator`, `smart-agent`

### GitHub & Repository
`github-modes`, `pr-manager`, `code-review-swarm`, `issue-tracker`, `release-manager`, `workflow-automation`, `project-board-sync`, `repo-architect`, `multi-repo-swarm`

### SPARC Methodology
`sparc-coord`, `sparc-coder`, `specification`, `pseudocode`, `architecture`, `refinement`

### Specialized Development
`backend-dev`, `mobile-dev`, `ml-developer`, `cicd-engineer`, `api-docs`, `system-architect`, `code-analyzer`, `base-template-generator`

### Testing & Validation
`tdd-london-swarm`, `production-validator`

### Migration & Planning
`migration-planner`, `swarm-init`

## 🎯 Claude Code vs MCP Tools

### Claude Code Handles ALL EXECUTION:
- **Task tool**: Spawn and run agents concurrently for actual work
- File operations (Read, Write, Edit, MultiEdit, Glob, Grep)
- Code generation and programming
- Bash commands and system operations
- Implementation work
- Project navigation and analysis
- TodoWrite and task management
- Git operations
- Package management
- Testing and debugging

### MCP Tools ONLY COORDINATE:
- Swarm initialization (topology setup)
- Agent type definitions (coordination patterns)
- Task orchestration (high-level planning)
- Memory management
- Neural features
- Performance tracking
- GitHub integration

**KEY**: MCP coordinates the strategy, Claude Code's Task tool executes with real agents.

## 🚀 Quick Setup

```bash
# Add MCP servers (Claude Flow required, others optional)
claude mcp add claude-flow npx claude-flow@alpha mcp start
claude mcp add ruv-swarm npx ruv-swarm mcp start  # Optional: Enhanced coordination
claude mcp add flow-nexus npx flow-nexus@latest mcp start  # Optional: Cloud features
```

## MCP Tool Categories

### Coordination
`swarm_init`, `agent_spawn`, `task_orchestrate`

### Monitoring
`swarm_status`, `agent_list`, `agent_metrics`, `task_status`, `task_results`

### Memory & Neural
`memory_usage`, `neural_status`, `neural_train`, `neural_patterns`

### GitHub Integration
`github_swarm`, `repo_analyze`, `pr_enhance`, `issue_triage`, `code_review`

### System
`benchmark_run`, `features_detect`, `swarm_monitor`

### Flow-Nexus MCP Tools (Optional Advanced Features)
Flow-Nexus extends MCP capabilities with 70+ cloud-based orchestration tools:

**Key MCP Tool Categories:**
- **Swarm & Agents**: `swarm_init`, `swarm_scale`, `agent_spawn`, `task_orchestrate`
- **Sandboxes**: `sandbox_create`, `sandbox_execute`, `sandbox_upload` (cloud execution)
- **Templates**: `template_list`, `template_deploy` (pre-built project templates)
- **Neural AI**: `neural_train`, `neural_patterns`, `seraphina_chat` (AI assistant)
- **GitHub**: `github_repo_analyze`, `github_pr_manage` (repository management)
- **Real-time**: `execution_stream_subscribe`, `realtime_subscribe` (live monitoring)
- **Storage**: `storage_upload`, `storage_list` (cloud file management)

**Authentication Required:**
- Register: `mcp__flow-nexus__user_register` or `npx flow-nexus@latest register`
- Login: `mcp__flow-nexus__user_login` or `npx flow-nexus@latest login`
- Access 70+ specialized MCP tools for advanced orchestration

## 🚀 Agent Execution Flow with Claude Code

### The Correct Pattern:

1. **Optional**: Use MCP tools to set up coordination topology
2. **REQUIRED**: Use Claude Code's Task tool to spawn agents that do actual work
3. **REQUIRED**: Each agent runs hooks for coordination
4. **REQUIRED**: Batch all operations in single messages

### Example Full-Stack Development:

```javascript
// Single message with all agent spawning via Claude Code's Task tool
[Parallel Agent Execution]:
  Task("Backend Developer", "Build REST API with Express. Use hooks for coordination.", "backend-dev")
  Task("Frontend Developer", "Create React UI. Coordinate with backend via memory.", "coder")
  Task("Database Architect", "Design PostgreSQL schema. Store schema in memory.", "code-analyzer")
  Task("Test Engineer", "Write Jest tests. Check memory for API contracts.", "tester")
  Task("DevOps Engineer", "Setup Docker and CI/CD. Document in memory.", "cicd-engineer")
  Task("Security Auditor", "Review authentication. Report findings via hooks.", "reviewer")
  
  // All todos batched together
  TodoWrite { todos: [...8-10 todos...] }
  
  // All file operations together
  Write "backend/server.js"
  Write "frontend/App.jsx"
  Write "database/schema.sql"
```

## 📋 Agent Coordination Protocol

### Every Agent Spawned via Task Tool MUST:

**1️⃣ BEFORE Work:**
```bash
npx claude-flow@alpha hooks pre-task --description "[task]"
npx claude-flow@alpha hooks session-restore --session-id "swarm-[id]"
```

**2️⃣ DURING Work:**
```bash
npx claude-flow@alpha hooks post-edit --file "[file]" --memory-key "swarm/[agent]/[step]"
npx claude-flow@alpha hooks notify --message "[what was done]"
```

**3️⃣ AFTER Work:**
```bash
npx claude-flow@alpha hooks post-task --task-id "[task]"
npx claude-flow@alpha hooks session-end --export-metrics true
```

## 🎯 Concurrent Execution Examples

### ✅ CORRECT WORKFLOW: MCP Coordinates, Claude Code Executes

```javascript
// Step 1: MCP tools set up coordination (optional, for complex tasks)
[Single Message - Coordination Setup]:
  mcp__claude-flow__swarm_init { topology: "mesh", maxAgents: 6 }
  mcp__claude-flow__agent_spawn { type: "researcher" }
  mcp__claude-flow__agent_spawn { type: "coder" }
  mcp__claude-flow__agent_spawn { type: "tester" }

// Step 2: Claude Code Task tool spawns ACTUAL agents that do the work
[Single Message - Parallel Agent Execution]:
  // Claude Code's Task tool spawns real agents concurrently
  Task("Research agent", "Analyze API requirements and best practices. Check memory for prior decisions.", "researcher")
  Task("Coder agent", "Implement REST endpoints with authentication. Coordinate via hooks.", "coder")
  Task("Database agent", "Design and implement database schema. Store decisions in memory.", "code-analyzer")
  Task("Tester agent", "Create comprehensive test suite with 90% coverage.", "tester")
  Task("Reviewer agent", "Review code quality and security. Document findings.", "reviewer")
  
  // Batch ALL todos in ONE call
  TodoWrite { todos: [
    {id: "1", content: "Research API patterns", status: "in_progress", priority: "high"},
    {id: "2", content: "Design database schema", status: "in_progress", priority: "high"},
    {id: "3", content: "Implement authentication", status: "pending", priority: "high"},
    {id: "4", content: "Build REST endpoints", status: "pending", priority: "high"},
    {id: "5", content: "Write unit tests", status: "pending", priority: "medium"},
    {id: "6", content: "Integration tests", status: "pending", priority: "medium"},
    {id: "7", content: "API documentation", status: "pending", priority: "low"},
    {id: "8", content: "Performance optimization", status: "pending", priority: "low"}
  ]}
  
  // Parallel file operations
  Bash "mkdir -p app/{src,tests,docs,config}"
  Write "app/package.json"
  Write "app/src/server.js"
  Write "app/tests/server.test.js"
  Write "app/docs/API.md"
```

### ❌ WRONG (Multiple Messages):
```javascript
Message 1: mcp__claude-flow__swarm_init
Message 2: Task("agent 1")
Message 3: TodoWrite { todos: [single todo] }
Message 4: Write "file.js"
// This breaks parallel coordination!
```

## Performance Benefits

- **84.8% SWE-Bench solve rate**
- **32.3% token reduction**
- **2.8-4.4x speed improvement**
- **27+ neural models**

## Hooks Integration

### Pre-Operation
- Auto-assign agents by file type
- Validate commands for safety
- Prepare resources automatically
- Optimize topology by complexity
- Cache searches

### Post-Operation
- Auto-format code
- Train neural patterns
- Update memory
- Analyze performance
- Track token usage

### Session Management
- Generate summaries
- Persist state
- Track metrics
- Restore context
- Export workflows

## Advanced Features (v2.0.0)

- 🚀 Automatic Topology Selection
- ⚡ Parallel Execution (2.8-4.4x speed)
- 🧠 Neural Training
- 📊 Bottleneck Analysis
- 🤖 Smart Auto-Spawning
- 🛡️ Self-Healing Workflows
- 💾 Cross-Session Memory
- 🔗 GitHub Integration

## Integration Tips

1. Start with basic swarm init
2. Scale agents gradually
3. Use memory for context
4. Monitor progress regularly
5. Train patterns from success
6. Enable hooks automation
7. Use GitHub tools first

## Support

- Documentation: https://github.com/ruvnet/claude-flow
- Issues: https://github.com/ruvnet/claude-flow/issues
- Flow-Nexus Platform: https://flow-nexus.ruv.io (registration required for cloud features)

---

Remember: **Claude Flow coordinates, Claude Code creates!**

# important-instruction-reminders
Do what has been asked; nothing more, nothing less.
NEVER create files unless they're absolutely necessary for achieving your goal.
ALWAYS prefer editing an existing file to creating a new one.
NEVER proactively create documentation files (*.md) or README files. Only create documentation files if explicitly requested by the User.
Never save working files, text/mds and tests to the root folder.
## Project Overview

Orleans.GpuBridge.Core is a .NET library that enables **GPU-native distributed computing** for Microsoft Orleans. This project represents a **paradigm shift** from traditional CPU-based actor systems to actors that live permanently on the GPU.

### The GPU-Native Actor Paradigm

**Revolutionary Concept**: Instead of CPU actors that offload work to GPU, Orleans.GpuBridge.Core enables actors that **reside entirely in GPU memory** and process messages at sub-microsecond latencies.

**Key Technologies**:
- **Ring Kernels**: Persistent GPU kernels running infinite dispatch loops (launched once, run forever)
- **Temporal Alignment on GPU**: HLC and Vector Clocks maintained entirely on GPU (20ns vs 50ns CPU)
- **GPU-to-GPU Messaging**: Actors communicate without CPU involvement (100-500ns latency)
- **Hypergraph Actors**: Multi-way relationships with GPU-accelerated pattern matching
- **Knowledge Organisms**: Emergent intelligence from actor interactions

**Performance Breakthrough**:
- Message latency: 100-500ns (GPU-native) vs 10-100μs (CPU actors) = **20-200× faster**
- Throughput: 2M messages/s/actor vs 15K messages/s = **133× improvement**
- Memory bandwidth: 1,935 GB/s (on-die GPU) vs 200 GB/s (CPU) = **10× improvement**
- Temporal ordering: 20ns (GPU) vs 50ns (CPU) = **2.5× faster**

This enables entirely new application classes:
- Real-time hypergraph analytics (<100μs pattern detection)
- Digital twins as living entities (physics-accurate at 100-500ns latency)
- Temporal pattern detection (fraud detection with causal ordering)
- Knowledge organisms (emergent intelligence from distributed actors)

## Architecture Overview

### Core Components

1. **Orleans.GpuBridge.Abstractions** - Defines core interfaces and contracts:
   - `IGpuBridge` - Main bridge interface for GPU operations
   - `IGpuKernel<TIn,TOut>` - Kernel execution contract
   - `[GpuAccelerated]` attribute for grain marking
   - Configuration via `GpuBridgeOptions`
   - Temporal clock interfaces (HLC, Vector Clocks)

2. **Orleans.GpuBridge.Runtime** - Runtime implementation:
   - `KernelCatalog` - Manages kernel registration and execution
   - `DeviceBroker` - GPU device management
   - DI integration via `AddGpuBridge()` extension method
   - Placement strategies for GPU-aware grain placement
   - Ring kernel lifecycle management

3. **Orleans.GpuBridge.BridgeFX** - High-level pipeline API:
   - `GpuPipeline<TIn,TOut>` - Fluent API for batch processing
   - Automatic partitioning and result aggregation
   - Temporal pattern detection pipelines

4. **Orleans.GpuBridge.Grains** - Orleans grain implementations:
   - `GpuBatchGrain` - Batch processing grain (GPU-offload model)
   - `GpuResidentGrain` - GPU-resident data grain (GPU-native model)
   - `GpuStreamGrain` - Stream processing grain
   - `HypergraphVertexGrain` - Vertex actor with GPU-native state
   - `HypergraphHyperedgeGrain` - Hyperedge actor for multi-way relationships

5. **GPU-Native Actor Components**:
   - Ring kernel dispatch loops (persistent GPU threads)
   - GPU-resident message queues (lock-free on GPU)
   - Temporal clock state (HLC/Vector Clocks in GPU memory)
   - Hypergraph structures (CSR format in GPU memory)

### Key Design Patterns

**Service Registration Pattern:**
```csharp
services.AddGpuBridge(options => options.PreferGpu = true)
        .AddKernel(k => k.Id("kernels/VectorAdd")
                        .In<float[]>().Out<float>()
                        .FromFactory(sp => new CustomKernel()));
```

**Pipeline Execution Pattern:**
```csharp
var results = await GpuPipeline<TIn,TOut>
    .For(grainFactory, "kernel-id")
    .WithBatchSize(batchSize)
    .ExecuteAsync(data);
```

## Development Commands

Since the project currently lacks build configuration files, you'll need to create them first:

### Initial Setup (Required)
```bash
# Create solution file
dotnet new sln -n Orleans.GpuBridge.Core

# Create project files for each component
cd src/Orleans.GpuBridge.Abstractions
dotnet new classlib -n Orleans.GpuBridge.Abstractions -f net9.0
cd ../Orleans.GpuBridge.Runtime
dotnet new classlib -n Orleans.GpuBridge.Runtime -f net9.0
cd ../Orleans.GpuBridge.BridgeFX
dotnet new classlib -n Orleans.GpuBridge.BridgeFX -f net9.0
cd ../Orleans.GpuBridge.Grains
dotnet new classlib -n Orleans.GpuBridge.Grains -f net9.0

# Add projects to solution
cd ../..
dotnet sln add src/**/*.csproj
```

### Standard Commands (after setup)
```bash
# Build the solution
dotnet build

# Run tests (when added)
dotnet test

# Create NuGet packages
dotnet pack

# Clean build artifacts
dotnet clean
```

## Project Status and Implementation Notes

### Current State
- **Core abstractions**: Complete
- **Runtime infrastructure**: Basic implementation with CPU fallbacks
- **GPU execution**: Not yet implemented (all kernels use CPU fallback)
- **Testing**: No test projects exist yet
- **Build configuration**: Missing .csproj and .sln files

### CPU Fallback System
All GPU kernels currently fall back to CPU implementations. The `KernelCatalog` manages this through:
```csharp
public async Task<TOut> ExecuteAsync<TIn, TOut>(string kernelId, TIn input)
{
    // Currently always uses CPU fallback
    var kernel = ResolveKernel<TIn, TOut>(kernelId);
    return await kernel.ExecuteAsync(input);
}
```

### GPU-Native Actor Implementation
The project implements two deployment models:

**GPU-Offload Model** (Traditional):
- CPU actors offload compute to GPU
- Kernel launch overhead: ~10-50μs
- Best for: Batch processing, infrequent GPU usage

**GPU-Native Model** (Revolutionary):
- Actors live permanently in GPU memory
- Ring kernels process messages on GPU
- Zero kernel launch overhead
- Sub-microsecond latency: 100-500ns
- Best for: High-frequency messaging, temporal graphs, real-time analytics

**Implementation Status**:
- Ring kernel infrastructure: ✅ Implemented
- GPU-resident message queues: ✅ Implemented
- Temporal alignment on GPU: ✅ Implemented (HLC, Vector Clocks)
- Hypergraph actors: ✅ Implemented
- DotCompute backend: 🚧 In progress
- Queue-depth aware placement: 🚧 In progress
- GPUDirect Storage: 📋 Planned

## Key Files to Understand

### Service Registration and DI
- `src/Orleans.GpuBridge.Runtime/ServiceCollectionExtensions.cs` - Entry point for service configuration
- `src/Orleans.GpuBridge.Runtime/KernelCatalog.cs` - Kernel registration and resolution

### Core Interfaces
- `src/Orleans.GpuBridge.Abstractions/IGpuBridge.cs` - Main bridge contract
- `src/Orleans.GpuBridge.Abstractions/IGpuKernel.cs` - Kernel execution interface

### High-Level API
- `src/Orleans.GpuBridge.BridgeFX/GpuPipeline.cs` - Fluent pipeline API implementation

### Orleans Integration
- `src/Orleans.GpuBridge.Grains/GpuBatchGrain.cs` - Primary grain for batch processing
- `src/Orleans.GpuBridge.Runtime/Placement/*.cs` - GPU-aware placement strategies

## Development Priorities

When implementing new features:

1. **Maintain CPU fallback**: Always provide CPU implementations for GPU kernels
2. **Follow Orleans patterns**: Use grain state, activation lifecycle properly
3. **Async throughout**: All GPU operations should be async
4. **Batch optimization**: Design for batch processing efficiency
5. **Resource management**: Proper GPU resource cleanup in `Dispose()` methods

## Testing Strategy

When adding tests:
```bash
# Create test projects
dotnet new xunit -n Orleans.GpuBridge.Abstractions.Tests
dotnet new xunit -n Orleans.GpuBridge.Runtime.Tests

# Add Orleans TestingHost for grain testing
dotnet add package Microsoft.Orleans.TestingHost
```

Test priorities:
1. Kernel registration and resolution
2. CPU fallback execution
3. Pipeline batch processing
4. Grain activation and placement
5. Resource cleanup

## Documentation Structure

- `docs/starter-kit/DESIGN.md` - Core architecture overview
- `docs/starter-kit/ABSTRACTION.md` - BridgeFX pipeline details
- `docs/starter-kit/KERNELS.md` - Kernel implementation guide
- `docs/starter-kit/OPERATIONS.md` - Operational considerations
- `docs/starter-kit/ROADMAP.md` - Future development plans

## Dependencies

Key NuGet packages to add when creating .csproj files:
- Microsoft.Orleans.Core
- Microsoft.Orleans.Runtime
- Microsoft.Extensions.DependencyInjection
- Microsoft.Extensions.Hosting
- Microsoft.Extensions.Logging
- Microsoft.Extensions.Options

## Important Implementation Considerations

1. **Thread Safety**: All kernel implementations must be thread-safe
2. **Memory Management**: GPU memory must be properly managed and released
3. **Error Handling**: GPU operations can fail - proper fallback to CPU required
4. **Performance**: Batch size optimization is critical for GPU efficiency
5. **Orleans Constraints**: Respect grain single-threaded execution model

QUALITY CODE USING LATEST .NET9 PATTERNS ONLY. NO SHORT CUTS. NO COMPROMISES. ALWAYS PRODUCTION GRADE CODE!
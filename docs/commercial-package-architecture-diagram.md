# Orleans.GpuBridge.Enterprise - Architecture Diagrams

**Visual Reference Guide**
**Version:** 1.0
**Date:** 2025-11-11

---

## 1. Overall Package Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│                  ORLEANS.GPUBRIDGE.ENTERPRISE                       │
│                      (Commercial Add-on)                            │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                                 │
                    ┌────────────┴────────────┐
                    │                         │
        ┌───────────▼──────────┐   ┌──────────▼──────────┐
        │  KERNEL LIBRARIES    │   │  GRAIN LIBRARIES    │
        │   (Domain Logic)     │   │  (Actor Runtime)    │
        └───────────┬──────────┘   └──────────┬──────────┘
                    │                         │
        ┌───────────┴────────────┬────────────┴───────────┐
        │                        │                        │
   ┌────▼─────┐           ┌─────▼────┐           ┌──────▼──────┐
   │ Process  │           │ Banking  │           │ Accounting  │
   │Intelligence│         │          │           │             │
   └────┬─────┘           └─────┬────┘           └──────┬──────┘
        │                        │                       │
        └────────────┬───────────┴───────┬───────────────┘
                     │                   │
            ┌────────▼─────────┐  ┌──────▼─────────┐
            │  ENTERPRISE      │  │  LICENSING &   │
            │  RUNTIME         │  │  TELEMETRY     │
            └────────┬─────────┘  └──────┬─────────┘
                     │                   │
                     └────────┬──────────┘
                              │
            ┌─────────────────▼─────────────────┐
            │                                   │
            │  ORLEANS.GPUBRIDGE.CORE           │
            │  (Open Source Foundation)         │
            │                                   │
            │  ┌─────────┐  ┌──────────┐       │
            │  │Abstractions│ │ Runtime │       │
            │  └─────────┘  └──────────┘       │
            │  ┌─────────┐  ┌──────────┐       │
            │  │ Grains  │  │ Backends │       │
            │  └─────────┘  └──────────┘       │
            │                                   │
            └───────────────────────────────────┘
                              │
            ┌─────────────────▼─────────────────┐
            │  MICROSOFT ORLEANS 9.0             │
            │  ┌──────┐ ┌─────────┐ ┌────────┐  │
            │  │ Core │ │ Runtime │ │ Hosting│  │
            │  └──────┘ └─────────┘ └────────┘  │
            └────────────────────────────────────┘
                              │
            ┌─────────────────▼─────────────────┐
            │  GPU COMPUTE BACKENDS              │
            │  ┌────────────┐  ┌──────────────┐ │
            │  │ DotCompute │  │ ILGPU (OSS)  │ │
            │  │(Commercial)│  │              │ │
            │  └────────────┘  └──────────────┘ │
            └────────────────────────────────────┘
```

---

## 2. Domain Package Structure

```
┌────────────────────────────────────────────────────────────────┐
│                    DOMAIN PACKAGE                              │
│              (e.g., Banking, Accounting)                       │
└────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              │                               │
    ┌─────────▼─────────┐          ┌─────────▼─────────┐
    │ KERNEL LIBRARY    │          │ GRAIN LIBRARY     │
    │                   │          │                   │
    │ ┌───────────────┐ │          │ ┌───────────────┐ │
    │ │ FraudDetection│ │          │ │ AccountGrain  │ │
    │ │    Kernel     │ │          │ │               │ │
    │ └───────────────┘ │          │ └───────┬───────┘ │
    │ ┌───────────────┐ │          │         │         │
    │ │   Payment     │ │          │    Uses │         │
    │ │ Processing    │◄├──────────┼─────────┘         │
    │ └───────────────┘ │          │                   │
    │ ┌───────────────┐ │          │ ┌───────────────┐ │
    │ │ Risk Analysis │ │          │ │TransactionGrain││
    │ │    Kernel     │ │          │ │               │ │
    │ └───────────────┘ │          │ └───────┬───────┘ │
    └───────────────────┘          │         │         │
                                   │    Uses │         │
                                   │         │         │
                                   └─────────┼─────────┘
                                             │
                    ┌────────────────────────┴───────────────┐
                    │                                        │
          ┌─────────▼─────────┐                  ┌──────────▼──────────┐
          │ Enterprise        │                  │ Enterprise          │
          │ Kernel Base       │                  │ Grain Base          │
          │                   │                  │                     │
          │ • Licensing       │                  │ • Licensing         │
          │ • Telemetry       │                  │ • Telemetry         │
          │ • GPU Execution   │                  │ • GPU-Resident      │
          │ • Error Handling  │                  │ • Temporal HLC      │
          └───────────────────┘                  └─────────────────────┘
```

---

## 3. Kernel Execution Flow

```
┌──────────────┐
│ Grain Method │
│    Call      │
└──────┬───────┘
       │
       │ 1. Resolve kernel from catalog
       ▼
┌──────────────────────────┐
│   Kernel Catalog         │
│                          │
│  • Discovery             │
│  • Registration          │
│  • License Validation    │
└──────┬───────────────────┘
       │
       │ 2. Create kernel instance
       ▼
┌──────────────────────────┐
│ Enterprise Kernel        │
│                          │
│  • Validate License      │
│  • Start Telemetry       │
└──────┬───────────────────┘
       │
       │ 3. Choose execution path
       ▼
       ├─────────────────┬─────────────────┐
       │                 │                 │
       │ GPU-Native      │ GPU-Offload     │ CPU Fallback
       ▼                 ▼                 ▼
┌─────────────┐   ┌──────────────┐   ┌──────────┐
│Ring Kernel  │   │Batch Kernel  │   │CPU Exec  │
│             │   │              │   │          │
│• Persistent │   │• Launch GPU  │   │• CPU     │
│  GPU thread │   │  kernel      │   │  compute │
│• 100-500ns  │   │• ~10-50μs    │   │• ~100μs  │
│  latency    │   │  latency     │   │  latency │
└─────┬───────┘   └──────┬───────┘   └────┬─────┘
      │                  │                 │
      └──────────────────┴─────────────────┘
                         │
                         │ 4. Return results
                         ▼
                  ┌──────────────┐
                  │   Results    │
                  │ (IAsyncEnum) │
                  └──────────────┘
```

---

## 4. Licensing Flow

```
┌─────────────────┐
│ Application     │
│   Startup       │
└────────┬────────┘
         │
         │ 1. Load configuration
         ▼
┌─────────────────────────┐
│ License Configuration   │
│                         │
│ • License Key           │
│ • Server URL            │
│ • Offline Mode          │
└────────┬────────────────┘
         │
         │ 2. Initialize validator
         ▼
┌──────────────────────────┐
│  License Validator       │
│                          │
│  Load License            │
└────────┬─────────────────┘
         │
         ├─────────────────┬──────────────┐
         │                 │              │
         │ Online          │ Offline      │ File-based
         ▼                 ▼              ▼
┌──────────────┐   ┌──────────────┐   ┌──────────┐
│License Server│   │Cached License│   │License   │
│              │   │              │   │  File    │
│• Validate    │   │• Load cache  │   │• Load    │
│• Telemetry   │   │• Verify sig  │   │• Verify  │
└──────┬───────┘   └──────┬───────┘   └────┬─────┘
       │                  │                 │
       └──────────────────┴─────────────────┘
                          │
                          │ 3. Validate features
                          ▼
                   ┌─────────────────┐
                   │ Feature Check   │
                   │                 │
                   │ • Banking.*     │
                   │ • Accounting.*  │
                   │ • Expiration    │
                   │ • Device limits │
                   └────────┬────────┘
                            │
              ┌─────────────┴─────────────┐
              │                           │
              ▼ Valid                     ▼ Invalid
       ┌─────────────┐             ┌──────────────┐
       │ Allow       │             │ Throw        │
       │ Execution   │             │ Exception    │
       └─────────────┘             └──────────────┘
```

---

## 5. GPU-Native Actor Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                         CPU MEMORY                             │
│                                                                │
│  ┌──────────────────┐         ┌──────────────────┐            │
│  │ Orleans Silo     │         │ Enterprise Grain │            │
│  │                  │         │                  │            │
│  │ • Grain Registry │◄────────┤ • Grain State    │            │
│  │ • Messaging      │         │ • HLC Timestamp  │            │
│  └──────────────────┘         └────────┬─────────┘            │
│                                        │                      │
└────────────────────────────────────────┼──────────────────────┘
                                         │
                         PCIe Bus        │
                                         │
┌────────────────────────────────────────┼──────────────────────┐
│                         GPU MEMORY     │                      │
│                                        │                      │
│  ┌────────────────────────────────────▼──────────────────┐   │
│  │              RING KERNEL (Persistent GPU Thread)      │   │
│  │                                                        │   │
│  │  ┌──────────────┐   ┌──────────────┐   ┌──────────┐  │   │
│  │  │ Message      │   │ Actor State  │   │ HLC      │  │   │
│  │  │ Queue        │   │              │   │ Clock    │  │   │
│  │  │ (Lock-free)  │   │ • Balance    │   │          │  │   │
│  │  │              │   │ • History    │   │ 20ns     │  │   │
│  │  │ 65K entries  │   │ • Metadata   │   │ precision│  │   │
│  │  └──────┬───────┘   └──────┬───────┘   └────┬─────┘  │   │
│  │         │                  │                 │        │   │
│  │         └──────────────────┴─────────────────┘        │   │
│  │                            │                          │   │
│  │         ┌──────────────────▼──────────────────┐       │   │
│  │         │  Message Processing Loop            │       │   │
│  │         │                                     │       │   │
│  │         │  while(true) {                      │       │   │
│  │         │    msg = dequeue();                 │       │   │
│  │         │    update_hlc(msg.timestamp);       │       │   │
│  │         │    process(msg);                    │       │   │
│  │         │    update_state();                  │       │   │
│  │         │  }                                  │       │   │
│  │         │                                     │       │   │
│  │         │  Latency: 100-500ns                 │       │   │
│  │         └─────────────────────────────────────┘       │   │
│  └────────────────────────────────────────────────────────┘   │
│                                                                │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │         DOMAIN KERNEL (e.g., Fraud Detection)           │ │
│  │                                                          │ │
│  │  • Pattern matching tables                              │ │
│  │  • ML model weights                                     │ │
│  │  • Temporal pattern buffers                             │ │
│  │  • Result buffers                                       │ │
│  └──────────────────────────────────────────────────────────┘ │
│                                                                │
└────────────────────────────────────────────────────────────────┘
                      NVIDIA RTX 4090
                      (24GB VRAM, 1,935 GB/s bandwidth)
```

---

## 6. Domain Kernel Discovery and Registration

```
┌────────────────────────────────────────────────────────┐
│              APPLICATION STARTUP                       │
└────────────────┬───────────────────────────────────────┘
                 │
                 │ 1. Configure enterprise
                 ▼
┌────────────────────────────────────────────────────────┐
│  services.AddGpuBridgeEnterprise()                     │
│    .EnableDomain("Banking")                            │
│    .EnableDomain("Accounting")                         │
└────────────────┬───────────────────────────────────────┘
                 │
                 │ 2. Discover assemblies
                 ▼
┌────────────────────────────────────────────────────────┐
│         Assembly Discovery                             │
│                                                        │
│  Orleans.GpuBridge.Enterprise.Kernels.Banking.dll      │
│  Orleans.GpuBridge.Enterprise.Kernels.Accounting.dll   │
│  MyApp.CustomKernels.dll                               │
└────────────────┬───────────────────────────────────────┘
                 │
                 │ 3. Scan for [EnterpriseKernel] attributes
                 ▼
┌────────────────────────────────────────────────────────┐
│         Kernel Type Discovery                          │
│                                                        │
│  [EnterpriseKernel("enterprise/banking/fraud-detect")] │
│  class FraudDetectionKernel { ... }                    │
│                                                        │
│  [EnterpriseKernel("custom/banking/my-fraud-detect")]  │
│  class MyFraudKernel { ... }                           │
└────────────────┬───────────────────────────────────────┘
                 │
                 │ 4. Validate licenses
                 ▼
┌────────────────────────────────────────────────────────┐
│         License Validation                             │
│                                                        │
│  FraudDetectionKernel → Banking.FraudDetection ✓       │
│  MyFraudKernel → CustomDomain.MyFraud ✓                │
└────────────────┬───────────────────────────────────────┘
                 │
                 │ 5. Create descriptors
                 ▼
┌────────────────────────────────────────────────────────┐
│         Kernel Catalog Registration                    │
│                                                        │
│  KernelId: "enterprise/banking/fraud-detect"           │
│  Domain: Banking                                       │
│  In: Transaction                                       │
│  Out: FraudScore                                       │
│  Factory: sp => new FraudDetectionKernel(...)          │
└────────────────┬───────────────────────────────────────┘
                 │
                 │ 6. Ready for execution
                 ▼
┌────────────────────────────────────────────────────────┐
│         Runtime Kernel Resolution                      │
│                                                        │
│  grainFactory.GetGrain<IAccountGrain>(id)              │
│    → calls kernel "enterprise/banking/fraud-detect"    │
│    → KernelCatalog.ResolveAsync()                      │
│    → Returns FraudDetectionKernel instance             │
└────────────────────────────────────────────────────────┘
```

---

## 7. Extension Points Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                    CUSTOMER APPLICATION                        │
└────────────────┬───────────────────────────────────────────────┘
                 │
      ┌──────────┴──────────┬────────────────┬──────────────────┐
      │                     │                │                  │
      │ Custom Kernels      │ Custom Grains  │ Custom Domains   │
      ▼                     ▼                ▼                  │
┌─────────────────┐  ┌──────────────┐  ┌────────────────┐      │
│ MyFraudKernel   │  │MyAccountGrain│  │MyHealthcare    │      │
│ extends         │  │extends       │  │Domain          │      │
│EnterpriseGpu    │  │Enterprise    │  │                │      │
│Kernel           │  │Grain         │  │• Kernels       │      │
│                 │  │              │  │• Grains        │      │
│• Custom logic   │  │• Custom      │  │• Config        │      │
│• Inherits:      │  │  business    │  │• License       │      │
│  - Licensing    │  │  logic       │  │  features      │      │
│  - Telemetry    │  │• Inherits:   │  └────────────────┘      │
│  - GPU exec     │  │  - Licensing │                          │
│  - Error        │  │  - Telemetry │                          │
│    handling     │  │  - GPU state │                          │
└─────────────────┘  └──────────────┘                          │
                                                                │
      │                     │                │                  │
      └─────────────────────┴────────────────┴──────────────────┘
                            │
                            │ Registration API
                            ▼
┌────────────────────────────────────────────────────────────────┐
│              ENTERPRISE EXTENSION FRAMEWORK                    │
│                                                                │
│  services.AddGpuBridgeEnterprise()                             │
│    .AddCustomKernel<Transaction, FraudScore>(                  │
│        "custom/banking/fraud",                                 │
│        "Banking",                                              │
│        sp => new MyFraudKernel(...)                            │
│    )                                                           │
│    .AddCustomGrain<MyAccountGrain, IMyAccountGrain>(           │
│        requiredFeatures: ["CustomDomain.Accounts"]             │
│    )                                                           │
│    .AddCustomDomain("Healthcare", config => {                  │
│        config.LicenseFeatures = ["Healthcare.Base"];           │
│        config.KernelAssemblies = [typeof(PatientKernels)       │
│                                   .Assembly];                  │
│    });                                                         │
└────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────────────┐
│            ENTERPRISE RUNTIME INFRASTRUCTURE                   │
│                                                                │
│  • Automatic kernel discovery                                 │
│  • License validation for custom features                     │
│  • Telemetry for custom kernels/grains                        │
│  • GPU execution for custom kernels                           │
│  • Orleans integration                                        │
└────────────────────────────────────────────────────────────────┘
```

---

## 8. Deployment Topology

### Single-Node Development

```
┌────────────────────────────────────────┐
│        Developer Workstation           │
│                                        │
│  ┌──────────────────────────────────┐  │
│  │  Orleans Silo (Localhost)        │  │
│  │                                  │  │
│  │  • GpuBridge.Enterprise          │  │
│  │  • Banking Domain                │  │
│  │  • License: Trial (30 days)      │  │
│  │                                  │  │
│  │  ┌───────────────────────────┐   │  │
│  │  │ GPU: NVIDIA RTX 4090      │   │  │
│  │  │ • Ring kernels active     │   │  │
│  │  │ • 24GB VRAM               │   │  │
│  │  └───────────────────────────┘   │  │
│  └──────────────────────────────────┘  │
└────────────────────────────────────────┘
```

### Production Cluster (Small)

```
┌────────────────────────────────────────────────────────────┐
│                   Orleans Cluster                          │
│                                                            │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   │
│  │  Silo 1      │   │  Silo 2      │   │  Silo 3      │   │
│  │              │   │              │   │              │   │
│  │  GPU: RTX    │   │  GPU: RTX    │   │  GPU: RTX    │   │
│  │  4090 (24GB) │   │  4090 (24GB) │   │  4090 (24GB) │   │
│  │              │   │              │   │              │   │
│  │  Domains:    │   │  Domains:    │   │  Domains:    │   │
│  │  • Banking   │   │  • Banking   │   │  • Banking   │   │
│  │  • Acct      │   │  • Acct      │   │  • Acct      │   │
│  └──────┬───────┘   └──────┬───────┘   └──────┬───────┘   │
│         │                  │                  │           │
│         └──────────────────┴──────────────────┘           │
│                            │                              │
└────────────────────────────┼──────────────────────────────┘
                             │
                    ┌────────▼────────┐
                    │ License Server  │
                    │                 │
                    │ • Validation    │
                    │ • Telemetry     │
                    │ • Feature gates │
                    └─────────────────┘
```

### Production Cluster (Large - Multi-Datacenter)

```
┌─────────────────────────────────────────────────────────────────┐
│                       DATACENTER 1 (East)                       │
│                                                                 │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐   │
│  │  Silo 1-1  │ │  Silo 1-2  │ │  Silo 1-3  │ │  Silo 1-4  │   │
│  │  GPU: A100 │ │  GPU: A100 │ │  GPU: A100 │ │  GPU: A100 │   │
│  │  80GB      │ │  80GB      │ │  80GB      │ │  80GB      │   │
│  └────────────┘ └────────────┘ └────────────┘ └────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                               │
                    ┌──────────┴──────────┐
                    │  Orleans Gossip     │
                    │  + License Server   │
                    └──────────┬──────────┘
                               │
┌─────────────────────────────────────────────────────────────────┐
│                       DATACENTER 2 (West)                       │
│                                                                 │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐   │
│  │  Silo 2-1  │ │  Silo 2-2  │ │  Silo 2-3  │ │  Silo 2-4  │   │
│  │  GPU: A100 │ │  GPU: A100 │ │  GPU: A100 │ │  GPU: A100 │   │
│  │  80GB      │ │  80GB      │ │  80GB      │ │  80GB      │   │
│  └────────────┘ └────────────┘ └────────────┘ └────────────┘   │
└─────────────────────────────────────────────────────────────────┘

License: Enterprise Edition
  • All domains enabled
  • 8 GPUs per DC
  • Unlimited silos (within limit)
  • 24/7 support
```

---

## 9. Data Flow: Real-Time Fraud Detection Example

```
1. Transaction arrives
   │
   ├─► Orleans Grain (IAccountGrain)
   │
   ├─► Resolve fraud detection kernel
   │   └─► KernelCatalog.ResolveAsync<Transaction, FraudScore>(
   │         "enterprise/banking/fraud-detection")
   │
   ├─► Validate license for Banking.FraudDetection
   │   └─► LicenseValidator.ValidateFeatureAsync()
   │
   ├─► Submit to GPU kernel
   │   └─► FraudDetectionKernel.SubmitBatchAsync([transaction])
   │
   ├─► Enqueue to ring kernel (GPU-resident)
   │   │
   │   ▼ GPU Memory
   │   ┌────────────────────────────────────────────┐
   │   │ Ring Kernel Message Queue                  │
   │   │                                            │
   │   │ 1. Dequeue message (100ns)                 │
   │   │ 2. Update HLC timestamp (20ns)             │
   │   │ 3. Pattern matching on transaction         │
   │   │    - Velocity check                        │
   │   │    - Geographic anomaly                    │
   │   │    - Merchant risk                         │
   │   │    - Temporal patterns (150ns)             │
   │   │ 4. ML model inference (200ns)              │
   │   │ 5. Compute fraud score (50ns)              │
   │   │                                            │
   │   │ Total latency: ~500ns                      │
   │   └────────────────────────────────────────────┘
   │
   ├─► Read results
   │   └─► kernel.ReadResultsAsync(handle)
   │
   ├─► Apply business logic
   │   ├─► if (fraudScore > 0.8) Block transaction
   │   └─► else Process transaction
   │
   └─► Return response to client
       │
       └─► Total end-to-end latency: <10μs
           (vs ~100μs for CPU-based fraud detection)
```

---

## 10. Package Dependency Graph

```
                              Customer Application
                                      │
                    ┌─────────────────┼─────────────────┐
                    │                 │                 │
         ┌──────────▼────────┐ ┌──────▼──────┐ ┌───────▼────────┐
         │ Banking Domain    │ │ Accounting  │ │ Custom Kernels │
         │ Package           │ │ Domain Pkg  │ │ (optional)     │
         └──────────┬────────┘ └──────┬──────┘ └───────┬────────┘
                    │                 │                 │
                    └─────────────────┼─────────────────┘
                                      │
                          ┌───────────▼────────────┐
                          │ Enterprise Metapackage │
                          │                        │
                          │ • Abstractions         │
                          │ • Runtime              │
                          │ • Licensing            │
                          │ • Telemetry            │
                          └───────────┬────────────┘
                                      │
                    ┌─────────────────┼─────────────────┐
                    │                 │                 │
         ┌──────────▼────────┐ ┌──────▼──────┐ ┌───────▼────────┐
         │ Core Abstractions │ │ Core Runtime│ │ Core Grains    │
         └──────────┬────────┘ └──────┬──────┘ └───────┬────────┘
                    │                 │                 │
                    └─────────────────┼─────────────────┘
                                      │
                          ┌───────────▼────────────┐
                          │ Orleans.GpuBridge.Core │
                          │ (Open Source)          │
                          └───────────┬────────────┘
                                      │
                    ┌─────────────────┼─────────────────┐
                    │                 │                 │
         ┌──────────▼────────┐ ┌──────▼──────┐ ┌───────▼────────┐
         │ DotCompute Backend│ │ILGPU Backend│ │ Other Backends │
         │ (Commercial)      │ │ (OSS)       │ │ (Future)       │
         └───────────────────┘ └─────────────┘ └────────────────┘
```

---

## 11. Configuration Layers

```
┌────────────────────────────────────────────────────────────────┐
│                    APPLICATION CODE                            │
│                                                                │
│  services.AddGpuBridgeEnterprise(options => {                  │
│      options.License.LicenseKey = "...";                       │
│      options.EnabledDomains.Add("Banking");                    │
│  });                                                           │
└────────────────┬───────────────────────────────────────────────┘
                 │ Overrides
                 ▼
┌────────────────────────────────────────────────────────────────┐
│                    appsettings.json                            │
│                                                                │
│  {                                                             │
│    "GpuBridgeEnterprise": {                                    │
│      "EnabledDomains": ["Banking", "Accounting"],              │
│      "License": {                                              │
│        "LicenseKey": "...",                                    │
│        "LicenseServerUrl": "https://..."                       │
│      },                                                        │
│      "DomainConfigurations": {                                 │
│        "Banking": {                                            │
│          "KernelConfiguration": {                              │
│            "FraudDetection": {                                 │
│              "ThresholdScore": 0.8                             │
│            }                                                   │
│          }                                                     │
│        }                                                       │
│      }                                                         │
│    }                                                           │
│  }                                                             │
└────────────────┬───────────────────────────────────────────────┘
                 │ Overrides
                 ▼
┌────────────────────────────────────────────────────────────────┐
│              ENVIRONMENT VARIABLES                             │
│                                                                │
│  GpuBridgeEnterprise__License__LicenseKey=xxx                  │
│  GpuBridgeEnterprise__EnabledDomains__0=Banking                │
└────────────────┬───────────────────────────────────────────────┘
                 │ Overrides
                 ▼
┌────────────────────────────────────────────────────────────────┐
│                    DEFAULT VALUES                              │
│                                                                │
│  • PreferGpu: true                                             │
│  • RingBufferSize: 65536                                       │
│  • EnableTelemetry: true                                       │
│  • FallbackToCpu: true                                         │
└────────────────────────────────────────────────────────────────┘
```

---

**End of Architecture Diagrams**

For detailed implementation specifications, see:
- `/docs/commercial-package-architecture.md` - Full architecture specification
- `/docs/commercial-package-executive-summary.md` - Business overview and ROI

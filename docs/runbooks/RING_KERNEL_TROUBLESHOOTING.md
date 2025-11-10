# Operational Runbook: Ring Kernel Troubleshooting

## Purpose

Step-by-step guide for diagnosing and resolving GPU-native ring kernel issues in production.

**Target Audience**: On-call engineers, SREs, DevOps teams
**Severity Levels**: P0 (Critical), P1 (High), P2 (Medium), P3 (Low)

---

## Table of Contents

1. [Quick Reference](#quick-reference)
2. [Ring Kernel Hang/Crash](#ring-kernel-hangcrash)
3. [High Message Latency](#high-message-latency)
4. [GPU Memory Exhaustion](#gpu-memory-exhaustion)
5. [Clock Drift Issues](#clock-drift-issues)
6. [Actor Migration During Maintenance](#actor-migration-during-maintenance)
7. [Metrics & Monitoring](#metrics--monitoring)

---

## Quick Reference

### Critical Alerts

| Alert | Severity | Triage Time | MTTR Target |
|-------|----------|-------------|-------------|
| Ring kernel crashed | P0 | <5 min | <10 sec (auto-restart) |
| Queue overflow detected | P0 | <5 min | <30 sec (backpressure) |
| Clock drift >100μs | P1 | <15 min | <5 min (recalibration) |
| High message latency (>1μs p99) | P1 | <15 min | <10 min |
| GPU memory >90% | P1 | <15 min | <15 min |
| Watchdog restart loop | P2 | <30 min | <1 hour |

### Key Metrics Dashboard

**Grafana Dashboard**: `Orleans-GpuBridge-RingKernels`

**Critical Metrics**:
- `ring_kernels_active` - Should match expected count
- `ring_kernels_unhealthy` - Should be 0
- `actor_message_latency_nanoseconds{quantile="0.99"}` - Should be <1000ns
- `queue_utilization_percent` - Should be <85%
- `clock_drift_current_nanoseconds` - Should be <100000ns

### Emergency Contacts

- **Pager**: GPU-native-actors-oncall
- **Slack**: #gpu-actors-incidents
- **Runbook Owner**: [Team Name]
- **Last Updated**: 2025-01-10

---

## Ring Kernel Hang/Crash

### Symptoms

- Alert: `ring_kernels_unhealthy > 0`
- Alert: `ring_kernel_restarts_total increasing`
- Logs: "Ring kernel {id} appears hung - No progress for Xms"
- User Impact: Actors become unresponsive, messages not processed

### Severity: P0 (Critical)

### Step 1: Confirm the Issue (30 seconds)

```bash
# Check ring kernel health
kubectl exec -it <pod-name> -- \
  curl http://localhost:5000/health/ring-kernels

# Expected: Healthy status
# Unhealthy: "status": "Unhealthy", "description": "Ring kernel X has Y restarts"
```

**Check Grafana**:
- Dashboard: `Orleans-GpuBridge-RingKernels`
- Panel: `Ring Kernel Health`
- Look for: Red bars (unhealthy kernels)

### Step 2: Check Watchdog Status (1 minute)

```bash
# View watchdog logs
kubectl logs <pod-name> -c app --since=5m | grep -i watchdog

# Look for:
# - "Ring kernel {id} appears hung"
# - "Attempting restart {N}/{MaxAttempts}"
# - "Ring kernel {id} restart completed"
```

**Key Questions**:
- Is watchdog attempting restarts? → Normal, wait for auto-recovery
- Has watchdog given up (3+ restarts)? → Manual intervention needed
- Are all kernels affected? → GPU driver issue

### Step 3: Automatic Recovery (10 seconds)

**Most Common**: Watchdog auto-restarts kernel

```
Monitor for:
- "Ring kernel {id} restart completed"
- ring_kernels_unhealthy → 0
- Message processing resumes
```

**If auto-recovery succeeds**: ✅ **RESOLVED**
- Document incident in #gpu-actors-incidents
- Monitor for recurrence (30 minutes)
- If recurs, proceed to Step 4

### Step 4: Manual Kernel Restart (2 minutes)

**If watchdog failed or gave up**:

```bash
# Option A: Restart the pod (graceful)
kubectl delete pod <pod-name>
# Kubernetes will recreate with fresh ring kernels

# Option B: Force kernel restart via API (advanced)
curl -X POST http://localhost:5000/api/ring-kernels/{kernel-id}/restart

# Monitor recovery
kubectl logs <pod-name> -f | grep "Ring kernel.*initialized"
```

**Expected**: New kernel launches within 5 seconds, resumes processing

### Step 5: GPU Driver Check (5 minutes)

**If multiple pods/kernels affected simultaneously**:

```bash
# Check GPU driver status
kubectl exec -it <pod-name> -- nvidia-smi

# Look for:
# - GPU processes listed
# - GPU utilization normal (not 0% or 100%)
# - No error messages

# Check driver logs (on node)
ssh <node> "dmesg | tail -100 | grep -i gpu"

# Look for:
# - "GPU has fallen off the bus"
# - "Xid" errors (GPU exceptions)
# - Driver crashes
```

**If GPU driver issue detected**:
1. **Immediate**: Cordon node to prevent new pods
   ```bash
   kubectl cordon <node-name>
   ```

2. **Drain pods** to healthy nodes
   ```bash
   kubectl drain <node-name> --ignore-daemonsets
   ```

3. **Escalate** to infrastructure team for node/GPU replacement

### Step 6: Root Cause Analysis (Post-Incident)

**Collect evidence**:

```bash
# Export kernel metrics
kubectl exec -it <pod-name> -- \
  curl http://localhost:9090/metrics | grep ring_kernel

# Export watchdog stats
kubectl exec -it <pod-name> -- \
  curl http://localhost:5000/api/ring-kernels/watchdog/stats

# Export GPU diagnostics
kubectl exec -it <pod-name> -- nvidia-smi -q > gpu-diagnostics.txt
```

**Common Root Causes**:
- GPU driver bug → Kernel upgrade needed
- Kernel code bug → Code fix needed
- GPU hardware fault → Node replacement
- Memory corruption → Review kernel safety

---

## High Message Latency

### Symptoms

- Alert: `actor_message_latency_nanoseconds{quantile="0.99"} > 1000`
- User Impact: Slow response times, degraded throughput
- Logs: "Actor message latency: X ns (above threshold)"

### Severity: P1 (High)

### Step 1: Identify Scope (1 minute)

```bash
# Check if affecting all actors or specific ones
kubectl exec -it <pod-name> -- \
  curl http://localhost:9090/metrics | grep actor_message_latency

# Compare p50, p90, p99
# - All elevated → System-wide issue
# - Only p99 elevated → Outliers (some actors slow)
```

### Step 2: Check Queue Depth (1 minute)

**High queue depth = processing bottleneck**:

```bash
# Check queue utilization
kubectl exec -it <pod-name> -- \
  curl http://localhost:9090/metrics | grep queue_utilization_percent

# Expected: <70%
# Warning: 70-85%
# Critical: >85%
```

**If queue >85%**: Backpressure should activate

```bash
# Verify backpressure active
kubectl logs <pod-name> --since=5m | grep -i backpressure

# Should see:
# - "Backpressure activated - Queue: X%, Severity: Critical"
# - "Applying critical backpressure - Delay: 50ms"
```

### Step 3: Check GPU Utilization (2 minutes)

```bash
# Check if GPU is saturated
kubectl exec -it <pod-name> -- nvidia-smi

# GPU Utilization: Should be 70-95%
# - <70%: GPU underutilized (CPU bottleneck?)
# - >95%: GPU saturated (need more GPUs)

# Check memory bandwidth
kubectl exec -it <pod-name> -- nvidia-smi dmon -s m

# Memory Usage: Should be <90%
```

### Step 4: Check for CPU Bottleneck (2 minutes)

```bash
# Check CPU usage
kubectl top pod <pod-name>

# CPU: Should be <80%
# If >80%: CPU bottleneck slowing message dispatch

# Check for context switches (sign of contention)
kubectl exec -it <pod-name> -- cat /proc/stat | grep ctxt
```

### Step 5: Mitigation Options

**Option A: Scale horizontally** (if GPU utilization >90%)

```bash
# Add more pods
kubectl scale deployment gpu-actors --replicas=5

# Distribute actors across multiple GPUs
```

**Option B: Increase queue capacity** (if frequent overflow)

```yaml
# Update configuration
apiVersion: v1
kind: ConfigMap
metadata:
  name: gpu-actor-config
data:
  queueCapacity: "20000"  # Double capacity
```

**Option C: Reduce temporal ordering overhead** (if causality not critical)

```yaml
# Disable temporal ordering (saves 15% overhead)
temporalOrdering:
  enabled: false
```

**Option D: Enable message batching** (future optimization)

```yaml
# Batch messages for same target
messageBatching:
  enabled: true
  maxBatchSize: 100
  maxWaitMicros: 1
```

### Step 6: Monitor Recovery (5 minutes)

```bash
# Watch latency metrics
watch -n 1 "kubectl exec -it <pod-name> -- \
  curl -s http://localhost:9090/metrics | grep 'actor_message_latency.*0.99'"

# Should see p99 drop below 1000ns within 5 minutes
```

---

## GPU Memory Exhaustion

### Symptoms

- Alert: `gpu_memory_used > 90%`
- Errors: "CUDA_ERROR_OUT_OF_MEMORY"
- User Impact: Actor creation fails, actors stop processing

### Severity: P1 (High)

### Step 1: Check Memory Usage (1 minute)

```bash
# Check GPU memory breakdown
kubectl exec -it <pod-name> -- nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv

# Get per-process memory
kubectl exec -it <pod-name> -- nvidia-smi pmon -c 1
```

**Identify**:
- Total memory available
- Memory used by actors
- Memory used by queues
- Memory fragmentation

### Step 2: Check for Memory Leaks (3 minutes)

```bash
# Compare memory usage over time
kubectl exec -it <pod-name> -- \
  curl http://localhost:9090/metrics | grep gpu_memory_used

# Get historical data from Prometheus
curl 'http://prometheus:9090/api/v1/query_range?query=gpu_memory_used&start=...&end=...&step=60s'

# Plot memory over time
# - Flat line: Good (no leak)
# - Steady increase: Memory leak!
```

**If memory leak suspected**:
1. **Immediate**: Restart pod to free memory
2. **Urgent**: File bug with memory growth charts
3. **Mitigation**: Schedule regular pod restarts (until fixed)

### Step 3: Reduce Memory Footprint (5 minutes)

**Option A: Reduce actor count per pod**

```bash
# Scale out instead of up
kubectl scale deployment gpu-actors --replicas=4

# Configure fewer actors per pod
# actors-per-pod: 5000 (was 10000)
```

**Option B: Reduce queue capacity**

```yaml
# Smaller queues = less memory
queueCapacity: 5000  # Was 10000
messageSize: 256     # Keep small
```

**Option C: Run memory defragmentation** (future feature)

```bash
# Trigger compaction
curl -X POST http://localhost:5000/api/gpu-memory/defragment

# Monitor progress
kubectl logs <pod-name> -f | grep defragmentation
```

### Step 4: Emergency Evacuation (if critically low)

```bash
# Gracefully shutdown actors on this pod
kubectl exec -it <pod-name> -- \
  curl -X POST http://localhost:5000/api/actors/shutdown-all

# Wait for actors to migrate to other pods
sleep 30

# Delete pod (force restart)
kubectl delete pod <pod-name> --force

# Verify actors redistributed
kubectl exec -it <new-pod> -- \
  curl http://localhost:5000/api/actors/count
```

---

## Clock Drift Issues

### Symptoms

- Alert: `clock_drift_current_nanoseconds > 100000` (>100μs)
- Logs: "Excessive clock drift detected - Drift: X ns"
- User Impact: Potential temporal ordering violations

### Severity: P1 (High - affects correctness)

### Step 1: Verify Drift Magnitude (1 minute)

```bash
# Check current drift
kubectl exec -it <pod-name> -- \
  curl http://localhost:9090/metrics | grep clock_drift_current_nanoseconds

# Thresholds:
# <50μs: Normal
# 50-100μs: Warning
# >100μs: CRITICAL
```

### Step 2: Trigger Manual Recalibration (30 seconds)

```bash
# Force immediate recalibration
kubectl exec -it <pod-name> -- \
  curl -X POST http://localhost:5000/api/temporal/recalibrate

# Monitor result
kubectl logs <pod-name> --since=1m | grep calibration

# Expected:
# "Clock calibration completed - Drift: X ns"
```

**If drift persists after recalibration**: GPU hardware issue

### Step 3: Check GPU Clock Stability (2 minutes)

```bash
# Check GPU clocks
kubectl exec -it <pod-name> -- nvidia-smi --query-gpu=clocks.sm,clocks.mem,clocks.gr --format=csv

# Monitor for clock throttling
kubectl exec -it <pod-name> -- nvidia-smi dmon -s c

# If clocks dropping: Thermal throttling or power limit
```

### Step 4: Mitigation

**Option A: Increase calibration frequency**

```yaml
# Recalibrate more often
temporalOrdering:
  calibrationInterval: 1m  # Was 5m
```

**Option B: Disable temporal ordering temporarily**

```yaml
# If drift cannot be fixed immediately
temporalOrdering:
  enabled: false

# WARNING: Loses causal ordering guarantees!
# Only use as last resort
```

### Step 5: Hardware Investigation

**If drift issue persists**:

```bash
# Check GPU health
kubectl exec -it <pod-name> -- nvidia-smi -q | grep -A 5 "GPU UUID"

# Look for:
# - Retired pages (memory errors)
# - ECC errors
# - Temperature issues
```

**Escalate to infrastructure** if hardware fault detected

---

## Actor Migration During Maintenance

### Scenario

Need to perform GPU node maintenance without downtime

### Severity: P3 (Planned)

### Prerequisites

- Multi-GPU setup (actors can migrate between nodes)
- Health checks enabled
- Graceful shutdown configured

### Step 1: Prepare Target Nodes (5 minutes)

```bash
# Ensure target nodes have capacity
kubectl top nodes | grep gpu

# Add extra capacity if needed
kubectl scale deployment gpu-actors --replicas=$(expr $(kubectl get pods -l app=gpu-actors --no-headers | wc -l) + 2)

# Wait for new pods ready
kubectl wait --for=condition=ready pod -l app=gpu-actors --timeout=300s
```

### Step 2: Cordon Source Node (30 seconds)

```bash
# Prevent new pods on maintenance node
kubectl cordon <node-name>

# Verify
kubectl get nodes | grep <node-name>
# Should show: SchedulingDisabled
```

### Step 3: Graceful Actor Shutdown (10 minutes)

```bash
# Get pods on maintenance node
kubectl get pods -o wide | grep <node-name>

# For each pod, gracefully shutdown actors
for POD in $(kubectl get pods -o name --field-selector spec.nodeName=<node-name>); do
  echo "Shutting down actors in $POD"
  kubectl exec -it $POD -- curl -X POST http://localhost:5000/api/actors/shutdown-all

  # Wait for queues to drain
  kubectl exec -it $POD -- \
    curl http://localhost:5000/api/metrics | grep queue_depth_current

  # Should drop to 0 within 30 seconds
done
```

### Step 4: Delete Pods (2 minutes)

```bash
# Delete pods on maintenance node
kubectl delete pods --field-selector spec.nodeName=<node-name>

# Verify actors redistributed
kubectl exec -it <other-pod> -- \
  curl http://localhost:5000/api/actors/count

# Should see actor count increased on other pods
```

### Step 5: Perform Maintenance

```bash
# Node is now safe for maintenance
ssh <node> "sudo reboot"  # Or whatever maintenance needed

# After maintenance complete
kubectl uncordon <node-name>

# Pods will automatically rebalance
```

### Step 6: Verify Recovery (5 minutes)

```bash
# Check cluster health
kubectl get pods -o wide | grep gpu-actors

# All pods should be Running
# Actors should be evenly distributed

# Check ring kernel health
kubectl exec -it <pod-name> -- curl http://localhost:5000/health/ring-kernels

# All should be Healthy
```

---

## Metrics & Monitoring

### Key Prometheus Queries

**Ring Kernel Health**:
```promql
# Unhealthy kernels
sum(ring_kernels_unhealthy)

# Restart rate (per 5 min)
rate(ring_kernel_restarts_total[5m])

# Crash rate
rate(ring_kernel_crashes_total[5m])
```

**Message Performance**:
```promql
# p99 latency
histogram_quantile(0.99, rate(actor_message_latency_nanoseconds_bucket[5m]))

# Throughput
rate(actor_messages_processed_total[1m])

# Message loss rate
rate(actor_messages_dropped_total[5m]) / rate(actor_messages_sent_total[5m])
```

**Queue Health**:
```promql
# Queue utilization
avg(queue_utilization_percent)

# Overflow rate
rate(queue_overflows_total[5m])

# Backpressure activation rate
rate(backpressure_activations_total[5m])
```

**Temporal Ordering**:
```promql
# Clock drift
clock_drift_current_nanoseconds

# Causal violations
rate(causal_violations_detected_total[5m])

# Time since last calibration
time_since_last_calibration_seconds
```

### Grafana Dashboard Panels

**Panel 1: Ring Kernel Health**
- Active kernels (gauge)
- Unhealthy kernels (gauge, red if >0)
- Restart rate (graph)
- Uptime distribution (heatmap)

**Panel 2: Message Performance**
- Latency percentiles (p50, p90, p99)
- Throughput (messages/s)
- Message loss rate

**Panel 3: Queue Health**
- Queue depth (graph)
- Queue utilization (gauge)
- Backpressure events (counter)
- Overflow events (counter)

**Panel 4: Resource Usage**
- GPU memory usage (gauge)
- GPU utilization (graph)
- CPU usage
- Network bandwidth

### Alerting Rules (Prometheus)

```yaml
groups:
  - name: gpu-native-actors
    interval: 30s
    rules:
      # P0 Alerts
      - alert: RingKernelCrashed
        expr: rate(ring_kernel_crashes_total[5m]) > 0
        for: 1m
        severity: critical
        annotations:
          summary: "Ring kernel crashed"

      - alert: QueueOverflow
        expr: rate(queue_overflows_total[5m]) > 0
        for: 1m
        severity: critical

      # P1 Alerts
      - alert: HighMessageLatency
        expr: histogram_quantile(0.99, rate(actor_message_latency_nanoseconds_bucket[5m])) > 1000
        for: 5m
        severity: warning

      - alert: ExcessiveClockDrift
        expr: abs(clock_drift_current_nanoseconds) > 100000
        for: 2m
        severity: warning

      - alert: HighGPUMemory
        expr: gpu_memory_used / gpu_memory_total > 0.9
        for: 5m
        severity: warning
```

---

## Troubleshooting Checklist

Use this checklist during incidents:

- [ ] Check Grafana dashboard for anomalies
- [ ] Review recent logs (last 5 minutes)
- [ ] Verify watchdog status (if kernel issue)
- [ ] Check queue depth/utilization (if latency issue)
- [ ] Verify GPU health with nvidia-smi
- [ ] Check for recent deployments/changes
- [ ] Review metrics for all affected components
- [ ] Document findings in incident channel
- [ ] Implement mitigation
- [ ] Verify recovery
- [ ] Schedule post-mortem (if P0/P1)

---

## Escalation Path

**L1 → L2 Escalation**: If not resolved in 15 minutes
**L2 → L3 Escalation**: If not resolved in 30 minutes
**L3 → Engineering**: For code bugs or architectural issues

**When to escalate**:
- Multiple restart loops (watchdog giving up)
- Widespread GPU hardware failures
- Data loss detected
- Unknown root cause after 30 minutes
- Requires code changes

---

## Related Documentation

- [Performance Tuning Guide](PERFORMANCE_TUNING.md)
- [Security Guide](SECURITY.md)
- [GPU-Native Actors Quick Start](../examples/GPU_NATIVE_ACTORS_QUICKSTART.md)
- [Production Readiness Analysis](../GPU_NATIVE_ACTORS_PRODUCTION_READINESS.md)

---

**Last Updated**: 2025-01-10
**Runbook Version**: 1.0
**Owner**: GPU-Native Actors Team

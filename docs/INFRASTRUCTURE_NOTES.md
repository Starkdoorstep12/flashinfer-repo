# Infrastructure Notes

This document tracks constraints and issues in the measurement environment
itself (Turing cluster) that are **not properties of the kernel being
studied**, but that affect what evidence could be collected and how it
should be interpreted. Kept separate from the optimization writeups
(`SPLITK_OPTIMIZATION.md`, `PROFILING.md`) to avoid conflating "what the
kernel does" with "what our test rig could measure."

## DCGM / Nsight Compute (ncu) resource conflict

### What happened

Attempting `ncu --set basic` profiling of `dsa_fwd_kernel_splitk_v3`
failed with:

==ERROR== An error was reported by the driver:
==ERROR== Profiling failed because a driver resource was unavailable.
Ensure that no other tool (like DCGM) is concurrently collecting
profiling data.


The same command had worked earlier in this project (successfully
profiled the baseline `dsa_fwd_kernel` and the v1 `dsa_fwd_kernel_splitk`
on prior sessions/nodes), so this was investigated rather than assumed to
be a code issue.

### Diagnosis

```bash
nvidia-smi              # showed "No running processes found" — no
                         # competing compute job, ruling out another
                         # user's job on this GPU
ps aux | grep dcgm       # showed nv-hostengine and dcgm-exporter running
                         # as root, continuously, since before this
                         # session started
systemctl status nvidia-dcgm
                         # confirmed: active, running for 1+ week,
                         # a persistent system service, not tied to any
                         # user job
```

**Root cause confirmed: `nvidia-dcgm` (NVIDIA Data Center GPU Manager) is a
permanent, admin-installed system service on this node**, running
`dcgm-exporter` for continuous cluster-wide GPU health/utilization
monitoring (standard practice on shared HPC/GPU clusters, independent of
this project). DCGM's counter polling holds exclusive access to the same
low-level hardware performance-counter registers that Nsight Compute
(`ncu`) requires for its detailed metrics (occupancy, throughput, stalls,
etc.). This is a documented, known class of conflict — NVIDIA's own
profiling-guide FAQ addresses it directly — not something specific to our
kernel, job, or account.

### Why this matters for the research, not just as a debugging note

- **It is not a finding about the kernel or GPU architecture.** It is a
  constraint of the *measurement environment* (a monitored, shared
  cluster), and belongs in a methodology/limitations discussion, not mixed
  into performance-result narratives where a reader might mistake it for a
  property of the code being studied.
- **It explains an asymmetry in the evidence base**: `ncu`
  occupancy/throughput data exists for the original baseline kernel and
  for `kernel_splitk` (v1), but was not obtainable for `kernel_splitk_v3`
  during the session it was built, due to this conflict. This is stated
  explicitly here and in `SPLITK_OPTIMIZATION.md` rather than left as a
  silent gap a reviewer would have to notice and question.
- **`nsys` (Nsight Systems) is unaffected** — it does not require the same
  exclusive hardware-counter lock, and every wall-clock latency
  measurement throughout this project (which is what the v1/v2/v3
  performance comparisons are actually built on) used `nsys` or
  `torch.cuda.Event`, not `ncu`. The core performance claims in this
  project are not compromised by this issue; only some *supplementary*
  occupancy/throughput data collection was blocked.
- **Relevant to the multi-hardware plan.** If A100/L40S/DGX Spark are also
  shared, centrally-monitored cluster nodes (likely, given they are
  described as institutional/lab hardware rather than dedicated
  workstations), this same conflict may recur. Having diagnosed it once
  means it can be recognized and worked around faster next time, rather
  than re-investigated from scratch.

### Handling going forward

- Retry `ncu` profiling opportunistically (the conflict may be
  intermittent depending on DCGM's polling schedule/configuration, though
  it was observed as a persistent, always-running service here).
- If `ncu` access is needed reliably for future work (e.g., a fuller
  occupancy comparison across kernel versions or across hardware), this is
  worth raising with the cluster administrators — DCGM can typically be
  configured to release hardware counters on request, or a maintenance
  window could be arranged, but this requires admin action and is outside
  what can be resolved from a user account.
- Document, for every `ncu`-based result in this project, which node and
  session it was collected on, so any future gaps or inconsistencies can
  be traced back to this kind of environmental cause rather than assumed
  to be a code regression.

### Retry attempted

Retried `ncu` profiling of v3 in a later session — same error, same
conflict. Confirms this is a persistent lock on this node during this
period, not a one-off timing fluke. Will retry again later (different
session/time) or raise with cluster admins if it continues to block
occupancy-comparison work.

### Attempted fix: dcgmi profile --pause/--resume

NVIDIA's own DCGM documentation (and HPC center guides, e.g. NERSC's
Perlmutter docs) recommend `dcgmi profile --pause` before running `ncu`,
then `dcgmi profile --resume` afterward, as the standard sanctioned
workaround for exactly this conflict.

```bash
dcgmi profile --pause
ncu --set basic --devices 0 -o ~/dsa_splitk_v3_ncu_test python profile_kernel_splitk_v3.py 16
dcgmi profile --resume
```

**Result: both `dcgmi` commands reported success ("Successfully paused
profiling." / "Successfully resumed profiling."), but `ncu` failed
immediately afterward with the identical error**, even with the GPU
device explicitly specified (`--devices 0`). This was reproduced twice.

**Root cause hypothesis (unresolved without admin access):** `dcgmi`'s own
output includes a warning on every invocation:

WARNING: CUDA_VISIBLE_DEVICES is set to 0 on dcgmi process. This does not
affect which GPUs are selected for diagnostics.
...
Status: Failed to get hostengine environment variable
Please set the CUDA_VISIBLE_DEVICES env on nv-hostengine


This suggests a device-context mismatch between the `dcgmi` client process
(user-level) and the actual `nv-hostengine` daemon (running as root,
system-wide — confirmed via `ps aux`/`systemctl status nvidia-dcgm`
earlier in this document). If the client and daemon disagree about which
GPU index corresponds to "device 0" — plausible on a shared SLURM node
where per-job GPU allocation may not map 1:1 with DCGM's global device
view — the `--pause` command may report success while not actually
releasing the hardware counters for the specific GPU the job is using.

**Conclusion: this is beyond what can be diagnosed or fixed from a user
account.** The client-side pause/resume commands work as documented, but
do not resolve the underlying conflict, which likely requires either
pausing DCGM at the `nv-hostengine` level directly (root-only) or a
cluster-side fix to GPU device-index mapping between SLURM job allocation
and DCGM's device view. Raised with the department's technical support
group; awaiting response.

### Additional detail: node-specific, not intermittent

Reproduced the `dcgmi profile --pause`/`ncu` failure a third time,
identical result each time on `node03`. Notably, `ncu` **did** work
successfully earlier in this project on different nodes (`node01`,
`node02`) — see the baseline and v1 profiling results in this project,
both collected without issue. This suggests the conflict may be specific
to `node03`'s DCGM configuration/instance rather than a universal property
of every node on the Turing cluster, though the driver version reported by
`nvidia-smi` looked identical across nodes when checked. Worth including
this detail (node-specific reproduction) when raising with cluster admins,
since it narrows the likely cause and may point to a configuration
difference on this particular node rather than a cluster-wide policy.

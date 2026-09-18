"""
Measures bucketing benefit with TRUE process isolation per workload
(each workload's timing runs in its own subprocess via time_one_workload.py),
avoiding the rare cross-shape state issue seen when many distinct
compiled shapes coexist in one long-lived process.
"""
import json, subprocess, re

trace_path = "/home/vedant.tejas/mlsys26-contest/workloads/dsa_paged/dsa_topk_indexer_fp8_h64_d128_topk2048_ps64.jsonl"
workloads = []
with open(trace_path) as f:
    for line in f:
        workloads.append(json.loads(line)['workload'])
workloads = workloads[:30]

def time_workload(uuid, mode):
    result = subprocess.run(
        ["python3", "time_one_workload.py", uuid[:8], mode],
        capture_output=True, text=True, timeout=180,
    )
    m = re.search(r"TIME=([\d.]+)", result.stdout)
    if m:
        return float(m.group(1))
    return None, result.stdout, result.stderr

print(f"Measuring {len(workloads)} workloads with per-workload process isolation\n")

for mode in ["unbucketed", "bucketed"]:
    print(f"=== {mode.upper()} ===")
    total = 0.0
    n_failed = 0
    for i, wl in enumerate(workloads):
        r = time_workload(wl['uuid'], mode)
        if isinstance(r, tuple):
            n_failed += 1
            print(f"  [{i+1}/{len(workloads)}] batch={wl['axes']['batch_size']:3d} pages={wl['axes']['max_num_pages']:3d}  FAILED (see stderr)")
        else:
            total += r
            print(f"  [{i+1}/{len(workloads)}] batch={wl['axes']['batch_size']:3d} pages={wl['axes']['max_num_pages']:3d}  {r:.2f}s")
    print(f"Total {mode} time: {total:.1f}s ({n_failed} failures)\n")

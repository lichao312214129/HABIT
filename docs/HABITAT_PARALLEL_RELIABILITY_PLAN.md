# Habitat batch parallel reliability plan

## Problem

Large cohort habitat training (`get-habitat` / `fit()`) occasionally fails a few
subjects per run. Re-running the batch may fail different subjects; re-running
failed subjects alone usually succeeds. This indicates **resource contention**
under parallel Stage-1 processing, not bad subject data.

Root causes identified in code:

| Layer | Issue |
|-------|--------|
| GPU assignment | `gpuSlotIndex` reserved but never wired from parallel workers; concurrent workers map by subject hash and can share one GPU |
| Process count | `processes` can exceed configured `torchGpus` pool size when `cap_processes_to_gpu_pool: false` |
| Spawn model | Default `individual_subject_timeout_sec=900` forces spawn even when `processes=1` (500 subjects → 500 spawns) |
| OOM handling | `oom_backoff` only reacts to Python `MemoryError`, not native CUDA / Windows crashes |
| Mitigation (done) | `individual_subject_auto_retry_rounds` — same-run retry of checkpoint failures |

## Strategy

Combine **root-cause scheduling fixes** with **same-run auto-retry** (already
shipped). Auto-retry remains the safety net; Phase 1+ reduce how often it is needed.

## Phase 1 — Worker GPU slots and process cap (implemented)

**Goal:** At most one active Stage-1 worker per configured GPU by default; worker slot
index drives `gpuSlotIndex` instead of subject hash under parallelism. Set
`cap_processes_to_gpu_pool: false` to keep full `processes` and share GPUs via
`gpuSlotIndex % len(gpu_pool)` when CPU-heavy steps should use all cores on 1-GPU machines.

### Changes

1. **`habit/utils/parallel_gpu_utils.py`**
   - `HABIT_GPU_SLOT_INDEX` env var contract for spawn children
   - `read_worker_gpu_slot_index()` / `inject_worker_gpu_slot_index()`
   - `resolve_habitat_torch_gpu_pool(config)` — parse `torchGpus` / implicit `[0]` for torch auto on CUDA
   - `cap_processes_to_gpu_pool(requested, pool_size)` — cap with logging (skipped when config flag is false)
   - `apply_gpu_pool_process_cap(requested, config)` — honor `cap_processes_to_gpu_pool` before capping

2. **`habit/utils/isolated_runner.py`**
   - Assign monotonic `gpu_slot_index` (`slot_counter % max_workers`) per spawned child
   - Set `HABIT_GPU_SLOT_INDEX` in child before running user func

3. **`habit/core/habitat_analysis/services/feature_service.py`**
   - Inject `gpuSlotIndex` from worker env into voxel / supervoxel step params

4. **`habit/core/habitat_analysis/checkpoint/stage.py`**
   - Cap `n_processes` to `len(torch_gpu_pool)` when pool is non-empty **and**
     `cap_processes_to_gpu_pool: true` (default)

5. **`habit/utils/parallel_utils.py`**
   - Log when using in-process sequential path (no spawn); existing rule unchanged:
     `processes=1` and `individual_subject_timeout_sec: null` avoids per-subject spawn

### Verification

- `pytest tests/utils/test_parallel_gpu_utils.py`
- `pytest tests/habitat/test_torch_radiomics_utils.py`
- `pytest tests/habitat/test_checkpoint_stage.py tests/utils/test_isolated_runner.py`

## Phase 2 — Stability (planned)

- Extend OOM backoff to native exit codes (`3221225477`, `os._exit(2)`)
- Timeout policy: optional scale-by-concurrency or disable default timeout for large cohorts
- `torch.cuda.empty_cache()` after GPU radiomics in child processes

## Phase 3 — Persistent worker pool (implemented)

### 3.0 Goals and non-goals

**Goals**

- Remove per-subject `spawn` + full import cost (500 subjects should not mean 500 cold starts).
- Keep Phase-1 guarantees: **one active GPU worker per GPU slot**, `gpuSlotIndex` binding, `processes` cap.
- Preserve existing outward behaviour: checkpoint manifest, auto-retry, `on_item_done`, progress bar, failure lists.
- Default is **`persistent`** (long-lived workers); use **`isolated`** when pipeline pickle fails or for spawn debugging.

**Non-goals (Phase 3.0 scope)**

- Cross-machine scheduling (Ray/Dask/Celery).
- Splitting one subject pipeline into separate CPU-pool + GPU-queue stages (Phase 3B, optional later).
- Replacing legacy `multiprocessing.Pool` in `traditional_radiomics_extractor` / `habitat_analyzer`.

### 3.1 Target architecture

```text
IndividualCheckpointStage._run_parallel_pass()
  │
  ├─ parallel_mode == "isolated"  →  IsolatedTaskRunner (current)
  │
  └─ parallel_mode == "persistent"  →  PersistentWorkerPoolRunner
         │
         Parent process
         ├─ Start W = effective_processes workers (W capped by GPU pool)
         │     Worker-0: spawn once, HABIT_GPU_SLOT_INDEX=0, bind cuda:torchGpus[0]
         │     Worker-1: spawn once, HABIT_GPU_SLOT_INDEX=1, ...
         ├─ Task queue(s): dispatch pending (subject_id, payload) to idle workers
         ├─ Result queue: ProcessingResult per finished subject
         ├─ on_item_done → checkpoint record_success / record_failure
         └─ Shutdown: send STOP, join workers, drain queues

         Worker child (long-lived)
         ├─ Once: restore_logging, set gpu slot, unpickle worker_init (pipeline handle)
         ├─ Loop: receive Task → _process_single_subject(item) → put ProcessingResult
         ├─ After each task: optional torch.cuda.empty_cache() on GPU paths
         └─ On STOP: exit cleanly
```

**Why not ProcessPoolExecutor**

- Pool workers survive, but a hung task blocks a pool worker with weak timeout story.
- Persistent design keeps **explicit parent control**: per-task timer, kill + respawn one slot.

**Single GPU behaviour**

- W = 1: one persistent worker on `cuda:0`, subjects processed **serially** inside that worker.
- Gain vs isolated: **no repeated import / FeatureService init** between subjects.
- Throughput win is mostly **amortized startup**, not more GPU parallelism.

**Multi GPU behaviour**

- W = len(torchGpus): each worker pinned to one GPU for its lifetime.
- Pending subjects assigned to **first idle worker** (work-stealing queue or shared task queue).

### 3.2 Public configuration (new)

Add to `HabitatAnalysisConfig`:

```yaml
# "persistent" (default): long-lived worker per GPU slot
# "isolated": spawn-per-subject IsolatedTaskRunner
individual_subject_parallel_mode: persistent

# Restart a persistent worker after N consecutive subject failures (default 1 for OOM/native crash)
persistent_worker_max_consecutive_failures: 1

# Optional: restart worker every N successful subjects to mitigate slow VRAM leak (0 = disabled)
persistent_worker_recycle_after_tasks: 0
```

| Field | Default | Notes |
|-------|---------|-------|
| `individual_subject_parallel_mode` | `persistent` | Default; use `isolated` if pipeline pickle fails |
| `persistent_worker_max_consecutive_failures` | `1` | Worker restart after fatal/OOM-class failure |
| `persistent_worker_recycle_after_tasks` | `0` | Periodic recycle; e.g. 50 for long GPU runs |

Not in hash (safe to toggle on resume), same as `processes`.

### 3.3 Module layout

| File | Responsibility |
|------|----------------|
| `habit/utils/persistent_worker_runner.py` | **New.** `PersistentWorkerPoolRunner.map_items()` — same return type as `IsolatedTaskRunner` |
| `habit/utils/persistent_worker_protocol.py` | **New.** Task/Result/Command dataclasses, queue message types |
| `habit/utils/persistent_worker_entry.py` | **New.** Top-level `persistent_worker_main()` for spawn pickling |
| `habit/utils/parallel_utils.py` | Route `parallel_map(..., parallel_mode=...)` to isolated vs persistent |
| `habit/core/habitat_analysis/checkpoint/stage.py` | Pass `parallel_mode` from config into `_run_parallel_pass` |
| `habit/core/habitat_analysis/config_schemas.py` | New fields + validators |
| `tests/utils/test_persistent_worker_runner.py` | **New.** Unit + integration tests |

Keep `IsolatedTaskRunner` unchanged as fallback.

### 3.4 IPC contract

Messages over `multiprocessing.Queue` (spawn-safe):

```python
@dataclass
class WorkerTask:
    task_id: str          # subject_id
    payload: Any          # HabitatSubjectData or initial empty payload

@dataclass
class WorkerCommand:
    kind: Literal["RUN", "STOP", "PING"]

@dataclass
class WorkerReply:
    kind: Literal["RESULT", "READY", "WORKER_DIED"]
    result: Optional[ProcessingResult]
    worker_slot: int
    error: Optional[str]
```

**Worker init blob** (sent once per worker at start):

```python
@dataclass
class PersistentWorkerInit:
    worker_slot: int
    log_file_path: Optional[str]
    log_level: int
    # Pickled callable OR pipeline snapshot — see 3.5
    worker_target: Any
```

Parent never sends the full pipeline per subject — only `(subject_id, payload)`.

### 3.5 Worker initialization strategy (critical design choice)

**Recommended for Phase 3.0: pickle bound worker target once**

At pool creation in parent:

```python
init = PersistentWorkerInit(
    worker_slot=slot,
    worker_target=pipeline._process_single_subject,  # bound method
    ...
)
```

Worker unpickles once, caches `process_fn`, loops.

**Risks**

- Pipeline / steps must be picklable (joblib already serializes pipeline for `.pkl`).
- `FeatureService` logger paths must be subprocess-safe (already handled via `restore_logging_in_subprocess`).

**Fallback if pickle fails**

- Log clear error: use `individual_subject_parallel_mode: isolated`.
- Optional Phase 3.1b: rebuild from `config_file` path inside worker (slower init, more robust).

**Acceptance test before merge**

- Pickle `HabitatPipeline` built from demo two_step config in worker on Windows spawn.

### 3.6 Timeout, OOM, worker restart

| Event | Persistent mode action |
|-------|------------------------|
| Per-subject wall timeout | Parent kills **that worker process**, spawns replacement on same slot, subject → `failed_subjects` |
| Python MemoryError in worker | Return failure; parent **restarts worker** before next task; optional oom_backoff on pool size |
| Native crash / queue hang | Treat as worker death; restart slot; subject failed |
| Consecutive failures on one slot | Restart worker after `persistent_worker_max_consecutive_failures` |
| Successful task | `on_item_done` → checkpoint; optional `empty_cache()` in worker |

Timeout semantics stay **per subject**, not per pool lifetime.

### 3.7 Integration with checkpoint + auto-retry

No change to retry logic:

- `_run_parallel_pass` still calls `parallel_map` with `on_item_done`.
- Auto-retry rounds re-enter `_run_parallel_pass` with smaller pending set.
- Persistent pool is **created per parallel pass** (simpler lifecycle) or **reused across retry rounds** within one `stage.run()` (preferred for performance).

**Recommended lifecycle**

```text
stage.run():
  create PersistentWorkerPool once (after pending resolved)
  for each pass (initial + auto-retry rounds):
      parallel_map via same pool
  shutdown pool
```

### 3.8 Implementation slices (land incrementally)

#### Slice 3A — Skeleton (1–2 days)

- [x] `persistent_worker_protocol.py` message types
- [x] `persistent_worker_entry.py` loop with dummy `func` (module-level)
- [x] `PersistentWorkerPoolRunner.map_items()` — 2 workers, 10 items, no timeout
- [x] Tests: success path, worker restart on synthetic crash

#### Slice 3B — Habitat wiring (2–3 days)

- [x] Config fields + schema tests
- [x] `parallel_utils.parallel_map(..., parallel_mode=)` dispatch
- [x] `IndividualCheckpointStage` passes config mode
- [x] Pickle `_process_single_subject` / pipeline in worker init
- [ ] Integration test: mock pipeline 3 subjects, checkpoint manifest updates

#### Slice 3C — Production hardening (2–3 days)

- [x] Per-subject timeout with worker kill + respawn
- [x] OOM / exit-code classification (align with Phase 2 helpers)
- [x] `torch.cuda.empty_cache()` after GPU tasks in worker loop
- [ ] Windows spawn stress test (50 dummy tasks)
- [x] Logging: `Using persistent worker pool: W workers, mode=persistent`

#### Slice 3D — Docs + opt-in default (0.5 day)

- [x] Update `HABITAT_PARALLEL_RELIABILITY_PLAN.md` status
- [x] `configuration_zh.rst` / example YAML comment
- [x] Keep default `persistent`; document when to use `isolated`

**Total estimate:** ~6–9 dev days + soak test on real 500-subject cohort.

### 3.9 Phase 3B (future, optional) — CPU/GPU pipeline split

Only pursue if 3A–3D still leaves GPU idle while CPU steps run.

```text
Stage-1 refactor (large):
  CPU phase (processes = cpu_workers): load images, mask, preprocessing
  GPU phase (workers = gpu_pool): voxel_radiomics / supervoxel_radiomics only
  CPU phase: clustering, merge, checkpoint_save
```

Requires splitting `_process_single_subject` into schedulable steps with intermediate serializable state. **Not part of 3.0.**

### 3.10 Phase 3C (future, optional) — Dynamic admission

- Parent polls `psutil.virtual_memory()` / optional `torch.cuda.mem_get_info()`.
- `_fill_slots` only when free RAM > threshold and GPU free VRAM > floor.
- Adds complexity; defer until persistent pool stable.

### 3.11 Testing matrix

| Test | Purpose |
|------|---------|
| `test_persistent_pool_success` | N subjects, W workers, all succeed |
| `test_persistent_pool_worker_restart_on_timeout` | Hung worker killed, slot replaced |
| `test_persistent_pool_pickle_pipeline` | Real HabitatPipeline pickles on Windows |
| `test_checkpoint_with_persistent_mode` | manifest success/failure |
| `test_parallel_mode_isolated_unchanged` | Default config regression |
| Benchmark script (manual) | 50 subjects isolated vs persistent, compare wall time |

### 3.12 Rollout recommendation

1. Ship 3A–3D behind `individual_subject_parallel_mode: persistent`.
2. Internal soak: `.cursor/test` cohort, compare failure rate + total time vs isolated.
3. If stable for 2 weeks, consider flipping default to `persistent` in a **major** release with changelog.
4. Never remove `isolated` mode (escape hatch for pickle/debug issues).

### 3.13 Risks and mitigations

| Risk | Mitigation |
|------|------------|
| Pipeline not picklable on some configs | Pre-flight pickle test; fallback to isolated |
| Worker memory leak across subjects | `empty_cache()` + optional `persistent_worker_recycle_after_tasks` |
| Hung GPU kernel | Wall timeout + kill worker |
| Stale worker after code change mid-run | Document: restart run after code update; pool is per `fit()` |
| Windows queue races | Reuse Phase-1 queue flush patterns; integration tests on win32 |

## Configuration guidance

```yaml
processes: 2
cap_processes_to_gpu_pool: true   # default; set false on 1-GPU / many-CPU hosts
FeatureConstruction:
  voxel_level:
    params:
      useTorchRadiomics: auto
      torchGpus: [0, 1]   # explicit pool; processes auto-capped when cap_processes_to_gpu_pool: true
      torchGpuCount: 2

individual_subject_auto_retry_rounds: 2  # same-run retry (default)
individual_subject_parallel_mode: persistent  # default; use isolated if pickle fails
persistent_worker_max_consecutive_failures: 1
persistent_worker_recycle_after_tasks: 0
individual_subject_timeout_sec: null     # with processes: 1, avoids 500 spawns
oom_backoff: true
```

### Verification (Phase 3)

- `pytest tests/utils/test_persistent_worker_runner.py`
- `pytest tests/habitat/test_checkpoint_stage.py tests/utils/test_isolated_runner.py`


- `habit/utils/torch_radiomics_utils.py` — `select_torch_gpu_device`, `gpu_slot_index` priority
- `habit/core/habitat_analysis/checkpoint/stage.py` — Stage-1 parallel orchestration
- `docs/source/user_guide/habitat_segmentation_zh.rst` — checkpoint / retry docs

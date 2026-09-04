#!/usr/bin/env python3
# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
# Multi-GPU Cohort Acceleration Benchmark
from __future__ import annotations

import json
import os
import resource
import sys
import time
from typing import Any, Dict, List, Optional
import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from habit.datasets import make_synthetic_cohort
from habit.execution import ProcessPoolBackend
from habit.spec import RunPolicy
from habit.voxel_features.voxel_radiomics import VoxelRadiomicsFeatures

OUTPUT_DIR = os.environ.get("HABIT_BENCH_OUT", "out")
os.makedirs(OUTPUT_DIR, exist_ok=True)
JSON_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "multi_gpu_cohort_benchmark.json")
LOG_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "multi_gpu_cohort_benchmark.log")


def get_peak_rss_mb() -> float:
    try:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        return float(usage.ru_maxrss) / 1024.0
    except Exception:
        return float('nan')


class CohortVoxelExtractor:
    def __init__(self, use_torch_radiomics: bool, use_gpu_matrices: bool, torch_device: str = 'auto') -> None:
        self.use_torch_radiomics = use_torch_radiomics
        self.use_gpu_matrices = use_gpu_matrices
        self.torch_device = torch_device

    def __call__(self, subject: Any) -> Dict[str, Any]:
        ext = VoxelRadiomicsFeatures(
            modalities=('T1',),
            roi='tumor',
            kernel_radius=1,
            voxel_batch=10000,
            use_torch_radiomics=self.use_torch_radiomics,
            use_gpu_matrices=self.use_gpu_matrices,
            torch_device=self.torch_device,
        )
        field = ext(subject)
        import torch
        cur_dev = torch.cuda.current_device() if torch.cuda.is_available() else -1
        return {
            'subject_id': subject.subject_id,
            'n_voxels': int(field.values.shape[0]),
            'n_features': int(field.values.shape[1]),
            'cuda_device': cur_dev,
            'cuda_vis': os.environ.get('CUDA_VISIBLE_DEVICES', ''),
            'gpu_slot': os.environ.get('HABIT_GPU_SLOT_INDEX', ''),
        }


def log(msg: str) -> None:
    ts = time.strftime('%Y-%m-%d %H:%M:%S')
    formatted = f'[{ts}] {msg}'
    print(formatted, flush=True)
    with open(LOG_OUTPUT_PATH, 'a', encoding='utf-8') as f:
        f.write(formatted + '\n')


def save_json(data: Dict[str, Any]) -> None:
    with open(JSON_OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)


def main() -> int:
    log('=== Starting HABIT Multi-GPU Cohort Acceleration Benchmark ===')
    n_subjects = 16
    shape = (80, 80, 48)
    seed = 21

    log(f'Creating synthetic cohort: n_subjects={n_subjects}, shape={shape}, seed={seed}...')
    cohort = make_synthetic_cohort(n_subjects=n_subjects, shape=shape, rng=seed)
    n_roi_voxels = int(np.sum(cohort[0].mask('tumor').data > 0))
    total_roi_voxels = n_roi_voxels * n_subjects
    log(f'Cohort ready: {n_subjects} subjects, ~{n_roi_voxels:,} tumor ROI voxels/subject, total ~{total_roi_voxels:,} ROI voxels.')

    # GPU Warmup
    log('Performing GPU warmup on cuda:0...')
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    warmup_ext = VoxelRadiomicsFeatures(
        modalities=('T1',), roi='tumor', kernel_radius=1,
        voxel_batch=10000, use_torch_radiomics=True, use_gpu_matrices=True, torch_device='cuda:0'
    )
    _ = warmup_ext(cohort[0])
    log('GPU Warmup complete.')

    hardware_info = {
        'gpu': '5x NVIDIA GeForce RTX 4080 SUPER (32 GB each)',
        'cpu': '144 logical CPUs (2x Intel Xeon Platinum 8352V)',
        'ram_gb': 503,
        'torch_version': '2.6.0+cu124',
    }

    all_results: List[Dict[str, Any]] = []
    bench_data = {
        'metadata': {
            'hardware': hardware_info,
            'n_subjects': n_subjects,
            'shape': list(shape),
            'roi_voxels_per_subj': n_roi_voxels,
            'total_roi_voxels': total_roi_voxels,
            'n_features': 90,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        },
        'runs': all_results,
    }
    save_json(bench_data)

    runs_plan = [
        # Scenario B: 1 GPU
        {'scenario': '1gpu_multicpu', 'device_env': '0', 'workers': 1, 'use_torch': True, 'use_gpu': True, 'cap_to_pool': False},
        {'scenario': '1gpu_multicpu', 'device_env': '0', 'workers': 2, 'use_torch': True, 'use_gpu': True, 'cap_to_pool': False},
        {'scenario': '1gpu_multicpu', 'device_env': '0', 'workers': 4, 'use_torch': True, 'use_gpu': True, 'cap_to_pool': False},

        # Scenario C: 5 GPUs
        {'scenario': '5gpu_multicpu', 'device_env': '0,1,2,3,4', 'workers': 1, 'use_torch': True, 'use_gpu': True, 'cap_to_pool': True},
        {'scenario': '5gpu_multicpu', 'device_env': '0,1,2,3,4', 'workers': 2, 'use_torch': True, 'use_gpu': True, 'cap_to_pool': True},
        {'scenario': '5gpu_multicpu', 'device_env': '0,1,2,3,4', 'workers': 4, 'use_torch': True, 'use_gpu': True, 'cap_to_pool': True},
        {'scenario': '5gpu_multicpu', 'device_env': '0,1,2,3,4', 'workers': 5, 'use_torch': True, 'use_gpu': True, 'cap_to_pool': True},

        # Scenario A: 0 GPU (multi-CPU)
        {'scenario': '0gpu_multicpu', 'device_env': '-1', 'workers': 8, 'use_torch': False, 'use_gpu': False, 'cap_to_pool': False},
        {'scenario': '0gpu_multicpu', 'device_env': '-1', 'workers': 4, 'use_torch': False, 'use_gpu': False, 'cap_to_pool': False},
        {'scenario': '0gpu_multicpu', 'device_env': '-1', 'workers': 2, 'use_torch': False, 'use_gpu': False, 'cap_to_pool': False},
        {'scenario': '0gpu_multicpu', 'device_env': '-1', 'workers': 1, 'use_torch': False, 'use_gpu': False, 'cap_to_pool': False},
    ]

    for plan in runs_plan:
        sc = plan['scenario']
        w = plan['workers']
        dev_env = plan['device_env']
        use_torch = plan['use_torch']
        use_gpu = plan['use_gpu']
        cap = plan['cap_to_pool']

        log(f'Running [{sc}] workers={w} (CUDA={dev_env}) ...')
        os.environ['CUDA_VISIBLE_DEVICES'] = dev_env
        policy = RunPolicy(
            workers=w,
            backend='process',
            parallel_mode='persistent',
            cap_workers_to_gpu_pool=cap,
            on_subject_failure='fail_fast',
        )
        backend = ProcessPoolBackend.from_policy(policy)
        op = CohortVoxelExtractor(use_torch_radiomics=use_torch, use_gpu_matrices=use_gpu)

        t0 = time.perf_counter()
        results = list(backend.map(op, cohort))
        wall_s = time.perf_counter() - t0
        spm = (60.0 * n_subjects / wall_s) if wall_s > 0 else 0.0
        ok = len(results) == n_subjects and all(r.error is None for r in results)
        err = '' if ok else str([r.error for r in results if r.error is not None])

        row = {
            'scenario': sc,
            'device': f'CUDA={dev_env}',
            'workers': w,
            'wall_s': round(wall_s, 2),
            'subjects_per_min': round(spm, 2),
            'peak_rss_mb': round(get_peak_rss_mb(), 1),
            'ok': ok,
            'error': err,
        }
        all_results.append(row)
        save_json(bench_data)
        log(f'  -> wall={wall_s:.2f}s, subjects/min={spm:.2f}, ok={ok} {err}')

    # Compute speedups
    cpu_w1_row = next((r for r in all_results if r['scenario'] == '0gpu_multicpu' and r['workers'] == 1 and r['ok']), None)
    base_cpu_wall = cpu_w1_row['wall_s'] if cpu_w1_row else 1.0

    for r in all_results:
        r['speedup_vs_cpu_serial'] = round(base_cpu_wall / r['wall_s'], 2) if r['wall_s'] > 0 else 0.0

    save_json(bench_data)
    log('=== Benchmark completed successfully! ===')
    log(json.dumps(all_results, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

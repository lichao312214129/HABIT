/*
 * Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * sv_cmatrices.c  –  Multi-label supervoxel texture matrix calculations.
 *
 * Computes GLCM / GLRLM / GLSZM / NGTDM / GLDM / first-order statistics
 * for multiple supervoxel labels in a single C pass, avoiding Python-level
 * per-label loops.
 *
 * Algorithm references follow PyRadiomics (Haralick GLCM, Galloway GLRLM,
 * Thibault GLSZM, Amadasun NGTDM, Sun GLDM).
 */

#include "sv_cmatrices.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/*
 * Optional thread cap for the volume loops. HABIT_SV_OMP_THREADS wins when
 * set to a positive integer; otherwise the OpenMP runtime honours
 * OMP_NUM_THREADS (or its own default). No-op when compiled without OpenMP.
 */
static void
sv_omp_apply_thread_limit(void)
{
#ifdef _OPENMP
    const char *habit = getenv("HABIT_SV_OMP_THREADS");
    if (habit != NULL && habit[0] != '\0') {
        int n = atoi(habit);
        if (n > 0)
            omp_set_num_threads(n);
    }
#endif
}

/*
 * Add ``src`` into ``dst`` (length ``n``). Used to reduce thread-local
 * integer histograms without atomics on the hot voxel increment.
 */
static void
sv_add_ll_into(long long *dst, const long long *src, size_t n)
{
    for (size_t i = 0; i < n; i++)
        dst[i] += src[i];
}


/* ── helpers ─────────────────────────────────────────────────────────── */

#define SV_MIN(a, b) ((a) < (b) ? (a) : (b))
#define SV_MAX(a, b) ((a) > (b) ? (a) : (b))
#define N_FIRSTORDER_STATS 17

static int
_dbl_cmp(const void *a, const void *b)
{
    double da = *(const double *)a;
    double db = *(const double *)b;
    if (da < db) return -1;
    if (da > db) return 1;
    return 0;
}

static double
_percentile(const double *sorted, int n, double percent)
{
    if (n <= 0) return NAN;
    if (n == 1) return sorted[0];
    double rank = (percent / 100.0) * (n - 1);
    int lo = (int)floor(rank);
    int hi = (int)ceil(rank);
    if (lo == hi) return sorted[lo];
    return sorted[lo] * (hi - rank) + sorted[hi] * (rank - lo);
}

/*
 * Build Chebyshev-1 (infinity norm) neighbor offsets for GLSZM region growing.
 * 3D: 26-connected; 2D: 8-connected; 1D: 2-connected.
 * When force2D is set, skip offsets along force2Ddimension (PyRadiomics convention:
 * dim 0 = z, dim 1 = y, dim 2 = x for SimpleITK array order).
 */
static int
sv_build_chessboard_neighbors(int ndim, int force2D, int force2Ddimension,
                              int *dz_out, int *dy_out, int *dx_out,
                              int max_neighbors)
{
    int n = 0;

    for (int dz = -1; dz <= 1; dz++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                if (dz == 0 && dy == 0 && dx == 0)
                    continue;

                /* Keep only face/edge/corner neighbors at Chebyshev distance 1. */
                int cheb = SV_MAX(SV_MAX(abs(dz), abs(dy)), abs(dx));
                if (cheb != 1)
                    continue;

                if (ndim == 1) {
                    if (dz != 0 || dy != 0)
                        continue;
                } else if (ndim == 2) {
                    if (dz != 0)
                        continue;
                } else if (force2D) {
                    if (force2Ddimension == 0 && dz != 0)
                        continue;
                    if (force2Ddimension == 1 && dy != 0)
                        continue;
                    if (force2Ddimension == 2 && dx != 0)
                        continue;
                }

                if (n >= max_neighbors)
                    return n;

                dz_out[n] = dz;
                dy_out[n] = dy;
                dx_out[n] = dx;
                n++;
            }
        }
    }
    return n;
}

/* Increment P[label, gray, run_length, angle]. ``rl`` is 0-based (length-1). */
static void
sv_glrlm_record_run(long long *P, int li, int Ng, int Nr, int n_angles,
                    int ai, int gl, int rl)
{
    if (gl <= 0 || gl > Ng || rl < 0 || rl >= Nr)
        return;
    P[((li * Ng + (gl - 1)) * Nr + rl) * n_angles + ai]++;
}

/* ── angle generation ────────────────────────────────────────────────── */

static int
sv_generate_angles_internal(int *size, int ndim, int force2D, int force2Ddimension,
                            int bidirectional,
                            int **angles_out, int *n_angles_out)
{
    int *buf = (int *)malloc(3 * 27 * sizeof(int));
    if (!buf) { PyErr_NoMemory(); return -1; }

    int n = 0;
    int dz_start, dz_end, dy_start, dy_end, dx_start, dx_end;

    if (ndim == 3 && !force2D) {
        dz_start = -1; dz_end = 1;
    } else {
        dz_start = 0; dz_end = 0;
    }
    if (ndim >= 2) {
        dy_start = -1; dy_end = 1;
        dx_start = -1; dx_end = 1;
    } else {
        dy_start = 0; dy_end = 0;
        dx_start = -1; dx_end = 1;
    }

    /* PyRadiomics force2Ddimension skips that axis (0=z, 1=y, 2=x). */
    if (force2D) {
        if (force2Ddimension == 0) { dz_start = 0; dz_end = 0; }
        else if (force2Ddimension == 1) { dy_start = 0; dy_end = 0; }
        else if (force2Ddimension == 2) { dx_start = 0; dx_end = 0; }
    }

    for (int dz = dz_start; dz <= dz_end; dz++) {
        for (int dy = dy_start; dy <= dy_end; dy++) {
            for (int dx = dx_start; dx <= dx_end; dx++) {
                if (dz == 0 && dy == 0 && dx == 0)
                    continue;

                int cheb = SV_MAX(SV_MAX(abs(dz), abs(dy)), abs(dx));
                if (cheb != 1)
                    continue;

                if (!bidirectional) {
                    if (dz < 0) continue;
                    if (dz == 0 && dy < 0) continue;
                    if (dz == 0 && dy == 0 && dx < 0) continue;
                }

                int *row = buf + n * 3;
                row[0] = dz; row[1] = dy; row[2] = dx;
                n++;
            }
        }
    }

    *angles_out = buf;
    *n_angles_out = n;
    return 0;
}

int
sv_generate_angles(int *size, int ndim, int force2D, int force2Ddimension,
                   int **angles_out, int *n_angles_out)
{
    /* GLCM / GLRLM use PyRadiomics asymmetric angle set (half-space). */
    return sv_generate_angles_internal(
        size, ndim, force2D, force2Ddimension, 0, angles_out, n_angles_out);
}

static int
sv_generate_angles_bidirectional(int *size, int ndim, int force2D, int force2Ddimension,
                                 int **angles_out, int *n_angles_out)
{
    /* GLSZM / NGTDM / GLDM use PyRadiomics bidirectional angle set. */
    return sv_generate_angles_internal(
        size, ndim, force2D, force2Ddimension, 1, angles_out, n_angles_out);
}

/* ── GLCM ────────────────────────────────────────────────────────────── */

/*
 * Accumulate GLCM counts for one angle into the shared ``P``.
 *
 * Parallelising over angles (not z) is required for memory: a thread-local
 * copy of P is ``n_labels * Ng * Ng * n_angles`` long longs (often >100 MB
 * per thread). Each angle writes a distinct last index of P, so there is
 * no data race and no reduction. Neighbor bounds are hoisted so OOB voxels
 * are not visited. Counts are identical to a serial full-volume walk.
 */
static void
sv_glcm_accumulate_angle(long long *P, const int *image, const int *sv_map,
                         int sz, int sy, int sx,
                         int max_label, const int *label_to_idx,
                         const int *distances, int n_distances,
                         const int *angles, int n_angles, int ai, int Ng)
{
    for (int di = 0; di < n_distances; di++) {
        int dist = distances[di];
        int adz = angles[ai * 3] * dist;
        int ady = angles[ai * 3 + 1] * dist;
        int adx = angles[ai * 3 + 2] * dist;
        int z0 = (adz >= 0) ? 0 : -adz;
        int z1 = (adz <= 0) ? sz : sz - adz;
        int y0 = (ady >= 0) ? 0 : -ady;
        int y1 = (ady <= 0) ? sy : sy - ady;
        int x0 = (adx >= 0) ? 0 : -adx;
        int x1 = (adx <= 0) ? sx : sx - adx;
        for (int z = z0; z < z1; z++) {
            int nz = z + adz;
            for (int y = y0; y < y1; y++) {
                int ny = y + ady;
                int row = z * sy * sx + y * sx;
                int nrow = nz * sy * sx + ny * sx;
                for (int x = x0; x < x1; x++) {
                    int idx = row + x;
                    int lbl = sv_map[idx];
                    if (lbl <= 0 || lbl > max_label)
                        continue;
                    int li = label_to_idx[lbl];
                    if (li < 0)
                        continue;
                    int nidx = nrow + x + adx;
                    if (sv_map[nidx] != lbl)
                        continue;
                    int gi = image[idx];
                    if (gi <= 0 || gi > Ng)
                        continue;
                    int gj = image[nidx];
                    if (gj <= 0 || gj > Ng)
                        continue;
                    P[((li * Ng + (gi - 1)) * Ng + (gj - 1)) * n_angles + ai]++;
                }
            }
        }
    }
}

int
sv_calculate_glcm(int *image, int *sv_map, int *size, int ndim,
                  int *labels, int n_labels, int max_label, int *label_to_idx,
                  int *distances, int n_distances,
                  int Ng, int force2D, int force2Ddimension,
                  long long **P_glcm_out, int **angles_out, int *n_angles_out)
{
    int *angles = NULL;
    int n_angles = 0;
    if (sv_generate_angles(size, ndim, force2D, force2Ddimension,
                           &angles, &n_angles) < 0)
        return -1;

    size_t p_len = (size_t)n_labels * (size_t)Ng * (size_t)Ng * (size_t)n_angles;
    long long *P = (long long *)calloc(p_len, sizeof(long long));
    if (!P) { free(angles); PyErr_NoMemory(); return -1; }

    int sz = (ndim >= 3) ? size[0] : 1;
    int sy = (ndim >= 2) ? size[ndim - 2] : 1;
    int sx = size[ndim - 1];

    /* One shared P: each angle writes a distinct last index. */
    sv_omp_apply_thread_limit();
    {
        int ai;
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (ai = 0; ai < n_angles; ai++) {
            sv_glcm_accumulate_angle(
                P, image, sv_map, sz, sy, sx,
                max_label, label_to_idx, distances, n_distances,
                angles, n_angles, ai, Ng);
        }
    }

    *P_glcm_out = P;
    *angles_out = angles;
    *n_angles_out = n_angles;
    return 0;
}

/* ── GLRLM ───────────────────────────────────────────────────────────── */

int
sv_calculate_glrlm(int *image, int *sv_map, int *size, int ndim,
                   int *labels, int n_labels, int max_label, int *label_to_idx,
                   int Ng, int Nr, int force2D, int force2Ddimension,
                   long long **P_glrlm_out, int **angles_out, int *n_angles_out)
{
    int *angles = NULL;
    int n_angles = 0;
    if (sv_generate_angles(size, ndim, force2D, force2Ddimension,
                           &angles, &n_angles) < 0)
        return -1;

    long long *P = (long long *)calloc(
        (size_t)n_labels * Ng * Nr * n_angles, sizeof(long long));
    if (!P) { free(angles); PyErr_NoMemory(); return -1; }

    int *multi_element = (int *)calloc((size_t)n_labels * n_angles, sizeof(int));
    if (!multi_element) {
        free(P);
        free(angles);
        PyErr_NoMemory();
        return -1;
    }

    int sz = (ndim >= 3) ? size[0] : 1;
    int sy = (ndim >= 2) ? size[ndim - 2] : 1;
    int sx = size[ndim - 1];

    /* PyRadiomics ``calculate_glrlm`` starts each line on the incoming
     * bounding-box face and walks it once. Mask holes end a run but the
     * walk continues, so a later island is not double-counted. Starting
     * a new walk at every "previous voxel is another label" site would
     * re-walk that line and inflate GLRLM versus execute(). */
    sv_omp_apply_thread_limit();
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
        int *elements_line = (int *)calloc((size_t)n_labels, sizeof(int));
        int ai;
#ifdef _OPENMP
        #pragma omp for schedule(static)
#endif
        for (ai = 0; ai < n_angles; ai++) {
            int dz = angles[ai * 3];
            int dy = angles[ai * 3 + 1];
            int dx = angles[ai * 3 + 2];
            if (!elements_line)
                continue;

            for (int z = 0; z < sz; z++) {
                for (int y = 0; y < sy; y++) {
                    for (int x = 0; x < sx; x++) {
                        int pz = z - dz;
                        int py = y - dy;
                        int px = x - dx;
                        /* Incoming face only: one walk per discrete line. */
                        if (pz >= 0 && pz < sz && py >= 0 && py < sy &&
                            px >= 0 && px < sx)
                            continue;

                        memset(elements_line, 0, (size_t)n_labels * sizeof(int));
                        int cz = z, cy = y, cx = x;
                        int cur_li = -1;
                        int gl = -1;
                        int rl = 0;

                        for (;;) {
                            if (cz < 0 || cz >= sz || cy < 0 || cy >= sy ||
                                cx < 0 || cx >= sx)
                                break;

                            int cidx = cz * sy * sx + cy * sx + cx;
                            int lbl = sv_map[cidx];
                            int li = -1;
                            int gi = image[cidx];
                            int in_roi = 0;
                            if (lbl > 0 && lbl <= max_label) {
                                li = label_to_idx[lbl];
                                in_roi = (li >= 0 && gi > 0 && gi <= Ng);
                            }

                            if (in_roi) {
                                elements_line[li]++;
                                if (cur_li != li) {
                                    if (gl >= 0 && cur_li >= 0)
                                        sv_glrlm_record_run(
                                            P, cur_li, Ng, Nr, n_angles, ai, gl, rl
                                        );
                                    cur_li = li;
                                    gl = gi;
                                    rl = 0;
                                } else if (gi == gl) {
                                    rl++;
                                } else {
                                    sv_glrlm_record_run(
                                        P, cur_li, Ng, Nr, n_angles, ai, gl, rl
                                    );
                                    gl = gi;
                                    rl = 0;
                                }
                            } else if (gl >= 0 && cur_li >= 0) {
                                sv_glrlm_record_run(
                                    P, cur_li, Ng, Nr, n_angles, ai, gl, rl
                                );
                                cur_li = -1;
                                gl = -1;
                                rl = 0;
                            }

                            cz += dz;
                            cy += dy;
                            cx += dx;
                        }

                        if (gl >= 0 && cur_li >= 0)
                            sv_glrlm_record_run(
                                P, cur_li, Ng, Nr, n_angles, ai, gl, rl
                            );
                        for (int li = 0; li < n_labels; li++) {
                            if (elements_line[li] > 1)
                                multi_element[li * n_angles + ai] = 1;
                        }
                    }
                }
            }

            for (int li = 0; li < n_labels; li++) {
                if (multi_element[li * n_angles + ai])
                    continue;
                for (int gi = 0; gi < Ng; gi++)
                    P[((li * Ng + gi) * Nr + 0) * n_angles + ai] = 0;
            }
        }
        free(elements_line);
    }

    free(multi_element);
    *P_glrlm_out = P;
    *angles_out = angles;
    *n_angles_out = n_angles;
    return 0;
}

/* ── GLSZM ───────────────────────────────────────────────────────────── */

/*
 * Grow one same-gray zone from a seed voxel using PyRadiomics angle neighbours.
 * The mutable mask marks unprocessed ROI voxels; processed voxels are cleared to 0.
 */
static int
sv_glszm_grow_zone(int *image, char *mask, int *size, int *strides, int ndim,
                   int *angles, int n_angles, size_t seed,
                   size_t *region_stack, size_t *stack_top, int *cur_idx)
{
    int gl = image[seed];
    int region = 0;
    size_t top = *stack_top;
    region_stack[top++] = seed;
    mask[seed] = 0;

    while (top > 0) {
        size_t k = region_stack[--top];
        region++;

        cur_idx[0] = (int)(k / (size_t)strides[0]);
        for (int d = 1; d < ndim; d++)
            cur_idx[d] = (int)((k % (size_t)strides[d - 1]) / (size_t)strides[d]);

        for (int ai = 0; ai < n_angles; ai++) {
            size_t j = k;
            for (int d = 0; d < ndim; d++) {
                int offset = angles[ai * 3 + d];
                if (cur_idx[d] + offset < 0 || cur_idx[d] + offset >= size[d]) {
                    j = k;
                    break;
                }
                j += (size_t)offset * (size_t)strides[d];
            }

            if (j != k && mask[j] && image[j] == gl) {
                region_stack[top++] = j;
                mask[j] = 0;
            }
        }
    }

    *stack_top = top;
    return region;
}

/*
 * Fill one label's GLSZM slice using the PyRadiomics per-ROI mask workflow.
 * When p_label is NULL, only the largest zone size is returned.
 */
static int
sv_glszm_single_label(int *image, char *mask, int *size, int *strides, int ndim,
                      size_t ni, int *angles, int n_angles, int ng,
                      long long *p_label, int p_zone_dim)
{
    int max_zone = 1;
    size_t *region_stack = (size_t *)malloc(ni * sizeof(size_t));
    int *cur_idx = (int *)malloc((size_t)ndim * sizeof(int));
    if (!region_stack || !cur_idx) {
        free(region_stack);
        free(cur_idx);
        PyErr_NoMemory();
        return -1;
    }

    for (size_t i = 0; i < ni; i++) {
        if (!mask[i])
            continue;

        int gl = image[i];
        if (gl <= 0 || gl > ng)
            continue;

        size_t stack_top = 0;
        int region = sv_glszm_grow_zone(
            image, mask, size, strides, ndim, angles, n_angles, i,
            region_stack, &stack_top, cur_idx);

        if (region > max_zone)
            max_zone = region;

        if (p_label != NULL && region > 0 && region <= p_zone_dim)
            p_label[(gl - 1) * p_zone_dim + (region - 1)]++;
    }

    free(region_stack);
    free(cur_idx);
    return max_zone;
}

int
sv_calculate_glszm(int *image, int *sv_map, int *size, int ndim,
                   int *labels, int n_labels, int max_label, int *label_to_idx,
                   int Ng, int force2D, int force2Ddimension,
                   long long **P_glszm_out, int *max_zone_out)
{
    (void)max_label;
    (void)label_to_idx;

    int *angles = NULL;
    int n_angles = 0;
    if (sv_generate_angles_bidirectional(size, ndim, force2D, force2Ddimension,
                                         &angles, &n_angles) < 0)
        return -1;

    int *strides = (int *)malloc((size_t)ndim * sizeof(int));
    if (!strides) {
        free(angles);
        PyErr_NoMemory();
        return -1;
    }
    strides[ndim - 1] = 1;
    for (int d = ndim - 2; d >= 0; d--)
        strides[d] = strides[d + 1] * size[d + 1];

    size_t ni = 1;
    for (int d = 0; d < ndim; d++)
        ni *= (size_t)size[d];

    char *mask = (char *)malloc(ni * sizeof(char));
    if (!mask) {
        free(strides);
        free(angles);
        PyErr_NoMemory();
        return -1;
    }

    /* Per-label flood fill: each ``li`` writes a distinct P slice. Parallelize
     * over labels with a private mutable mask so two threads never grow the
     * same zone. */
    int *label_max_zones = (int *)calloc((size_t)n_labels, sizeof(int));
    if (!label_max_zones) {
        free(mask);
        free(strides);
        free(angles);
        PyErr_NoMemory();
        return -1;
    }

    int failed = 0;
    sv_omp_apply_thread_limit();
#ifdef _OPENMP
    #pragma omp parallel
    {
        char *local_mask = (char *)malloc(ni * sizeof(char));
        if (local_mask == NULL) {
            #pragma omp atomic
            failed += 1;
        } else {
            int li;
            #pragma omp for schedule(static)
            for (li = 0; li < n_labels; li++) {
                int lbl = labels[li];
                for (size_t i = 0; i < ni; i++)
                    local_mask[i] = (sv_map[i] == lbl) ? 1 : 0;
                int label_max_zone = sv_glszm_single_label(
                    image, local_mask, size, strides, ndim, ni, angles, n_angles, Ng,
                    NULL, 0);
                if (label_max_zone < 0) {
                    #pragma omp atomic
                    failed += 1;
                } else {
                    label_max_zones[li] = label_max_zone;
                }
            }
            free(local_mask);
        }
    }
#else
    for (int li = 0; li < n_labels; li++) {
        int lbl = labels[li];
        for (size_t i = 0; i < ni; i++)
            mask[i] = (sv_map[i] == lbl) ? 1 : 0;
        int label_max_zone = sv_glszm_single_label(
            image, mask, size, strides, ndim, ni, angles, n_angles, Ng,
            NULL, 0);
        if (label_max_zone < 0) {
            failed = 1;
            break;
        }
        label_max_zones[li] = label_max_zone;
    }
#endif
    if (failed) {
        free(label_max_zones);
        free(mask);
        free(strides);
        free(angles);
        PyErr_NoMemory();
        return -1;
    }

    int max_zone = 1;
    for (int li = 0; li < n_labels; li++) {
        if (label_max_zones[li] > max_zone)
            max_zone = label_max_zones[li];
    }
    free(label_max_zones);

    long long *final_P = (long long *)calloc(
        (size_t)n_labels * (size_t)Ng * (size_t)max_zone, sizeof(long long));
    if (!final_P) {
        free(mask);
        free(strides);
        free(angles);
        PyErr_NoMemory();
        return -1;
    }

    failed = 0;
#ifdef _OPENMP
    #pragma omp parallel
    {
        char *local_mask = (char *)malloc(ni * sizeof(char));
        if (local_mask == NULL) {
            #pragma omp atomic
            failed += 1;
        } else {
            int li;
            #pragma omp for schedule(static)
            for (li = 0; li < n_labels; li++) {
                int lbl = labels[li];
                for (size_t i = 0; i < ni; i++)
                    local_mask[i] = (sv_map[i] == lbl) ? 1 : 0;
                int label_max_zone = sv_glszm_single_label(
                    image, local_mask, size, strides, ndim, ni, angles, n_angles, Ng,
                    final_P + (size_t)li * (size_t)Ng * (size_t)max_zone, max_zone);
                if (label_max_zone < 0) {
                    #pragma omp atomic
                    failed += 1;
                }
            }
            free(local_mask);
        }
    }
#else
    for (int li = 0; li < n_labels; li++) {
        int lbl = labels[li];
        for (size_t i = 0; i < ni; i++)
            mask[i] = (sv_map[i] == lbl) ? 1 : 0;

        int label_max_zone = sv_glszm_single_label(
            image, mask, size, strides, ndim, ni, angles, n_angles, Ng,
            final_P + (size_t)li * (size_t)Ng * (size_t)max_zone, max_zone);
        if (label_max_zone < 0) {
            free(final_P);
            free(mask);
            free(strides);
            free(angles);
            return -1;
        }
        (void)label_max_zone;
    }
#endif
    if (failed) {
        free(final_P);
        free(mask);
        free(strides);
        free(angles);
        PyErr_NoMemory();
        return -1;
    }

    free(mask);
    free(strides);
    free(angles);

    *P_glszm_out = final_P;
    *max_zone_out = max_zone;
    return 0;
}

/* ── NGTDM ───────────────────────────────────────────────────────────── */

static void
sv_ngtdm_accumulate_z(double *P, int *image, int *sv_map,
                      int z, int sz, int sy, int sx,
                      int max_label, int *label_to_idx,
                      int *distances, int n_distances,
                      int *angles, int n_angles, int Ng)
{
    for (int y = 0; y < sy; y++) {
        for (int x = 0; x < sx; x++) {
            int idx = z * sy * sx + y * sx + x;
            int lbl = sv_map[idx];
            if (lbl <= 0 || lbl > max_label) continue;
            int li = label_to_idx[lbl];
            if (li < 0) continue;

            int gi = image[idx];
            if (gi <= 0 || gi > Ng) continue;

            double neighbor_sum = 0.0;
            int neighbor_count = 0;

            for (int di = 0; di < n_distances; di++) {
                int dist = distances[di];
                for (int ai = 0; ai < n_angles; ai++) {
                    int dz = angles[ai * 3] * dist;
                    int dy = angles[ai * 3 + 1] * dist;
                    int dx = angles[ai * 3 + 2] * dist;

                    int nz = z + dz, ny = y + dy, nx = x + dx;
                    if (nz < 0 || nz >= sz || ny < 0 || ny >= sy || nx < 0 || nx >= sx)
                        continue;
                    int nidx = nz * sy * sx + ny * sx + nx;
                    if (sv_map[nidx] != lbl) continue;

                    neighbor_sum += (double)image[nidx];
                    neighbor_count++;
                }
            }

            double abs_diff = 0.0;
            if (neighbor_count > 0) {
                abs_diff = fabs((double)gi - neighbor_sum / neighbor_count);
            }

            int base = (li * Ng + (gi - 1)) * 3;
            P[base] += 1.0;
            P[base + 1] += abs_diff;
        }
    }
}

int
sv_calculate_ngtdm(int *image, int *sv_map, int *size, int ndim,
                   int *labels, int n_labels, int max_label, int *label_to_idx,
                   int *distances, int n_distances,
                   int Ng, int force2D, int force2Ddimension,
                   double **P_ngtdm_out)
{
    int *angles = NULL;
    int n_angles = 0;
    if (sv_generate_angles_bidirectional(size, ndim, force2D, force2Ddimension,
                                           &angles, &n_angles) < 0)
        return -1;

    /* P_ngtdm: [n_labels, Ng, 3]  columns: count, sum_abs_diff, gray_level */
    double *P = (double *)calloc((size_t)n_labels * Ng * 3, sizeof(double));
    if (!P) { free(angles); PyErr_NoMemory(); return -1; }

    for (int li = 0; li < n_labels; li++) {
        for (int gl = 0; gl < Ng; gl++)
            P[(li * Ng + gl) * 3 + 2] = (double)(gl + 1);
    }

    int sz = (ndim >= 3) ? size[0] : 1;
    int sy = (ndim >= 2) ? size[ndim - 2] : 1;
    int sx = size[ndim - 1];

    sv_omp_apply_thread_limit();

#ifdef _OPENMP
    {
        size_t p_len = (size_t)n_labels * (size_t)Ng * 3;
        int omp_failed = 0;
        #pragma omp parallel
        {
            double *local = (double *)calloc(p_len, sizeof(double));
            if (local == NULL) {
                #pragma omp atomic
                omp_failed += 1;
            } else {
                int z;
                #pragma omp for schedule(static)
                for (z = 0; z < sz; z++) {
                    sv_ngtdm_accumulate_z(
                        local, image, sv_map, z, sz, sy, sx,
                        max_label, label_to_idx, distances, n_distances,
                        angles, n_angles, Ng);
                }
                #pragma omp critical
                {
                    /* Only reduce count / sum_abs_diff; gray-level column is
                     * already written in ``P`` and must not be added. */
                    for (int li = 0; li < n_labels; li++) {
                        for (int gl = 0; gl < Ng; gl++) {
                            size_t base = ((size_t)li * (size_t)Ng + (size_t)gl) * 3;
                            P[base] += local[base];
                            P[base + 1] += local[base + 1];
                        }
                    }
                }
                free(local);
            }
        }
        if (omp_failed) {
            memset(P, 0, p_len * sizeof(double));
            for (int li = 0; li < n_labels; li++) {
                for (int gl = 0; gl < Ng; gl++)
                    P[(li * Ng + gl) * 3 + 2] = (double)(gl + 1);
            }
            for (int z = 0; z < sz; z++) {
                sv_ngtdm_accumulate_z(
                    P, image, sv_map, z, sz, sy, sx,
                    max_label, label_to_idx, distances, n_distances,
                    angles, n_angles, Ng);
            }
        }
    }
#else
    for (int z = 0; z < sz; z++) {
        sv_ngtdm_accumulate_z(
            P, image, sv_map, z, sz, sy, sx,
            max_label, label_to_idx, distances, n_distances,
            angles, n_angles, Ng);
    }
#endif

    free(angles);
    *P_ngtdm_out = P;
    return 0;
}

/* ── GLDM ────────────────────────────────────────────────────────────── */

static void
sv_gldm_accumulate_z(long long *P, int *image, int *sv_map,
                     int z, int sz, int sy, int sx,
                     int max_label, int *label_to_idx,
                     int *distances, int n_distances,
                     int *angles, int n_angles, int Ng, int alpha, int max_dep)
{
    for (int y = 0; y < sy; y++) {
        for (int x = 0; x < sx; x++) {
            int idx = z * sy * sx + y * sx + x;
            int lbl = sv_map[idx];
            if (lbl <= 0 || lbl > max_label) continue;
            int li = label_to_idx[lbl];
            if (li < 0) continue;

            int gi = image[idx];
            if (gi <= 0 || gi > Ng) continue;

            int dep = 0;
            for (int di = 0; di < n_distances; di++) {
                int dist = distances[di];
                for (int ai = 0; ai < n_angles; ai++) {
                    int dz = angles[ai * 3] * dist;
                    int dy = angles[ai * 3 + 1] * dist;
                    int dx = angles[ai * 3 + 2] * dist;

                    int nz = z + dz, ny = y + dy, nx = x + dx;
                    if (nz < 0 || nz >= sz || ny < 0 || ny >= sy || nx < 0 || nx >= sx)
                        continue;
                    int nidx = nz * sy * sx + ny * sx + nx;
                    if (sv_map[nidx] != lbl) continue;

                    int abs_diff = abs(image[nidx] - gi);
                    if (abs_diff <= alpha) dep++;
                }
            }

            if (dep >= 0 && dep < max_dep) {
                P[((li * Ng + (gi - 1)) * max_dep + dep)]++;
            }
        }
    }
}

int
sv_calculate_gldm(int *image, int *sv_map, int *size, int ndim,
                  int *labels, int n_labels, int max_label, int *label_to_idx,
                  int *distances, int n_distances,
                  int Ng, int alpha, int force2D, int force2Ddimension,
                  long long **P_gldm_out, int *max_dep_out)
{
    int *angles = NULL;
    int n_angles = 0;
    if (sv_generate_angles_bidirectional(size, ndim, force2D, force2Ddimension,
                           &angles, &n_angles) < 0)
        return -1;

    /* PyRadiomics GLDM column count per gray level: Na * 2 + 1 (bidirectional Na). */
    int max_dep = n_angles * 2 + 1;

    int sz = (ndim >= 3) ? size[0] : 1;
    int sy = (ndim >= 2) ? size[ndim - 2] : 1;
    int sx = size[ndim - 1];

    size_t p_len = (size_t)n_labels * (size_t)Ng * (size_t)max_dep;
    long long *P = (long long *)calloc(p_len, sizeof(long long));
    if (!P) { free(angles); PyErr_NoMemory(); return -1; }

    sv_omp_apply_thread_limit();

#ifdef _OPENMP
    {
        int omp_failed = 0;
        #pragma omp parallel
        {
            long long *local = (long long *)calloc(p_len, sizeof(long long));
            if (local == NULL) {
                #pragma omp atomic
                omp_failed += 1;
            } else {
                int z;
                #pragma omp for schedule(static)
                for (z = 0; z < sz; z++) {
                    sv_gldm_accumulate_z(
                        local, image, sv_map, z, sz, sy, sx,
                        max_label, label_to_idx, distances, n_distances,
                        angles, n_angles, Ng, alpha, max_dep);
                }
                #pragma omp critical
                sv_add_ll_into(P, local, p_len);
                free(local);
            }
        }
        if (omp_failed) {
            memset(P, 0, p_len * sizeof(long long));
            for (int z = 0; z < sz; z++) {
                sv_gldm_accumulate_z(
                    P, image, sv_map, z, sz, sy, sx,
                    max_label, label_to_idx, distances, n_distances,
                    angles, n_angles, Ng, alpha, max_dep);
            }
        }
    }
#else
    for (int z = 0; z < sz; z++) {
        sv_gldm_accumulate_z(
            P, image, sv_map, z, sz, sy, sx,
            max_label, label_to_idx, distances, n_distances,
            angles, n_angles, Ng, alpha, max_dep);
    }
#endif

    free(angles);
    *P_gldm_out = P;
    *max_dep_out = max_dep;
    return 0;
}

/* ── First-order statistics ──────────────────────────────────────────── */

int
sv_calculate_firstorder(double *image, int *sv_map, int *size, int ndim,
                        int *labels, int n_labels, int max_label, int *label_to_idx,
                        int Ng, double binWidth,
                        double voxelArrayShift, double voxelVolume,
                        double **stats_out, int *n_stats_out)
{
    int sz = (ndim >= 3) ? size[0] : 1;
    int sy = (ndim >= 2) ? size[ndim - 2] : 1;
    int sx = size[ndim - 1];
    int total_voxels = sz * sy * sx;

    double *stats = (double *)calloc((size_t)n_labels * N_FIRSTORDER_STATS,
                                      sizeof(double));
    int *counts = (int *)calloc((size_t)n_labels, sizeof(int));
    double *sums = (double *)calloc((size_t)n_labels, sizeof(double));
    double *sq_sums = (double *)calloc((size_t)n_labels, sizeof(double));
    double *abs_sums = (double *)calloc((size_t)n_labels, sizeof(double));

    double **sorted = (double **)malloc((size_t)n_labels * sizeof(double *));
    int *sorted_cap = (int *)calloc((size_t)n_labels, sizeof(int));
    int *sorted_len = (int *)calloc((size_t)n_labels, sizeof(int));

    if (!stats || !counts || !sums || !sq_sums || !abs_sums || !sorted ||
        !sorted_cap || !sorted_len) {
        free(stats); free(counts); free(sums); free(sq_sums); free(abs_sums);
        free(sorted); free(sorted_cap); free(sorted_len);
        PyErr_NoMemory(); return -1;
    }

    for (int i = 0; i < n_labels; i++) {
        sorted[i] = NULL;
        sorted_cap[i] = 0;
        sorted_len[i] = 0;
    }

    for (int idx = 0; idx < total_voxels; idx++) {
        int lbl = sv_map[idx];
        if (lbl <= 0 || lbl > max_label) continue;
        int li = label_to_idx[lbl];
        if (li < 0) continue;

        double val = image[idx];
        counts[li]++;
        sums[li] += val;
        sq_sums[li] += val * val;
        abs_sums[li] += fabs(val);

        if (sorted_len[li] >= sorted_cap[li]) {
            int new_cap = sorted_cap[li] == 0 ? 256 : sorted_cap[li] * 2;
            double *new_buf = (double *)realloc(sorted[li], (size_t)new_cap * sizeof(double));
            if (!new_buf) {
                for (int j = 0; j < n_labels; j++) free(sorted[j]);
                free(stats); free(counts); free(sums); free(sq_sums);
                free(abs_sums); free(sorted); free(sorted_cap); free(sorted_len);
                PyErr_NoMemory(); return -1;
            }
            sorted[li] = new_buf;
            sorted_cap[li] = new_cap;
        }
        sorted[li][sorted_len[li]++] = val;
    }

    /* Voxel collection stays serial (per-label realloc). Stats over labels
     * are independent: each ``li`` writes a distinct ``stats`` slice. */
    sv_omp_apply_thread_limit();
    {
    int li;
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (li = 0; li < n_labels; li++) {
        int n = counts[li];
        double *base = stats + li * N_FIRSTORDER_STATS;

        if (n == 0) {
            for (int s = 0; s < N_FIRSTORDER_STATS; s++) base[s] = NAN;
            continue;
        }

        double mean = sums[li] / n;
        double variance = (sq_sums[li] / n) - (mean * mean);
        if (variance < 0) variance = 0;
        double stddev = sqrt(variance);

        qsort(sorted[li], (size_t)n, sizeof(double), _dbl_cmp);

        double minimum = sorted[li][0];
        double maximum = sorted[li][n - 1];
        double range = maximum - minimum;

        double p10 = _percentile(sorted[li], n, 10.0);
        double p90 = _percentile(sorted[li], n, 90.0);
        double median = _percentile(sorted[li], n, 50.0);
        double q1 = _percentile(sorted[li], n, 25.0);
        double q3 = _percentile(sorted[li], n, 75.0);
        double iqr = q3 - q1;

        double mad = 0, rmad = 0;
        for (int i = 0; i < n; i++) mad += fabs(sorted[li][i] - mean);
        mad /= n;

        /* PyRadiomics RobustMAD: voxels in [p10, p90], MAD about the
         * subset mean (not the full-ROI mean). */
        {
            int r_n = 0;
            double r_sum = 0.0;
            for (int i = 0; i < n; i++) {
                if (sorted[li][i] >= p10 && sorted[li][i] <= p90) {
                    r_sum += sorted[li][i];
                    r_n++;
                }
            }
            if (r_n > 0) {
                double r_mean = r_sum / (double)r_n;
                for (int i = 0; i < n; i++) {
                    if (sorted[li][i] >= p10 && sorted[li][i] <= p90)
                        rmad += fabs(sorted[li][i] - r_mean);
                }
                rmad /= (double)r_n;
            }
        }

        /* PyRadiomics: Energy = sum((X + voxelArrayShift)^2);
         * TotalEnergy = Energy * prod(spacing). Expanding the square
         * reuses the raw-moment accumulators collected above. */
        double energy = sq_sums[li]
            + (2.0 * voxelArrayShift * sums[li])
            + ((double)n * voxelArrayShift * voxelArrayShift);
        double total_energy = energy * voxelVolume;
        double rms = (n > 0) ? sqrt(energy / (double)n) : 0.0;

        double skewness = 0, kurtosis = 0;
        if (stddev > 0) {
            double m3 = 0, m4 = 0;
            for (int i = 0; i < n; i++) {
                double d = (sorted[li][i] - mean) / stddev;
                m3 += d * d * d;
                m4 += d * d * d * d;
            }
            skewness = m3 / n;
            /* PyRadiomics kurtosis is NOT excess (no -3). */
            kurtosis = m4 / n;
        }

        double entropy = 0, uniformity = 0;
        if (range > 0 && binWidth > 0) {
            int n_bins = Ng;
            int *hist = (int *)calloc((size_t)n_bins, sizeof(int));
            if (hist) {
                for (int i = 0; i < n; i++) {
                    int bin = (int)((sorted[li][i] - minimum) / binWidth);
                    if (bin >= n_bins) bin = n_bins - 1;
                    if (bin < 0) bin = 0;
                    hist[bin]++;
                }
                for (int b = 0; b < n_bins; b++) {
                    if (hist[b] > 0) {
                        double p = (double)hist[b] / n;
                        entropy -= p * (log(p) / log(2.0));
                        uniformity += p * p;
                    }
                }
                free(hist);
            }
        }

        base[0] = energy;
        base[1] = total_energy;
        base[2] = entropy;
        base[3] = minimum;
        base[4] = p10;
        base[5] = p90;
        base[6] = maximum;
        base[7] = mean;
        base[8] = median;
        base[9] = iqr;
        base[10] = range;
        base[11] = mad;
        base[12] = rmad;
        base[13] = rms;
        base[14] = skewness;
        base[15] = kurtosis;
        base[16] = uniformity;
    }
    }

    for (int i = 0; i < n_labels; i++) free(sorted[i]);
    free(sorted); free(sorted_cap); free(sorted_len);
    free(counts); free(sums); free(sq_sums); free(abs_sums);

    *stats_out = stats;
    *n_stats_out = N_FIRSTORDER_STATS;
    return 0;
}

/* ── GLCM formulas (24 default features; MCC is last) ─────────────── */

#define N_GLCM_FORMULA_FEATURES 24
#define SV_GLCM_EPS 2.220446049250313e-16

static double sv_second_eig_symmetric(double *A, int n, double *d, double *e);
static double sv_mcc_from_norm_glcm(const double *S, const double *px, int Ng,
                                    double *Q, double *inv_px, double *inv_py,
                                    int *used);

int
sv_glcm_formulas(const double *P, int n_labels, int Ng, int n_angles,
                 int symmetrical, const double *gray, const double *ng_full,
                 double **out_features, int *n_features_out)
{
    if (n_labels <= 0 || Ng <= 0 || n_angles <= 0) {
        PyErr_SetString(PyExc_ValueError, "GLCM formula batch has an empty axis");
        return -1;
    }
    double *out = (double *)malloc((size_t)n_labels * N_GLCM_FORMULA_FEATURES * sizeof(double));
    size_t n_va_feat = (size_t)n_labels * (size_t)n_angles * N_GLCM_FORMULA_FEATURES;
    double *ang_feat = (double *)malloc(n_va_feat * sizeof(double));
    if (!out || !ang_feat) {
        free(out);
        free(ang_feat);
        PyErr_NoMemory();
        return -1;
    }
    for (size_t i = 0; i < n_va_feat; i++)
        ang_feat[i] = NAN;

    sv_omp_apply_thread_limit();
#ifdef _OPENMP
    #pragma omp parallel
#endif
    {
        double *S = (double *)malloc((size_t)Ng * (size_t)Ng * sizeof(double));
        double *Q = (double *)malloc((size_t)Ng * (size_t)Ng * sizeof(double));
        double *px = (double *)malloc((size_t)Ng * sizeof(double));
        double *py = (double *)malloc((size_t)Ng * sizeof(double));
        double *px_sub = (double *)malloc((size_t)Ng * sizeof(double));
        double *px_add = (double *)malloc((size_t)(2 * Ng - 1) * sizeof(double));
        double *inv_px = (double *)malloc((size_t)Ng * sizeof(double));
        double *inv_py = (double *)malloc((size_t)Ng * sizeof(double));
        int *used = (int *)malloc((size_t)Ng * sizeof(int));
        if (!S || !Q || !px || !py || !px_sub || !px_add || !inv_px || !inv_py || !used) {
#ifdef _OPENMP
            #pragma omp critical
#endif
            {
                PyErr_NoMemory();
            }
            free(S); free(Q); free(px); free(py); free(px_sub); free(px_add);
            free(inv_px); free(inv_py); free(used);
        } else {
            int va;
#ifdef _OPENMP
            #pragma omp for schedule(static)
#endif
            for (va = 0; va < n_labels * n_angles; va++) {
                int v = va / n_angles;
                int a = va % n_angles;
                double ng_scale = (ng_full != NULL) ? ng_full[v] : (double)Ng;
                if (ng_scale <= 0.0)
                    ng_scale = (double)Ng;
                {
                    double sum = 0.0;
                    for (int i = 0; i < Ng; i++) {
                        for (int j = 0; j < Ng; j++) {
                            double val = P[(((size_t)v * Ng + i) * Ng + j) * n_angles + a];
                            if (symmetrical)
                                val += P[(((size_t)v * Ng + j) * Ng + i) * n_angles + a];
                            S[i * Ng + j] = val;
                            sum += val;
                        }
                    }
                    if (!(sum > 0.0))
                        continue;
                    for (int i = 0; i < Ng * Ng; i++)
                        S[i] /= sum;

                    for (int i = 0; i < Ng; i++) {
                        px[i] = 0.0;
                        py[i] = 0.0;
                        px_sub[i] = 0.0;
                    }
                    for (int k = 0; k < 2 * Ng - 1; k++)
                        px_add[k] = 0.0;

                    double ux = 0.0, uy = 0.0;
                    double joint_energy = 0.0, hxy = 0.0, maxprob = 0.0;
                    double ac = 0.0, contrast = 0.0;
                    for (int i = 0; i < Ng; i++) {
                        double gi = gray[i];
                        for (int j = 0; j < Ng; j++) {
                            double p = S[i * Ng + j];
                            double gj = gray[j];
                            px[i] += p;
                            py[j] += p;
                            ux += p * gi;
                            uy += p * gj;
                            joint_energy += p * p;
                            hxy -= p * (log(p + SV_GLCM_EPS) / log(2.0));
                            if (p > maxprob)
                                maxprob = p;
                            ac += p * gi * gj;
                            double d = fabs(gi - gj);
                            contrast += p * d * d;
                            int kd = (int)(fabs(gi - gj) + 0.5);
                            if (kd >= 0 && kd < Ng)
                                px_sub[kd] += p;
                            int ks = (int)(gi + gj + 0.5) - 2;
                            if (ks >= 0 && ks < 2 * Ng - 1)
                                px_add[ks] += p;
                        }
                    }

                    double cp = 0.0, cs = 0.0, ct = 0.0, sumsq = 0.0;
                    double corm = 0.0, varx = 0.0, vary = 0.0;
                    double hx = 0.0, hy = 0.0, hxy1 = 0.0, hxy2 = 0.0;
                    for (int i = 0; i < Ng; i++) {
                        double gi = gray[i];
                        hx -= px[i] * (log(px[i] + SV_GLCM_EPS) / log(2.0));
                        hy -= py[i] * (log(py[i] + SV_GLCM_EPS) / log(2.0));
                        for (int j = 0; j < Ng; j++) {
                            double p = S[i * Ng + j];
                            double gj = gray[j];
                            double t = gi + gj - ux - uy;
                            double t2 = t * t;
                            cp += p * t2 * t2;
                            cs += p * t2 * t;
                            ct += p * t2;
                            double dx = gi - ux;
                            double dy = gj - uy;
                            sumsq += p * dx * dx;
                            corm += p * dx * dy;
                            varx += p * dx * dx;
                            vary += p * dy * dy;
                            double pxpy = px[i] * py[j];
                            hxy1 -= p * (log(pxpy + SV_GLCM_EPS) / log(2.0));
                            hxy2 -= pxpy * (log(pxpy + SV_GLCM_EPS) / log(2.0));
                        }
                    }
                    double sigx = sqrt(varx);
                    double sigy = sqrt(vary);
                    double corr = (sigx * sigy == 0.0) ? 1.0 : (corm / (sigx * sigy + SV_GLCM_EPS));

                    double da = 0.0, dent = 0.0, idm = 0.0, idmn = 0.0, id = 0.0, idn = 0.0, inv = 0.0;
                    for (int k = 0; k < Ng; k++) {
                        double pk = px_sub[k];
                        double kd = (double)k;
                        da += kd * pk;
                        dent -= pk * (log(pk + SV_GLCM_EPS) / log(2.0));
                        idm += pk / (1.0 + kd * kd);
                        idmn += pk / (1.0 + (kd * kd) / (ng_scale * ng_scale));
                        id += pk / (1.0 + kd);
                        idn += pk / (1.0 + kd / ng_scale);
                        if (k > 0)
                            inv += pk / (kd * kd);
                    }
                    double dvar = 0.0;
                    for (int k = 0; k < Ng; k++)
                        dvar += px_sub[k] * (((double)k - da) * ((double)k - da));

                    double savg = 0.0, sent = 0.0;
                    for (int k = 0; k < 2 * Ng - 1; k++) {
                        double pk = px_add[k];
                        double kv = (double)(k + 2);
                        savg += kv * pk;
                        sent -= pk * (log(pk + SV_GLCM_EPS) / log(2.0));
                    }

                    double imc1;
                    double div = (hx > hy) ? hx : hy;
                    if (div != 0.0)
                        imc1 = (hxy - hxy1) / div;
                    else
                        imc1 = 0.0;
                    double imc2 = 0.0;
                    if (hxy2 != hxy) {
                        double inner = 1.0 - exp(-2.0 * (hxy2 - hxy));
                        if (inner > 0.0)
                            imc2 = sqrt(inner);
                    }

                    /* Reuse the already-normalised S and px; do not walk P again. */
                    double mcc_ang = sv_mcc_from_norm_glcm(
                        S, px, Ng, Q, inv_px, inv_py, used);
                    double vals[N_GLCM_FORMULA_FEATURES] = {
                        ac, ux, cp, cs, ct, contrast, corr,
                        da, dent, dvar, joint_energy, hxy, imc1, imc2,
                        idm, idmn, id, idn, inv, maxprob, savg, sent, sumsq,
                        mcc_ang
                    };
                    double *slot = ang_feat
                        + ((size_t)v * n_angles + (size_t)a) * N_GLCM_FORMULA_FEATURES;
                    for (int f = 0; f < N_GLCM_FORMULA_FEATURES; f++)
                        slot[f] = vals[f];
                }
            }
            free(S); free(Q); free(px); free(py); free(px_sub); free(px_add);
            free(inv_px); free(inv_py); free(used);
        }
    }

    if (PyErr_Occurred()) {
        free(out);
        free(ang_feat);
        return -1;
    }
    for (int v = 0; v < n_labels; v++) {
        double *row = out + (size_t)v * N_GLCM_FORMULA_FEATURES;
        for (int f = 0; f < N_GLCM_FORMULA_FEATURES; f++) {
            double acc = 0.0;
            int cnt = 0;
            for (int a = 0; a < n_angles; a++) {
                double val = ang_feat[
                    ((size_t)v * n_angles + (size_t)a) * N_GLCM_FORMULA_FEATURES + (size_t)f];
                if (val == val) {
                    acc += val;
                    cnt += 1;
                }
            }
            row[f] = (cnt > 0) ? (acc / (double)cnt) : NAN;
        }
    }
    free(ang_feat);
    *out_features = out;
    *n_features_out = N_GLCM_FORMULA_FEATURES;
    return 0;
}

/* ── GLCM MCC (explicit Q + cyclic Jacobi, OpenMP over label×angle) ── */

#define SV_QL_MAX_ITER 40

/*
 * Householder reduction of symmetric row-major A[n,n] to tridiagonal form.
 * Eigenvalues-only (no eigenvector accumulation). On exit, d is the
 * diagonal and e[1..n-1] is the sub-diagonal (e[0] = 0). A is destroyed.
 * Translation of the public-domain EISPACK TRED1 recurrence.
 */
static void
sv_tred1(double *A, int n, double *d, double *e)
{
    int i, j, k;
    for (i = 0; i < n; i++)
        d[i] = A[(size_t)(n - 1) * n + i];
    for (i = n - 1; i >= 1; i--) {
        int l = i - 1;
        double h = 0.0, scale = 0.0;
        for (k = 0; k <= l; k++)
            scale += fabs(d[k]);
        if (scale == 0.0) {
            e[i] = d[l];
            for (j = 0; j <= l; j++) {
                d[j] = A[(size_t)(i - 1) * n + j];
                A[(size_t)i * n + j] = 0.0;
                A[(size_t)j * n + i] = 0.0;
            }
        } else {
            for (k = 0; k <= l; k++) {
                d[k] /= scale;
                h += d[k] * d[k];
            }
            double f = d[l];
            double g = (f >= 0.0) ? -sqrt(h) : sqrt(h);
            e[i] = scale * g;
            h -= f * g;
            d[l] = f - g;
            for (j = 0; j <= l; j++)
                e[j] = 0.0;
            for (j = 0; j <= l; j++) {
                f = d[j];
                A[(size_t)j * n + i] = f;
                g = e[j] + A[(size_t)j * n + j] * f;
                for (k = j + 1; k <= l; k++) {
                    g += A[(size_t)k * n + j] * d[k];
                    e[k] += A[(size_t)k * n + j] * f;
                }
                e[j] = g;
            }
            f = 0.0;
            for (j = 0; j <= l; j++) {
                e[j] /= h;
                f += e[j] * d[j];
            }
            double hh = f / (h + h);
            for (j = 0; j <= l; j++)
                e[j] -= hh * d[j];
            for (j = 0; j <= l; j++) {
                f = d[j];
                g = e[j];
                for (k = j; k <= l; k++)
                    A[(size_t)k * n + j] -= f * e[k] + g * d[k];
                d[j] = A[(size_t)(i - 1) * n + j];
                A[(size_t)i * n + j] = 0.0;
            }
        }
        d[i] = h;
    }
    e[0] = 0.0;
    for (i = 0; i < n; i++) {
        d[i] = A[(size_t)i * n + i];
        A[(size_t)i * n + i] = 0.0;
    }
}

/*
 * Implicit QL on a symmetric tridiagonal matrix (eigenvalues only).
 * d is the diagonal, e[1..n-1] the sub-diagonal. Returns 0 on success.
 * Recurrence follows the public-domain EISPACK IMTQL1 algorithm.
 */
static int
sv_imtql1(int n, double *d, double *e)
{
    int l, i, m, iter;
    if (n == 1)
        return 0;
    for (i = 1; i < n; i++)
        e[i - 1] = e[i];
    e[n - 1] = 0.0;
    for (l = 0; l < n; l++) {
        iter = 0;
        for (;;) {
            for (m = l; m < n - 1; m++) {
                double testd = fabs(d[m]) + fabs(d[m + 1]);
                if (fabs(e[m]) + testd == testd)
                    break;
            }
            if (m == l)
                break;
            if (iter++ >= SV_QL_MAX_ITER)
                return -1;
            double g = (d[l + 1] - d[l]) / (2.0 * e[l]);
            double r = hypot(g, 1.0);
            g = d[m] - d[l] + e[l] / (g + ((g >= 0.0) ? r : -r));
            double s = 1.0, c = 1.0, p = 0.0;
            for (i = m - 1; i >= l; i--) {
                double f = s * e[i];
                double b = c * e[i];
                r = hypot(f, g);
                e[i + 1] = r;
                if (r == 0.0) {
                    d[i + 1] -= p;
                    e[m] = 0.0;
                    break;
                }
                s = f / r;
                c = g / r;
                g = d[i + 1] - p;
                r = (d[i] - g) * s + 2.0 * c * b;
                p = s * r;
                d[i + 1] = g + p;
                g = c * r - b;
            }
            if (r == 0.0 && i >= l)
                continue;
            d[l] -= p;
            e[l] = g;
            e[m] = 0.0;
        }
    }
    return 0;
}

/*
 * Second-largest eigenvalue of a symmetric row-major A[n,n].
 * Householder + implicit QL (O(n^3) once). A is destroyed. ``d`` and
 * ``e`` are length-n workspaces. If QL does not converge the caller
 * receives 0 (finite MCC).
 *
 * Applied to the Gram matrix that is diagonally similar to the
 * (non-symmetric) PyRadiomics Q, not to Q itself.
 */
static double
sv_second_eig_symmetric(double *A, int n, double *d, double *e)
{
    int i;
    if (n < 2)
        return 0.0;
    sv_tred1(A, n, d, e);
    if (sv_imtql1(n, d, e) != 0) {
        /* Rare QL failure: recover via Jacobi on the now-destroyed A is
         * impossible. Return 0 so the caller keeps a finite MCC. */
        return 0.0;
    }
    double top = -1.0e300;
    double second = -1.0e300;
    for (i = 0; i < n; i++) {
        double ev = d[i];
        if (ev > top) {
            second = top;
            top = ev;
        } else if (ev > second) {
            second = ev;
        }
    }
    return second;
}

/*
 * MCC for one already-normalised GLCM ``S`` with row marginals ``px``.
 *
 * Q is not symmetric. After GLCM symmetrisation (px == py) it is
 * diagonally similar to the Gram matrix
 *   G[i, j] = sum_k P[i,k] P[j,k] / (sqrt(px[i]) sqrt(px[j]) px[k])
 * which shares eigenvalues with Q. Jacobi runs on G.
 */
static double
sv_mcc_from_norm_glcm(const double *S, const double *px, int Ng,
                      double *Q, double *inv_px, double *inv_py, int *used)
{
    int i, j, k, n_used;
    if (Ng < 2)
        return 1.0;
    n_used = 0;
    for (i = 0; i < Ng; i++) {
        if (px[i] > 0.0)
            used[n_used++] = i;
    }
    if (n_used < 2)
        return 1.0;
    for (i = 0; i < n_used; i++) {
        double pxi = px[used[i]];
        inv_px[i] = 1.0 / sqrt(pxi);
        inv_py[i] = 1.0 / (pxi + SV_GLCM_EPS);
    }
    for (i = 0; i < n_used; i++) {
        int ii = used[i];
        double isi = inv_px[i];
        for (j = 0; j <= i; j++) {
            int jj = used[j];
            double acc = 0.0;
            for (k = 0; k < n_used; k++) {
                int kk = used[k];
                acc += S[ii * Ng + kk] * S[jj * Ng + kk] * inv_py[k];
            }
            acc *= isi * inv_px[j];
            Q[i * n_used + j] = acc;
            Q[j * n_used + i] = acc;
        }
    }
    {
        /* inv_px / inv_py are free after Q is assembled; reuse as QL work. */
        double lam2 = sv_second_eig_symmetric(Q, n_used, inv_px, inv_py);
        return (lam2 > 0.0) ? sqrt(lam2) : 0.0;
    }
}

int
sv_glcm_mcc(const double *P, int n_labels, int Ng, int n_angles,
            int symmetrical, double **out_mcc)
{
    if (n_labels <= 0 || Ng <= 0 || n_angles <= 0) {
        PyErr_SetString(PyExc_ValueError, "GLCM MCC batch has an empty axis");
        return -1;
    }
    double *out = (double *)malloc((size_t)n_labels * sizeof(double));
    double *ang = (double *)malloc((size_t)n_labels * (size_t)n_angles * sizeof(double));
    if (!out || !ang) {
        free(out);
        free(ang);
        PyErr_NoMemory();
        return -1;
    }
    for (int i = 0; i < n_labels * n_angles; i++)
        ang[i] = NAN;

    sv_omp_apply_thread_limit();
#ifdef _OPENMP
    #pragma omp parallel
#endif
    {
        double *S = (double *)malloc((size_t)Ng * (size_t)Ng * sizeof(double));
        double *Q = (double *)malloc((size_t)Ng * (size_t)Ng * sizeof(double));
        double *px = (double *)malloc((size_t)Ng * sizeof(double));
        double *inv_px = (double *)malloc((size_t)Ng * sizeof(double));
        double *inv_py = (double *)malloc((size_t)Ng * sizeof(double));
        int *used = (int *)malloc((size_t)Ng * sizeof(int));
        if (!S || !Q || !px || !inv_px || !inv_py || !used) {
#ifdef _OPENMP
            #pragma omp critical
#endif
            {
                PyErr_NoMemory();
            }
            free(S); free(Q); free(px); free(inv_px); free(inv_py); free(used);
        } else {
            int va;
#ifdef _OPENMP
            #pragma omp for schedule(static)
#endif
            for (va = 0; va < n_labels * n_angles; va++) {
                int v = va / n_angles;
                int a = va % n_angles;
                int i, j;
                double sum = 0.0;
                for (i = 0; i < Ng; i++) {
                    for (j = 0; j < Ng; j++) {
                        double val = P[(((size_t)v * Ng + i) * Ng + j) * n_angles + a];
                        if (symmetrical)
                            val += P[(((size_t)v * Ng + j) * Ng + i) * n_angles + a];
                        S[i * Ng + j] = val;
                        sum += val;
                    }
                }
                if (!(sum > 0.0))
                    continue;
                double inv_sum = 1.0 / sum;
                for (i = 0; i < Ng * Ng; i++)
                    S[i] *= inv_sum;
                for (i = 0; i < Ng; i++)
                    px[i] = 0.0;
                for (i = 0; i < Ng; i++) {
                    for (j = 0; j < Ng; j++)
                        px[i] += S[i * Ng + j];
                }
                ang[(size_t)v * n_angles + a] = sv_mcc_from_norm_glcm(
                    S, px, Ng, Q, inv_px, inv_py, used);
            }
            free(S); free(Q); free(px); free(inv_px); free(inv_py); free(used);
        }
    }

    if (PyErr_Occurred()) {
        free(out);
        free(ang);
        return -1;
    }
    for (int v = 0; v < n_labels; v++) {
        double acc = 0.0;
        int cnt = 0;
        for (int a = 0; a < n_angles; a++) {
            double val = ang[(size_t)v * n_angles + a];
            if (val == val) {
                acc += val;
                cnt += 1;
            }
        }
        out[v] = (cnt > 0) ? (acc / (double)cnt) : NAN;
    }
    free(ang);
    *out_mcc = out;
    return 0;
}

/* ── GLRLM formulas ──────────────────────────────────────────────────── */

#define N_GLRLM_FORMULA_FEATURES 16

int
sv_glrlm_formulas(const double *P, int n_labels, int Ng, int Nr, int n_angles,
                  const double *gray, double **out_features, int *n_features_out)
{
    if (n_labels <= 0 || Ng <= 0 || Nr <= 0 || n_angles <= 0) {
        PyErr_SetString(PyExc_ValueError, "GLRLM formula batch has an empty axis");
        return -1;
    }
    double *out = (double *)malloc((size_t)n_labels * N_GLRLM_FORMULA_FEATURES * sizeof(double));
    if (!out) {
        PyErr_NoMemory();
        return -1;
    }

    sv_omp_apply_thread_limit();
#ifdef _OPENMP
    #pragma omp parallel
#endif
    {
        double *pr = (double *)malloc((size_t)Nr * sizeof(double));
        double *pg = (double *)malloc((size_t)Ng * sizeof(double));
        if (!pr || !pg) {
#ifdef _OPENMP
            #pragma omp critical
#endif
            {
                PyErr_NoMemory();
            }
            free(pr); free(pg);
        } else {
            int v;
#ifdef _OPENMP
            #pragma omp for schedule(static)
#endif
            for (v = 0; v < n_labels; v++) {
                double acc[N_GLRLM_FORMULA_FEATURES];
                int cnt[N_GLRLM_FORMULA_FEATURES];
                for (int f = 0; f < N_GLRLM_FORMULA_FEATURES; f++) {
                    acc[f] = 0.0;
                    cnt[f] = 0;
                }
                for (int a = 0; a < n_angles; a++) {
                    for (int j = 0; j < Nr; j++)
                        pr[j] = 0.0;
                    for (int i = 0; i < Ng; i++)
                        pg[i] = 0.0;
                    double nr = 0.0;
                    double n_p = 0.0;
                    double run_ent = 0.0;
                    double srlge = 0.0, srhge = 0.0, lrlge = 0.0, lrhge = 0.0;
                    for (int i = 0; i < Ng; i++) {
                        double gi = gray[i];
                        double i2 = gi * gi;
                        if (i2 <= 0.0)
                            i2 = SV_GLCM_EPS;
                        for (int j = 0; j < Nr; j++) {
                            double p = P[(((size_t)v * Ng + i) * Nr + j) * n_angles + a];
                            pr[j] += p;
                            pg[i] += p;
                            nr += p;
                            double rlen = (double)(j + 1);
                            n_p += p * rlen;
                            double j2 = rlen * rlen;
                            srlge += p / (i2 * j2);
                            srhge += p * i2 / j2;
                            lrlge += p * j2 / i2;
                            lrhge += p * i2 * j2;
                        }
                    }
                    if (!(nr > 0.0))
                        continue;
                    double sre = 0.0, lre = 0.0, rlnu = 0.0;
                    for (int j = 0; j < Nr; j++) {
                        double rlen = (double)(j + 1);
                        double j2 = rlen * rlen;
                        sre += pr[j] / j2;
                        lre += pr[j] * j2;
                        rlnu += pr[j] * pr[j];
                    }
                    double glnu = 0.0, lgre = 0.0, hgre = 0.0;
                    double u_i = 0.0;
                    for (int i = 0; i < Ng; i++) {
                        glnu += pg[i] * pg[i];
                        double gi = gray[i];
                        double i2 = gi * gi;
                        if (i2 <= 0.0)
                            i2 = SV_GLCM_EPS;
                        lgre += pg[i] / i2;
                        hgre += pg[i] * i2;
                        u_i += (pg[i] / nr) * gi;
                    }
                    double glv = 0.0;
                    for (int i = 0; i < Ng; i++) {
                        double d = gray[i] - u_i;
                        glv += (pg[i] / nr) * d * d;
                    }
                    double u_j = 0.0;
                    for (int j = 0; j < Nr; j++)
                        u_j += (pr[j] / nr) * (double)(j + 1);
                    double rv = 0.0;
                    for (int j = 0; j < Nr; j++) {
                        double d = (double)(j + 1) - u_j;
                        rv += (pr[j] / nr) * d * d;
                    }
                    for (int i = 0; i < Ng; i++) {
                        for (int j = 0; j < Nr; j++) {
                            double p = P[(((size_t)v * Ng + i) * Nr + j) * n_angles + a] / nr;
                            if (p > 0.0)
                                run_ent -= p * (log(p) / log(2.0));
                        }
                    }
                    double vals[N_GLRLM_FORMULA_FEATURES] = {
                        sre / nr,
                        lre / nr,
                        glnu / nr,
                        glnu / (nr * nr),
                        rlnu / nr,
                        rlnu / (nr * nr),
                        (n_p > 0.0) ? (nr / n_p) : NAN,
                        glv,
                        rv,
                        run_ent,
                        lgre / nr,
                        hgre / nr,
                        srlge / nr,
                        srhge / nr,
                        lrlge / nr,
                        lrhge / nr,
                    };
                    for (int f = 0; f < N_GLRLM_FORMULA_FEATURES; f++) {
                        if (vals[f] == vals[f]) {
                            acc[f] += vals[f];
                            cnt[f] += 1;
                        }
                    }
                }
                double *row = out + (size_t)v * N_GLRLM_FORMULA_FEATURES;
                for (int f = 0; f < N_GLRLM_FORMULA_FEATURES; f++)
                    row[f] = (cnt[f] > 0) ? (acc[f] / (double)cnt[f]) : NAN;
            }
            free(pr); free(pg);
        }
    }

    if (PyErr_Occurred()) {
        free(out);
        return -1;
    }
    *out_features = out;
    *n_features_out = N_GLRLM_FORMULA_FEATURES;
    return 0;
}

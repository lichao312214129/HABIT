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

/*
 * Return 1 when (z, y, x) is a valid GLRLM run start voxel for angle (dz, dy, dx),
 * matching PyRadiomics moving-dimension start rules on [0, size-1] bounds.
 */
static int
sv_glrlm_is_start_voxel(int z, int y, int x,
                        int sz, int sy, int sx,
                        int ndim, int dz, int dy, int dx)
{
    int valid = 0;

    if (ndim >= 3 && dz != 0) {
        if ((dz > 0 && z == 0) || (dz < 0 && z == sz - 1))
            valid = 1;
    }
    if (ndim >= 2 && dy != 0) {
        if ((dy > 0 && y == 0) || (dy < 0 && y == sy - 1))
            valid = 1;
    }
    if (dx != 0) {
        if ((dx > 0 && x == 0) || (dx < 0 && x == sx - 1))
            valid = 1;
    }
    return valid;
}

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

    long long *P = (long long *)calloc(
        (size_t)n_labels * Ng * Ng * n_angles, sizeof(long long));
    if (!P) { free(angles); PyErr_NoMemory(); return -1; }

    int sz = (ndim >= 3) ? size[0] : 1;
    int sy = (ndim >= 2) ? size[ndim - 2] : 1;
    int sx = size[ndim - 1];

    for (int z = 0; z < sz; z++) {
        for (int y = 0; y < sy; y++) {
            for (int x = 0; x < sx; x++) {
                int idx = z * sy * sx + y * sx + x;
                int lbl = sv_map[idx];
                if (lbl <= 0 || lbl > max_label) continue;
                int li = label_to_idx[lbl];
                if (li < 0) continue;

                int gi = image[idx];
                if (gi <= 0 || gi > Ng) continue;

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

                        int gj = image[nidx];
                        if (gj <= 0 || gj > Ng) continue;

                        P[((li * Ng + (gi - 1)) * Ng + (gj - 1)) * n_angles + ai]++;
                    }
                }
            }
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

    for (int ai = 0; ai < n_angles; ai++) {
        int dz = angles[ai * 3];
        int dy = angles[ai * 3 + 1];
        int dx = angles[ai * 3 + 2];

        for (int z = 0; z < sz; z++) {
            for (int y = 0; y < sy; y++) {
                for (int x = 0; x < sx; x++) {
                    if (!sv_glrlm_is_start_voxel(z, y, x, sz, sy, sx, ndim, dz, dy, dx))
                        continue;

                    int idx = z * sy * sx + y * sx + x;
                    int lbl = sv_map[idx];
                    if (lbl <= 0 || lbl > max_label)
                        continue;
                    int li = label_to_idx[lbl];
                    if (li < 0)
                        continue;

                    int cz = z, cy = y, cx = x;
                    int gl = -1;
                    int rl = 0;
                    int elements = 0;

                    /* PyRadiomics-style ray march: collect runs of constant gray level
                     * within the same supervoxel label along this angle. */
                    for (;;) {
                        if (cz < 0 || cz >= sz || cy < 0 || cy >= sy || cx < 0 || cx >= sx)
                            break;

                        int cidx = cz * sy * sx + cy * sx + cx;
                        int in_label = (sv_map[cidx] == lbl);
                        int gi = image[cidx];

                        if (in_label && gi > 0 && gi <= Ng) {
                            elements++;
                            if (gl < 0) {
                                gl = gi;
                                rl = 0;
                            } else if (gi == gl) {
                                rl++;
                            } else {
                                sv_glrlm_record_run(P, li, Ng, Nr, n_angles, ai, gl, rl);
                                gl = gi;
                                rl = 0;
                            }
                        } else if (gl >= 0) {
                            sv_glrlm_record_run(P, li, Ng, Nr, n_angles, ai, gl, rl);
                            gl = -1;
                            rl = 0;
                        }

                        cz += dz;
                        cy += dy;
                        cx += dx;
                    }

                    if (gl >= 0)
                        sv_glrlm_record_run(P, li, Ng, Nr, n_angles, ai, gl, rl);

                    if (elements > 1)
                        multi_element[li * n_angles + ai] = 1;
                }
            }
        }

        /* Remove angle columns that are degenerate for a label (2D segmentation). */
        for (int li = 0; li < n_labels; li++) {
            if (multi_element[li * n_angles + ai])
                continue;
            for (int gi = 0; gi < Ng; gi++)
                P[((li * Ng + gi) * Nr + 0) * n_angles + ai] = 0;
        }
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

    int max_zone = 1;
    for (int li = 0; li < n_labels; li++) {
        int lbl = labels[li];
        for (size_t i = 0; i < ni; i++)
            mask[i] = (sv_map[i] == lbl) ? 1 : 0;

        int label_max_zone = sv_glszm_single_label(
            image, mask, size, strides, ndim, ni, angles, n_angles, Ng,
            NULL, 0);
        if (label_max_zone < 0) {
            free(mask);
            free(strides);
            free(angles);
            return -1;
        }
        if (label_max_zone > max_zone)
            max_zone = label_max_zone;
    }

    long long *final_P = (long long *)calloc(
        (size_t)n_labels * (size_t)Ng * (size_t)max_zone, sizeof(long long));
    if (!final_P) {
        free(mask);
        free(strides);
        free(angles);
        PyErr_NoMemory();
        return -1;
    }

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

    free(mask);
    free(strides);
    free(angles);

    *P_glszm_out = final_P;
    *max_zone_out = max_zone;
    return 0;
}

/* ── NGTDM ───────────────────────────────────────────────────────────── */

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

    for (int z = 0; z < sz; z++) {
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

    free(angles);
    *P_ngtdm_out = P;
    return 0;
}

/* ── GLDM ────────────────────────────────────────────────────────────── */

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

    long long *P = (long long *)calloc(
        (size_t)n_labels * Ng * max_dep, sizeof(long long));
    if (!P) { free(angles); PyErr_NoMemory(); return -1; }

    for (int z = 0; z < sz; z++) {
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

    for (int li = 0; li < n_labels; li++) {
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

        int r_lo = (int)floor(n * 0.1);
        int r_hi = (int)ceil(n * 0.9);
        int r_n = r_hi - r_lo;
        if (r_n > 0) {
            for (int i = r_lo; i < r_hi; i++) rmad += fabs(sorted[li][i] - mean);
            rmad /= r_n;
        }

        double rms = sqrt(sq_sums[li] / n);

        double skewness = 0, kurtosis = 0;
        if (stddev > 0) {
            double m3 = 0, m4 = 0;
            for (int i = 0; i < n; i++) {
                double d = (sorted[li][i] - mean) / stddev;
                m3 += d * d * d;
                m4 += d * d * d * d;
            }
            skewness = m3 / n;
            kurtosis = (m4 / n) - 3.0;
        }

        double energy = sq_sums[li];
        double total_energy = energy * n;

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

    for (int i = 0; i < n_labels; i++) free(sorted[i]);
    free(sorted); free(sorted_cap); free(sorted_len);
    free(counts); free(sums); free(sq_sums); free(abs_sums);

    *stats_out = stats;
    *n_stats_out = N_FIRSTORDER_STATS;
    return 0;
}

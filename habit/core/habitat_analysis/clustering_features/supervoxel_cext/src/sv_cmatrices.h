/*
 * Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
 *
 * This file is part of HABIT (Habitat Analysis: Biomedical Imaging Toolkit).
 * Use is governed by the HABIT Software License — see the LICENSE file in the
 * project root for the full text. Summary:
 *
 *   - Non-commercial use (academic, research, education, personal) is permitted
 *     provided that copyright notices are retained and HABIT usage is
 *     acknowledged in publications, reports, or documentation.
 *   - Commercial use requires prior written consent from the copyright holder
 *     (lichao19870617@163.com) and public acknowledgment of HABIT usage in
 *     product documentation or user-facing materials.
 *   - Unauthorized commercial use or removal of attribution is prohibited.
 */

#ifndef SV_CMATRICES_H
#define SV_CMATRICES_H

#include <Python.h>
#include <numpy/arrayobject.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Multi-label supervoxel matrix calculation C API.
 *
 * All functions take a discretized image array and a multi-label supervoxel
 * map, computing texture matrices for every requested label in a single C
 * pass.  This eliminates Python-level per-label loops and redundant image
 * scans, yielding significant speed-ups for large supervoxel maps.
 *
 * Output arrays have leading dimension n_labels.  Angles are shared across
 * labels (generated once from image geometry).
 *
 * Return: 0 on success, -1 on error (Python exception already set).
 */

int
sv_generate_angles(int *size, int ndim, int force2D, int force2Ddimension,
                   int **angles_out, int *n_angles_out);

int
sv_calculate_glcm(int *image, int *sv_map, int *size, int ndim,
                  int *labels, int n_labels, int max_label, int *label_to_idx,
                  int *distances, int n_distances,
                  int Ng, int force2D, int force2Ddimension,
                  long long **P_glcm_out, int **angles_out, int *n_angles_out);

int
sv_calculate_glrlm(int *image, int *sv_map, int *size, int ndim,
                   int *labels, int n_labels, int max_label, int *label_to_idx,
                   int Ng, int Nr, int force2D, int force2Ddimension,
                   long long **P_glrlm_out, int **angles_out, int *n_angles_out);

int
sv_calculate_glszm(int *image, int *sv_map, int *size, int ndim,
                   int *labels, int n_labels, int max_label, int *label_to_idx,
                   int Ng, int force2D, int force2Ddimension,
                   long long **P_glszm_out, int *max_zone_out);

int
sv_calculate_ngtdm(int *image, int *sv_map, int *size, int ndim,
                   int *labels, int n_labels, int max_label, int *label_to_idx,
                   int *distances, int n_distances,
                   int Ng, int force2D, int force2Ddimension,
                   double **P_ngtdm_out);

int
sv_calculate_gldm(int *image, int *sv_map, int *size, int ndim,
                  int *labels, int n_labels, int max_label, int *label_to_idx,
                  int *distances, int n_distances,
                  int Ng, int alpha, int force2D, int force2Ddimension,
                  long long **P_gldm_out, int *max_dep_out);

int
sv_calculate_firstorder(double *image, int *sv_map, int *size, int ndim,
                        int *labels, int n_labels, int max_label, int *label_to_idx,
                        int Ng, double binWidth,
                        double **stats_out, int *n_stats_out);

#ifdef __cplusplus
}
#endif

#endif /* SV_CMATRICES_H */

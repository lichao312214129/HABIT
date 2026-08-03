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

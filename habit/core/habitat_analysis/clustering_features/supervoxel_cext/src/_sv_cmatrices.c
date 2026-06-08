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

/*
 * _sv_cmatrices.c  –  Python C extension bridge for multi-label supervoxel
 *                     texture matrix calculations.
 *
 * Exposes sv_cmatrices.calculate_glcm, calculate_glrlm, calculate_glszm,
 * calculate_ngtdm, calculate_gldm, calculate_firstorder to Python.
 *
 * Build: pip install -e .  (or python setup.py build_ext --inplace)
 */

#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include "sv_cmatrices.h"

/* ── array validation helper ─────────────────────────────────────────── */

static PyArrayObject *
_require_int32_array(PyObject *obj, int min_ndim, int max_ndim)
{
    PyArrayObject *arr = (PyArrayObject *)PyArray_FROM_OTF(
        obj, NPY_INT32, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_FORCECAST);
    if (arr == NULL)
        return NULL;
    if (!PyArray_ISCONTIGUOUS(arr)) {
        Py_DECREF(arr);
        PyErr_SetString(PyExc_ValueError, "Array must be C-contiguous.");
        return NULL;
    }
    int ndim = (int)PyArray_NDIM(arr);
    if (ndim < min_ndim || ndim > max_ndim) {
        Py_DECREF(arr);
        PyErr_SetString(PyExc_ValueError, "Unexpected array dimensionality.");
        return NULL;
    }
    return arr;
}

static PyArrayObject *
_require_float64_array(PyObject *obj, int min_ndim, int max_ndim)
{
    PyArrayObject *arr = (PyArrayObject *)PyArray_FROM_OTF(
        obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_FORCECAST);
    if (arr == NULL)
        return NULL;
    if (!PyArray_ISCONTIGUOUS(arr)) {
        Py_DECREF(arr);
        PyErr_SetString(PyExc_ValueError, "Array must be C-contiguous.");
        return NULL;
    }
    int ndim = (int)PyArray_NDIM(arr);
    if (ndim < min_ndim || ndim > max_ndim) {
        Py_DECREF(arr);
        PyErr_SetString(PyExc_ValueError, "Unexpected array dimensionality.");
        return NULL;
    }
    return arr;
}

/* ── label lookup helper ─────────────────────────────────────────────── */

static int *
_build_label_to_idx(int *labels, int n_labels, int *max_label_out)
{
    int max_label = 0;
    for (int i = 0; i < n_labels; i++)
        if (labels[i] > max_label) max_label = labels[i];
    *max_label_out = max_label;

    int *lut = (int *)malloc((size_t)(max_label + 1) * sizeof(int));
    if (!lut) { PyErr_NoMemory(); return NULL; }
    for (int i = 0; i <= max_label; i++) lut[i] = -1;
    for (int i = 0; i < n_labels; i++) lut[labels[i]] = i;
    return lut;
}

/* ── calculate_glcm ──────────────────────────────────────────────────── */

static PyObject *
py_calculate_glcm(PyObject *self, PyObject *args)
{
    PyObject *py_image_obj, *py_sv_map_obj, *py_labels_obj, *py_distances_obj;
    PyArrayObject *py_image, *py_sv_map, *py_labels, *py_distances;
    int Ng, force2D = 0, force2Ddimension = 0;

    if (!PyArg_ParseTuple(args, "OOOOi|ii",
                          &py_image_obj, &py_sv_map_obj, &py_labels_obj,
                          &py_distances_obj, &Ng, &force2D, &force2Ddimension))
        return NULL;

    py_image = _require_int32_array(py_image_obj, 1, 3);
    py_sv_map = _require_int32_array(py_sv_map_obj, 1, 3);
    py_labels = _require_int32_array(py_labels_obj, 1, 1);
    py_distances = _require_int32_array(py_distances_obj, 1, 1);
    if (!py_image || !py_sv_map || !py_labels || !py_distances) {
        Py_XDECREF(py_image);
        Py_XDECREF(py_sv_map);
        Py_XDECREF(py_labels);
        Py_XDECREF(py_distances);
        return NULL;
    }
    if (PyArray_SIZE(py_image) != PyArray_SIZE(py_sv_map)) {
        Py_DECREF(py_image);
        Py_DECREF(py_sv_map);
        Py_DECREF(py_labels);
        Py_DECREF(py_distances);
        PyErr_SetString(PyExc_ValueError, "image and sv_map must have the same shape.");
        return NULL;
    }

    int *image = (int *)PyArray_DATA(py_image);
    int *sv_map = (int *)PyArray_DATA(py_sv_map);
    int *labels = (int *)PyArray_DATA(py_labels);
    int n_labels = (int)PyArray_SIZE(py_labels);
    int *distances = (int *)PyArray_DATA(py_distances);
    int n_distances = (int)PyArray_SIZE(py_distances);

    int ndim = (int)PyArray_NDIM(py_image);
    int *size = (int *)malloc((size_t)ndim * sizeof(int));
    if (!size) { PyErr_NoMemory(); return NULL; }
    for (int d = 0; d < ndim; d++) size[d] = (int)PyArray_DIM(py_image, d);

    int max_label = 0;
    int *label_to_idx = _build_label_to_idx(labels, n_labels, &max_label);
    if (!label_to_idx) { free(size); return NULL; }

    long long *P = NULL;
    int *angles = NULL;
    int n_angles = 0;

    int rc = sv_calculate_glcm(image, sv_map, size, ndim,
                               labels, n_labels, max_label, label_to_idx,
                               distances, n_distances,
                               Ng, force2D, force2Ddimension,
                               &P, &angles, &n_angles);
    free(size);
    free(label_to_idx);

    if (rc < 0) return NULL;

    npy_intp glcm_dims[4] = {n_labels, Ng, Ng, n_angles};
    PyArrayObject *py_P = (PyArrayObject *)PyArray_SimpleNew(4, glcm_dims, NPY_LONGLONG);
    if (!py_P) { free(P); free(angles); Py_DECREF(py_image); Py_DECREF(py_sv_map); Py_DECREF(py_labels); Py_DECREF(py_distances); return NULL; }
    memcpy(PyArray_DATA(py_P), P, (size_t)n_labels * Ng * Ng * n_angles * sizeof(long long));
    free(P);

    npy_intp angle_dims[2] = {n_angles, 3};
    PyArrayObject *py_angles = (PyArrayObject *)PyArray_SimpleNew(2, angle_dims, NPY_INT);
    if (!py_angles) { Py_DECREF(py_P); free(angles); Py_DECREF(py_image); Py_DECREF(py_sv_map); Py_DECREF(py_labels); Py_DECREF(py_distances); return NULL; }
    memcpy(PyArray_DATA(py_angles), angles, (size_t)n_angles * 3 * sizeof(int));
    free(angles);

    Py_DECREF(py_image);
    Py_DECREF(py_sv_map);
    Py_DECREF(py_labels);
    Py_DECREF(py_distances);

    PyObject *result = PyTuple_New(2);
    PyTuple_SET_ITEM(result, 0, (PyObject *)py_P);
    PyTuple_SET_ITEM(result, 1, (PyObject *)py_angles);
    return result;
}

/* ── calculate_glrlm ─────────────────────────────────────────────────── */

static PyObject *
py_calculate_glrlm(PyObject *self, PyObject *args)
{
    PyArrayObject *py_image, *py_sv_map, *py_labels;
    int Ng, Nr, force2D = 0, force2Ddimension = 0;

    if (!PyArg_ParseTuple(args, "O!O!O!ii|ii",
                          &PyArray_Type, &py_image,
                          &PyArray_Type, &py_sv_map,
                          &PyArray_Type, &py_labels,
                          &Ng, &Nr, &force2D, &force2Ddimension))
        return NULL;

    int *image = (int *)PyArray_DATA(py_image);
    int *sv_map = (int *)PyArray_DATA(py_sv_map);
    int *labels = (int *)PyArray_DATA(py_labels);
    int n_labels = (int)PyArray_SIZE(py_labels);

    int ndim = (int)PyArray_NDIM(py_image);
    int *size = (int *)malloc((size_t)ndim * sizeof(int));
    if (!size) { PyErr_NoMemory(); return NULL; }
    for (int d = 0; d < ndim; d++) size[d] = (int)PyArray_DIM(py_image, d);

    int max_label = 0;
    int *label_to_idx = _build_label_to_idx(labels, n_labels, &max_label);
    if (!label_to_idx) { free(size); return NULL; }

    long long *P = NULL;
    int *angles = NULL;
    int n_angles = 0;

    int rc = sv_calculate_glrlm(image, sv_map, size, ndim,
                                labels, n_labels, max_label, label_to_idx,
                                Ng, Nr, force2D, force2Ddimension,
                                &P, &angles, &n_angles);
    free(size);
    free(label_to_idx);

    if (rc < 0) return NULL;

    npy_intp glrlm_dims[4] = {n_labels, Ng, Nr, n_angles};
    PyArrayObject *py_P = (PyArrayObject *)PyArray_SimpleNew(4, glrlm_dims, NPY_LONGLONG);
    if (!py_P) { free(P); free(angles); return NULL; }
    memcpy(PyArray_DATA(py_P), P, (size_t)n_labels * Ng * Nr * n_angles * sizeof(long long));
    free(P);

    npy_intp angle_dims[2] = {n_angles, 3};
    PyArrayObject *py_angles = (PyArrayObject *)PyArray_SimpleNew(2, angle_dims, NPY_INT);
    if (!py_angles) { Py_DECREF(py_P); free(angles); return NULL; }
    memcpy(PyArray_DATA(py_angles), angles, (size_t)n_angles * 3 * sizeof(int));
    free(angles);

    PyObject *result = PyTuple_New(2);
    PyTuple_SET_ITEM(result, 0, (PyObject *)py_P);
    PyTuple_SET_ITEM(result, 1, (PyObject *)py_angles);
    return result;
}

/* ── calculate_glszm ─────────────────────────────────────────────────── */

static PyObject *
py_calculate_glszm(PyObject *self, PyObject *args)
{
    PyArrayObject *py_image, *py_sv_map, *py_labels;
    int Ng, force2D = 0, force2Ddimension = 0;

    if (!PyArg_ParseTuple(args, "O!O!O!i|ii",
                          &PyArray_Type, &py_image,
                          &PyArray_Type, &py_sv_map,
                          &PyArray_Type, &py_labels,
                          &Ng, &force2D, &force2Ddimension))
        return NULL;

    int *image = (int *)PyArray_DATA(py_image);
    int *sv_map = (int *)PyArray_DATA(py_sv_map);
    int *labels = (int *)PyArray_DATA(py_labels);
    int n_labels = (int)PyArray_SIZE(py_labels);

    int ndim = (int)PyArray_NDIM(py_image);
    int *size = (int *)malloc((size_t)ndim * sizeof(int));
    if (!size) { PyErr_NoMemory(); return NULL; }
    for (int d = 0; d < ndim; d++) size[d] = (int)PyArray_DIM(py_image, d);

    int max_label = 0;
    int *label_to_idx = _build_label_to_idx(labels, n_labels, &max_label);
    if (!label_to_idx) { free(size); return NULL; }

    long long *P = NULL;
    int max_zone = 0;

    int rc = sv_calculate_glszm(image, sv_map, size, ndim,
                                labels, n_labels, max_label, label_to_idx,
                                Ng, force2D, force2Ddimension,
                                &P, &max_zone);
    free(size);
    free(label_to_idx);

    if (rc < 0) return NULL;

    npy_intp glszm_dims[3] = {n_labels, Ng, max_zone};
    PyArrayObject *py_P = (PyArrayObject *)PyArray_SimpleNew(3, glszm_dims, NPY_LONGLONG);
    if (!py_P) { free(P); return NULL; }
    memcpy(PyArray_DATA(py_P), P, (size_t)n_labels * Ng * max_zone * sizeof(long long));
    free(P);

    return (PyObject *)py_P;
}

/* ── calculate_ngtdm ─────────────────────────────────────────────────── */

static PyObject *
py_calculate_ngtdm(PyObject *self, PyObject *args)
{
    PyArrayObject *py_image, *py_sv_map, *py_labels, *py_distances;
    int Ng, force2D = 0, force2Ddimension = 0;

    if (!PyArg_ParseTuple(args, "O!O!O!O!i|ii",
                          &PyArray_Type, &py_image,
                          &PyArray_Type, &py_sv_map,
                          &PyArray_Type, &py_labels,
                          &PyArray_Type, &py_distances,
                          &Ng, &force2D, &force2Ddimension))
        return NULL;

    int *image = (int *)PyArray_DATA(py_image);
    int *sv_map = (int *)PyArray_DATA(py_sv_map);
    int *labels = (int *)PyArray_DATA(py_labels);
    int n_labels = (int)PyArray_SIZE(py_labels);
    int *distances = (int *)PyArray_DATA(py_distances);
    int n_distances = (int)PyArray_SIZE(py_distances);

    int ndim = (int)PyArray_NDIM(py_image);
    int *size = (int *)malloc((size_t)ndim * sizeof(int));
    if (!size) { PyErr_NoMemory(); return NULL; }
    for (int d = 0; d < ndim; d++) size[d] = (int)PyArray_DIM(py_image, d);

    int max_label = 0;
    int *label_to_idx = _build_label_to_idx(labels, n_labels, &max_label);
    if (!label_to_idx) { free(size); return NULL; }

    double *P = NULL;

    int rc = sv_calculate_ngtdm(image, sv_map, size, ndim,
                                labels, n_labels, max_label, label_to_idx,
                                distances, n_distances,
                                Ng, force2D, force2Ddimension,
                                &P);
    free(size);
    free(label_to_idx);

    if (rc < 0) return NULL;

    npy_intp ngtdm_dims[3] = {n_labels, Ng, 3};
    PyArrayObject *py_P = (PyArrayObject *)PyArray_SimpleNew(3, ngtdm_dims, NPY_DOUBLE);
    if (!py_P) { free(P); return NULL; }
    memcpy(PyArray_DATA(py_P), P, (size_t)n_labels * Ng * 3 * sizeof(double));
    free(P);

    return (PyObject *)py_P;
}

/* ── calculate_gldm ──────────────────────────────────────────────────── */

static PyObject *
py_calculate_gldm(PyObject *self, PyObject *args)
{
    PyArrayObject *py_image, *py_sv_map, *py_labels, *py_distances;
    int Ng, alpha, force2D = 0, force2Ddimension = 0;

    if (!PyArg_ParseTuple(args, "O!O!O!O!ii|ii",
                          &PyArray_Type, &py_image,
                          &PyArray_Type, &py_sv_map,
                          &PyArray_Type, &py_labels,
                          &PyArray_Type, &py_distances,
                          &Ng, &alpha, &force2D, &force2Ddimension))
        return NULL;

    int *image = (int *)PyArray_DATA(py_image);
    int *sv_map = (int *)PyArray_DATA(py_sv_map);
    int *labels = (int *)PyArray_DATA(py_labels);
    int n_labels = (int)PyArray_SIZE(py_labels);
    int *distances = (int *)PyArray_DATA(py_distances);
    int n_distances = (int)PyArray_SIZE(py_distances);

    int ndim = (int)PyArray_NDIM(py_image);
    int *size = (int *)malloc((size_t)ndim * sizeof(int));
    if (!size) { PyErr_NoMemory(); return NULL; }
    for (int d = 0; d < ndim; d++) size[d] = (int)PyArray_DIM(py_image, d);

    int max_label = 0;
    int *label_to_idx = _build_label_to_idx(labels, n_labels, &max_label);
    if (!label_to_idx) { free(size); return NULL; }

    long long *P = NULL;
    int max_dep = 0;

    int rc = sv_calculate_gldm(image, sv_map, size, ndim,
                               labels, n_labels, max_label, label_to_idx,
                               distances, n_distances,
                               Ng, alpha, force2D, force2Ddimension,
                               &P, &max_dep);
    free(size);
    free(label_to_idx);

    if (rc < 0) return NULL;

    npy_intp gldm_dims[3] = {n_labels, Ng, max_dep};
    PyArrayObject *py_P = (PyArrayObject *)PyArray_SimpleNew(3, gldm_dims, NPY_LONGLONG);
    if (!py_P) { free(P); return NULL; }
    memcpy(PyArray_DATA(py_P), P, (size_t)n_labels * Ng * max_dep * sizeof(long long));
    free(P);

    return (PyObject *)py_P;
}

/* ── calculate_firstorder ────────────────────────────────────────────── */

static PyObject *
py_calculate_firstorder(PyObject *self, PyObject *args)
{
    PyArrayObject *py_image, *py_sv_map, *py_labels;
    int Ng;
    double binWidth;

    if (!PyArg_ParseTuple(args, "O!O!O!id",
                          &PyArray_Type, &py_image,
                          &PyArray_Type, &py_sv_map,
                          &PyArray_Type, &py_labels,
                          &Ng, &binWidth))
        return NULL;

    double *image = (double *)PyArray_DATA(py_image);
    int *sv_map = (int *)PyArray_DATA(py_sv_map);
    int *labels = (int *)PyArray_DATA(py_labels);
    int n_labels = (int)PyArray_SIZE(py_labels);

    int ndim = (int)PyArray_NDIM(py_image);
    int *size = (int *)malloc((size_t)ndim * sizeof(int));
    if (!size) { PyErr_NoMemory(); return NULL; }
    for (int d = 0; d < ndim; d++) size[d] = (int)PyArray_DIM(py_image, d);

    int max_label = 0;
    int *label_to_idx = _build_label_to_idx(labels, n_labels, &max_label);
    if (!label_to_idx) { free(size); return NULL; }

    double *stats = NULL;
    int n_stats = 0;

    int rc = sv_calculate_firstorder(image, sv_map, size, ndim,
                                     labels, n_labels, max_label, label_to_idx,
                                     Ng, binWidth,
                                     &stats, &n_stats);
    free(size);
    free(label_to_idx);

    if (rc < 0) return NULL;

    npy_intp stats_dims[2] = {n_labels, n_stats};
    PyArrayObject *py_stats = (PyArrayObject *)PyArray_SimpleNew(2, stats_dims, NPY_DOUBLE);
    if (!py_stats) { free(stats); return NULL; }
    memcpy(PyArray_DATA(py_stats), stats, (size_t)n_labels * n_stats * sizeof(double));
    free(stats);

    return (PyObject *)py_stats;
}

/* ── module definition ───────────────────────────────────────────────── */

static PyMethodDef SvCmatricesMethods[] = {
    {"calculate_glcm", py_calculate_glcm, METH_VARARGS,
     "Calculate GLCM matrices for multiple supervoxel labels.\n\n"
     "Args:\n"
     "  image: int32 ndarray (discretized, 1-indexed gray levels)\n"
     "  sv_map: int32 ndarray (multi-label supervoxel map)\n"
     "  labels: int32 1D ndarray (label IDs to process)\n"
     "  distances: int32 1D ndarray\n"
     "  Ng: number of gray levels\n"
     "  force2D: bool (default 0)\n"
     "  force2Ddimension: int (default 0)\n\n"
     "Returns:\n"
     "  (P_glcm, angles) where P_glcm has shape (n_labels, Ng, Ng, n_angles)\n"
     "  and angles has shape (n_angles, 3)"},

    {"calculate_glrlm", py_calculate_glrlm, METH_VARARGS,
     "Calculate GLRLM matrices for multiple supervoxel labels.\n\n"
     "Args:\n"
     "  image: int32 ndarray\n  sv_map: int32 ndarray\n  labels: int32 1D ndarray\n"
     "  Ng: int, Nr: int\n  force2D: int, force2Ddimension: int\n\n"
     "Returns:\n"
     "  (P_glrlm, angles) where P_glrlm has shape (n_labels, Ng, Nr, n_angles)"},

    {"calculate_glszm", py_calculate_glszm, METH_VARARGS,
     "Calculate GLSZM matrices for multiple supervoxel labels.\n\n"
     "Args:\n"
     "  image: int32 ndarray\n  sv_map: int32 ndarray\n  labels: int32 1D ndarray\n"
     "  Ng: int\n  force2D: int, force2Ddimension: int\n\n"
     "Returns:\n"
     "  P_glszm with shape (n_labels, Ng, max_zone)"},

    {"calculate_ngtdm", py_calculate_ngtdm, METH_VARARGS,
     "Calculate NGTDM matrices for multiple supervoxel labels.\n\n"
     "Args:\n"
     "  image: int32 ndarray\n  sv_map: int32 ndarray\n  labels: int32 1D ndarray\n"
     "  distances: int32 1D ndarray\n  Ng: int\n  force2D: int, force2Ddimension: int\n\n"
     "Returns:\n"
     "  P_ngtdm with shape (n_labels, Ng, 3)"},

    {"calculate_gldm", py_calculate_gldm, METH_VARARGS,
     "Calculate GLDM matrices for multiple supervoxel labels.\n\n"
     "Args:\n"
     "  image: int32 ndarray\n  sv_map: int32 ndarray\n  labels: int32 1D ndarray\n"
     "  distances: int32 1D ndarray\n  Ng: int, alpha: int\n  force2D: int, force2Ddimension: int\n\n"
     "Returns:\n"
     "  P_gldm with shape (n_labels, Ng, max_dependence)"},

    {"calculate_firstorder", py_calculate_firstorder, METH_VARARGS,
     "Calculate first-order statistics for multiple supervoxel labels.\n\n"
     "Args:\n"
     "  image: float64 ndarray\n  sv_map: int32 ndarray\n  labels: int32 1D ndarray\n"
     "  Ng: int, binWidth: float\n\n"
     "Returns:\n"
     "  stats with shape (n_labels, 17) — Energy, TotalEnergy, Entropy, Minimum,\n"
     "  10Percentile, 90Percentile, Maximum, Mean, Median, InterquartileRange,\n"
     "  Range, MeanAbsoluteDeviation, RobustMeanAbsoluteDeviation, RootMeanSquared,\n"
     "  Skewness, Kurtosis, Uniformity"},

    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef sv_cmatrices_module = {
    PyModuleDef_HEAD_INIT,
    "_sv_cmatrices",
    "Multi-label supervoxel texture matrix C extension for HABIT.\n\n"
    "Computes GLCM, GLRLM, GLSZM, NGTDM, GLDM and first-order statistics\n"
    "for all requested supervoxel labels in a single C pass, avoiding\n"
    "Python-level per-label loops.",
    -1,
    SvCmatricesMethods
};

PyMODINIT_FUNC
PyInit__sv_cmatrices(void)
{
    import_array();
    return PyModule_Create(&sv_cmatrices_module);
}

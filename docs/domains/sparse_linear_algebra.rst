.. _onemkl_sparse_linear_algebra:

Sparse Linear Algebra
=====================

See the latest specification for the sparse domain `here
<https://oneapi-spec.uxlfoundation.org/specifications/oneapi/latest/elements/onemkl/source/domains/spblas/spblas>`_.

This page documents implementation specific or backend specific details of the
sparse domain.

OneMKL Intel CPU and GPU backends
---------------------------------

Currently known limitations:

- All operations' algorithms except ``no_optimize_alg`` map to the default
  algorithm.
- The required external workspace size is always 0 bytes.
- ``oneapi::mkl::sparse::set_csr_data`` and
  ``oneapi::mkl::sparse::set_coo_data`` functions cannot be used on a handle
  that has already been used for an operation or its optimize function. Doing so
  will throw an ``oneapi::mkl::unimplemented`` exception.
- Using ``spsv`` with the ``oneapi::mkl::sparse::spsv_alg::no_optimize_alg`` and
  a sparse matrix that does not have the
  ``oneapi::mkl::sparse::matrix_property::sorted`` property will throw an
  ``oneapi::mkl::unimplemented`` exception.
- Using ``spmm`` on Intel GPU with a sparse matrix that is
  ``oneapi::mkl::transpose::conjtrans`` and has the
  ``oneapi::mkl::sparse::matrix_property::symmetric`` property will throw an
  ``oneapi::mkl::unimplemented`` exception.
- Using ``spmv`` with a sparse matrix that is
  ``oneapi::mkl::transpose::conjtrans`` with a ``type_view``
  ``matrix_descr::symmetric`` or ``matrix_descr::hermitian`` will throw an
  ``oneapi::mkl::unimplemented`` exception.
- Using ``spsv`` on Intel GPU with a sparse matrix that is
  ``oneapi::mkl::transpose::conjtrans`` and will throw an
  ``oneapi::mkl::unimplemented`` exception.
- Scalar parameters ``alpha`` and ``beta`` should be host pointers to prevent
  synchronizations and copies to the host.


cuSPARSE backend
----------------

Currently known limitations:

- Using ``spmv`` with a ``type_view`` other than ``matrix_descr::general`` will
  throw an ``oneapi::mkl::unimplemented`` exception.
- The COO format requires the indices to be sorted by row. See the `cuSPARSE
  documentation
  <https://docs.nvidia.com/cuda/cusparse/index.html#coordinate-coo>`_.


Operation algorithms mapping
----------------------------

The following tables describe how a oneMKL SYCL Interface algorithm maps to the
backend's algorithms. Refer to the backend's documentation for a more detailed
explanation of the algorithms.

Backends with no equivalent algorithms will fallback to the backend's default
behavior.


spmm
^^^^

.. list-table::
   :header-rows: 1
   :widths: 10 30 45

   * - Value
     - Description
     - Backend equivalent
   * - ``default_optimize_alg``
     - Default algorithm.
     - | MKL: none
       | cuSPARSE: ``CUSPARSE_SPMM_ALG_DEFAULT``
   * - ``no_optimize_alg``
     - Default algorithm but may skip some optimizations. Useful only if an
       operation with the same configuration is run once.
     - | MKL: none
       | cuSPARSE: ``CUSPARSE_SPMM_ALG_DEFAULT``
   * - ``coo_alg1``
     - Should provide best performance for COO format, small ``nnz`` and
       column-major layout.
     - | MKL: none
       | cuSPARSE: ``CUSPARSE_SPMM_COO_ALG1``
   * - ``coo_alg2``
     - Should provide best performance for COO format and column-major layout.
       Produces deterministic results.
     - | MKL: none
       | cuSPARSE: ``CUSPARSE_SPMM_COO_ALG2``
   * - ``coo_alg3``
     - Should provide best performance for COO format and large ``nnz``.
     - | MKL: none
       | cuSPARSE: ``CUSPARSE_SPMM_COO_ALG3``
   * - ``coo_alg4``
     - Should provide best performance for COO format and row-major layout.
     - | MKL: none
       | cuSPARSE: ``CUSPARSE_SPMM_COO_ALG4``
   * - ``csr_alg1``
     - Should provide best performance for CSR format and column-major layout.
     - | MKL: none
       | cuSPARSE: ``CUSPARSE_SPMM_CSR_ALG1``
   * - ``csr_alg2``
     - Should provide best performance for CSR format and row-major layout.
     - | MKL: none
       | cuSPARSE: ``CUSPARSE_SPMM_CSR_ALG2``
   * - ``csr_alg3``
     - Deterministic algorithm for CSR format.
     - | MKL: none
       | cuSPARSE: ``CUSPARSE_SPMM_CSR_ALG3``


spmv
^^^^

.. list-table::
   :header-rows: 1
   :widths: 10 30 45

   * - Value
     - Description
     - Backend equivalent
   * - ``default_alg``
     - Default algorithm.
     - | MKL: none
       | cuSPARSE: ``CUSPARSE_SPMV_ALG_DEFAULT``
   * - ``no_optimize_alg``
     - Default algorithm but may skip some optimizations. Useful only if an
       operation with the same configuration is run once.
     - | MKL: none
       | cuSPARSE: ``CUSPARSE_SPMM_ALG_DEFAULT``
   * - ``coo_alg1``
     - Default algorithm for COO format.
     - | MKL: none
       | cuSPARSE: ``CUSPARSE_SPMV_COO_ALG1``
   * - ``coo_alg2``
     - Deterministic algorithm for COO format.
     - | MKL: none
       | cuSPARSE: ``CUSPARSE_SPMV_COO_ALG2``
   * - ``csr_alg1``
     - Default algorithm for CSR format.
     - | MKL: none
       | cuSPARSE: ``CUSPARSE_SPMV_CSR_ALG1``
   * - ``csr_alg2``
     - Deterministic algorithm for CSR format.
     - | MKL: none
       | cuSPARSE: ``CUSPARSE_SPMV_CSR_ALG2``
   * - ``csr_alg3``
     - LRB variant of the algorithm for CSR format.
     - | MKL: none
       | cuSPARSE: none


spsv
^^^^

.. list-table::
   :header-rows: 1
   :widths: 10 30 45

   * - Value
     - Description
     - Backend equivalent
   * - ``default_optimize_alg``
     - Default algorithm.
     - | MKL: none
       | cuSPARSE: ``CUSPARSE_SPMM_ALG_DEFAULT``
   * - ``no_optimize_alg``
     - Default algorithm but may skip some optimizations. Useful only if an
       operation with the same configuration is run once.
     - | MKL: none
       | cuSPARSE: ``CUSPARSE_SPMM_ALG_DEFAULT``

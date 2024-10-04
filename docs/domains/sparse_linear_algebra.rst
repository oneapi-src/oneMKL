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
  will throw a ``oneapi::mkl::unimplemented`` exception.
- Using ``spsv`` with the ``oneapi::mkl::sparse::spsv_alg::no_optimize_alg`` and
  a sparse matrix that does not have the
  ``oneapi::mkl::sparse::matrix_property::sorted`` property will throw a
  ``oneapi::mkl::unimplemented`` exception.
- Using ``spmm`` on Intel GPU with a sparse matrix that is
  ``oneapi::mkl::transpose::conjtrans`` and has the
  ``oneapi::mkl::sparse::matrix_property::symmetric`` property will throw a
  ``oneapi::mkl::unimplemented`` exception.
- Using ``spmv`` with a sparse matrix that is
  ``oneapi::mkl::transpose::conjtrans`` with a ``type_view``
  ``matrix_descr::symmetric`` or ``matrix_descr::hermitian`` will throw a
  ``oneapi::mkl::unimplemented`` exception.
- Using ``spsv`` on Intel GPU with a sparse matrix that is
  ``oneapi::mkl::transpose::conjtrans`` and will throw a
  ``oneapi::mkl::unimplemented`` exception.
- Scalar parameters ``alpha`` and ``beta`` should be host pointers to prevent
  synchronizations and copies to the host.


cuSPARSE backend
----------------

Currently known limitations:

- The COO format requires the indices to be sorted by row. See the `cuSPARSE
  documentation
  <https://docs.nvidia.com/cuda/cusparse/index.html#coordinate-coo>`_. Sparse
  operations using matrices with the COO format without the property
  ``matrix_property::sorted_by_rows`` or ``matrix_property::sorted`` will throw
  a ``oneapi::mkl::unimplemented`` exception.
- Using ``spmm`` with the algorithm ``spmm_alg::csr_alg3`` and an ``opA`` other
  than ``transpose::nontrans`` or an ``opB`` ``transpose::conjtrans`` will throw
  a ``oneapi::mkl::unimplemented`` exception.
- Using ``spmv`` with a ``type_view`` other than ``matrix_descr::general`` will
  throw a ``oneapi::mkl::unimplemented`` exception.
- Using ``spsv`` with the algorithm ``spsv_alg::no_optimize_alg`` may still
  perform some mandatory preprocessing.
- oneMKL Interface does not provide a way to use non-default algorithms without
  calling preprocess functions such as ``cusparseSpMM_preprocess`` or
  ``cusparseSpMV_preprocess``. Feel free to create an issue if this is needed.


rocSPARSE backend
----------------

Currently known limitations:

- Using ``spmv`` with a ``type_view`` other than ``matrix_descr::general`` will
  throw a ``oneapi::mkl::unimplemented`` exception.
- The COO format requires the indices to be sorted by row then by column. See
  the `rocSPARSE COO documentation
  <https://rocm.docs.amd.com/projects/rocSPARSE/en/latest/how-to/basics.html#coo-storage-format>`_.
  Sparse operations using matrices with the COO format without the property
  ``matrix_property::sorted`` will throw a ``oneapi::mkl::unimplemented``
  exception.
- The CSR format requires the column indices to be sorted within each row. See
  the `rocSPARSE CSR documentation
  <https://rocm.docs.amd.com/projects/rocSPARSE/en/latest/how-to/basics.html#csr-storage-format>`_.
  Sparse operations using matrices with the CSR format without the property
  ``matrix_property::sorted`` will throw a ``oneapi::mkl::unimplemented``
  exception.
- The same sparse matrix handle cannot be reused for multiple operations
  ``spmm``, ``spmv``, or ``spsv``. Doing so will throw a
  ``oneapi::mkl::unimplemented`` exception. See `#332
  <https://github.com/ROCm/rocSPARSE/issues/332>`_.


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

   * - ``spmm_alg`` value
     - MKLCPU/MKLGPU
     - cuSPARSE
     - rocSPARSE
   * - ``default_alg``
     - none
     - ``CUSPARSE_SPMM_ALG_DEFAULT``
     - ``rocsparse_spmm_alg_default``
   * - ``no_optimize_alg``
     - none
     - ``CUSPARSE_SPMM_ALG_DEFAULT``
     - ``rocsparse_spmm_alg_default``
   * - ``coo_alg1``
     - none
     - ``CUSPARSE_SPMM_COO_ALG1``
     - ``rocsparse_spmm_alg_coo_segmented``
   * - ``coo_alg2``
     - none
     - ``CUSPARSE_SPMM_COO_ALG2``
     - ``rocsparse_spmm_alg_coo_atomic``
   * - ``coo_alg3``
     - none
     - ``CUSPARSE_SPMM_COO_ALG3``
     - ``rocsparse_spmm_alg_coo_segmented_atomic``
   * - ``coo_alg4``
     - none
     - ``CUSPARSE_SPMM_COO_ALG4``
     - ``rocsparse_spmm_alg_default``
   * - ``csr_alg1``
     - none
     - ``CUSPARSE_SPMM_CSR_ALG1``
     - ``rocsparse_spmm_alg_csr``
   * - ``csr_alg2``
     - none
     - ``CUSPARSE_SPMM_CSR_ALG2``
     - ``rocsparse_spmm_alg_csr_row_split``
   * - ``csr_alg3``
     - none
     - ``CUSPARSE_SPMM_CSR_ALG3``
     - ``rocsparse_spmm_alg_csr_merge``


spmv
^^^^

.. list-table::
   :header-rows: 1
   :widths: 10 30 45

   * - ``spmv_alg`` value
     - MKLCPU/MKLGPU
     - cuSPARSE
     - rocSPARSE
   * - ``default_alg``
     - none
     - ``CUSPARSE_SPMV_ALG_DEFAULT``
     - ``rocsparse_spmv_alg_default``
   * - ``no_optimize_alg``
     - none
     - ``CUSPARSE_SPMV_ALG_DEFAULT``
     - ``rocsparse_spmv_alg_default``
   * - ``coo_alg1``
     - none
     - ``CUSPARSE_SPMV_COO_ALG1``
     - ``rocsparse_spmv_alg_coo``
   * - ``coo_alg2``
     - none
     - ``CUSPARSE_SPMV_COO_ALG2``
     - ``rocsparse_spmv_alg_coo_atomic``
   * - ``csr_alg1``
     - none
     - ``CUSPARSE_SPMV_CSR_ALG1``
     - ``rocsparse_spmv_alg_csr_adaptive``
   * - ``csr_alg2``
     - none
     - ``CUSPARSE_SPMV_CSR_ALG2``
     - ``rocsparse_spmv_alg_csr_stream``
   * - ``csr_alg3``
     - none
     - ``CUSPARSE_SPMV_ALG_DEFAULT``
     - ``rocsparse_spmv_alg_csr_lrb``


spsv
^^^^

.. list-table::
   :header-rows: 1
   :widths: 10 30 45

   * - ``spsv_alg`` value
     - MKLCPU/MKLGPU
     - cuSPARSE
     - rocSPARSE
   * - ``default_alg``
     - none
     - ``CUSPARSE_SPSV_ALG_DEFAULT``
     - ``rocsparse_spsv_alg_default``
   * - ``no_optimize_alg``
     - none
     - ``CUSPARSE_SPSV_ALG_DEFAULT``
     - ``rocsparse_spsv_alg_default``

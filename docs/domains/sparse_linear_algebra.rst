.. _onemkl_sparse_linear_algebra:

Sparse Linear Algebra
=====================

See the latest specification for the sparse domain `here
<https://oneapi-spec.uxlfoundation.org/specifications/oneapi/latest/elements/onemkl/source/domains/spblas/spblas>`_.

This page documents implementation specific or backend specific details of the
sparse domain.

OneMKL Intel CPU and GPU backends
---------------------------------

Known limitations as of Intel oneMKL product release 2024.1:

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
- Scalar parameters ``alpha`` and ``beta`` should be host pointers to prevent
  synchronizations and copies to the host.

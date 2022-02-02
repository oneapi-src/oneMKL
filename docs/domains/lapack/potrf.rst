.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemkl_lapack_potrf:

potrf
=====

Computes the Cholesky factorization of a symmetric (Hermitian)
positive-definite matrix.

.. container:: section

  .. rubric:: Description
      
``potrf`` supports the following precisions.

     .. list-table:: 
        :header-rows: 1

        * -  T 
        * -  ``float`` 
        * -  ``double`` 
        * -  ``std::complex<float>`` 
        * -  ``std::complex<double>`` 

The routine forms the Cholesky factorization of a symmetric
positive-definite or, for complex data, Hermitian positive-definite
matrix :math:`A`:

    .. list-table:: 
       :header-rows: 1
 
       * -  :math:`A` = :math:`U^{T}U` for real data, :math:`A = U^{H}U` for complex data
         -  if upper_lower=\ ``oneapi::mkl::uplo::upper`` 
       * -  :math:`A` = :math:`LL^{T}` for real data, :math:`A = LL^{H}` for complex data
         -  if upper_lower=\ ``oneapi::mkl::uplo::lower`` 

where :math:`L` is a lower triangular matrix and :math:`U` is upper
triangular.

potrf (Buffer Version)
----------------------

.. container:: section

  .. rubric:: Syntax

.. code-block:: cpp

    namespace oneapi::mkl::lapack {
      void potrf(sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n, sycl::buffer<T,1> &a, std::int64_t lda, sycl::buffer<T,1> &scratchpad, std::int64_t scratchpad_size)
    }

.. container:: section

  .. rubric:: Input Parameters

queue
   The queue where the routine should be executed.

upper_lower
   Indicates whether the upper or lower triangular part of :math:`A` is
   stored and how :math:`A` is factored:

   If upper_lower=\ ``oneapi::mkl::uplo::upper``, the array ``a`` stores the
   upper triangular part of the matrix :math:`A`, and the strictly lower
   triangular part of the matrix is not referenced.

   If upper_lower=\ ``oneapi::mkl::uplo::lower``, the array ``a`` stores the
   lower triangular part of the matrix :math:`A`, and the strictly upper
   triangular part of the matrix is not referenced.

n
   Specifies the order of the matrix :math:`A` (:math:`0 \le n`).

a
   Buffer holding input matrix :math:`A`. The buffer ``a`` contains either
   the upper or the lower triangular part of the matrix :math:`A` (see
   upper_lower). The second dimension of ``a`` must be at least
   :math:`\max(1, n)`.

lda
   The leading dimension of ``a``.

scratchpad_size
   Size of scratchpad memory as a number of floating point elements of type ``T``.
   Size should not be less than the value returned by :ref:`onemkl_lapack_potrf_scratchpad_size` function.

.. container:: section

  .. rubric:: Output Parameters

a
   The buffer ``a`` is overwritten by the Cholesky factor :math:`U` or :math:`L`,
   as specified by ``upper_lower``.

scratchpad
   Buffer holding scratchpad memory to be used by routine for storing intermediate results.

potrf (USM Version)
----------------------

.. container:: section

  .. rubric:: Syntax
         
.. code-block:: cpp

    namespace oneapi::mkl::lapack {
      sycl::event potrf(sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n, T *a, std::int64_t lda, T *scratchpad, std::int64_t scratchpad_size, const std::vector<sycl::event> &events = {})
    }

.. container:: section

  .. rubric:: Input Parameters
      
queue
   The queue where the routine should be executed.

upper_lower
   Indicates whether the upper or lower triangular part of :math:`A` is
   stored and how :math:`A` is factored:

   If upper_lower=\ ``oneapi::mkl::uplo::upper``, the array ``a`` stores the
   upper triangular part of the matrix :math:`A`, and the strictly lower
   triangular part of the matrix is not referenced.

   If upper_lower=\ ``oneapi::mkl::uplo::lower``, the array ``a`` stores the
   lower triangular part of the matrix :math:`A`, and the strictly upper
   triangular part of the matrix is not referenced.

n
   Specifies the order of the matrix :math:`A` (:math:`0 \le n`).

a
   Pointer to input matrix :math:`A`. The array ``a`` contains either
   the upper or the lower triangular part of the matrix :math:`A` (see
   upper_lower). The second dimension of ``a`` must be at least
   :math:`\max(1, n)`.

lda
   The leading dimension of ``a``.

scratchpad_size
   Size of scratchpad memory as a number of floating point elements of type ``T``.
   Size should not be less than the value returned by :ref:`onemkl_lapack_potrf_scratchpad_size` function.

events
   List of events to wait for before starting computation. Defaults to empty list.

.. container:: section

  .. rubric:: Output Parameters

a
   The memory pointer to by pointer ``a`` is overwritten by the Cholesky factor :math:`U` or :math:`L`,
   as specified by ``upper_lower``.

scratchpad
   Pointer to scratchpad memory to be used by routine for storing intermediate results.

.. container:: section

  .. rubric:: Return Values

Output event to wait on to ensure computation is complete.

**Parent topic:** :ref:`onemkl_lapack-linear-equation-routines`



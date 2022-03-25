.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemkl_lapack_potrs_batch:

potrs_batch
===========

Computes the LU factorizations of a batch of general matrices.

.. container:: section

  .. rubric:: Description

``potrs_batch`` supports the following precisions.

   .. list-table:: 
      :header-rows: 1

      * -  T 
      * -  ``float`` 
      * -  ``double`` 
      * -  ``std::complex<float>`` 
      * -  ``std::complex<double>`` 

.. _onemkl_lapack_potrs_batch_buffer:

potrs_batch (Buffer Version)
----------------------------

.. container:: section

  .. rubric:: Description

The buffer version of ``potrs_batch`` supports only the strided API. 
   
**Strided API**

 | The routine solves for :math:`X_i` the systems of linear equations :math:`A_iX_i = B_i` with a symmetric positive-definite or, for complex data, Hermitian positive-definite matrices :math:`A_i`, given the Cholesky factorization of :math:`A_i`, :math:`i \in \{1...batch\_size\}`:
 | :math:`A_i = U_i^TU_i` for real data, :math:`A_i = U_i^HU_i` for complex data if ``uplo = mkl::uplo::upper``,
 | :math:`A_i = L_iL_i^T` for real data, :math:`A_i = L_iL_i^H` for complex data if ``uplo = mkl::uplo::lower``,
 | where :math:`L_i` is a lower triangular matrix and :math:`U_i` is upper triangular.
 | The systems are solved with multiple right-hand sides stored in the columns of the matrices :math:`B_i`.
 | Before calling this routine, matrices :math:`A_i` should be factorized by call to the Strided API of the :ref:`onemkl_lapack_potrf_batch_buffer` function.

.. container:: section

  .. rubric:: Syntax

.. code-block:: cpp

    namespace oneapi::mkl::lapack {
      void potrs_batch(sycl::queue &queue, mkl::uplo uplo, std::int64_t n, std::int64_t nrhs, sycl::buffer<T> &a, std::int64_t lda, std::int64_t stride_a, sycl::buffer<T> &b, std::int64_t ldb, std::int64_t stride_b, std::int64_t batch_size, sycl::buffer<T> &scratchpad, std::int64_t scratchpad_size)
    }

.. container:: section

  .. rubric:: Input Parameters

queue
  Device queue where calculations will be performed.

uplo
 | Indicates how the input matrices have been factored:
 | If ``uplo = mkl::uplo::upper``, the upper triangle :math:`U_i` of :math:`A_i` is stored, where :math:`A_i = U_i^TU_i` for real data, :math:`A_i = U_i^HU_i` for complex data.
 | If ``uplo = mkl::uplo::lower``, the upper triangle :math:`L_i` of :math:`A_i` is stored, where :math:`A_i = L_iL_i^T` for real data, :math:`A_i = L_iL_i^H` for complex data.

n
  The order of matrices :math:`A_i` (:math:`0 \le n`).

nrhs
  The number of right-hand sides (:math:`0 \le \text{nrhs}`).

a
  Array containing batch of factorizations of the matrices :math:`A_i`, as returned by the Strided API of the :ref:`onemkl_lapack_potrf_batch_buffer` function.

lda
  Leading dimension of :math:`A_i`.

stride_a
  Stride between the beginnings of matrices inside the batch array ``a``.

b
  Array containing batch of matrices :math:`B_i` whose columns are the right-hand sides for the systems of equations.

ldb
  Leading dimension of :math:`B_i`.

stride_b
  Stride between the beginnings of matrices :math:`B_i` inside the batch array ``b``.

batch_size
  Number of problems in a batch.

scratchpad
  Scratchpad memory to be used by routine for storing intermediate results.

scratchpad_size
  Size of scratchpad memory as a number of floating point elements of type ``T``. Size should not be less then the value returned by the Strided API of the :ref:`onemkl_lapack_potrs_batch_scratchpad_size` function.

.. container:: section

  .. rubric:: Output Parameters

b
  Solution matrices :math:`X_i`.

.. _onemkl_lapack_potrs_batch_usm:

potrs_batch (USM Version)
-------------------------

.. container:: section

  .. rubric:: Description

The USM version of ``potrs_batch`` supports the group API and strided API. 

**Group API**

.. container:: section

  .. rubric:: Syntax

.. code-block:: cpp

    namespace oneapi::mkl::lapack {
      sycl::event potrs_batch(sycl::queue &queue, mkl::uplo *uplo, std::int64_t *n, std::int64_t *nrhs, T **a, std::int64_t *lda, T **b, std::int64_t *ldb, std::int64_t group_count, std::int64_t *group_sizes, T *scratchpad, std::int64_t scratchpad_size, const std::vector<sycl::event> &events = {})
    }

.. container:: section

  .. rubric:: Input Parameters

queue
  Device queue where calculations will be performed.

uplo  
 | Array of ``group_count`` :math:`\text{uplo}_g` parameters.
 | Each of :math:`\text{uplo}_g` indicates whether the upper or lower triangular parts of the input matrices are provided:
 | If :math:`\text{uplo}_g` is ``mkl::uplo::upper``, input matrices from array ``a`` belonging to group :math:`g` store the upper triangular parts,
 | If :math:`\text{uplo}_g` is ``mkl::uplo::lower``, input matrices from array ``a`` belonging to group :math:`g` store the lower triangular parts.

n
 | Array of ``group_count`` :math:`n_g` parameters.
 | Each :math:`n_g` specifies the order of the input matrices from array ``a`` belonging to group :math:`g`.

nrhs
 | Array of ``group_count`` :math:`\text{nrhs}_g` parameters.
 | Each :math:`\text{nrhs}_g` specifies the number of right-hand sides supplied for group :math:`g` in corresponding part of array ``b``.

a
  Array of ``batch_size`` pointers to Cholesky factored matrices :math:`A_i` as returned by the Group API of the :ref:`onemkl_lapack_potrf_batch_usm` function.

lda
 | Array of ``group_count`` :math:`\text{lda}_g` parameters.
 | Each :math:`\text{lda}_g` specifies the leading dimensions of the matrices from ``a`` belonging to group :math:`g`.

b
  Array of ``batch_size`` pointers to right-hand side matrices :math:`B_i`, each of size :math:`\text{ldb}_g \cdot \text{nrhs}_g`, where :math:`g` is an index of group corresponding to :math:`B_i`.

ldb
 | Array of ``group_count`` :math:`\text{ldb}_g` parameters.
 | Each :math:`\text{ldb}_g` specifies the leading dimensions of the matrices from ``b`` belonging to group :math:`g`.

group_count
  Number of groups of parameters. Must be at least 0.

group_sizes
  Array of ``group_count`` integers. Array element with index :math:`g` specifies the number of problems to solve for each of the groups of parameters :math:`g`. So the total number of problems to solve, ``batch_size``, is a sum of all parameter group sizes.

scratchpad
  Scratchpad memory to be used by routine for storing intermediate results.

scratchpad_size
  Size of scratchpad memory as a number of floating point elements of type ``T``. Size should not be less then the value returned by the Group API of the :ref:`onemkl_lapack_potrs_batch_scratchpad_size` function.

events
  List of events to wait for before starting computation. Defaults to empty list.

.. container:: section

  .. rubric:: Output Parameters

b
  Solution matrices :math:`X_i`.

.. container:: section
   
  .. rubric:: Return Values

Output event to wait on to ensure computation is complete.

**Strided API**

 | The routine solves for :math:`X_i` the systems of linear equations :math:`A_iX_i = B_i` with a symmetric positive-definite or, for complex data, Hermitian positive-definite matrices :math:`A_i`, given the Cholesky factorization of :math:`A_i`, :math:`i \in \{1...batch\_size\}`:
 | :math:`A_i = U_i^TU_i` for real data, :math:`A_i = U_i^HU_i` for complex data if ``uplo = mkl::uplo::upper``,
 | :math:`A_i = L_iL_i^T` for real data, :math:`A_i = L_iL_i^H` for complex data if ``uplo = mkl::uplo::lower``,
 | where :math:`L_i` is a lower triangular matrix and :math:`U_i` is upper triangular.
 | The systems are solved with multiple right-hand sides stored in the columns of the matrices :math:`B_i`.
 | Before calling this routine, matrices :math:`A_i` should be factorized by call to the Strided API of the :ref:`onemkl_lapack_potrf_batch_usm` function.

.. container:: section

  .. rubric:: Syntax

.. code-block:: cpp

    namespace oneapi::mkl::lapack {
      sycl::event potrs_batch(sycl::queue &queue, mkl::uplo uplo, std::int64_t n, std::int64_t nrhs, T *a, std::int64_t lda, std::int64_t stride_a, T *b, std::int64_t ldb, std::int64_t stride_b, std::int64_t batch_size, T *scratchpad, std::int64_t scratchpad_size, const std::vector<sycl::event> &events = {})
    };

.. container:: section

  .. rubric:: Input Parameters

queue
  Device queue where calculations will be performed.

uplo
 | Indicates how the input matrices have been factored:
 | If ``uplo = mkl::uplo::upper``, the upper triangle :math:`U_i` of :math:`A_i` is stored, where :math:`A_i = U_i^TU_i` for real data, :math:`A_i = U_i^HU_i` for complex data.
 | If ``uplo = mkl::uplo::lower``, the upper triangle :math:`L_i` of :math:`A_i` is stored, where :math:`A_i = L_iL_i^T` for real data, :math:`A_i = L_iL_i^H` for complex data.

n
  The order of matrices :math:`A_i` (:math:`0 \le n`).

nrhs
  The number of right-hand sides (:math:`0 \le nrhs`).

a
  Array containing batch of factorizations of the matrices :math:`A_i`, as returned by the Strided API of the :ref:`onemkl_lapack_potrf_batch_usm` function.

lda
  Leading dimension of :math:`A_i`.

stride_a
  Stride between the beginnings of matrices inside the batch array ``a``.

b
  Array containing batch of matrices :math:`B_i` whose columns are the right-hand sides for the systems of equations.

ldb
  Leading dimension of :math:`B_i`.

stride_b
  Stride between the beginnings of matrices :math:`B_i` inside the batch array ``b``.

batch_size
  Number of problems in a batch.

scratchpad
  Scratchpad memory to be used by routine for storing intermediate results.

scratchpad_size
  Size of scratchpad memory as a number of floating point elements of type ``T``. Size should not be less then the value returned by the Strided API of the :ref:`onemkl_lapack_potrs_batch_scratchpad_size` function.

events
  List of events to wait for before starting computation. Defaults to empty list.

.. container:: section

  .. rubric:: Output Parameters

b
  Solution matrices :math:`X_i`.

.. container:: section
   
  .. rubric:: Return Values

Output event to wait on to ensure computation is complete.

**Parent topic:** :ref:`onemkl_lapack-like-extensions-routines`


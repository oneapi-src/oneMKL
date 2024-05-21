.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemkl_lapack_getrf_batch:

getrf_batch
===========

Computes the LU factorizations of a batch of general matrices.

.. rubric:: Description

``getrf_batch`` supports the following precisions.

   .. list-table:: 
      :header-rows: 1

      * -  T 
      * -  ``float`` 
      * -  ``double`` 
      * -  ``std::complex<float>`` 
      * -  ``std::complex<double>`` 

.. _onemkl_lapack_getrf_batch_buffer:

getrf_batch (Buffer Version)
----------------------------

.. rubric:: Description

The buffer version of ``getrf_batch`` supports only the strided API. 

**Strided API**

The routine computes the LU factorizations of general :math:`m \times n` matrices :math:`A_i` as :math:`A_i = P_iL_iU_i`, where :math:`P_i` is a permutation matrix, :math:`L_i` is lower triangular with unit diagonal elements (lower trapezoidal if :math:`m > n`) and :math:`U_i` is upper triangular (upper trapezoidal if :math:`m < n`). The routine uses partial pivoting, with row interchanges.

.. rubric:: Syntax

.. code-block:: cpp

    namespace oneapi::mkl::lapack {
      void getrf_batch(sycl::queue &queue, std::int64_t m, std::int64_t n, sycl::buffer<T> &a, std::int64_t lda, std::int64_t stride_a, sycl::buffer<std::int64_t> &ipiv, std::int64_t stride_ipiv, std::int64_t batch_size, sycl::buffer<T> &scratchpad, std::int64_t scratchpad_size)
    }

.. container:: section

   .. rubric:: Input Parameters

queue
  Device queue where calculations will be performed.

m
  Number of rows in matrices :math:`A_i` (:math:`0 \le m`).

n
  Number of columns in matrices :math:`A_i` (:math:`0 \le n`).

a
  Array holding input matrices :math:`A_i`.

lda
  Leading dimension of matrices :math:`A_i`.

stride_a
  Stride between the beginnings of matrices :math:`A_i` inside the batch array ``a``.

stride_ipiv
  Stride between the beginnings of arrays :math:`ipiv_i` inside the array ``ipiv``.

batch_size
  Number of problems in a batch.

scratchpad
  Scratchpad memory to be used by routine for storing intermediate results.

scratchpad_size
  Size of scratchpad memory as a number of floating point elements of type ``T``. Size should not be less than the value returned by the Strided API of the :ref:`onemkl_lapack_getrf_batch_scratchpad_size` function.

.. container:: section

   .. rubric:: Output Parameters

a
  :math:`L_i` and :math:`U_i`. The unit diagonal elements of :math:`L_i` are not stored.

ipiv
  Array containing batch of the pivot indices :math:`\text{ipiv}_i` each of size at least :math:`\max(1,\min(m,n))`; for :math:`1 \le k \le \min(m,n)`, where row :math:`k` of :math:`A_i` was interchanged with row :math:`\text{ipiv}_i(k)`.

.. _onemkl_lapack_getrf_batch_usm:

getrf_batch (USM Version)
-------------------------

.. rubric:: Description

The USM version of ``getrf_batch`` supports the group API and strided API. 

**Group API**

The routine computes the batch of LU factorizations of general :math:`m \times n` matrices :math:`A_i` (:math:`i \in \{1...batch\_size\}`) as :math:`A_i = P_iL_iU_i`, where :math:`P_i` is a permutation matrix, :math:`L_i` is lower triangular with unit diagonal elements (lower trapezoidal if :math:`m > n`) and :math:`U_i` is upper triangular (upper trapezoidal if :math:`m < n`). The routine uses partial pivoting, with row interchanges. Total number of problems to solve, ``batch_size``, is a sum of sizes of all of the groups of parameters as provided by ``group_sizes`` array.

.. rubric:: Syntax

.. code-block:: cpp

    namespace oneapi::mkl::lapack {
      sycl::event getrf_batch(sycl::queue &queue, std::int64_t *m, std::int64_t *n, T **a, std::int64_t *lda, std::int64_t **ipiv, std::int64_t group_count, std::int64_t *group_sizes, T *scratchpad, std::int64_t scratchpad_size, const std::vector<sycl::event> &events = {})
    }

.. container:: section

   .. rubric:: Input Parameters

queue
  Device queue where calculations will be performed.

m
  Array of ``group_count`` parameters :math:`m_g` specifying the number of rows in matrices :math:`A_i` (:math:`0 \le m_g`) belonging to group :math:`g`.

n
  Array of ``group_count`` parameters :math:`n_g` specifying the number of columns in matrices :math:`A_i` (:math:`0 \le n_g`) belonging to group :math:`g`.

a
  Array holding ``batch_size`` pointers to input matrices :math:`A_i`.

lda
  Array of ``group_count`` parameters :math:`lda_g` specifying the leading dimensions of :math:`A_i` belonging to group :math:`g`.

group_count
  Number of groups of parameters. Must be at least 0.

group_sizes
  Array of group_count integers. Array element with index :math:`g` specifies the number of problems to solve for each of the groups of parameters :math:`g`. So the total number of problems to solve, ``batch_size``, is a sum of all parameter group sizes.

scratchpad
  Scratchpad memory to be used by routine for storing intermediate results.

scratchpad_size
  Size of scratchpad memory as a number of floating point elements of type ``T``. Size should not be less then the value returned by the Group API of the :ref:`onemkl_lapack_getrf_batch_scratchpad_size` function.

events
  List of events to wait for before starting computation. Defaults to empty list.

.. container:: section

   .. rubric:: Output Parameters

a
  :math:`L_i` and :math:`U_i`. The unit diagonal elements of :math:`L_i` are not stored.

ipiv
  Arrays of batch_size pointers to arrays containing pivot indices :math:`\text{ipiv}_i` each of size at least :math:`\max(1,\min(m_g,n_g))`; for :math:`1 \le k \le \min(m_g,n_g)`, where row :math:`k` of :math:`A_i` was interchanged with row :math:`\text{ipiv}_i(k)`.

.. container:: section
   
   .. rubric:: Return Values

Output event to wait on to ensure computation is complete.

**Strided API**

The routine computes the LU factorizations of general :math:`m \times n` matrices :math:`A_i` as :math:`A_i = P_iL_iU_i`, where :math:`P_i` is a permutation matrix, :math:`L_i` is lower triangular with unit diagonal elements (lower trapezoidal if :math:`m > n`) and :math:`U_i` is upper triangular (upper trapezoidal if :math:`m < n`). The routine uses partial pivoting, with row interchanges.

.. rubric:: Syntax

.. code-block:: cpp

    namespace oneapi::mkl::lapack {
      sycl::event getrf_batch(sycl::queue &queue, std::int64_t m, std::int64_t n, T *a, std::int64_t lda, std::int64_t stride_a, std::int64_t *ipiv, std::int64_t stride_ipiv, std::int64_t batch_size, T *scratchpad, std::int64_t scratchpad_size, const std::vector<sycl::event> &events = {})
    };

.. container:: section

   .. rubric:: Input Parameters

queue
  Device queue where calculations will be performed.

m
  Number of rows in matrices :math:`A_i` (:math:`0 \le m`).

n
  Number of columns in matrices :math:`A_i` (:math:`0 \le n`).

a
  Array holding input matrices :math:`A_i`.

lda
  Leading dimension of matrices :math:`A_i`.

stride_a
  Stride between the beginnings of matrices :math:`A_i` inside the batch array ``a``.

stride_ipiv
  Stride between the beginnings of arrays :math:`\text{ipiv}_i` inside the array ``ipiv``.

batch_size
  Number of problems in a batch.

scratchpad
  Scratchpad memory to be used by routine for storing intermediate results.

scratchpad_size
  Size of scratchpad memory as a number of floating point elements of type ``T``. Size should not be less then the value returned by the Strided API of the :ref:`onemkl_lapack_getrf_batch_scratchpad_size` function.

events
  List of events to wait for before starting computation. Defaults to empty list.

.. container:: section

   .. rubric:: Output Parameters

a
  :math:`L_i` and :math:`U_i`. The unit diagonal elements of :math:`L_i` are not stored.

ipiv
  Array containing batch of the pivot indices :math:`\text{ipiv}_i` each of size at least :math:`\max(1,\min(m,n))`; for :math:`1 \le k \le \min(m,n)`, where row :math:`k` of :math:`A_i` was interchanged with row :math:`\text{ipiv}_i(k)`.

.. container:: section
   
   .. rubric:: Return Values

Output event to wait on to ensure computation is complete.

**Parent topic:** :ref:`onemkl_lapack-like-extensions-routines`

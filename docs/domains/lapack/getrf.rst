.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemkl_lapack_getrf:

getrf
=====

Computes the LU factorization of a general :math:`m \times n` matrix.

.. container:: section

   .. rubric:: Description

``getrf`` supports the following precisions.

   .. list-table:: 
      :header-rows: 1

      * -  T 
      * -  ``float`` 
      * -  ``double`` 
      * -  ``std::complex<float>`` 
      * -  ``std::complex<double>`` 

The routine computes the LU factorization of a general
:math:`m \times n` matrix :math:`A` as :math:`A = PLU`,

where :math:`P` is a permutation matrix, :math:`L` is lower triangular with
unit diagonal elements (lower trapezoidal if :math:`m > n`) and :math:`U` is
upper triangular (upper trapezoidal if :math:`m < n`). The routine uses
partial pivoting, with row interchanges.

getrf (BUFFER Version)
----------------------

.. container:: section

   .. rubric:: Syntax

.. code-block:: cpp

    namespace oneapi::mkl::lapack {
      void getrf(sycl::queue &queue, std::int64_t m, std::int64_t n, sycl::buffer<T,1> &a, std::int64_t lda, sycl::buffer<std::int64_t,1> &ipiv, sycl::buffer<T,1> &scratchpad, std::int64_t scratchpad_size)
    }

.. container:: section

  .. rubric:: Input Parameters

queue
   The queue where the routine should be executed.

m
    The number of rows in the matrix :math:`A` (:math:`0 \le m`).

n
    The number of columns in :math:`A` (:math:`0 \le n`).

a
   Buffer holding input matrix :math:`A`. The buffer a contains    the matrix :math:`A`. The second dimension of a must be at least   :math:`\max(1, n)`.

lda
   The leading dimension of ``a``.

scratchpad_size
      Size of scratchpad memory as a number of floating point elements of type ``T``.
      Size should not be less than the value returned by :ref:`onemkl_lapack_getrf_scratchpad_size` function.

.. container:: section

  .. rubric:: Output Parameters

a
   Overwritten by :math:`L` and :math:`U`. The unit diagonal    elements of :math:`L` are not stored.

ipiv
   Array, size at least :math:`\max(1,\min(m, n))`. Contains the    pivot indices; for :math:`1 \le i \le \min(m, n)`, row :math:`i` was interchanged with   row :math:`\text{ipiv}(i)`.

scratchpad
   Buffer holding scratchpad memory to be used by routine for storing intermediate results.

getrf (USM Version)
----------------------

.. container:: section

   .. rubric:: Syntax

.. code-block:: cpp

    namespace oneapi::mkl::lapack {
      sycl::event getrf(sycl::queue &queue, std::int64_t m, std::int64_t n, T *a, std::int64_t lda, std::int64_t *ipiv, T *scratchpad, std::int64_t scratchpad_size, const std::vector<sycl::event> &events = {})
    }

.. container:: section

  .. rubric:: Input Parameters

queue
   The queue where the routine should be executed.

m
    The number of rows in the matrix :math:`A` (:math:`0 \le m`).

n
    The number of columns in :math:`A` (:math:`0 \le n`).

a
   Pointer to array holding input matrix :math:`A`. The second dimension of ``a`` must be at least   :math:`\max(1, n)`.

lda
   The leading dimension of ``a``.

scratchpad_size
   Size of scratchpad memory as a number of floating point elements of type ``T``.
   Size should not be less than the value returned by :ref:`onemkl_lapack_getrf_scratchpad_size` function.

events
   List of events to wait for before starting computation. Defaults to empty list.

.. container:: section

  .. rubric:: Output Parameters

a
   Overwritten by :math:`L` and :math:`U`. The unit diagonal    elements of :math:`L` are not stored.

ipiv
   Array, size at least :math:`\max(1,\min(m, n))`. Contains the    pivot indices; for :math:`1 \le i \le \min(m, n)`, row :math:`i` was interchanged with   row :math:`\text{ipiv}(i)`.

scratchpad
   Pointer to scratchpad memory to be used by routine for storing intermediate results.

.. container:: section

  .. rubric:: Return Values

Output event to wait on to ensure computation is complete.

**Parent topic:** :ref:`onemkl_lapack-linear-equation-routines`



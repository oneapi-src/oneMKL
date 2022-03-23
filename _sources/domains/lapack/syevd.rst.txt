.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemkl_lapack_syevd:

syevd
=====

Computes all eigenvalues and, optionally, all eigenvectors of a real
symmetric matrix using divide and conquer algorithm.

.. container:: section

  .. rubric:: Description
      
``syevd`` supports the following precisions.

    .. list-table:: 
       :header-rows: 1

       * -  T 
       * -  ``float`` 
       * -  ``double`` 

The routine computes all the eigenvalues, and optionally all the
eigenvectors, of a real symmetric matrix :math:`A`. In other words, it
can compute the spectral factorization of :math:`A` as: :math:`A = Z\lambda Z^T`.

Here :math:`\Lambda` is a diagonal matrix whose diagonal elements are the
eigenvalues :math:`\lambda_i`, and :math:`Z` is the orthogonal matrix whose
columns are the eigenvectors :math:`z_{i}`. Thus,

:math:`A z_i = \lambda_i z_i` for :math:`i = 1, 2, ..., n`.

If the eigenvectors are requested, then this routine uses a divide
and conquer algorithm to compute eigenvalues and eigenvectors.
However, if only eigenvalues are required, then it uses the
Pal-Walker-Kahan variant of the QL or QR algorithm.

syevd (Buffer Version)
----------------------

.. container:: section

  .. rubric:: Syntax

.. code-block:: cpp

    namespace oneapi::mkl::lapack {
      void syevd(sycl::queue &queue, jobz jobz, oneapi::mkl::uplo upper_lower, std::int64_t n, sycl::buffer<T,1> &a, std::int64_t lda, sycl::buffer<T,1> &w, sycl::buffer<T,1> &scratchpad, std::int64_t scratchpad_size)
    }

.. container:: section

  .. rubric:: Input Parameters
      
queue
   The queue where the routine should be executed.

jobz
   Must be ``job::novec`` or ``job::vec``.

   If ``jobz = job::novec``, then only eigenvalues are computed.

   If ``jobz = job::vec``, then eigenvalues and eigenvectors are
   computed.

upper_lower
   Must be ``uplo::upper`` or ``uplo::lower``.

   If ``upper_lower = job::upper``, a stores the upper triangular
   part of :math:`A`.

   If ``upper_lower = job::lower``, a stores the lower triangular
   part of :math:`A`.

n
   The order of the matrix :math:`A` (:math:`0 \le n`).

a
   The buffer ``a``, size (``lda,*``). The buffer ``a`` contains the matrix
   :math:`A`. The second dimension of ``a`` must be at least :math:`\max(1, n)`.

lda
   The leading dimension of ``a``. Must be at least :math:`\max(1,n)`.

scratchpad_size
   Size of scratchpad memory as a number of floating point elements of type ``T``.
   Size should not be less than the value returned by :ref:`onemkl_lapack_syevd_scratchpad_size` function.

.. container:: section

  .. rubric:: Output Parameters
      
a
   If ``jobz = job::vec``, then on exit this buffer is overwritten by
   the orthogonal matrix :math:`Z` which contains the eigenvectors of
   :math:`A`.

w
   Buffer, size at least :math:`n`. Contains the eigenvalues
   of the matrix :math:`A` in ascending order.

scratchpad
   Buffer holding scratchpad memory to be used by routine for storing intermediate results.

syevd (USM Version)
----------------------

.. container:: section

  .. rubric:: Syntax
         
.. code-block:: cpp

    namespace oneapi::mkl::lapack {
      sycl::event syevd(sycl::queue &queue, jobz jobz, oneapi::mkl::uplo upper_lower, std::int64_t n, T *a, std::int64_t lda, T *w, T *scratchpad, std::int64_t scratchpad_size, const std::vector<sycl::event> &events = {})
    }

.. container:: section

  .. rubric:: Input Parameters

queue
   The queue where the routine should be executed.

jobz
   Must be ``job::novec`` or ``job::vec``.

   If ``jobz = job::novec``, then only eigenvalues are computed.

   If ``jobz = job::vec``, then eigenvalues and eigenvectors are
   computed.

upper_lower
   Must be ``uplo::upper`` or ``uplo::lower``.

   If ``upper_lower = job::upper``, a stores the upper triangular
   part of :math:`A`.

   If ``upper_lower = job::lower``, a stores the lower triangular
   part of :math:`A`.

n
   The order of the matrix :math:`A` (:math:`0 \le n`).

a
   Pointer to array containing :math:`A`, size (``lda,*``).
   The second dimension of ``a`` must be at least :math:`\max(1, n)`.

lda
   The leading dimension of ``a``. Must be at least :math:`\max(1,n)`.

scratchpad_size
   Size of scratchpad memory as a number of floating point elements of type ``T``.
   Size should not be less than the value returned by :ref:`onemkl_lapack_syevd_scratchpad_size` function.

events
   List of events to wait for before starting computation. Defaults to empty list.

.. container:: section

  .. rubric:: Output Parameters

a
   If ``jobz = job::vec``, then on exit this array is overwritten by
   the orthogonal matrix :math:`Z` which contains the eigenvectors of
   :math:`A`.

w
   Pointer to array of size at least :math:`n`. Contains the eigenvalues
   of the matrix :math:`A` in ascending order.

scratchpad
   Pointer to scratchpad memory to be used by routine for storing intermediate results.

.. container:: section

  .. rubric:: Return Values
         
Output event to wait on to ensure computation is complete.

**Parent topic:** :ref:`onemkl_lapack-singular-value-eigenvalue-routines`



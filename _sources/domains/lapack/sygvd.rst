.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemkl_lapack_sygvd:

sygvd
=====

Computes all eigenvalues and, optionally, eigenvectors of a real
generalized symmetric definite eigenproblem using a divide and
conquer method.

.. container:: section

  .. rubric:: Description
      
``sygvd`` supports the following precisions.

     .. list-table:: 
        :header-rows: 1

        * -  T 
        * -  ``float`` 
        * -  ``double`` 

The routine computes all the eigenvalues, and optionally, the
eigenvectors of a real generalized symmetric-definite eigenproblem,
of the form

:math:`Ax = \lambda Bx`, :math:`ABx = \lambda x`, or :math:`BAx = \lambda x` .

Here :math:`A` and :math:`B` are assumed to be symmetric and :math:`B` is also
positive definite.

It uses a divide and conquer algorithm.

sygvd (Buffer Version)
----------------------

.. container:: section

  .. rubric:: Syntax
         
.. code-block:: cpp

    namespace oneapi::mkl::lapack {
      void sygvd(sycl::queue &queue, std::int64_t itype, oneapi::mkl::job jobz, oneapi::mkl::uplo upper_lower, std::int64_t n, sycl::buffer<T,1> &a, std::int64_t lda, sycl::buffer<T,1> &b, std::int64_t ldb, sycl::buffer<T,1> &w, sycl::buffer<T,1> &scratchpad, std::int64_t scratchpad_size)
    }

.. container:: section

  .. rubric:: Input Parameters
      
queue
   The queue where the routine should be executed.

itype
   Must be 1 or 2 or 3. Specifies the problem type to be solved:

   if :math:`\text{itype} = 1`, the problem type is :math:`Ax =  \lambda Bx`;

   if :math:`\text{itype} = 2`, the problem type is :math:`ABx = \lambda x`;

   if :math:`\text{itype} = 3`, the problem type is :math:`BAx = \lambda x`.

jobz
   Must be ``job::novec`` or ``job::vec``.

   If ``jobz = job::novec``, then only eigenvalues are computed.

   If ``jobz = job::vec``, then eigenvalues and eigenvectors are
   computed.

upper_lower
   Must be ``uplo::upper`` or ``uplo::lower``.

   If ``upper_lower = job::upper``, ``a`` and ``b`` store the upper
   triangular part of :math:`A` and :math:`B`.

   If ``upper_lower = job::lower``, ``a`` and ``b`` stores the lower
   triangular part of :math:`A` and :math:`B`.

n
   The order of the matrices :math:`A` and :math:`B` :math:`(0 \le n)`.

a
   Buffer, size a\ ``(lda,*)`` contains the upper or lower triangle
   of the symmetric matrix :math:`A`, as specified by ``upper_lower``. The
   second dimension of ``a`` must be at least :math:`\max(1, n)`.

lda
   The leading dimension of ``a``; at least :math:`\max(1, n)`.

b
   Buffer, size ``b`` ``(ldb,*)`` contains the upper or lower triangle
   of the symmetric matrix :math:`B`, as specified by ``upper_lower``. The
   second dimension of ``b`` must be at least :math:`\max(1, n)`.

ldb
   The leading dimension of ``b``; at least :math:`\max(1, n)`.

scratchpad_size
   Size of scratchpad memory as a number of floating point elements of type ``T``.
   Size should not be less than the value returned by :ref:`onemkl_lapack_sygvd_scratchpad_size` function.

.. container:: section

  .. rubric:: Output Parameters
      
a
   On exit, if ``jobz = job::vec``, then if :math:`\text{info} = 0`, ``a``
   contains the matrix :math:`Z` of eigenvectors. The eigenvectors are
   normalized as follows:

   if :math:`\text{itype} = 1` or :math:`2` , :math:`Z^{T}BZ = I`;

   if :math:`\text{itype} = 3` , :math:`Z^{T}B^{-1}Z = I`;

   If ``jobz = job::novec``, then on exit the upper triangle (if
   ``upper_lower = uplo::upper``) or the lower triangle (if
   ``upper_lower = uplo::lower``) of :math:`A`, including the diagonal,
   is destroyed.

b
   On exit, if :math:`\text{info} \le n`, the part of ``b`` containing the matrix is
   overwritten by the triangular factor :math:`U` or :math:`L` from the
   Cholesky factorization :math:`B = U^{T}U` or
   :math:`B = LL^{T}`.

w
   Buffer, size at least :math:`n`. If :math:`\text{info} = 0`, contains the
   eigenvalues of the matrix :math:`A` in ascending order.

scratchpad
   Buffer holding scratchpad memory to be used by routine for storing intermediate results.

sygvd (USM Version)
----------------------

.. container:: section

  .. rubric:: Syntax

.. code-block:: cpp

    namespace oneapi::mkl::lapack {
      sycl::event sygvd(sycl::queue &queue, std::int64_t itype, oneapi::mkl::job jobz, oneapi::mkl::uplo upper_lower, std::int64_t n, T *a, std::int64_t lda, T *b, std::int64_t ldb, T *w, T *scratchpad, std::int64_t scratchpad_size, const std::vector<sycl::event> &events = {})
    }

.. container:: section

  .. rubric:: Input Parameters
      
queue
   The queue where the routine should be executed.

itype
   Must be 1 or 2 or 3. Specifies the problem type to be solved:

   if :math:`\text{itype} = 1`, the problem type is :math:`Ax =  \lambda Bx`;

   if :math:`\text{itype} = 2`, the problem type is :math:`ABx = \lambda x`;

   if :math:`\text{itype} = 3`, the problem type is :math:`BAx = \lambda x`.

jobz
   Must be ``job::novec`` or ``job::vec``.

   If ``jobz = job::novec``, then only eigenvalues are computed.

   If ``jobz = job::vec``, then eigenvalues and eigenvectors are
   computed.

upper_lower
   Must be ``uplo::upper`` or ``uplo::lower``.

   If ``upper_lower = job::upper``, ``a`` and ``b`` store the upper
   triangular part of :math:`A` and :math:`B`.

   If ``upper_lower = job::lower``, ``a`` and ``b`` stores the lower
   triangular part of :math:`A` and :math:`B`.

n
   The order of the matrices :math:`A` and :math:`B` :math:`(0 \le n)`.

a
   Pointer to array of size a\ ``(lda,*)`` containing the upper or lower triangle
   of the symmetric matrix :math:`A`, as specified by ``upper_lower``. The
   second dimension of ``a`` must be at least :math:`\max(1, n)`.

lda
   The leading dimension of ``a``; at least :math:`\max(1, n)`.

b
   Pointer to array of size ``b`` ``(ldb,*)`` contains the upper or lower triangle
   of the symmetric matrix :math:`B`, as specified by ``upper_lower``. The
   second dimension of ``b`` must be at least :math:`\max(1, n)`.

ldb
   The leading dimension of ``b``; at least :math:`\max(1, n)`.

scratchpad_size
   Size of scratchpad memory as a number of floating point elements of type ``T``.
   Size should not be less than the value returned by :ref:`onemkl_lapack_sygvd_scratchpad_size` function.

events
   List of events to wait for before starting computation. Defaults to empty list.

.. container:: section

  .. rubric:: Output Parameters

a
   On exit, if ``jobz = job::vec``, then if :math:`\text{info} = 0`, :math:`a`
   contains the matrix :math:`Z` of eigenvectors. The eigenvectors are
   normalized as follows:

   if :math:`\text{itype} = 1` or :math:`2`, :math:`Z^{T}BZ = I`;
   
   if :math:`\text{itype} = 3`, :math:`Z^{T}B^{-1}Z = I`;

   If ``jobz = job::novec``, then on exit the upper triangle (if
   ``upper_lower = uplo::upper``) or the lower triangle (if
   ``upper_lower = uplo::lower``) of :math:`A`, including the diagonal,
   is destroyed.

b
   On exit, if :math:`\text{info} \le n`, the part of ``b`` containing the matrix is
   overwritten by the triangular factor :math:`U` or :math:`L` from the
   Cholesky factorization :math:`B` = :math:`U^{T}U` or
   :math:`B = LL^{T}`.

w
   Pointer to array of size at least ``n``. If :math:`\text{info} = 0`, contains the
   eigenvalues of the matrix :math:`A` in ascending order.

scratchpad
   Pointer to scratchpad memory to be used by routine for storing intermediate results.

.. container:: section

  .. rubric:: Return Values

Output event to wait on to ensure computation is complete

**Parent topic:** :ref:`onemkl_lapack-singular-value-eigenvalue-routines`



.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemkl_lapack_gesvd:

gesvd
=====

Computes the singular value decomposition of a general rectangular matrix.

.. container:: section

  .. rubric:: Description

``gesvd`` supports the following precisions.

    .. list-table::
       :header-rows: 1

       * -  T
       * -  ``float``
       * -  ``double``
       * -  ``std::complex<float>``
       * -  ``std::complex<double>``

.. _onemkl_lapack_gesvd_batch_buffer:

gesvd (Buffer Version)
----------------------

.. container:: section

  .. rubric:: Description

The routine computes the singular value decomposition (SVD) of a
real/complex :math:`m \times n` matrix :math:`A`, optionally computing the
left and/or right singular vectors. The SVD is written as

:math:`A = U\Sigma V^T` for real routines

:math:`A = U\Sigma V^H` for complex routines

where :math:`\Sigma` is an :math:`m \times n` diagonal matrix, :math:`U` is an
:math:`m \times m` orthogonal/unitary matrix, and :math:`V` is an
:math:`n \times n` orthogonal/unitary matrix. The diagonal elements of :math:`\Sigma`
are the singular values of :math:`A`; they are real and non-negative, and
are returned in descending order. The first :math:`\min(m, n)` columns of
:math:`U` and :math:`V` are the left and right singular vectors of :math:`A`.

.. container:: section

  .. rubric:: Syntax

.. code-block:: cpp

    namespace oneapi::mkl::lapack {
      void gesvd(sycl::queue &queue, oneapi::mkl::job jobu, oneapi::mkl::job jobvt, std::int64_t m, std::int64_t n, sycl::buffer<T,1> &a, std::int64_t lda, sycl::buffer<realT,1> &s, sycl::buffer<T,1> &u, std::int64_t ldu, sycl::buffer<T,1> &vt, std::int64_t ldvt, sycl::buffer<T,1> &scratchpad, std::int64_t scratchpad_size)
    }

.. container:: section

  .. rubric:: Input Parameters

queue
   The queue where the routine should be executed.

jobu
   Must be ``job::allvec``, ``job::somevec``, ``job::overwritevec``,
   or ``job::novec``. Specifies options for computing all or part of
   the matrix :math:`U`.

   If ``jobu = job::allvec``, all :math:`m` columns of :math:`U` are returned
   in the buffer ``u``;

   if ``jobu = job::somevec``, the first :math:`\min(m, n)` columns of
   :math:`U` (the left singular vectors) are returned in the buffer ``u``;

   if ``jobu = job::overwritevec``, the first :math:`\min(m, n)` columns
   of :math:`U` (the left singular vectors) are overwritten on the buffer
   a;

   if ``jobu = job::novec``, no columns of :math:`U` (no left singular
   vectors) are computed.

jobvt
   Must be ``job::allvec, job::somevec``, ``job::overwritevec``, or
   ``job::novec``. Specifies options for computing all or part of the
   matrix :math:`V^T/V^H`.

   If ``jobvt = job::allvec``, all :math:`n` columns of :math:`V^T/V^H` are
   returned in the buffer vt;

   if ``jobvt = job::somevec``, the first :math:`\min(m, n)` columns of
   :math:`V^T/V^H` (the left singular vectors) are returned in the buffer
   vt;

   if ``jobvt = job::overwritevec``, the first :math:`\min(m, n)` columns
   of :math:`V^T/V^H` (the left singular vectors) are overwritten on the
   buffer ``a``;

   if ``jobvt = job::novec``, no columns of :math:`V^T/V^H` (no left
   singular vectors) are computed.

   ``jobvt`` and ``jobu`` cannot both be ``job::overwritevec``.

m
   The number of rows in the matrix :math:`A` (:math:`0 \le m`).

a
   The buffer ``a``, size ``(lda,*)``. The buffer ``a`` contains the
   matrix :math:`A`. The second dimension of ``a`` must be at least
   :math:`\max(1, m)`.

lda
   The leading dimension of ``a``.

ldu
   The leading dimension of ``u``.

ldvt
   The leading dimension of ``vt``.

scratchpad_size
   Size of scratchpad memory as a number of floating point elements of type ``T``.
   Size should not be less than the value returned by :ref:`onemkl_lapack_gesvd_scratchpad_size` function.

.. container:: section

  .. rubric:: Output Parameters

a
   On exit,

   If ``jobu = job::overwritevec``, ``a`` is overwritten with the first
   :math:`\min(m,n)` columns of :math:`U` (the left singular vectors stored
   columnwise);

   If ``jobvt = job::overwritevec``, ``a`` is overwritten with the first
   :math:`\min(m, n)` rows of :math:`V^{T}`/:math:`V^{H}` (the right
   singular vectors stored rowwise);

   If ``jobu`` :math:`\ne` ``job::overwritevec`` and ``jobvt`` :math:`\ne` ``job::overwritevec``,
   the contents of a are destroyed.

s
   Buffer containing the singular values, size at least
   :math:`\max(1, \min(m,n))`. Contains the singular values of :math:`A` sorted
   so that :math:`s(i) \ge s(i+1)`.

u
   Buffer containing :math:`U`; the second dimension of ``u`` must be at
   least :math:`\max(1, m)` if ``jobu = job::allvec``, and at least
   :math:`\max(1, \min(m, n))` if ``jobu = job::somevec``.

   If ``jobu = job::allvec``, ``u`` contains the :math:`m \times m`
   orthogonal/unitary matrix :math:`U`.

   If ``jobu = job::somevec``, ``u`` contains the first :math:`\min(m, n)`
   columns of :math:`U` (the left singular vectors stored column-wise).

   If ``jobu = job::novec`` or ``job::overwritevec``, ``u`` is not
   referenced.

vt
   Buffer containing :math:`V^{T}`; the second dimension of ``vt`` must
   be at least :math:`\max(1, n)`.

   If ``jobvt = job::allvec``, ``vt`` contains the :math:`n \times n`
   orthogonal/unitary matrix :math:`V^{T}`/:math:`V^{H}`.

   If ``jobvt = job::somevec``, ``vt`` contains the first :math:`\min(m, n)`
   rows of :math:`V^{T}`/:math:`V^{H}` (the right singular
   vectors stored row-wise).

   If ``jobvt = job::novec`` or ``job::overwritevec``, ``vt`` is not
   referenced.

scratchpad
   Buffer holding scratchpad memory to be used by routine for storing intermediate results.

gesvd (USM Version)
----------------------

.. container:: section

  .. rubric:: Description

The routine computes the singular value decomposition (SVD) of a
real/complex :math:`m \times n` matrix :math:`A`, optionally computing the
left and/or right singular vectors. The SVD is written as

:math:`A = U\Sigma V^T` for real routines

:math:`A = U\Sigma V^H` for complex routines

where :math:`\Sigma` is an :math:`m \times n` diagonal matrix, :math:`U` is an
:math:`m \times m` orthogonal/unitary matrix, and :math:`V` is an
:math:`n \times n` orthogonal/unitary matrix. The diagonal elements of :math:`\Sigma`
are the singular values of :math:`A`; they are real and non-negative, and
are returned in descending order. The first :math:`\min(m, n)` columns of
:math:`U` and :math:`V` are the left and right singular vectors of :math:`A`.

.. container:: section
  
  .. rubric:: Syntax

.. code-block:: cpp

    namespace oneapi::mkl::lapack {
      sycl::event gesvd(sycl::queue &queue, oneapi::mkl::job jobu, oneapi::mkl::job jobvt, std::int64_t m, std::int64_t n, T *a, std::int64_t lda, RealT *s, T *u, std::int64_t ldu, T *vt, std::int64_t ldvt, T *scratchpad, std::int64_t scratchpad_size, const std::vector<sycl::event> &events = {})
    }

.. container:: section

  .. rubric:: Input Parameters

queue
   The queue where the routine should be executed.

jobu
   Must be ``job::allvec``, ``job::somevec``, ``job::overwritevec``,
   or ``job::novec``. Specifies options for computing all or part of
   the matrix :math:`U`.

   If ``jobu = job::allvec``, all :math:`m` columns of :math:`U` are returned
   in the array ``u``;

   if ``jobu = job::somevec``, the first :math:`\min(m, n)` columns of
   :math:`U` (the left singular vectors) are returned in the array ``u``;

   if ``jobu = job::overwritevec``, the first :math:`\min(m, n)` columns
   of :math:`U` (the left singular vectors) are overwritten on the array
   a;

   if ``jobu = job::novec``, no columns of :math:`U` (no left singular
   vectors) are computed.

jobvt
   Must be ``job::allvec, job::somevec``, ``job::overwritevec``, or
   ``job::novec``. Specifies options for computing all or part of the
   matrix :math:`V^T/V^H`.

   If ``jobvt = job::allvec``, all :math:`n` columns of :math:`V^T/V^H` are
   returned in the array ``vt``;

   if ``jobvt = job::somevec``, the first :math:`\min(m, n)` columns of
   :math:`V^T/V^H` (the left singular vectors) are returned in the array
   vt;

   if ``jobvt = job::overwritevec``, the first :math:`\min(m, n)` columns
   of :math:`V^T/V^H` (the left singular vectors) are overwritten on the
   array ``a``;

   if ``jobvt = job::novec``, no columns of :math:`V^T/V^H` (no left
   singular vectors) are computed.

   ``jobvt`` and ``jobu`` cannot both be ``job::overwritevec``.

m
   The number of rows in the matrix :math:`A` (:math:`0 \le m`).

a
   Pointer to array ``a``, size ``(lda,*)``, containing the
   matrix :math:`A`. The second dimension of ``a`` must be at least
   :math:`\max(1, m)`.

lda
   The leading dimension of ``a``.

ldu
   The leading dimension of ``u``.

ldvt
   The leading dimension of ``vt``.

scratchpad_size
   Size of scratchpad memory as a number of floating point elements of type ``T``.
   Size should not be less than the value returned by :ref:`onemkl_lapack_gesvd_scratchpad_size` function.

events
   List of events to wait for before starting computation. Defaults to empty list.

.. container:: section

  .. rubric:: Output Parameters

a
   On exit,

   If ``jobu = job::overwritevec``, ``a`` is overwritten with the first
   :math:`\min(m,n)` columns of :math:`U` (the left singular vectors stored
   columnwise);

   If ``jobvt = job::overwritevec``, ``a`` is overwritten with the first
   :math:`\min(m, n)` rows of :math:`V^{T}`/:math:`V^{H}` (the right
   singular vectors stored rowwise);

   If ``jobu`` :math:`\ne` ``job::overwritevec`` and ``jobvt`` :math:`\ne` ``job::overwritevec``,
   the contents of a are destroyed.

s
   Array containing the singular values, size at least
   :math:`\max(1, \min(m,n))`. Contains the singular values of :math:`A` sorted
   so that :math:`s(i) \ge s(i+1)`.

u
   Array containing :math:`U`; the second dimension of ``u`` must be at
   least :math:`\max(1, m)` if ``jobu = job::allvec``, and at least
   :math:`\max(1, \min(m, n))` if ``jobu = job::somevec``.

   If ``jobu = job::allvec``, ``u`` contains the :math:`m \times m`
   orthogonal/unitary matrix :math:`U`.

   If ``jobu = job::somevec``, ``u`` contains the first :math:`\min(m, n)`
   columns of :math:`U` (the left singular vectors stored column-wise).

   If ``jobu = job::novec`` or ``job::overwritevec``, ``u`` is not
   referenced.

vt
   Array containing :math:`V^{T}`; the second dimension of ``vt`` must
   be at least :math:`\max(1, n)`.

   If ``jobvt = job::allvec``, ``vt`` contains the :math:`n \times n`
   orthogonal/unitary matrix :math:`V^{T}`/:math:`V^{H}`.

   If ``jobvt = job::somevec``, ``vt`` contains the first :math:`\min(m, n)`
   rows of :math:`V^{T}`/:math:`V^{H}` (the right singular
   vectors stored row-wise).

   If ``jobvt = job::novec`` or ``job::overwritevec``, ``vt`` is not
   referenced.

scratchpad
   Pointer to scratchpad memory to be used by routine for storing intermediate results.

.. container:: section

  .. rubric:: Return Values

Output event to wait on to ensure computation is complete.

**Parent topic:** :ref:`onemkl_lapack-singular-value-eigenvalue-routines`

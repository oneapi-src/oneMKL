.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemkl_lapack_unmqr:

unmqr
=====

Multiplies a complex matrix by the unitary matrix :math:`Q` of the QR
factorization formed by :ref:`onemkl_lapack_geqrf`.

.. container:: section

  .. rubric:: Description

``unmqr`` supports the following precisions.

    .. list-table::
       :header-rows: 1

       * -  T
       * -  ``std::complex<float>``
       * -  ``std::complex<double>``

The routine multiplies a rectangular complex :math:`m \times n` matrix :math:`C` by
:math:`Q` or :math:`Q^H`, where :math:`Q` is the complex unitary matrix defined
as a product of :math:`k` elementary reflectors :math:`H(i)` of order :math:`n`:
:math:`Q = H(1)^HH(2)^H ... H(k)^H` as returned by the RQ factorization routine
:ref:`onemkl_lapack_gerqf`.

Depending on the parameters ``side`` and ``trans``, the routine can form one of
the matrix products :math:`QC`, :math:`Q^HC`, :math:`CQ`, or :math:`CQ^H`
(overwriting the result over :math:`C`).

unmqr (Buffer Version)
----------------------

.. container:: section

  .. rubric:: Syntax

.. code-block:: cpp

    namespace oneapi::mkl::lapack {
      void unmqr(sycl::queue &queue, oneapi::mkl::side side, oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n, std::int64_t k, sycl::buffer<T,1> &a, std::int64_t lda, sycl::buffer<T,1> &tau, sycl::buffer<T,1> &c, std::int64_t ldc, sycl::buffer<T,1> &scratchpad, std::int64_t scratchpad_size)
    }

.. container:: section

  .. rubric:: Input Parameters

queue
    The queue where the routine should be executed.

side
    If ``side = oneapi::mkl::side::left``, :math:`Q` or :math:`Q^{H}` is applied
    to :math:`C` from the left.

    If ``side = oneapi::mkl::side::right``, :math:`Q` or :math:`Q^{H}` is
    applied to :math:`C` from the right.

trans
    If ``trans = oneapi::mkl::transpose::nontrans``, the routine multiplies
    :math:`C` by :math:`Q`.

    If ``trans = oneapi::mkl::transpose::conjtrans``, the routine multiplies :math:`C`
    by :math:`Q^{H}`.

m
    The number of rows in the matrix :math:`C` (:math:`0 \le m`).

n
    The number of columns in the matrix :math:`C` (:math:`0 \le n`).

k
    The number of elementary reflectors whose product defines the
    matrix :math:`Q` 

    If ``side = oneapi::mkl::side::left``, :math:`0 \le k \le m`

    If ``side = oneapi::mkl::side::right``, :math:`0 \le k \le n`

a
    The buffer ``a`` as returned by :ref:`onemkl_lapack_geqrf`.
    The second dimension of ``a`` must be at least :math:`\max(1,k)`.

lda
    The leading dimension of ``a``.

tau
    The buffer ``tau`` as returned by :ref:`onemkl_lapack_geqrf`.

c
    The buffer ``c`` contains the matrix :math:`C`. The second dimension of
    ``c`` must be at least :math:`\max(1,n)`.

ldc
    The leading dimension of ``c``.

scratchpad_size
    Size of scratchpad memory as a number of floating point elements of type
    ``T``. Size should not be less than the value returned by the
    :ref:`onemkl_lapack_unmqr_scratchpad_size` function.

.. container:: section

  .. rubric:: Output Parameters

c
    Overwritten by the product :math:`QC`, :math:`Q^{H}C`, :math:`CQ`, or
    :math:`CQ^H` (as specified by ``side`` and ``trans``).

scratchpad
    Buffer holding scratchpad memory to be used by routine for storing intermediate results.

unmqr (USM Version)
----------------------

.. container:: section

  .. rubric:: Syntax

.. code-block:: cpp

    namespace oneapi::mkl::lapack {
      sycl::event unmqr(sycl::queue &queue, oneapi::mkl::side side, oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n, std::int64_t k, T *a, std::int64_t lda, T *tau, T *c, std::int64_t ldc, T *scratchpad, std::int64_t scratchpad_size, const std::vector<sycl::event> &events = {})
    }

.. container:: section

  .. rubric:: Input Parameters

queue
    The queue where the routine should be executed.

side
    If ``side = oneapi::mkl::side::left``, :math:`Q` or :math:`Q^{H}` is applied
    to :math:`C` from the left.

    If ``side = oneapi::mkl::side::right``, :math:`Q` or :math:`Q^{H}` is
    applied to :math:`C` from the right.

trans
    If ``trans = oneapi::mkl::transpose::nontrans``, the routine multiplies
    :math:`C` by :math:`Q`.

    If ``trans = oneapi::mkl::transpose::conjtrans``, the routine multiplies :math:`C`
    by :math:`Q^{H}`.

m
    The number of rows in the matrix :math:`C` (:math:`0 \le m`).

n
    The number of columns in the matrix :math:`C` (:math:`0 \le n`).

k
    The number of elementary reflectors whose product defines the
    matrix :math:`Q`

    If ``side = oneapi::mkl::side::left``, :math:`0 \le k \le m`

    If ``side = oneapi::mkl::side::right``, :math:`0 \le k \le n`

a
    The pointer to ``a`` as returned by :ref:`onemkl_lapack_geqrf`.
    The second dimension of ``a`` must be at least :math:`\max(1,k)`.

lda
    The leading dimension of ``a``.

tau
    The pointer to ``tau`` as returned by :ref:`onemkl_lapack_geqrf`.

c
    The pointer ``c`` points to the matrix :math:`C`. The second dimension of
    ``c`` must be at least :math:`\max(1,n)`.

ldc
    The leading dimension of ``c``.

scratchpad_size
    Size of scratchpad memory as a number of floating point elements of type
    ``T``. Size should not be less than the value returned by
    :ref:`onemkl_lapack_unmqr_scratchpad_size` function.

events
    List of events to wait for before starting computation. Defaults to empty list.

.. container:: section

  .. rubric:: Output Parameters

c
    Overwritten by the product :math:`QC`, :math:`Q^{H}C`, :math:`CQ`, or
    :math:`CQ^{H}` (as specified by ``side`` and ``trans``).

scratchpad
    Pointer to scratchpad memory to be used by routine for storing intermediate results.

.. container:: section

  .. rubric:: Return Values

Output event to wait on to ensure computation is complete.

**Parent topic:** :ref:`onemkl_lapack-linear-equation-routines`

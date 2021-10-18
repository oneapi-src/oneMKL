.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemkl_lapack_unmrq:

unmrq
=====

Multiplies a complex matrix by the unitary matrix :math:`Q` of the RQ
factorization formed by :ref:`onemkl_lapack_gerqf`.

.. container:: section

  .. rubric:: Description

``unmrq`` supports the following precisions.

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

unmrq (Buffer Version)
----------------------

.. container:: section

  .. rubric:: Syntax

.. code-block:: cpp

    namespace oneapi::mkl::lapack {
      void unmrq(cl::sycl::queue &queue, oneapi::mkl::side side, oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n, std::int64_t k, cl::sycl::buffer<T,1> &a, std::int64_t lda, cl::sycl::buffer<T,1> &tau, cl::sycl::buffer<T,1> &c, std::int64_t ldc, cl::sycl::buffer<T,1> &scratchpad, std::int64_t scratchpad_size)
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
    The buffer ``a`` as returned by :ref:`onemkl_lapack_gerqf`.
    The second dimension of ``a`` must be at least :math:`\max(1,k)`.

lda
    The leading dimension of ``a``.

tau
    The buffer ``tau`` as returned by :ref:`onemkl_lapack_gerqf`.

c
    The buffer ``c`` contains the matrix :math:`C`. The second dimension of
    ``c`` must be at least :math:`\max(1,n)`.

ldc
    The leading dimension of ``c``.

scratchpad_size
    Size of scratchpad memory as a number of floating point elements of type
    ``T``. Size should not be less than the value returned by
    :ref:`onemkl_lapack_unmrq_scratchpad_size` function.

.. container:: section

  .. rubric:: Output Parameters

c
    Overwritten by the product :math:`QC`, :math:`Q^{H}C`, :math:`CQ`, or
    :math:`CQ^H` (as specified by ``side`` and ``trans``).

scratchpad
    Buffer holding scratchpad memory to be used by routine for storing intermediate results.

.. container:: section

  .. rubric:: Throws

This routine shall throw the following exceptions if the associated condition is detected. An implementation may throw additional implementation-specific exception(s) in case of error conditions not covered here.

:ref:`oneapi::mkl::host_bad_alloc<onemkl_exception_host_bad_alloc>`

:ref:`oneapi::mkl::device_bad_alloc<onemkl_exception_device_bad_alloc>`

:ref:`oneapi::mkl::unimplemented<onemkl_exception_unimplemented>`

:ref:`oneapi::mkl::unsupported_device<onemkl_exception_unsupported_device>`

:ref:`oneapi::mkl::lapack::invalid_argument<onemkl_lapack_exception_invalid_argument>`

:ref:`oneapi::mkl::lapack::computation_error<onemkl_lapack_exception_computation_error>`

    Exception is thrown in case of problems during calculations. The ``info`` code of the problem can be obtained by `info()` method of exception object:

    If :math:`\text{info}=-i`, the :math:`i`-th parameter had an illegal value.

    If ``info`` equals to value passed as scratchpad size, and `detail()` returns non zero, then passed scratchpad is of insufficient size, and required size should not be less than value return by `detail()` method of exception object.

unmrq (USM Version)
----------------------

.. container:: section

  .. rubric:: Syntax

.. code-block:: cpp

    namespace oneapi::mkl::lapack {
      cl::sycl::event unmrq(cl::sycl::queue &queue, oneapi::mkl::side side, oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n, std::int64_t k, T *a, std::int64_t lda, T *tau, T *c, std::int64_t ldc, T *scratchpad, std::int64_t scratchpad_size, const cl::sycl::vector_class<cl::sycl::event> &events = {})
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
    The pointer to ``a`` as returned by :ref:`onemkl_lapack_gerqf`.
    The second dimension of ``a`` must be at least :math:`\max(1,k)`.

lda
    The leading dimension of ``a``.

tau
    The pointer to ``tau`` as returned by :ref:`onemkl_lapack_gerqf`.

c
    The pointer ``c`` points to the matrix :math:`C`. The second dimension of
    ``c`` must be at least :math:`\max(1,n)`.

ldc
    The leading dimension of ``c``.

scratchpad_size
    Size of scratchpad memory as a number of floating point elements of type
    ``T``. Size should not be less than the value returned by
    :ref:`onemkl_lapack_unmrq_scratchpad_size` function.

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

  .. rubric:: Throws

This routine shall throw the following exceptions if the associated condition is detected. An implementation may throw additional implementation-specific exception(s) in case of error conditions not covered here.

:ref:`oneapi::mkl::host_bad_alloc<onemkl_exception_host_bad_alloc>`

:ref:`oneapi::mkl::device_bad_alloc<onemkl_exception_device_bad_alloc>`

:ref:`oneapi::mkl::unimplemented<onemkl_exception_unimplemented>`

:ref:`oneapi::mkl::unsupported_device<onemkl_exception_unsupported_device>`

:ref:`oneapi::mkl::lapack::invalid_argument<onemkl_lapack_exception_invalid_argument>`

:ref:`oneapi::mkl::lapack::computation_error<onemkl_lapack_exception_computation_error>`

    Exception is thrown in case of problems during calculations. The ``info`` code of the problem can be obtained by `info()` method of exception object:

    If :math:`\text{info}=-i`, the :math:`i`-th parameter had an illegal value.

    If ``info`` equals to value passed as scratchpad size, and `detail()` returns non zero, then passed scratchpad is of insufficient size, and required size should not be less than value return by `detail()` method of exception object.

.. container:: section

  .. rubric:: Return Values

Output event to wait on to ensure computation is complete.

**Parent topic:** :ref:`onemkl_lapack-linear-equation-routines`

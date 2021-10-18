.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemkl_lapack_ormtr:

ormtr
=====

Multiplies a real matrix by the real orthogonal matrix :math:`Q` determined by
:ref:`onemkl_lapack_sytrd`.

.. container:: section

  .. rubric:: Description
      
``ormtr`` supports the following precisions.

     .. list-table:: 
        :header-rows: 1

        * -  T 
        * -  ``float`` 
        * -  ``double`` 

The routine multiplies a real matrix :math:`C` by :math:`Q` or :math:`Q^{T}`, 
where :math:`Q` is the orthogonal matrix :math:`Q` formed by:ref:`onemkl_lapack_sytrd` 
when reducing a real symmetric matrix :math:`A` to tridiagonal form:
:math:`A = QTQ^T`. Use this routine after a call to :ref:`onemkl_lapack_sytrd`.

Depending on the parameters side and trans, the routine can
form one of the matrix products :math:`QC`, :math:`Q^TC`, :math:`CQ`, or
:math:`CQ^T` (overwriting the result on :math:`C`).

ormtr (Buffer Version)
----------------------

.. container:: section

  .. rubric:: Syntax
         
.. code-block:: cpp

    namespace oneapi::mkl::lapack {
      void ormtr(cl::sycl::queue &queue, oneapi::mkl::side side, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n, cl::sycl::buffer<T,1> &a, std::int64_t lda, cl::sycl::buffer<T,1> &tau, cl::sycl::buffer<T,1> &c, std::int64_t ldc, cl::sycl::buffer<T,1> &scratchpad, std::int64_t scratchpad_size)
    }

.. container:: section

  .. rubric:: Input Parameters

In the descriptions below, ``r`` denotes the order of :math:`Q`:

.. container:: tablenoborder

     .. list-table:: 
        :header-rows: 1

        * -  :math:`r = m` 
          -  if ``side = side::left`` 
        * -  :math:`r = n` 
          -  if ``side = side::right`` 

queue
   The queue where the routine should be executed.

side
   Must be either ``side::left`` or ``side::right``.

   If ``side = side::left``, :math:`Q` or :math:`Q^{T}` is
   applied to :math:`C` from the left.

   If ``side = side::right``, :math:`Q` or :math:`Q^{T}` is
   applied to :math:`C` from the right.

upper_lower
   Must be either ``uplo::upper`` or ``uplo::lower``. Uses the same
   ``upper_lower`` as supplied to
   :ref:`onemkl_lapack_sytrd`.

trans
   Must be either ``transpose::nontrans`` or ``transpose::trans``.

   If ``trans = transpose::nontrans``, the routine multiplies :math:`C`
   by :math:`Q`.

   If ``trans = transpose::trans``, the routine multiplies :math:`C` by
   :math:`Q^{T}`.

m
   The number of rows in the matrix :math:`C` :math:`(m \ge 0)`.

n
   The number of columns in the matrix :math:`C` :math:`(n \ge 0)`.

a
   The buffer ``a`` as returned by   :ref:`onemkl_lapack_sytrd`.

lda
   The leading dimension of ``a`` :math:`(\max(1, r) \le \text{lda})`.

tau
   The buffer ``tau`` as returned bya   :ref:`onemkl_lapack_sytrd`. The
   dimension of ``tau`` must be at least :math:`\max(1, r-1)`.

c
   The buffer ``c`` contains the matrix :math:`C`. The second dimension of ``c``
   must be at least :math:`\max(1, n)`.

ldc
   The leading dimension of ``c`` :math:`(\max(1, n) \le \text{ldc})`.

scratchpad_size
   Size of scratchpad memory as a number of floating point elements of type ``T``.
   Size should not be less than the value returned by :ref:`onemkl_lapack_ormtr_scratchpad_size` function.

.. container:: section

  .. rubric:: Output Parameters

c
   Overwritten by the product :math:`QC`, :math:`Q^TC`, :math:`CQ`, or :math:`CQ^T`
   (as specified by ``side`` and ``trans``).

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

ormtr (USM Version)
----------------------

.. container:: section

  .. rubric:: Syntax
         
.. code-block:: cpp

    namespace oneapi::mkl::lapack {
      cl::sycl::event ormtr(cl::sycl::queue &queue, oneapi::mkl::side side, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n, T *a, std::int64_t lda, T *tau, T *c, std::int64_t ldc, T *scratchpad, std::int64_t scratchpad_size, const cl::sycl::vector_class<cl::sycl::event> &events = {})
    }

.. container:: section

  .. rubric:: Input Parameters

In the descriptions below, ``r`` denotes the order of :math:`Q`:

.. container:: tablenoborder

     .. list-table:: 
        :header-rows: 1

        * -  :math:`r = m` 
          -  if ``side = side::left`` 
        * -  :math:`r = n` 
          -  if ``side = side::right`` 

queue
   The queue where the routine should be executed.

side
   Must be either ``side::left`` or ``side::right``.

   If ``side = side::left``, :math:`Q` or :math:`Q^{T}` is
   applied to :math:`C` from the left.

   If ``side = side::right``, :math:`Q` or :math:`Q^{T}` is
   applied to :math:`C` from the right.

upper_lower
   Must be either ``uplo::upper`` or ``uplo::lower``. Uses the same
   ``upper_lower`` as supplied to   :ref:`onemkl_lapack_sytrd`.

trans
   Must be either ``transpose::nontrans`` or ``transpose::trans``.

   If ``trans = transpose::nontrans``, the routine multiplies :math:`C`
   by :math:`Q`.

   If ``trans = transpose::trans``, the routine multiplies :math:`C` by
   :math:`Q^{T}`.

m
   The number of rows in the matrix :math:`C` :math:`(m \ge 0)`.

n
   The number of columns in the matrix :math:`C` :math:`(n \ge 0)`.

a
   The pointer to ``a`` as returned by   :ref:`onemkl_lapack_sytrd`.

lda
   The leading dimension of ``a`` :math:`(\max(1, r) \le \text{lda})`.

tau
   The buffer ``tau`` as returned by   :ref:`onemkl_lapack_sytrd`. The
   dimension of ``tau`` must be at least :math:`\max(1, r-1)`.

c
   The pointer to memory containing the matrix :math:`C`. The second dimension of ``c``
   must be at least :math:`\max(1, n)`.

ldc
   The leading dimension of ``c`` :math:`(\max(1, n) \le \text{ldc})`.

scratchpad_size
   Size of scratchpad memory as a number of floating point elements of type ``T``.
   Size should not be less than the value returned by :ref:`onemkl_lapack_ormtr_scratchpad_size` function.

events
   List of events to wait for before starting computation. Defaults to empty list.

.. container:: section

  .. rubric:: Output Parameters

c
   Overwritten by the product :math:`QC`, :math:`Q^TC`, :math:`CQ`, or :math:`CQ^T`
   (as specified by ``side`` and ``trans``).

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

**Parent topic:** :ref:`onemkl_lapack-singular-value-eigenvalue-routines`


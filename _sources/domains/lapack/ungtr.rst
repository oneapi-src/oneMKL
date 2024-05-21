.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemkl_lapack_ungtr:

ungtr
=====

Generates the complex unitary matrix :math:`Q` determined by
:ref:`onemkl_lapack_hetrd`.

.. container:: section

  .. rubric:: Description
      
``ungtr`` supports the following precisions.

     .. list-table:: 
        :header-rows: 1

        * -  T 
        * -  ``std::complex<float>`` 
        * -  ``std::complex<double>`` 

The routine explicitly generates the :math:`n \times n` unitary matrix
:math:`Q` formed by :ref:`onemkl_lapack_hetrd` when
reducing a complex Hermitian matrix :math:`A` to tridiagonal form:
:math:`A = QTQ^H`. Use this routine after a call to
:ref:`onemkl_lapack_hetrd`.

ungtr (Buffer Version)
----------------------

.. container:: section

  .. rubric:: Syntax

.. code-block:: cpp

    namespace oneapi::mkl::lapack {
      void ungtr(sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n, sycl::buffer<T,1> &a, std::int64_t lda, sycl::buffer<T,1> &tau, sycl::buffer<T,1> &scratchpad, std::int64_t scratchpad_size)
    }

.. container:: section

  .. rubric:: Input Parameters
      
queue
   The queue where the routine should be executed.

upper_lower
   Must be ``uplo::upper`` or ``uplo::lower``. Uses the same
   ``upper_lower`` as supplied to
   :ref:`onemkl_lapack_hetrd`.

n
   The order of the matrix :math:`Q` :math:`(0 \le n)`.

a
   The buffer ``a`` as returned by
   :ref:`onemkl_lapack_hetrd`. The
   second dimension of ``a`` must be at least :math:`\max(1, n)`.

lda
   The leading dimension of ``a`` :math:`(n \le \text{lda})`.

tau
   The buffer ``tau`` as returned by
   :ref:`onemkl_lapack_hetrd`. The
   dimension of ``tau`` must be at least :math:`\max(1, n-1)`.

scratchpad_size
   Size of scratchpad memory as a number of floating point elements of type ``T``.
   Size should not be less than the value returned by :ref:`onemkl_lapack_ungtr_scratchpad_size` function.

.. container:: section

  .. rubric:: Output Parameters

a
   Overwritten by the unitary matrix :math:`Q`.

scratchpad
   Buffer holding scratchpad memory to be used by routine for storing intermediate results.

ungtr (USM Version)
----------------------

.. container:: section

  .. rubric:: Syntax
         
.. code-block:: cpp

    namespace oneapi::mkl::lapack {
      sycl::event ungtr(sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n, T *a, std::int64_t lda, T *tau, T *scratchpad, std::int64_t scratchpad_size, const std::vector<sycl::event> &events = {})
    }

.. container:: section

  .. rubric:: Input Parameters

queue
   The queue where the routine should be executed.

upper_lower
   Must be ``uplo::upper`` or ``uplo::lower``. Uses the same
   ``upper_lower`` as supplied to
   :ref:`onemkl_lapack_hetrd`.

n
   The order of the matrix :math:`Q` :math:`(0 \le n)`.

a
   The pointer to ``a`` as returned by
   :ref:`onemkl_lapack_hetrd`. The
   second dimension of ``a`` must be at least :math:`\max(1, n)`.

lda
   The leading dimension of ``a`` :math:`(n \le \text{lda})`.

tau
   The pointer to ``tau`` as returned by
   :ref:`onemkl_lapack_hetrd`. The
   dimension of ``tau`` must be at least :math:`\max(1, n-1)`.

scratchpad_size
   Size of scratchpad memory as a number of floating point elements of type ``T``.
   Size should not be less than the value returned by :ref:`onemkl_lapack_ungtr_scratchpad_size` function.

events
   List of events to wait for before starting computation. Defaults to empty list.

.. container:: section

  .. rubric:: Output Parameters

a
   Overwritten by the unitary matrix :math:`Q`.

scratchpad
   Pointer to scratchpad memory to be used by routine for storing intermediate results.

.. container:: section

  .. rubric:: Return Values

Output event to wait on to ensure computation is complete.

**Parent topic:** :ref:`onemkl_lapack-singular-value-eigenvalue-routines`



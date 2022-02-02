.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemkl_lapack_orgtr:

orgtr
=====

Generates the real orthogonal matrix :math:`Q` determined by
:ref:`onemkl_lapack_sytrd`.

.. container:: section

  .. rubric:: Description
      
``orgtr`` supports the following precisions.

    .. list-table:: 
       :header-rows: 1

       * -  T 
       * -  ``float`` 
       * -  ``double`` 

The routine explicitly generates the :math:`n \times n` orthogonal matrix
:math:`Q` formed by :ref:`onemkl_lapack_sytrd` when
reducing a real symmetric matrix :math:`A` to tridiagonal form:
:math:`A = QTQ^T`. Use this routine after a call to
:ref:`onemkl_lapack_sytrd`.

orgtr (Buffer Version)
----------------------

.. container:: section

  .. rubric:: Syntax
         
.. code-block:: cpp

    namespace oneapi::mkl::lapack {
      void orgtr(sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n, sycl::buffer<T,1> &a, std::int64_t lda, sycl::buffer<T,1> &tau, sycl::buffer<T,1> &scratchpad, std::int64_t scratchpad_size)
    }

.. container:: section

  .. rubric:: Input Parameters
      
queue
   The queue where the routine should be executed.

upper_lower
   Must be ``uplo::upper`` or ``uplo::lower``. Uses the same
   ``upper_lower`` as supplied to :ref:`onemkl_lapack_sytrd`.

n
   The order of the matrix :math:`Q` :math:`(0 \le n)`.

a
   The buffer ``a`` as returned by :ref:`onemkl_lapack_sytrd`. The
   second dimension of ``a`` must be at least :math:`\max(1,n)`.

lda
   The leading dimension of ``a`` :math:`(n \le \text{lda})`.

tau
   The buffer ``tau`` as returned by :ref:`onemkl_lapack_sytrd`. The
   dimension of ``tau`` must be at least :math:`\max(1, n-1)`.

scratchpad_size
   Size of scratchpad memory as a number of floating point elements of type ``T``.
   Size should not be less than the value returned by :ref:`onemkl_lapack_orgtr_scratchpad_size` function.

.. container:: section

  .. rubric:: Output Parameters

a
   Overwritten by the orthogonal matrix :math:`Q`.

scratchpad
   Buffer holding scratchpad memory to be used by routine for storing intermediate results.

orgtr (USM Version)
----------------------

.. container:: section

  .. rubric:: Syntax

.. code-block:: cpp

    namespace oneapi::mkl::lapack {
      sycl::event orgtr(sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n, T *a, std::int64_t lda, T *tau, T *scratchpad, std::int64_t scratchpad_size, const std::vector<sycl::event> &events = {})
    }

.. container:: section

  .. rubric:: Input Parameters
      
queue
   The queue where the routine should be executed.

upper_lower
   Must be ``uplo::upper`` or ``uplo::lower``. Uses the same
   ``upper_lower`` as supplied
   to :ref:`onemkl_lapack_sytrd`.

n
   The order of the matrix :math:`Q` :math:`(0 \le n)`.

a
   The pointer to ``a`` as returned by
   :ref:`onemkl_lapack_sytrd`. The
   second dimension of ``a`` must be at least :math:`\max(1,n)`.

lda
   The leading dimension of ``a`` :math:`(n \le \text{lda})`.

tau
   The pointer to ``tau`` as returned by :ref:`onemkl_lapack_sytrd`. The
   dimension of ``tau`` must be at least :math:`\max(1, n-1)`.

scratchpad_size
   Size of scratchpad memory as a number of floating point elements of type ``T``.
   Size should not be less than the value returned by :ref:`onemkl_lapack_orgtr_scratchpad_size` function.

events
   List of events to wait for before starting computation. Defaults to empty list.

.. container:: section

  .. rubric:: Output Parameters
      
a
   Overwritten by the orthogonal matrix :math:`Q`.

scratchpad
   Pointer to scratchpad memory to be used by routine for storing intermediate results.

.. container:: section

  .. rubric:: Return Values
         
Output event to wait on to ensure computation is complete.

**Parent topic:** :ref:`onemkl_lapack-singular-value-eigenvalue-routines`


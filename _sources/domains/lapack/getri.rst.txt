.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemkl_lapack_getri:

getri
=====

Computes the inverse of an LU-factored general matrix determined by
:ref:`onemkl_lapack_getrf`.

.. container:: section

  .. rubric:: Description

``getri`` supports the following precisions.

     .. list-table:: 
        :header-rows: 1
  
        * -  T 
        * -  ``float`` 
        * -  ``double`` 
        * -  ``std::complex<float>`` 
        * -  ``std::complex<double>`` 

The routine computes the inverse :math:`A^{-1}` of a general matrix
:math:`A`. Before calling this routine, call :ref:`onemkl_lapack_getrf`
to factorize :math:`A`.

getri (BUFFER Version)
----------------------

.. container:: section

  .. rubric:: Syntax

.. code-block:: cpp

    namespace oneapi::mkl::lapack {
      void getri(sycl::queue &queue, std::int64_t n, sycl::buffer<T,1> &a, std::int64_t lda, sycl::buffer<std::int64_t,1> &ipiv, sycl::buffer<T,1> &scratchpad, std::int64_t scratchpad_size)
    }

.. container:: section

  .. rubric:: Input Parameters

queue
   The queue where the routine should be executed.

n
   The order of the matrix :math:`A` :math:`(0 \le n)`.

a
   The buffer ``a`` as returned by :ref:`onemkl_lapack_getrf`. Must
   be of size at least :math:`\text{lda} \cdot \max(1,n)`.

lda
   The leading dimension of ``a`` :math:`(n \le \text{lda})`.

ipiv
   The buffer as returned by :ref:`onemkl_lapack_getrf`. The
   dimension of ``ipiv`` must be at least :math:`\max(1, n)`.

scratchpad_size
   Size of scratchpad memory as a number of floating point elements of type ``T``.
   Size should not be less than the value returned by :ref:`onemkl_lapack_getri_scratchpad_size` function.

.. container:: section

  .. rubric:: Output Parameters

a
   Overwritten by the :math:`n \times n` matrix :math:`A`.

scratchpad
   Buffer holding scratchpad memory to be used by routine for storing intermediate results.

getri (USM Version)
----------------------

.. container:: section

  .. rubric:: Syntax

.. code-block:: cpp

    namespace oneapi::mkl::lapack {
      sycl::event getri(sycl::queue &queue, std::int64_t n, T *a, std::int64_t lda, std::int64_t *ipiv, T *scratchpad, std::int64_t scratchpad_size, const std::vector<sycl::event> &events = {})
    }

.. container:: section

  .. rubric:: Input Parameters

queue
   The queue where the routine should be executed.

n
   The order of the matrix :math:`A` :math:`(0 \le n)`.

a
   The array as returned by :ref:`onemkl_lapack_getrf`. Must
   be of size at least :math:`\text{lda} \cdot \max(1,n)`.

lda
   The leading dimension of ``a`` :math:`(n \le \text{lda})`.

ipiv
   The array as returned by :ref:`onemkl_lapack_getrf`. The
   dimension of ``ipiv`` must be at least :math:`\max(1, n)`.

scratchpad_size
   Size of scratchpad memory as a number of floating point elements of type ``T``.
   Size should not be less than the value returned by :ref:`onemkl_lapack_getri_scratchpad_size` function.

events
   List of events to wait for before starting computation. Defaults to empty list.

.. container:: section

  .. rubric:: Output Parameters

a
   Overwritten by the :math:`n \times n` matrix :math:`A`.

scratchpad
   Pointer to scratchpad memory to be used by routine for storing intermediate results.

.. container:: section

  .. rubric:: Return Values

Output event to wait on to ensure computation is complete.

**Parent topic:** :ref:`onemkl_lapack-linear-equation-routines`


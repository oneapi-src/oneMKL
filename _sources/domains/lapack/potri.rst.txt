.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemkl_lapack_potri:

potri
=====

Computes the inverse of a symmetric (Hermitian) positive-definite
matrix using the Cholesky factorization.

.. container:: section

  .. rubric:: Description

``potri`` supports the following precisions.

      .. list-table:: 
         :header-rows: 1

         * -  T 
         * -  ``float`` 
         * -  ``double`` 
         * -  ``std::complex<float>`` 
         * -  ``std::complex<double>`` 

The routine computes the inverse :math:`A^{-1}` of a symmetric positive
definite or, for complex flavors, Hermitian positive-definite matrix
:math:`A`. Before calling this routine, call :ref:`onemkl_lapack_potrf`
to factorize :math:`A`.

potri (Buffer Version)
----------------------

.. container:: section

  .. rubric:: Syntax

.. code-block:: cpp

    namespace oneapi::mkl::lapack {
      void potri(sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n, sycl::buffer<T,1> &a, std::int64_t lda, sycl::buffer<T,1> &scratchpad, std::int64_t scratchpad_size)
    }

.. container:: section

  .. rubric:: Input Parameters
      
queue
   The queue where the routine should be executed.

upper_lower
   Indicates how the input matrix :math:`A` has been    factored:

   If ``upper_lower = oneapi::mkl::uplo::upper``, the upper   triangle of :math:`A` is stored.

   If   ``upper_lower = oneapi::mkl::uplo::lower``, the lower triangle of :math:`A` is   stored.

n
   Specifies the order of the matrix    :math:`A` (:math:`0 \le n`).

a
   Contains the factorization of the matrix :math:`A`, as    returned by   :ref:`onemkl_lapack_potrf`.   The second dimension of ``a`` must be at least :math:`\max(1, n)`.

lda
   The leading dimension of ``a``.

scratchpad_size
   Size of scratchpad memory as a number of floating point elements of type ``T``.
   Size should not be less than the value returned by :ref:`onemkl_lapack_potri_scratchpad_size` function.

.. container:: section

  .. rubric:: Output Parameters
      
a
   Overwritten by the upper or lower triangle of the inverse    of :math:`A`. Specified by ``upper_lower``.

scratchpad
   Buffer holding scratchpad memory to be used by routine for storing intermediate results.

potri (USM Version)
----------------------

.. container:: section

  .. rubric:: Syntax
         
.. code-block:: cpp

    namespace oneapi::mkl::lapack {
      sycl::event potri(sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n, T *a, std::int64_t lda, T *scratchpad, std::int64_t scratchpad_size, const std::vector<sycl::event> &events = {})
    }

.. container:: section

  .. rubric:: Input Parameters
      
queue
   The queue where the routine should be executed.

upper_lower
   Indicates how the input matrix :math:`A` has been    factored:

   If ``upper_lower = oneapi::mkl::uplo::upper``, the upper   triangle of :math:`A` is stored.

   If   ``upper_lower = oneapi::mkl::uplo::lower``, the lower triangle of :math:`A` is   stored.

n
   Specifies the order of the matrix    :math:`A` (:math:`0 \le n`).

a
   Contains the factorization of the matrix :math:`A`, as    returned by   :ref:`onemkl_lapack_potrf`.   The second dimension of ``a`` must be at least :math:`\max(1, n)`.

lda
   The leading dimension of ``a``.

scratchpad_size
   Size of scratchpad memory as a number of floating point elements of type ``T``.
   Size should not be less than the value returned by :ref:`onemkl_lapack_potri_scratchpad_size` function.

events
   List of events to wait for before starting computation. Defaults to empty list.

.. container:: section

  .. rubric:: Output Parameters
      
a
   Overwritten by the upper or lower triangle of the inverse    of :math:`A`. Specified by ``upper_lower``.

scratchpad
   Pointer to scratchpad memory to be used by routine for storing intermediate results.

.. container:: section

  .. rubric:: Return Values
         
Output event to wait on to ensure computation is complete.

**Parent topic:** :ref:`onemkl_lapack-linear-equation-routines`



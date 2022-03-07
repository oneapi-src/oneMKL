.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemkl_lapack_hetrf_scratchpad_size:

hetrf_scratchpad_size
=====================

Computes size of scratchpad memory required for :ref:`onemkl_lapack_hetrf` function.

.. container:: section

  .. rubric:: Description
         
``hetrf_scratchpad_size`` supports the following precisions.

     .. list-table:: 
        :header-rows: 1
  
        * -  T 
        * -  ``std::complex<float>`` 
        * -  ``std::complex<double>`` 

Computes the number of elements of type ``T`` the scratchpad memory to be passed to :ref:`onemkl_lapack_hetrf` function should be able to hold.
Calls to this routine must specify the template parameter explicitly.

hetrf_scratchpad_size
---------------------

.. container:: section

  .. rubric:: Syntax

.. code-block:: cpp

    namespace oneapi::mkl::lapack {
      template <typename T>
      std::int64_t hetrf_scratchpad_size(sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n, std::int64_t lda) 
    }

.. container:: section

  .. rubric:: Input Parameters
         
queue
   Device queue where calculations by :ref:`onemkl_lapack_hetrf` function will be performed.

upper_lower
   Indicates whether the upper or lower triangular part of :math:`A` is
   stored and how :math:`A` is factored:

   If ``upper_lower=uplo::upper``, the buffer ``a`` stores the
   upper triangular part of the matrix :math:`A`, and :math:`A` is
   factored as :math:`UDU^H`.

   If ``upper_lower=uplo::lower``, the buffer ``a`` stores the
   lower triangular part of the matrix :math:`A`, and :math:`A` is
   factored as :math:`LDL^H`

n
   The order of the matrix :math:`A` (:math:`0 \le n`).

lda
   The leading dimension of ``a``.

.. container:: section

  .. rubric:: Return Value

The number of elements of type ``T`` the scratchpad memory to be passed to :ref:`onemkl_lapack_hetrf` function should be able to hold.

**Parent topic:** :ref:`onemkl_lapack-linear-equation-routines`


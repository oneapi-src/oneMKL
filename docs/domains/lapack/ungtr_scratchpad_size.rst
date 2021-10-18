.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemkl_lapack_ungtr_scratchpad_size:

ungtr_scratchpad_size
=====================

Computes size of scratchpad memory required for :ref:`onemkl_lapack_ungtr` function.

.. container:: section

  .. rubric:: Description

``ungtr_scratchpad_size`` supports the following precisions.

     .. list-table:: 
        :header-rows: 1

        * -  T 
        * -  ``std::complex<float>`` 
        * -  ``std::complex<double>`` 

Computes the number of elements of type ``T`` the scratchpad memory to be passed to :ref:`onemkl_lapack_ungtr` function should be able to hold.
Calls to this routine must specify the template parameter explicitly.

ungtr_scratchpad_size
---------------------

.. container:: section

  .. rubric:: Syntax
         
.. code-block:: cpp

    namespace oneapi::mkl::lapack {
      template <typename T>
      std::int64_t ungtr_scratchpad_size(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n, std::int64_t lda) 
    }

.. container:: section

  .. rubric:: Input Parameters
         
queue
   Device queue where calculations by :ref:`onemkl_lapack_ungtr` function will be performed.

upper_lower
   Must be ``uplo::upper`` or ``uplo::lower``. Uses the same
   ``upper_lower`` as supplied to
   :ref:`onemkl_lapack_hetrd`.

n
   The order of the matrix :math:`Q` :math:`(0 \le n)`.

lda
   The leading dimension of ``a`` :math:`(n \le lda)`.

.. container:: section

  .. rubric:: Throws

This routine shall throw the following exceptions if the associated condition is detected. An implementation may throw additional implementation-specific exception(s) in case of error conditions not covered here.

:ref:`oneapi::mkl::unimplemented<onemkl_exception_unimplemented>`

:ref:`oneapi::mkl::unsupported_device<onemkl_exception_unsupported_device>`

:ref:`oneapi::mkl::lapack::invalid_argument<onemkl_lapack_exception_invalid_argument>`

   Exception is thrown in case of incorrect supplied argument value.
   Position of wrong argument can be determined by `info()` method of exception object.

.. container:: section

  .. rubric:: Return Value

The number of elements of type T the scratchpad memory to be passed to :ref:`onemkl_lapack_ungtr` function should be able to hold.

**Parent topic:** :ref:`onemkl_lapack-singular-value-eigenvalue-routines`


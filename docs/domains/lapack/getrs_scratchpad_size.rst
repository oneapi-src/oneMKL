.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemkl_lapack_getrs_scratchpad_size:

getrs_scratchpad_size
=====================

Computes size of scratchpad memory required for :ref:`onemkl_lapack_getrs` function.

.. container:: section

  .. rubric:: Description

``getrs_scratchpad_size`` supports the following precisions.

    .. list-table:: 
       :header-rows: 1

       * -  T 
       * -  ``float`` 
       * -  ``double`` 
       * -  ``std::complex<float>`` 
       * -  ``std::complex<double>`` 
      
Computes the number of elements of type ``T`` the scratchpad memory to be passed to :ref:`onemkl_lapack_getrs` function should be able to hold.
Calls to this routine must specify the template parameter
explicitly.

getrs_scratchpad_size
---------------------

.. container:: section

  .. rubric:: Syntax
      
.. code-block:: cpp

    namespace oneapi::mkl::lapack {
      template <typename T>
      std::int64_t getrs_scratchpad_size(cl::sycl::queue &queue, oneapi::mkl::transpose trans, std::int64_t n, std::int64_t nrhs, std::int64_t lda, std::int64_t ldb) 
    }

.. container:: section

  .. rubric:: Input Parameters

queue
   Device queue where calculations by :ref:`onemkl_lapack_getrs` function will be performed.

trans
   Indicates the form of the equations:

   If ``trans=oneapi::mkl::transpose::nontrans``, then :math:`AX = B` is solved
   for :math:`X`.

   If ``trans=oneapi::mkl::transpose::trans``, then :math:`A^TX = B` is solved
   for :math:`X`.

   If ``trans=oneapi::mkl::transpose::conjtrans``, then :math:`A^HX = B` is
   solved for :math:`X`.

n
   The order of the matrix :math:`A` :math:`(0 \le n)` and the number of rows in matrix
   :math:`B(0 \le n)`.

nrhs
   The number of right-hand sides (:math:`0 \le \text{nrhs}`).

lda
   The leading dimension of ``a``.

ldb
   The leading dimension of ``b``.

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

The number of elements of type ``T`` the scratchpad memory to be passed to :ref:`onemkl_lapack_getrs` function should be able to hold.

**Parent topic:** :ref:`onemkl_lapack-linear-equation-routines` 



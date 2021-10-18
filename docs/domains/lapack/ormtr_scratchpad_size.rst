.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemkl_lapack_ormtr_scratchpad_size:

ormtr_scratchpad_size
=====================

Computes size of scratchpad memory required for :ref:`onemkl_lapack_ormtr` function.

.. container:: section

  .. rubric:: Description

``ormtr_scratchpad_size`` supports the following precisions.

     .. list-table:: 
        :header-rows: 1

        * -  T 
        * -  ``float`` 
        * -  ``double`` 

Computes the number of elements of type ``T`` the scratchpad memory to be passed to :ref:`onemkl_lapack_ormtr` function should be able to hold.
Calls to this routine must specify the template parameter
explicitly.

ormtr_scratchpad_size
---------------------

.. container:: section

  .. rubric:: Syntax
         
.. code-block:: cpp

    namespace oneapi::mkl::lapack {
      template <typename T>
      std::int64_t ormtr_scratchpad_size(cl::sycl::queue &queue, oneapi::mkl::side side, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n, std::int64_t lda, std::int64_t ldc) 
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
   Device queue where calculations by :ref:`onemkl_lapack_ormtr` function will be performed.

side
   Must be either ``side::left`` or ``side::right``.

   If ``side = side::left``, :math:`Q` or :math:`Q^{T}` is
   applied to :math:`C` from the left.

   If ``side = side::right``, :math:`Q` or :math:`Q^{T}` is
   applied to :math:`C` from the right.

upper_lower
   Must be either ``uplo::upper`` or ``uplo::lower``. Uses the
   same ``upper_lower`` as supplied to
   :ref:`onemkl_lapack_sytrd`.

trans
   Must be either ``transpose::nontrans`` or ``transpose::trans``.

   If ``trans = transpose::nontrans``, the routine multiplies
   :math:`C` by :math:`Q`.

   If ``trans = transpose::trans``, the routine multiplies :math:`C`
   by :math:`Q^{T}`.

m
   The number of rows in the matrix :math:`C` :math:`(m \ge 0)`.

n
   The number of rows in the matrix :math:`C` :math:`(n \ge 0)`.

lda
   The leading dimension of ``a`` :math:`(\max(1, r) \le \text{lda})`.

ldc
   The leading dimension of ``c`` :math:`(\max(1, n) \le \text{ldc})`.

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

The number of elements of type ``T`` the scratchpad memory to be passed to :ref:`onemkl_lapack_ormtr` function should be able to hold.

**Parent topic:** :ref:`onemkl_lapack-singular-value-eigenvalue-routines`



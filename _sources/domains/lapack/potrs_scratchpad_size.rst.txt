.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemkl_lapack_potrs_scratchpad_size:

potrs_scratchpad_size
=====================

Computes size of scratchpad memory required for :ref:`onemkl_lapack_potrs` function.

.. container:: section

  .. rubric:: Description
         
``potrs_scratchpad_size`` supports the following precisions.

    .. list-table:: 
       :header-rows: 1
    
       * -  T 
       * -  ``float`` 
       * -  ``double`` 
       * -  ``std::complex<float>`` 
       * -  ``std::complex<double>`` 

Computes the number of elements of type ``T`` the scratchpad memory to be passed to :ref:`onemkl_lapack_potrs` function should be able to hold.
Calls to this routine must specify the template parameter explicitly.

potrs_scratchpad_size
---------------------

.. container:: section

  .. rubric:: Syntax

.. code-block:: cpp

    namespace oneapi::mkl::lapack {
      template <typename T>
      std::int64_t potrs_scratchpad_size(sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n, std::int64_t nrhs, std::int64_t lda, std::int64_t ldb) 
    }

.. container:: section

  .. rubric:: Input Parameters

queue
   Device queue where calculations by :ref:`onemkl_lapack_potrs` function will be performed.

upper_lower
   Indicates how the input matrix has been factored:

   If ``upper_lower = oneapi::mkl::uplo::upper``, the upper triangle   :math:`U` of :math:`A` is stored, where :math:`A = U^{T}U`   for real data, :math:`A = U^{H}U` for complex data.

   If ``upper_lower = oneapi::mkl::uplo::lower``, the lower triangle   :math:`L` of :math:`A` is stored, where :math:`A = LL^{T}`   for real data, :math:`A = LL^{H}` for complex   data.

n
   The order of matrix :math:`A` (:math:`0 \le n`).

nrhs
   The number of right-hand sides (:math:`0 \le nrhs`).

lda
   The leading dimension of ``a``.

ldb
   The leading dimension of ``b``.

.. container:: section

  .. rubric:: Return Value
         
The number of elements of type ``T`` the scratchpad memory to be passed to :ref:`onemkl_lapack_potrs` function should be able to hold.

**Parent topic:** :ref:`onemkl_lapack-linear-equation-routines`


.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemkl_lapack_trtrs_scratchpad_size:

trtrs_scratchpad_size
=====================

Computes size of scratchpad memory required for :ref:`onemkl_lapack_trtrs` function.

.. container:: section

  .. rubric:: Description
         
``trtrs_scratchpad_size`` supports the following precisions.

    .. list-table:: 
       :header-rows: 1

       * -  T 
       * -  ``float`` 
       * -  ``double`` 
       * -  ``std::complex<float>`` 
       * -  ``std::complex<double>`` 

Computes the number of elements of type ``T`` the scratchpad memory to be passed to :ref:`onemkl_lapack_trtrs` function should be able to hold.
Calls to this routine must specify the template parameter explicitly.

trtrs_scratchpad_size
---------------------

.. container:: section

  .. rubric:: Syntax
         
.. code-block:: cpp

    namespace oneapi::mkl::lapack {
      template <typename T>
      std::int64_t trtrs_scratchpad_size(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans, oneapi::mkl::diag diag, std::int64_t n, std::int64_t nrhs, std::int64_t lda, std::int64_t ldb) 
    }

.. container:: section

  .. rubric:: Input Parameters
         
queue
   Device queue where calculations by :ref:`onemkl_lapack_trtrs` function will be performed.

upper_lower
   Indicates whether :math:`A` is upper or lower    triangular:

   If upper_lower = ``uplo::upper``, then   :math:`A` is upper triangular.

   If upper_lower =   ``uplo::lower``, then :math:`A` is lower triangular.

trans
   Indicates the form of the equations:

   If ``trans=oneapi::mkl::transpose::nontrans``, then :math:`AX = B` is solved
   for :math:`X`.

   If ``trans=oneapi::mkl::transpose::trans``, then :math:`A^TX = B` is solved
   for :math:`X`.

   If ``trans=oneapi::mkl::transpose::conjtrans``, then :math:`A^HX = B` is
   solved for :math:`X`.

diag
   If diag = ``oneapi::mkl::diag::nonunit``, then :math:`A` is not a    unit triangular matrix.

   If unit_diag = ``diag::unit``,   then :math:`A` is unit triangular: diagonal elements of :math:`A` are assumed   to be 1 and not referenced in the array ``a``.

n
   The order of :math:`A`; the number of rows in :math:`B`;    :math:`n \ge 0`.

nrhs
   The number of right-hand sides (:math:`0 \le \text{nrhs}`).

lda
   The leading dimension of ``a``; :math:`\text{lda} \ge \max(1, n)`.

ldb
   The leading dimension of ``b``; :math:`\text{ldb} \ge \max(1, n)`.

.. container:: section

  .. rubric:: Return Value

The number of elements of type ``T`` the scratchpad memory to be passed to :ref:`onemkl_lapack_trtrs` function should be able to hold.

**Parent topic:** :ref:`onemkl_lapack-linear-equation-routines`


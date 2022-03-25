.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemkl_lapack_orgbr_scratchpad_size:

orgbr_scratchpad_size
=====================

Computes size of scratchpad memory required for :ref:`onemkl_lapack_orgbr` function.

``orgbr_scratchpad_size`` supports the following precisions.

    .. list-table:: 
       :header-rows: 1

       * -  T 
       * -  ``float`` 
       * -  ``double`` 

.. container:: section

  .. rubric:: Description

Computes the number of elements of type ``T`` the scratchpad memory to be passed to :ref:`onemkl_lapack_orgbr` function should be able to hold.
Calls to this routine must specify the template parameter explicitly.

orgbr_scratchpad_size
---------------------

.. container:: section

  .. rubric:: Syntax
         
.. code-block:: cpp

    namespace oneapi::mkl::lapack {
      template <typename T>
      std::int64_t orgbr_scratchpad_size(sycl::queue &queue, oneapi::mkl::generate gen, std::int64_t m, std::int64_t n, std::int64_t k, std::int64_t lda, std::int64_t &scratchpad_size) 
    }

.. container:: section

  .. rubric:: Input Parameters
         
queue
   Device queue where calculations by :ref:`onemkl_lapack_orgbr` function will be performed.

gen
   Must be ``generate::q`` or ``generate::p``.

   If ``gen = generate::q``, the routine generates the matrix
   :math:`Q`.

   If ``gen = generate::p``, the routine generates the matrix
   :math:`P^{T}`.

m
   The number of rows in the matrix :math:`Q` or :math:`P^{T}` to be
   returned :math:`(0 \le m)`.

   If ``gen = generate::q``, :math:`m \le  n \le \min(m, k)`.

   If ``gen = generate::p``, :math:`n \le m \le \min(n, k)`.

n
   The number of rows in the matrix :math:`Q` or :math:`P^{T}` to be
   returned :math:`(0 \le n)`. See ``m`` for constraints.

k
   If ``gen = generate::q``, the number of columns in the original
   :math:`m \times k` matrix returned by
   :ref:`onemkl_lapack_gebrd`.

   If ``gen = generate::p``, the number of rows in the original
   :math:`k \times n` matrix returned by
   :ref:`onemkl_lapack_gebrd`.

lda
   The leading dimension of ``a``.

.. container:: section

  .. rubric:: Return Value
         
The number of elements of type ``T`` the scratchpad memory to be passed to :ref:`onemkl_lapack_orgbr` function should be able to hold.

**Parent topic:** :ref:`onemkl_lapack-singular-value-eigenvalue-routines` 



.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemkl_lapack_ungbr:

ungbr
=====

Generates the complex unitary matrix :math:`Q` or :math:`P^{t}` determined by
:ref:`onemkl_lapack_gebrd`.

.. container:: section

  .. rubric:: Description
     
``ungbr`` supports the following precisions.

      .. list-table:: 
         :header-rows: 1

         * -  T 
         * -  ``std::complex<float>`` 
         * -  ``std::complex<double>`` 

The routine generates the whole or part of the unitary matrices :math:`Q`
and :math:`P^{H}` formed by the routines
:ref:`onemkl_lapack_gebrd`.
All valid combinations of arguments are described in *Input Parameters*; in
most cases you need the following:

To compute the whole :math:`m \times m` matrix :math:`Q`, use:

::

   oneapi::mkl::lapack::ungbr(queue, generate::q, m, m, n, a, ...)

(note that the buffer ``a`` must have at least :math:`m` columns).

To form the :math:`n` leading columns of :math:`Q` if :math:`m > n`, use:

::

   oneapi::mkl::lapack::ungbr(queue, generate::q, m, n, n, a, ...)

To compute the whole :math:`n \times n` matrix :math:`P^{T}`, use:

::

   oneapi::mkl::lapack::ungbr(queue, generate::p, n, n, m, a, ...)

(note that the array ``a`` must have at least :math:`n` rows).

To form the :math:`m` leading rows of :math:`P^{T}` if :math:`m < n`, use:

::

   oneapi::mkl::lapack::ungbr(queue, generate::p, m, n, m, a, ...)

ungbr (Buffer Version)
----------------------

.. container:: section

  .. rubric:: Syntax

.. code-block:: cpp

    namespace oneapi::mkl::lapack {
      void ungbr(sycl::queue &queue, oneapi::mkl::generate gen, std::int64_t m, std::int64_t n, std::int64_t k, sycl::buffer<T,1> &a, std::int64_t lda, sycl::buffer<T,1> &tau, sycl::buffer<T,1> &scratchpad, std::int64_t scratchpad_size)
    }

.. container:: section

  .. rubric:: Input Parameters

queue
   The queue where the routine should be executed.

gen
   Must be ``generate::q`` or ``generate::p``.

   If ``gen = generate::q``, the routine generates the matrix :math:`Q`.

   If ``gen = generate::p``, the routine generates the matrix
   :math:`P^{T}`.

m
   The number of rows in the matrix :math:`Q` or :math:`P^{T}` to be
   returned :math:`(0 \le m)`.

   If ``gen = generate::q``, :math:`m \ge n \ge \min(m, k)`.

   If ``gen = generate::p``, :math:`n \ge m \ge \min(n, k)`.

n
   The number of columns in the matrix :math:`Q` or :math:`P^{T}` to be
   returned :math:`(0 \le n)`. See ``m`` for constraints.

k
   If ``gen = generate::q``, the number of columns in the original
   :math:`m \times k` matrix returned by
   :ref:`onemkl_lapack_gebrd`.

   If ``gen = generate::p``, the number of rows in the original
   :math:`k \times n` matrix returned by
   :ref:`onemkl_lapack_gebrd`.

a
   The buffer ``a`` as returned by
   :ref:`onemkl_lapack_gebrd`.

lda
   The leading dimension of ``a``.

tau
   For ``gen = generate::q``, the array ``tauq`` as returned by :ref:`onemkl_lapack_gebrd`.
   For ``gen = generate::p``, the array ``taup`` as returned by :ref:`onemkl_lapack_gebrd`.

   The dimension of ``tau`` must be at least :math:`\max(1, \min(m, k))` for
   ``gen = generate::q``, or :math:`\max(1, \min(m, k))` for
   ``gen = generate::p``.

scratchpad_size
   Size of scratchpad memory as a number of floating point elements of type :math:`T`.
   Size should not be less than the value returned by :ref:`onemkl_lapack_ungbr_scratchpad_size` function.

.. container:: section

  .. rubric:: Output Parameters

a
   Overwritten by :math:`n` leading columns of the :math:`m \times m` unitary matrix
   :math:`Q` or :math:`P^{T}`, (or the leading rows or columns thereof)
   as specified by ``gen``, ``m``, and ``n``.

scratchpad
   Buffer holding scratchpad memory to be used by routine for storing intermediate results.

ungbr (USM Version)
----------------------

.. container:: section

  .. rubric:: Syntax

.. code-block:: cpp

    namespace oneapi::mkl::lapack {
      sycl::event ungbr(sycl::queue &queue, oneapi::mkl::generate gen, std::int64_t m, std::int64_t n, std::int64_t k, T *a, std::int64_t lda, T *tau, T *scratchpad, std::int64_t scratchpad_size, const sycl::vector_class<sycl::event> &events = {})
    }

.. container:: section

  .. rubric:: Input Parameters
      
queue
   The queue where the routine should be executed.

gen
   Must be ``generate::q`` or ``generate::p``.

   If ``gen = generate::q``, the routine generates the matrix :math:`Q`.

   If ``gen = generate::p``, the routine generates the matrix
   :math:`P^{T}`.

m
   The number of rows in the matrix :math:`Q` or :math:`P^{T}` to be
   returned :math:`(0 \ge m)`.

   If ``gen = generate::q``, :math:`m \ge n \ge \min(m, k)`.

   If ``gen = generate::p``, :math:`n \ge m \ge \min(n, k)`.

n
   The number of columns in the matrix :math:`Q` or :math:`P^{T}` to be
   returned :math:`(0 \le n)`. See ``m`` for constraints.

k
   If ``gen = generate::q``, the number of columns in the original
   :math:`m \times k` matrix returned by
   :ref:`onemkl_lapack_gebrd`.

   If ``gen = generate::p``, the number of rows in the original
   :math:`k \times n` matrix returned by
   :ref:`onemkl_lapack_gebrd`.

a
   The pointer to ``a`` as returned by
   :ref:`onemkl_lapack_gebrd`.

lda
   The leading dimension of ``a``.

tau
   For ``gen = generate::q``, the array ``tauq`` as returned by :ref:`onemkl_lapack_gebrd`.
   For ``gen = generate::p``, the array ``taup`` as returned by :ref:`onemkl_lapack_gebrd`.

   The dimension of ``tau`` must be at least :math:`\max(1, \min(m, k))` for
   ``gen = generate::q``, or :math:`\max(1, \min(m, k))` for
   ``gen = generate::p``.

scratchpad_size
   Size of scratchpad memory as a number of floating point elements of type :math:`T`.
   Size should not be less than the value returned by :ref:`onemkl_lapack_ungbr_scratchpad_size` function.

events
   List of events to wait for before starting computation. Defaults to empty list.

.. container:: section

  .. rubric:: Output Parameters
      
a
   Overwritten by :math:`n` leading columns of the :math:`m \times m` unitary matrix
   :math:`Q` or :math:`P^{T}`, (or the leading rows or columns thereof)
   as specified by ``gen``, ``m``, and ``n``.

scratchpad
   Pointer to scratchpad memory to be used by routine for storing intermediate results.

.. container:: section

  .. rubric:: Return Values
         
Output event to wait on to ensure computation is complete.

**Parent topic:** :ref:`onemkl_lapack-singular-value-eigenvalue-routines`



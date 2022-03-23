.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemkl_lapack_orgqr:

orgqr
=====

Generates the real orthogonal matrix :math:`Q` of the QR factorization formed
by :ref:`onemkl_lapack_geqrf`.

.. container:: section

  .. rubric:: Description

``orgqr`` supports the following precisions.

    .. list-table:: 
       :header-rows: 1

       * -  T 
       * -  ``float`` 
       * -  ``double`` 

The routine generates the whole or part of :math:`m \times m` orthogonal
matrix :math:`Q` of the QR factorization formed by the routine
:ref:`onemkl_lapack_geqrf`.

Usually :math:`Q` is determined from the QR factorization of an ``m``
by ``p`` matrix :math:`A` with :math:`m \ge p`. To compute the whole matrix
:math:`Q`, use:

::

   oneapi::mkl::lapack::orgqr(queue, m, m, p, a, lda, tau, scratchpad, scratchpad_size)

To compute the leading :math:`p` columns of :math:`Q` (which form an
orthonormal basis in the space spanned by the columns of :math:`A`):

::

   oneapi::mkl::lapack::orgqr(queue, m, p, p, a, lda, tau, scratchpad, scratchpad_size)

To compute the matrix :math:`Q^{k}` of the QR factorization of
leading :math:`k` columns of the matrix :math:`A`:

::

   oneapi::mkl::lapack::orgqr(queue, m, m, k, a, lda, tau, scratchpad, scratchpad_size)

To compute the leading :math:`k` columns of :math:`Q^{k}` (which form
an orthonormal basis in the space spanned by leading :math:`k` columns of
the matrix :math:`A`):

::

   oneapi::mkl::lapack::orgqr(queue, m, k, k, a, lda, tau, scratchpad, scratchpad_size)

orgqr (Buffer Version)
----------------------

.. container:: section

  .. rubric:: Syntax

.. code-block:: cpp

    namespace oneapi::mkl::lapack {
      void orgqr(sycl::queue &queue, std::int64_t m, std::int64_t n, std::int64_t k, sycl::buffer<T,1> &a, std::int64_t lda, sycl::buffer<T,1> &tau, sycl::buffer<T,1> &scratchpad, std::int64_t scratchpad_size)
    }

.. container:: section

  .. rubric:: Input Parameters
      
queue
   The queue where the routine should be executed.

m
   The number of rows in the matrix :math:`A` (:math:`0 \le m`).

n
   The number of columns in the matrix :math:`A` (:math:`0 \le n`).

k
   The number of elementary reflectors whose product defines the
   matrix :math:`Q` (:math:`0 \le k \le n`).

a
   The buffer ``a`` as returned by
   :ref:`onemkl_lapack_geqrf`.

lda
   The leading dimension of ``a`` (:math:`\text{lda} \le m`).

tau
   The buffer ``tau`` as returned by
   :ref:`onemkl_lapack_geqrf`.

scratchpad_size
   Size of scratchpad memory as a number of floating point elements of type ``T``.
   Size should not be less than the value returned by :ref:`onemkl_lapack_orgqr_scratchpad_size` function.

.. container:: section

  .. rubric:: Output Parameters

a
   Overwritten by :math:`n` leading columns of the :math:`m \times m` orthogonal matrix
   :math:`Q`.

scratchpad
   Buffer holding scratchpad memory to be used by routine for storing intermediate results.

orgqr (USM Version)
----------------------

.. container:: section

  .. rubric:: Syntax
         
.. code-block:: cpp

    namespace oneapi::mkl::lapack {
      sycl::event orgqr(sycl::queue &queue, std::int64_t m, std::int64_t n, std::int64_t k, T *a, std::int64_t lda, T *tau, T *scratchpad, std::int64_t scratchpad_size, const std::vector<sycl::event> &events = {})
    }

.. container:: section

  .. rubric:: Input Parameters
      
queue
   The queue where the routine should be executed.

m
   The number of rows in the matrix :math:`A` (:math:`0 \le m`).

n
   The number of columns in the matrix :math:`A` (:math:`0 \le n`).

k
   The number of elementary reflectors whose product defines the
   matrix :math:`Q` (:math:`0 \le k \le n`).

a
   The pointer to ``a`` as returned by
   :ref:`onemkl_lapack_geqrf`.

lda
   The leading dimension of ``a`` (:math:`\text{lda} \le m`).

tau
   The pointer to ``tau`` as returned by
   :ref:`onemkl_lapack_geqrf`.

scratchpad_size
   Size of scratchpad memory as a number of floating point elements of type ``T``.
   Size should not be less than the value returned by :ref:`onemkl_lapack_orgqr_scratchpad_size` function.

events
   List of events to wait for before starting computation. Defaults to empty list.

.. container:: section

  .. rubric:: Output Parameters

a
   Overwritten by :math:`n` leading columns of the :math:`m \times m` orthogonal matrix
   :math:`Q`.

scratchpad
   Pointer to scratchpad memory to be used by routine for storing intermediate results.

.. container:: section

  .. rubric:: Return Values
         
Output event to wait on to ensure computation is complete.

**Parent topic:** :ref:`onemkl_lapack-linear-equation-routines`



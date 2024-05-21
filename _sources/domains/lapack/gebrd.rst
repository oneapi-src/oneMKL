.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemkl_lapack_gebrd:

gebrd
=====

Reduces a general matrix to bidiagonal form.

.. container:: section

    .. rubric:: Description

``gebrd`` supports the following precisions.

     .. list-table:: 
        :header-rows: 1

        * -  T 
        * -  ``float`` 
        * -  ``double`` 
        * -  ``std::complex<float>`` 
        * -  ``std::complex<double>`` 

The routine reduces a general :math:`m \times n` matrix :math:`A` to a 
bidiagonal matrix :math:`B` by an orthogonal (unitary) transformation.


If :math:`m \ge n`, the reduction is given by :math:`A=QBP^H=\begin{pmatrix}B_1\\0\end{pmatrix}P^H=Q_1B_1P_H`

where :math:`B_{1}` is an :math:`n \times n` upper diagonal matrix,
:math:`Q` and :math:`P` are orthogonal or, for a complex :math:`A`, unitary
matrices; :math:`Q_{1}` consists of the first :math:`n` columns of
:math:`Q`.

If :math:`m < n`, the reduction is given by

:math:`A = QBP^H = Q\begin{pmatrix}B_1\\0\end{pmatrix}P^H = Q_1B_1P_1^H`,

where :math:`B_{1}` is an :math:`m \times m` lower diagonal matrix,
:math:`Q` and :math:`P` are orthogonal or, for a complex :math:`A`, unitary
matrices; :math:`P_{1}` consists of the first :math:`m` columns of
:math:`P`.

The routine does not form the matrices :math:`Q` and :math:`P` explicitly,
but represents them as products of elementary reflectors. Routines
are provided to work with the matrices :math:`Q` and :math:`P` in this
representation:

If the matrix :math:`A` is real,

-  to compute :math:`Q` and :math:`P` explicitly, call
   :ref:`onemkl_lapack_orgbr`.

If the matrix :math:`A` is complex,

-  to compute :math:`Q` and :math:`P` explicitly, call
   :ref:`onemkl_lapack_ungbr`

gebrd (Buffer Version)
----------------------

.. container:: section

  .. rubric:: Syntax

.. code-block:: cpp

    namespace oneapi::mkl::lapack {
      void gebrd(sycl::queue &queue, std::int64_t m, std::int64_t n, sycl::buffer<T,1> &a, std::int64_t lda, sycl::buffer<realT,1> &d, sycl::buffer<realT,1> &e, sycl::buffer<T,1> &tauq, sycl::buffer<T,1> &taup, sycl::buffer<T,1> &scratchpad, std::int64_t scratchpad_size)
    }

.. container:: section

  .. rubric:: Input Parameters

queue
   The queue where the routine should be executed.

m
   The number of rows in the matrix :math:`A` (:math:`0 \le m`).

n
   The number of columns in the matrix :math:`A` (:math:`0 \le n`).

a
   The buffer :math:`a`, size (``lda,*``). The buffer ``a`` contains the
   matrix :math:`A`. The second dimension of ``a`` must be at least
   :math:`\max(1, m)`.

lda
   The leading dimension of :math:`a`.

scratchpad_size
   Size of scratchpad memory as a number of floating point elements of type ``T``.
   Size should not be less than the value returned by :ref:`onemkl_lapack_gebrd_scratchpad_size` function.

.. container:: section

    .. rubric:: Output Parameters

a
   If :math:`m \ge n`, the diagonal and first super-diagonal of a are
   overwritten by the upper bidiagonal matrix :math:`B`. The elements
   below the diagonal, with the buffer tauq, represent the orthogonal
   matrix :math:`Q` as a product of elementary reflectors, and the
   elements above the first superdiagonal, with the buffer ``taup``,
   represent the orthogonal matrix :math:`P` as a product of elementary
   reflectors.

   If :math:`m<n`, the diagonal and first sub-diagonal of a are
   overwritten by the lower bidiagonal matrix :math:`B`. The elements
   below the first subdiagonal, with the buffer tauq, represent the
   orthogonal matrix :math:`Q` as a product of elementary reflectors, and
   the elements above the diagonal, with the buffer ``taup``, represent
   the orthogonal matrix :math:`P` as a product of elementary reflectors.

d
   Buffer, size at least :math:`\max(1, \min(m,n))`. Contains the diagonal
   elements of :math:`B`.

e
   Buffer, size at least :math:`\max(1, \min(m,n) - 1)`. Contains the
   off-diagonal elements of :math:`B`.

tauq
   Buffer, size at least :math:`\max(1, \min(m, n))`. The scalar factors of
   the elementary reflectors which represent the orthogonal or
   unitary matrix :math:`Q`.

taup
   Buffer, size at least :math:`\max(1, \min(m, n))`. The scalar factors of
   the elementary reflectors which represent the orthogonal or
   unitary matrix :math:`P`.

scratchpad
   Buffer holding scratchpad memory to be used by routine for storing intermediate results.

gebrd (USM Version)
-------------------

.. container:: section

  .. rubric:: Syntax

.. code-block:: cpp

    namespace oneapi::mkl::lapack {
      sycl::event gebrd(sycl::queue &queue, std::int64_t m, std::int64_t n, T *a, std::int64_t lda, RealT *d, RealT *e, T *tauq, T *taup, T *scratchpad, std::int64_t scratchpad_size, const std::vector<sycl::event> &events = {})
    }

.. container:: section

    .. rubric:: Input Parameters

queue
   The queue where the routine should be executed.

m
   The number of rows in the matrix :math:`A` (:math:`0 \le m`).

n
   The number of columns in the matrix :math:`A` (:math:`0 \le n`).

a
   Pointer to matrix :math:`A`. The second dimension of ``a`` must be at least
   :math:`\max(1, m)`.

lda
   The leading dimension of ``a``.

scratchpad_size
   Size of scratchpad memory as a number of floating point elements of type T.
   Size should not be less than the value returned by :ref:`onemkl_lapack_gebrd_scratchpad_size` function.

events
   List of events to wait for before starting computation. Defaults to empty list.

.. container:: section

    .. rubric:: Output Parameters

a
   If :math:`m \ge n`, the diagonal and first super-diagonal of a are
   overwritten by the upper bidiagonal matrix :math:`B`. The elements
   below the diagonal, with the array tauq, represent the orthogonal
   matrix :math:`Q` as a product of elementary reflectors, and the
   elements above the first superdiagonal, with the array ``taup``,
   represent the orthogonal matrix :math:`P` as a product of elementary
   reflectors.

   If :math:`m<n`, the diagonal and first sub-diagonal of a are
   overwritten by the lower bidiagonal matrix :math:`B`. The elements
   below the first subdiagonal, with the array tauq, represent the
   orthogonal matrix :math:`Q` as a product of elementary reflectors, and
   the elements above the diagonal, with the array ``taup``, represent
   the orthogonal matrix :math:`P` as a product of elementary reflectors.

d
   Pointer to memory of size at least :math:`\max(1, \min(m,n))`. Contains the diagonal
   elements of :math:`B`.

e
   Pointer to memory of size at least :math:`\max(1, \min(m,n) - 1)`. Contains the
   off-diagonal elements of :math:`B`.

tauq
   Pointer to memory of size at least :math:`\max(1, \min(m, n))`. The scalar factors of
   the elementary reflectors which represent the orthogonal or
   unitary matrix :math:`Q`.

taup
   Pointer to memory of size at least :math:`\max(1, \min(m, n))`. The scalar factors of
   the elementary reflectors which represent the orthogonal or
   unitary matrix :math:`P`.

scratchpad
   Pointer to scratchpad memory to be used by routine for storing intermediate results.

.. container:: section

    .. rubric:: Return Values

Output event to wait on to ensure computation is complete.

**Parent topic:** :ref:`onemkl_lapack-singular-value-eigenvalue-routines`



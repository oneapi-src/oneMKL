.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemkl_lapack_gerqf:

gerqf
=====

Computes the RQ factorization of a general :math:`m \times n` matrix.

.. container:: section

  .. rubric:: Description
      
``gerqf`` supports the following precisions.

     .. list-table:: 
        :header-rows: 1

        * -  T 
        * -  ``float`` 
        * -  ``double`` 
        * -  ``std::complex<float>`` 
        * -  ``std::complex<double>``

The routine forms the RQ factorization of a general :math:`m \times n` matrix :math:`A`. No pivoting is performed.
The routine does not form the matrix :math:`Q` explicitly. Instead, :math:`Q` is represented as a product of :math:`\min(m, n)` elementary reflectors. Routines are provided to work with :math:`Q` in this representation

gerqf (Buffer Version)
----------------------

.. container:: section

  .. rubric:: Syntax

.. code-block:: cpp

    namespace oneapi::mkl::lapack {
      void gerqf(sycl::queue &queue, std::int64_t m, std::int64_t n, sycl::buffer<T> &a, std::int64_t lda, sycl::buffer<T> &tau, sycl::buffer<T> &scratchpad, std::int64_t scratchpad_size)
    }

.. container:: section

  .. rubric:: Input Parameters

queue
   Device queue where calculations will be performed.
   
m
   The number of rows in the matrix :math:`A` (:math:`0 \le m`).
   
n
   The number of columns in the matrix :math:`A` (:math:`0 \le n`).
   
a
   Buffer holding input matrix :math:`A`. The second dimension of ``a`` must be at least :math:`\max(1, n)`.
   
lda
   The leading dimension of ``a``, at least :math:`\max(1, m)`.
      
scratchpad
   Buffer holding scratchpad memory to be used by the routine for storing intermediate results.
   
scratchpad_size
   Size of scratchpad memory as a number of floating point elements of type ``T``. Size should not be less than the value returned by the :ref:`onemkl_lapack_gerqf_scratchpad_size` function.

.. container:: section

  .. rubric:: Output Parameters

a
   Output buffer, overwritten by the factorization data as follows:

   If :math:`m \le n`, the upper triangle of the subarray ``a(1:m, n-m+1:n)`` contains the :math:`m \times m` upper triangular matrix :math:`R`; if :math:`m \ge n`, the elements on and above the :math:`(m-n)`-th subdiagonal contain the :math:`m \times n` upper trapezoidal matrix :math:`R`

   In both cases, the remaining elements, with the array ``tau``, represent the orthogonal/unitary matrix :math:`Q` as a product of :math:`\min(m,n)` elementary reflectors.

tau
   Array, size at least :math:`\min(m,n)`.

   Contains scalars that define elementary reflectors for the matrix :math:`Q` in its decomposition in a product of elementary reflectors.

gerqf (USM Version)
----------------------

.. container:: section

  .. rubric:: Syntax

.. code-block:: cpp

    namespace oneapi::mkl::lapack {
      sycl::event gerqf(sycl::queue &queue, std::int64_t m, std::int64_t n, T *a, std::int64_t lda, T *tau, T *scratchpad, std::int64_t scratchpad_size, const std::vector<sycl::event> &events = {})
    }

.. container:: section

  .. rubric:: Input Parameters

queue
   Device queue where calculations will be performed.
   
m
   The number of rows in the matrix :math:`A` (:math:`0 \le m`).
   
n
   The number of columns in the matrix :math:`A` (:math:`0 \le n`).
   
a
   Buffer holding input matrix :math:`A`. The second dimension of ``a`` must be at least :math:`\max(1, n)`.
   
lda
   The leading dimension of ``a``, at least :math:`\max(1, m)`.
      
scratchpad
   Buffer holding scratchpad memory to be used by the routine for storing intermediate results.
   
scratchpad_size
   Size of scratchpad memory as a number of floating point elements of type ``T``. Size should not be less than the value returned by the :ref:`onemkl_lapack_gerqf_scratchpad_size` function.

events
   List of events to wait for before starting computation. Defaults to empty list.

.. container:: section

  .. rubric:: Output Parameters

a
   Output buffer, overwritten by the factorization data as follows:

   If :math:`m \le n`, the upper triangle of the subarray ``a(1:m, n-m+1:n)`` contains the :math:`m \times m` upper triangular matrix :math:`R`; if :math:`m \ge n`, the elements on and above the :math:`(m-n)`-th subdiagonal contain the :math:`m \times n` upper trapezoidal matrix :math:`R`

   In both cases, the remaining elements, with the array ``tau``, represent the orthogonal/unitary matrix :math:`Q` as a product of :math:`\min(m,n)` elementary reflectors.

tau
   Array, size at least :math:`\min(m,n)`.

   Contains scalars that define elementary reflectors for the matrix :math:`Q` in its decomposition in a product of elementary reflectors.

.. container:: section

  .. rubric:: Return Values
         
Output event to wait on to ensure computation is complete.

**Parent topic:** :ref:`onemkl_lapack-linear-equation-routines`


.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemkl_lapack_ungqr_batch:

ungqr_batch
===========

Generates the complex unitary matrices :math:`Q_i` of the batch of QR factorizations formed by the :ref:`onemkl_lapack_geqrf_batch` function.

.. container:: section

  .. rubric:: Description

``ungqr_batch`` supports the following precisions.

   .. list-table:: 
      :header-rows: 1

      * -  T 
      * -  ``std::complex<float>`` 
      * -  ``std::complex<double>`` 

.. _onemkl_lapack_ungqr_batch_buffer:

ungqr_batch (Buffer Version)
----------------------------

.. container:: section

  .. rubric:: Description

The buffer version of ``ungqr_batch`` supports only the strided API. 
   
**Strided API**

 | The routine generates the wholes or parts of :math`m \times m` unitary matrices :math:`Q_i` of the batch of QR factorization formed by the Strided API of the :ref:`onemkl_lapack_geqrf_batch_buffer`.
 | Usually :math:`Q_i` is determined from the QR factorization of an :math:`m \times p` matrix :math:`A_i` with :math`m \ge p`.
 | To compute the whole matrices :math:`Q_i`, use:
 | ``ungqr_batch(queue, m, m, p, a, ...)``
 | To compute the leading :math:`p` columns of :math:`Q_i` (which form an orthonormal basis in the space spanned by the columns of :math:`A_i`):
 | ``ungqr_batch(queue, m, p, p, a, ...)``
 | To compute the matrices :math:`Q_i`^k` of the QR factorizations of leading :math:`k` columns of the matrices :math:`A_i`:
 | ``ungqr_batch(queue, m, m, k, a, ...)``
 | To compute the leading :math:`k` columns of :math:`Q_i^k` (which form an orthonormal basis in the space spanned by leading :math:`k` columns of the matrices :math:`A_i`):
 | ``ungqr_batch(queue, m, k, k, a, ...)``

.. container:: section

  .. rubric:: Syntax

.. code-block:: cpp

    namespace oneapi::mkl::lapack {
      void ungqr_batch(cl::sycl::queue &queue, std::int64_t m, std::int64_t n, std::int64_t k, cl::sycl::buffer<T> &a, std::int64_t lda, std::int64_t stride_a, cl::sycl::buffer<T> &tau, std::int64_t stride_tau, std::int64_t batch_size, cl::sycl::buffer<T> &scratchpad, std::int64_t scratchpad_size)
    }

.. container:: section

  .. rubric:: Input Parameters

queue
  Device queue where calculations will be performed.

m
  Number of rows in the matrices :math:`A_i` (:math:`0 \le m`).

n
  Number of columns in the matrices :math:`A_i` (:math:`0\le n`).

k
  Number of elementary reflectors whose product defines the matrices :math:`Q_i` (:math:`0 \le k \le n`).

a
  Array resulting after call to the Strided API of the :ref:`onemkl_lapack_geqrf_batch_usm` function.

lda
  Leading dimension of :math:`A_i` (:math:`\text{lda} \le m`).

stride_a
  Stride between the beginnings of matrices :math:`A_i` inside the batch array ``a``.

tau
  Array resulting after call to the Strided API of the :ref:`onemkl_lapack_geqrf_batch_usm` function.

stride_tau
  Stride between the beginnings of arrays :math:`tau_i` inside the array ``tau``.

batch_size
  Number of problems in a batch.

scratchpad
  Scratchpad memory to be used by routine for storing intermediate results.

scratchpad_size 
  Size of scratchpad memory as a number of floating point elements of type ``T``. Size should not be less then the value returned by strided version of the Strided API of the :ref:`onemkl_lapack_ungqr_batch_scratchpad_size` function.

.. container:: section

  .. rubric:: Output Parameters

a
  Array data is overwritten by ``a`` batch of n leading columns of the :math:`m \times m` unitary matrices :math:`Q_i`.

.. container:: section

  .. rubric:: Throws

This routine shall throw the following exceptions if the associated condition is detected. An implementation may throw additional implementation-specific exception(s) in case of error conditions not covered here.

:ref:`oneapi::mkl::lapack::batch_error<onemkl_lapack_exception_batch_error>`

:ref:`oneapi::mkl::unimplemented<onemkl_exception_unimplemented>`

:ref:`oneapi::mkl::unsupported_device<onemkl_exception_unsupported_device>`

:ref:`oneapi::mkl::lapack::invalid_argument<onemkl_lapack_exception_invalid_argument>`
  
   Exception is thrown in case of problems during calculations. The info code of the problem can be obtained by info() method of exception object:

    If ``info = -n``, the :math:`n`-th parameter had an illegal value.

    If ``info`` equals to value passed as scratchpad size, and `detail()` returns non zero, then passed scratchpad is of insufficient size, and required size should be not less then value returned by `detail()` method of exception object.

    If ``info`` is not zero and `detail()` returns zero, then there were some errors for some of the problems in the supplied batch and ``info`` code contains the number of failed calculations in a batch.

.. _onemkl_lapack_ungqr_batch_usm:

ungqr_batch (USM Version)
-------------------------

.. container:: section

  .. rubric:: Description

The USM version of ``ungqr_batch`` supports the group API and strided API. 

**Group API**

 | The routine generates the wholes or parts of :math`m \times m` unitary matrices :math:`Q_i` of the batch of QR factorization formed by the Group API of the :ref:`onemkl_lapack_geqrf_batch_buffer`.
 | Usually :math:`Q_i` is determined from the QR factorization of an :math:`m \times p` matrix :math:`A_i` with :math`m \ge p`.
 | To compute the whole matrices :math:`Q_i`, use:
 | ``ungqr_batch(queue, m, m, p, a, ...)``
 | To compute the leading :math:`p` columns of :math:`Q_i` (which form an orthonormal basis in the space spanned by the columns of :math:`A_i`):
 | ``ungqr_batch(queue, m, p, p, a, ...)``
 | To compute the matrices :math:`Q_i`^k` of the QR factorizations of leading :math:`k` columns of the matrices :math:`A_i`:
 | ``ungqr_batch(queue, m, m, k, a, ...)``
 | To compute the leading :math:`k` columns of :math:`Q_i^k` (which form an orthonormal basis in the space spanned by leading :math:`k` columns of the matrices :math:`A_i`):
 | ``ungqr_batch(queue, m, k, k, a, ...)``

.. container:: section

  .. rubric:: Syntax

.. code-block:: cpp

    namespace oneapi::mkl::lapack {
      cl::sycl::event ungqr_batch(cl::sycl::queue &queue, std::int64_t *m, std::int64_t *n, std::int64_t *k, T **a, std::int64_t *lda, T **tau, std::int64_t group_count, std::int64_t *group_sizes, T *scratchpad, std::int64_t scratchpad_size, const cl::sycl::vector_class<cl::sycl::event> &events = {})
    }

.. container:: section

  .. rubric:: Input Parameters

queue
  Device queue where calculations will be performed.

m
  Array of ``group_count`` :math:`m_g` parameters as previously supplied to the Group API of the :ref:`onemkl_lapack_geqrf_batch_usm` function.

n
  Array of ``group_count`` :math:`n_g` parameters as previously supplied to the Group API of the :ref:`onemkl_lapack_geqrf_batch_usm` function.

k
 | Array of ``group_count`` :math:`k_g` parameters as previously supplied to the Group API of the :ref:`onemkl_lapack_geqrf_batch_usm` function.
 | The number of elementary reflectors whose product defines the matrices :math:`Q_i` (:math:`0 \le k_g \le n_g`).

a
  Array resulting after call to the Group API of the :ref:`onemkl_lapack_geqrf_batch_usm` function.

lda
  Array of leading dimensions of :math:`A_i` as previously supplied to the Group API of the :ref:`onemkl_lapack_geqrf_batch_usm` function.

tau
  Array resulting after call to the Group API of the :ref:`onemkl_lapack_geqrf_batch_usm` function.

group_count
  Number of groups of parameters. Must be at least 0.

group_sizes
  Array of ``group_count`` integers. Array element with index :math:`g` specifies the number of problems to solve for each of the groups of parameters :math:`g`. So the total number of problems to solve, ``batch_size``, is a sum of all parameter group sizes.

scratchpad
  Scratchpad memory to be used by routine for storing intermediate results.

scratchpad_size
  Size of scratchpad memory as a number of floating point elements of type ``T``. Size should not be less then the value returned by Group API of the :ref:`onemkl_lapack_ungqr_batch_scratchpad_size` function.

events
  List of events to wait for before starting computation. Defaults to empty list.

.. container:: section

  .. rubric:: Output Parameters
   
a
  Matrices pointed to by array ``a`` are overwritten by :math:`n_g` leading columns of the :math:`m_g \times m_g` orthogonal matrices :math:`Q_i`, where :math:`g` is an index of group of parameters corresponding to :math:`Q_i`.

.. container:: section
   
  .. rubric:: Return Values

Output event to wait on to ensure computation is complete.

.. container:: section

  .. rubric:: Throws

This routine shall throw the following exceptions if the associated condition is detected. An implementation may throw additional implementation-specific exception(s) in case of error conditions not covered here.

:ref:`oneapi::mkl::lapack::batch_error<onemkl_lapack_exception_batch_error>`

:ref:`oneapi::mkl::unimplemented<onemkl_exception_unimplemented>`

:ref:`oneapi::mkl::unsupported_device<onemkl_exception_unsupported_device>`

:ref:`oneapi::mkl::lapack::invalid_argument<onemkl_lapack_exception_invalid_argument>`
 
   The ``info`` code of the problem can be obtained by `info()` method of exception object:

    If ``info = -n``, the :math:`n`-th parameter had an illegal value.
    If ``info`` equals to value passed as scratchpad size, and `detail()` returns non zero, then passed scratchpad is of insufficient size, and required size should be not less then value returned by `detail()` method of exception object.

    If ``info`` is not zero and `detail()` returns zero, then there were some errors for some of the problems in the supplied batch and ``info`` code contains the number of failed calculations in a batch.

**Strided API**

 | The routine generates the wholes or parts of :math`m \times m` unitary matrices :math:`Q_i` of the batch of QR factorization formed by the Strided API of the :ref:`onemkl_lapack_geqrf_batch_usm`.
 | Usually :math:`Q_i` is determined from the QR factorization of an :math:`m \times p` matrix :math:`A_i` with :math`m \ge p`.
 | To compute the whole matrices :math:`Q_i`, use:
 | ``ungqr_batch(queue, m, m, p, a, ...)``
 | To compute the leading :math:`p` columns of :math:`Q_i` (which form an orthonormal basis in the space spanned by the columns of :math:`A_i`):
 | ``ungqr_batch(queue, m, p, p, a, ...)``
 | To compute the matrices :math:`Q_i`^k` of the QR factorizations of leading :math:`k` columns of the matrices :math:`A_i`:
 | ``ungqr_batch(queue, m, m, k, a, ...)``
 | To compute the leading :math:`k` columns of :math:`Q_i^k` (which form an orthonormal basis in the space spanned by leading :math:`k` columns of the matrices :math:`A_i`):
 | ``ungqr_batch(queue, m, k, k, a, ...)``

.. container:: section

  .. rubric:: Syntax

.. code-block:: cpp

    namespace oneapi::mkl::lapack {
      cl::sycl::event ungqr_batch(cl::sycl::queue &queue, std::int64_t m, std::int64_t n, std::int64_t k, T *a, std::int64_t lda, std::int64_t stride_a, T *tau, std::int64_t stride_tau, std::int64_t batch_size, T *scratchpad, std::int64_t scratchpad_size, const cl::sycl::vector_class<cl::sycl::event> &events = {})
    };

.. container:: section

  .. rubric:: Input Parameters

queue
  Device queue where calculations will be performed.

m
  Number of rows in the matrices :math:`A_i` (:math:`0 \le m`).

n
  Number of columns in the matrices :math:`A_i` (:math:`0\le n`).

k
  Number of elementary reflectors whose product defines the matrices :math:`Q_i` (:math:`0 \le k \le n`).

a
  Array resulting after call to the Strided API of the :ref:`onemkl_lapack_geqrf_batch_usm` function.

lda
  Leading dimension of :math:`A_i` (:math:`\text{lda} \le m`).

stride_a
  Stride between the beginnings of matrices :math:`A_i` inside the batch array ``a``.

tau
  Array resulting after call to the Strided API of the :ref:`onemkl_lapack_geqrf_batch_usm` function.

stride_tau
  Stride between the beginnings of arrays :math:`tau_i` inside the array ``tau``.

batch_size
  Number of problems in a batch.

scratchpad
  Scratchpad memory to be used by routine for storing intermediate results.

scratchpad_size 
  Size of scratchpad memory as a number of floating point elements of type ``T``. Size should not be less then the value returned by strided version of the Strided API of the :ref:`onemkl_lapack_ungqr_batch_scratchpad_size` function.

events  
  List of events to wait for before starting computation. Defaults to empty list.

.. container:: section

  .. rubric:: Output Parameters

a
  Array data is overwritten by ``a`` batch of n leading columns of the :math:`m \times m` unitary matrices :math:`Q_i`.

.. container:: section
   
  .. rubric:: Return Values

Output event to wait on to ensure computation is complete.

.. container:: section

  .. rubric:: Throws

This routine shall throw the following exceptions if the associated condition is detected. An implementation may throw additional implementation-specific exception(s) in case of error conditions not covered here.

:ref:`oneapi::mkl::lapack::batch_error<onemkl_lapack_exception_batch_error>`

:ref:`oneapi::mkl::unimplemented<onemkl_exception_unimplemented>`

:ref:`oneapi::mkl::unsupported_device<onemkl_exception_unsupported_device>`

:ref:`oneapi::mkl::lapack::invalid_argument<onemkl_lapack_exception_invalid_argument>`

   The ``info`` code of the problem can be obtained by `info()` method of exception object:
  
   If ``info = -n``, the :math:`n`-th parameter had an illegal value.
   
   If ``info`` equals to value passed as scratchpad size, and `detail()` returns non zero, then passed scratchpad is of insufficient size, and required size should be not less then value returned by `detail()` method of exception object.

   If ``info`` is not zero and `detail()` returns zero, then there were some errors for some of the problems in the supplied batch and ``info`` code contains the number of failed calculations in a batch.

**Parent topic:** :ref:`onemkl_lapack-like-extensions-routines`


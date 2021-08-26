.. _onemkl_blas_dotc:

dotc
====

Computes the dot product of two complex vectors, conjugating the first vector.


Description
***********

The ``dotc`` routines perform a dot product between two complex
vectors, conjugating the first of them:

.. math::

   result = \sum_{i=1}^{n}\overline{X_i}Y_i 

``dotc`` supports the following precisions for data.

.. list-table::
   :header-rows: 1

   * -  T
   * -  ``std::complex<float>``
   * -  ``std::complex<double>``


dotc (Buffer Version)
*********************

Syntax
------

.. code-block:: cpp

   namespace oneapi::mkl::blas::column_major {
       void dotc(sycl::queue &queue,
                 std::int64_t n,
                 sycl::buffer<T,1> &x,
                 std::int64_t incx,
                 sycl::buffer<T,1> &y,
                 std::int64_t incy,
                 sycl::buffer<T,1> &result)
   }
.. code-block:: cpp

   namespace oneapi::mkl::blas::row_major {
       void dotc(sycl::queue &queue,
                 std::int64_t n,
                 sycl::buffer<T,1> &x,
                 std::int64_t incx,
                 sycl::buffer<T,1> &y,
                 std::int64_t incy,
                 sycl::buffer<T,1> &result)
   }


Input Parameters
----------------

queue
   The queue where the routine should be executed.

n
   The number of elements in vectors ``x`` and ``y``.

x
   Buffer holding input vector ``x``. The buffer must be of size at least (1 + (``n`` - 1)*abs(``incx``)). See :ref:`matrix-storage` for more details.

incx
   The stride of vector ``x``.

y
   Buffer holding input vector ``y``. The buffer must be of size at least (1 + (``n`` - 1)*abs(``incy``)). See :ref:`matrix-storage` for more details..

incy
   The stride of vector ``y``.


Output Parameters
-----------------

result
   The buffer where the result (a scalar) is stored.


dotc (USM Version)
******************

Syntax
------

.. code-block:: cpp

   namespace oneapi::mkl::blas::column_major {
       void dotc(sycl::queue &queue,
                 std::int64_t n,
                 const T *x,
                 std::int64_t incx,
                 const T *y,
                 std::int64_t incy,
                 T *result,
                 const sycl::vector_class<sycl::event> &dependencies = {})
   }
.. code-block:: cpp

   namespace oneapi::mkl::blas::row_major {
       void dotc(sycl::queue &queue,
                 std::int64_t n,
                 const T *x,
                 std::int64_t incx,
                 const T *y,
                 std::int64_t incy,
                 T *result,
                 const sycl::vector_class<sycl::event> &dependencies = {})
   }


Input Parameters
----------------

queue
   The queue where the routine should be executed.

n
   The number of elements in vectors ``x`` and ``y``.

x
   Pointer to input vector ``x``. The array holding the input vector ``x`` must be of size at least (1 + (``n`` - 1)*abs(``incx``)). See :ref:`matrix-storage` for more details.

incx
   The stride of vector ``x``.

y
   Pointer to input vector ``y``. The array holding the input vector ``y`` must be of size at least (1 + (``n`` - 1)*abs(``incy``)). See :ref:`matrix-storage` for more details..

incy
   The stride of vector ``y``.

dependencies
   List of events to wait for before starting computation, if any. If omitted, defaults to no dependencies.


Output Parameters
-----------------

result
   The pointer to where the result (a scalar) is stored.


Return Values
-------------

Output event to wait on to ensure computation is complete.
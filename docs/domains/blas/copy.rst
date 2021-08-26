.. _onemkl_blas_copy:

copy
====

Copies a vector to another vector.

Description
***********

The ``copy`` routines copy one vector to another:

.. math::
      
      y \leftarrow  x

where ``x`` and ``y`` are vectors of n elements.

``copy`` supports the following precisions.

.. list-table::
   :header-rows: 1

   * -  T
   * -  ``float``
   * -  ``double``
   * -  ``std::complex<float>``
   * -  ``std::complex<double>``


copy (Buffer Version)
*********************

Syntax
------

.. code-block:: cpp

   namespace oneapi::mkl::blas::column_major {
       void copy(sycl::queue &queue,
                 std::int64_t n,
                 sycl::buffer<T,1> &x,
                 std::int64_t incx,
                 sycl::buffer<T,1> &y,
                 std::int64_t incy)
   }
.. code-block:: cpp

   namespace oneapi::mkl::blas::row_major {
       void copy(sycl::queue &queue,
                 std::int64_t n,
                 sycl::buffer<T,1> &x,
                 std::int64_t incx,
                 sycl::buffer<T,1> &y,
                 std::int64_t incy)
   }


Input Parameters
----------------

queue
   The queue where the routine should be executed.

n
   Number of elements in vector ``x``.

x
   Buffer holding input vector ``x``. The buffer must be of size at least (1 + (``n`` – 1)*abs(``incx``)). See :ref:`matrix-storage` for more details.

incx
   Stride of vector ``x``.

incy
   Stride of vector ``y``.


Output Parameters
-----------------

y
   Buffer holding the updated vector ``y``.


copy (USM Version)
******************

Syntax
------

.. code-block:: cpp

   namespace oneapi::mkl::blas::column_major {
       sycl::event copy(sycl::queue &queue,
                        std::int64_t n,
                        const T *x,
                        std::int64_t incx,
                        T *y,
                        std::int64_t incy,
                        const sycl::vector_class<sycl::event> &dependencies = {})
   }
.. code-block:: cpp

   namespace oneapi::mkl::blas::row_major {
       sycl::event copy(sycl::queue &queue,
                        std::int64_t n,
                        const T *x,
                        std::int64_t incx,
                        T *y,
                        std::int64_t incy,
                        const sycl::vector_class<sycl::event> &dependencies = {})
   }


Input Parameters
----------------

queue
   The queue where the routine should be executed.

n
   Number of elements in vector ``x``.

x
   Pointer to the input vector ``x``. The array holding the vector ``x`` must be of size at least (1 + (``n`` – 1)*abs(``incx``)). See :ref:`matrix-storage` for more details.

incx
   Stride of vector ``x``.

incy
   Stride of vector ``y``.

dependencies
   List of events to wait for before starting computation, if any.
   If omitted, defaults to no dependencies.


Output Parameters
-----------------

y
   Pointer to the updated vector ``y``.


Return Values
-------------

Output event to wait on to ensure computation is complete.
.. _onemkl_blas_axpby:

axpby
=====

Computes a vector-scalar product added to a scaled-vector.

.. _onemkl_blas_axpby_description:

.. rubric:: Description

The ``axpby`` routines compute two scalar-vector product and add them:

.. math::

      y \leftarrow beta * y + alpha * x

where ``x`` and ``y`` are vectors of ``n`` elements and ``alpha`` and ``beta`` are scalars.

``axpby`` supports the following precisions.

   .. list-table::
      :header-rows: 1

      * -  T
      * -  ``float``
      * -  ``double``
      * -  ``std::complex<float>``
      * -  ``std::complex<double>``

.. _onemkl_blas_axpby_buffer:

axpby (Buffer Version)
----------------------

.. rubric:: Syntax

.. code-block:: cpp

   namespace oneapi::mkl::blas::column_major {
       void axpby(sycl::queue &queue,
                 std::int64_t n,
                 T alpha,
                 sycl::buffer<T,1> &x, std::int64_t incx,
                 T beta,
                 sycl::buffer<T,1> &y, std::int64_t incy)
   }
.. code-block:: cpp

   namespace oneapi::mkl::blas::row_major {
       void axpby(sycl::queue &queue,
                 std::int64_t n,
                 T alpha,
                 sycl::buffer<T,1> &x, std::int64_t incx,
                 T beta,
                 sycl::buffer<T,1> &y, std::int64_t incy)
   }

.. container:: section

   .. rubric:: Input Parameters

   queue
      The queue where the routine should be executed.

   n
      Number of elements in vector ``x`` and ``y``.

   alpha
      Specifies the scalar ``alpha``.

   x
      Buffer holding input vector ``x``. The buffer must be of size at least
      (1 + (``n`` – 1)*abs(``incx``)). See :ref:`matrix-storage` for
      more details.

   incx
      Stride between two consecutive elements of the ``x`` vector.

   beta
      Specifies the scalar ``beta``.

   y
      Buffer holding input vector ``y``. The buffer must be of size at least
      (1 + (``n`` – 1)*abs(``incy``)). See :ref:`matrix-storage` for
      more details.

   incy
      Stride between two consecutive elements of the ``y`` vector.

.. container:: section

   .. rubric:: Output Parameters

   y
      Buffer holding the updated vector ``y``.


.. _onemkl_blas_axpby_usm:

axpby (USM Version)
-------------------

.. rubric:: Syntax

.. code-block:: cpp

   namespace oneapi::mkl::blas::column_major {
       sycl::event axpby(sycl::queue &queue,
                        std::int64_t n,
                        T alpha,
                        const T *x, std::int64_t incx,
                        const T beta,
                        T *y, std::int64_t incy,
                        const std::vector<sycl::event> &dependencies = {})
   }
.. code-block:: cpp

   namespace oneapi::mkl::blas::row_major {
       sycl::event axpby(sycl::queue &queue,
                        std::int64_t n,
                        T alpha,
                        const T *x, std::int64_t incx,
                        const T beta,
                        T *y, std::int64_t incy,
                        const std::vector<sycl::event> &dependencies = {})
   }

.. container:: section

   .. rubric:: Input Parameters

   queue
      The queue where the routine should be executed.

   n
      Number of elements in vector ``x`` and ``y``.

   alpha
      Specifies the scalar alpha.

   beta
      Specifies the scalar beta.

   x
      Pointer to the input vector ``x``. The allocated memory must be
      of size at least (1 + (``n`` – 1)*abs(``incx``)). See
      :ref:`matrix-storage` for more details.

   incx
      Stride between consecutive elements of the ``x`` vector.

   y
      Pointer to the input vector ``y``. The allocated memory must be
      of size at least (1 + (``n`` – 1)*abs(``incy``)). See
      :ref:`matrix-storage` for more details.

   incy
      Stride between consecutive elements of the ``y`` vector.

   dependencies
      List of events to wait for before starting computation, if any.
      If omitted, defaults to no dependencies.

.. container:: section

   .. rubric:: Output Parameters

   y
      Array holding the updated vector ``y``.

.. container:: section

   .. rubric:: Return Values

   Output event to wait on to ensure computation is complete.


   **Parent topic:** :ref:`blas-like-extensions`


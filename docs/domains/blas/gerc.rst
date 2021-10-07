.. _onemkl_blas_gerc:

gerc
====

Computes a rank-1 update (conjugated) of a general complex matrix.

.. _onemkl_blas_gerc_description:

.. rubric:: Description

The ``gerc`` routines compute a scalar-vector-vector product and add the
result to a general matrix. The operation is defined as:

.. math::

      A \leftarrow alpha*x*y^H + A


where:

``alpha`` is a scalar,

``A`` is an ``m``-by-``n`` matrix,

``x`` is a vector of length ``m``,

``y`` is vector of length ``n``.

``gerc`` supports the following precisions.

   .. list-table:: 
      :header-rows: 1

      * -  T 
      * -  ``std::complex<float>`` 
      * -  ``std::complex<double>`` 

.. _onemkl_blas_gerc_buffer:

gerc (Buffer Version)
---------------------

.. rubric:: Syntax

.. code-block:: cpp

   namespace oneapi::mkl::blas::column_major {
       void gerc(sycl::queue &queue,
                 std::int64_t m,
                 std::int64_t n,
                 T alpha,
                 sycl::buffer<T,1> &x,
                 std::int64_t incx,
                 sycl::buffer<T,1> &y,
                 std::int64_t incy,
                 sycl::buffer<T,1> &a,
                 std::int64_t lda)
   }
.. code-block:: cpp

   namespace oneapi::mkl::blas::row_major {
       void gerc(sycl::queue &queue,
                 std::int64_t m,
                 std::int64_t n,
                 T alpha,
                 sycl::buffer<T,1> &x,
                 std::int64_t incx,
                 sycl::buffer<T,1> &y,
                 std::int64_t incy,
                 sycl::buffer<T,1> &a,
                 std::int64_t lda)
   }

.. container:: section

   .. rubric:: Input Parameters

   queue
      The queue where the routine should be executed.

   m
      Number of rows of ``A``. Must be at least zero.

   n
      Number of columns of ``A``. Must be at least zero.

   alpha
      Scaling factor for the matrix-vector product.

   x
      Buffer holding input vector ``x``. The buffer must be of size at
      least (1 + (``m`` - 1)*abs(``incx``)). See :ref:`matrix-storage` for
      more details.

   incx
      Stride of vector ``x``.

   y
      Buffer holding input/output vector ``y``. The buffer must be of
      size at least (1 + (``n`` - 1)*abs(``incy``)). See :ref:`matrix-storage`
      for more details.

   incy
      Stride of vector ``y``.

   a
      Buffer holding input matrix ``A``. Must have size at least
      ``lda``\ \*\ ``n`` if column major layout is used or at least ``lda``\ \*\ ``m``
      if row major layout is used. See :ref:`matrix-storage` for
      more details.

   lda
      Leading dimension of matrix ``A``. Must be positive and at least
      ``m`` if column major layout is used or at least ``n`` if row
      major layout is used.


.. container:: section

   .. rubric:: Output Parameters

   a
      Buffer holding the updated matrix ``A``.


.. _onemkl_blas_gerc_usm:

gerc (USM Version)
------------------

.. rubric:: Syntax

.. code-block:: cpp

   namespace oneapi::mkl::blas::column_major {
       sycl::event gerc(sycl::queue &queue,
                        std::int64_t m,
                        std::int64_t n,
                        T alpha,
                        const T *x,
                        std::int64_t incx,
                        const T *y,
                        std::int64_t incy,
                        T *a,
                        std::int64_t lda,
                        const std::vector<sycl::event> &dependencies = {})
   }
.. code-block:: cpp

   namespace oneapi::mkl::blas::row_major {
       sycl::event gerc(sycl::queue &queue,
                        std::int64_t m,
                        std::int64_t n,
                        T alpha,
                        const T *x,
                        std::int64_t incx,
                        const T *y,
                        std::int64_t incy,
                        T *a,
                        std::int64_t lda,
                        const std::vector<sycl::event> &dependencies = {})
   }

.. container:: section

   .. rubric:: Input Parameters

   queue
      The queue where the routine should be executed.

   m
      Number of rows of ``A``. Must be at least zero.

   n
      Number of columns of ``A``. Must be at least zero.

   alpha
      Scaling factor for the matrix-vector product.

   x
      Pointer to the input vector ``x``. The array holding input
      vector ``x`` must be of size at least (1 + (``m`` -
      1)*abs(``incx``)). See :ref:`matrix-storage` for
      more details.

   incx
      Stride of vector ``x``.

   y
      Pointer to the input/output vector ``y``. The array holding the
      input/output vector ``y`` must be of size at least (1 + (``n``
      - 1)*abs(``incy``)). See :ref:`matrix-storage` for
      more details.

   incy
      Stride of vector ``y``.

   a
      Pointer to input matrix ``A``. The array holding input matrix
      ``A``\ must have size at least ``lda``\ \*\ ``n`` if column
      major layout is used or at least ``lda``\ \*\ ``m`` if row
      major layout is used. See :ref:`matrix-storage` for more details.

   lda
      Leading dimension of matrix ``A``. Must be positive and at least
      ``m`` if column major layout is used or at least ``n`` if row
      major layout is used.

   dependencies
      List of events to wait for before starting computation, if any.
      If omitted, defaults to no dependencies.

.. container:: section

   .. rubric:: Output Parameters

   a
      Pointer to the updated matrix ``A``.

.. container:: section

   .. rubric:: Return Values

   Output event to wait on to ensure computation is complete.

   **Parent topic:** :ref:`blas-level-2-routines`

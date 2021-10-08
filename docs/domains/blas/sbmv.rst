.. _onemkl_blas_sbmv:

sbmv
====

Computes a matrix-vector product with a symmetric band matrix.

.. _onemkl_blas_sbmv_description:

.. rubric:: Description

The ``sbmv`` routines compute a scalar-matrix-vector product and add the
result to a scalar-vector product, with a symmetric band matrix. The
operation is defined as:

.. math::

      y \leftarrow alpha*A*x + beta*y

where:

``alpha`` and ``beta`` are scalars,

``A`` is an ``n``-by-``n`` symmetric matrix with ``k``
super-diagonals,

``x`` and ``y`` are vectors of length ``n``.

``sbmv`` supports the following precisions.

   .. list-table:: 
      :header-rows: 1

      * -  T 
      * -  ``float`` 
      * -  ``double`` 

.. _onemkl_blas_sbmv_buffer:

sbmv (Buffer Version)
---------------------

.. rubric:: Syntax

.. code-block:: cpp

   namespace oneapi::mkl::blas::column_major {
       void sbmv(sycl::queue &queue,
                 onemkl::uplo upper_lower,
                 std::int64_t n,
                 std::int64_t k,
                 T alpha,
                 sycl::buffer<T,1> &a,
                 std::int64_t lda,
                 sycl::buffer<T,1> &x,
                 std::int64_t incx,
                 T beta,
                 sycl::buffer<T,1> &y,
                 std::int64_t incy)
   }
.. code-block:: cpp

   namespace oneapi::mkl::blas::row_major {
       void sbmv(sycl::queue &queue,
                 onemkl::uplo upper_lower,
                 std::int64_t n,
                 std::int64_t k,
                 T alpha,
                 sycl::buffer<T,1> &a,
                 std::int64_t lda,
                 sycl::buffer<T,1> &x,
                 std::int64_t incx,
                 T beta,
                 sycl::buffer<T,1> &y,
                 std::int64_t incy)
   }

.. container:: section

   .. rubric:: Input Parameters

   queue
      The queue where the routine should be executed.

   upper_lower
      Specifies whether ``A`` is upper or lower triangular. See :ref:`onemkl_datatypes` for more details.

   n
      Number of rows and columns of ``A``. Must be at least zero.

   k
      Number of super-diagonals of the matrix ``A``. Must be at least
      zero.

   alpha
      Scaling factor for the matrix-vector product.

   a
      Buffer holding input matrix ``A``. Must have size at least
      ``lda``\ \*\ ``n``. See :ref:`matrix-storage` for
      more details.

   lda
      Leading dimension of matrix ``A``. Must be at least (``k`` + 1),
      and positive.

   x
      Buffer holding input vector ``x``. The buffer must be of size at
      least (1 + (``n`` - 1)*abs(``incx``)). See :ref:`matrix-storage` for
      more details.

   incx
      Stride of vector ``x``.

   beta
      Scaling factor for vector ``y``.

   y
      Buffer holding input/output vector ``y``. The buffer must be of
      size at least (1 + (``n`` - 1)*abs(``incy``)). See :ref:`matrix-storage`
      for more details.

   incy
      Stride of vector ``y``.

.. container:: section

   .. rubric:: Output Parameters

   y
      Buffer holding the updated vector ``y``.


.. _onemkl_blas_sbmv_usm:

sbmv (USM Version)
------------------

.. rubric:: Syntax

.. code-block:: cpp

   namespace oneapi::mkl::blas::column_major {
       sycl::event sbmv(sycl::queue &queue,
                        onemkl::uplo upper_lower,
                        std::int64_t n,
                        std::int64_t k,
                        T alpha,
                        const T *a,
                        std::int64_t lda,
                        const T *x,
                        std::int64_t incx,
                        T beta,
                        T *y,
                        std::int64_t incy,
                        const std::vector<sycl::event> &dependencies = {})
   }
.. code-block:: cpp

   namespace oneapi::mkl::blas::row_major {
       sycl::event sbmv(sycl::queue &queue,
                        onemkl::uplo upper_lower,
                        std::int64_t n,
                        std::int64_t k,
                        T alpha,
                        const T *a,
                        std::int64_t lda,
                        const T *x,
                        std::int64_t incx,
                        T beta,
                        T *y,
                        std::int64_t incy,
                        const std::vector<sycl::event> &dependencies = {})
   }

.. container:: section

   .. rubric:: Input Parameters

   queue
      The queue where the routine should be executed.

   upper_lower
      Specifies whether ``A`` is upper or lower triangular. See :ref:`onemkl_datatypes` for more details.

   n
      Number of rows and columns of ``A``. Must be at least zero.

   k
      Number of super-diagonals of the matrix ``A``. Must be at least
      zero.

   alpha
      Scaling factor for the matrix-vector product.

   a
      Pointer to input matrix ``A``. The array holding input matrix
      ``A`` must have size at least ``lda``\ \*\ ``n``. See :ref:`matrix-storage` for
      more details.

   lda
      Leading dimension of matrix ``A``. Must be at least (``k`` +
      1), and positive.

   x
      Pointer to input vector ``x``. The array holding input vector
      ``x`` must be of size at least (1 + (``n`` - 1)*abs(``incx``)).
      See :ref:`matrix-storage` for
      more details.

   incx
      Stride of vector ``x``.

   beta
      Scaling factor for vector ``y``.

   y
      Pointer to input/output vector ``y``. The array holding
      input/output vector ``y`` must be of size at least (1 + (``n``
      - 1)*abs(``incy``)). See :ref:`matrix-storage` for
      more details.

   incy
      Stride of vector ``y``.

   dependencies
      List of events to wait for before starting computation, if any.
      If omitted, defaults to no dependencies.

.. container:: section

   .. rubric:: Output Parameters

   y
      Pointer to the updated vector ``y``.

.. container:: section

   .. rubric:: Return Values

   Output event to wait on to ensure computation is complete.


   **Parent topic:** :ref:`blas-level-2-routines`

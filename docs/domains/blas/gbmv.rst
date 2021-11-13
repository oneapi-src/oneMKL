.. _onemkl_blas_gbmv:

gbmv
====

Computes a matrix-vector product with a general band matrix.

.. _onemkl_blas_gbmv_description:

.. rubric:: Description

The ``gbmv`` routines compute a scalar-matrix-vector product and add
the result to a scalar-vector product, with a general band matrix.
The operation is defined as

.. math::
      
      y \leftarrow alpha*op(A)*x + beta*y

where:

op(``A``) is one of op(``A``) = ``A``, or op(``A``) =
``A``\ :sup:`T`, or op(``A``) = ``A``\ :sup:`H`,

``alpha`` and ``beta`` are scalars,

``A`` is an ``m``-by-``n`` matrix with ``kl`` sub-diagonals and
``ku`` super-diagonals,

``x`` and ``y`` are vectors.

``gbmv`` supports the following precisions.

   .. list-table:: 
      :header-rows: 1

      * -  T 
      * -  ``float`` 
      * -  ``double`` 
      * -  ``std::complex<float>`` 
      * -  ``std::complex<double>`` 

.. _onemkl_blas_gbmv_buffer:

gbmv (Buffer Version)
---------------------

.. rubric:: Syntax

.. code-block:: cpp

   namespace oneapi::mkl::blas::column_major {
       void gbmv(sycl::queue &queue,
                 onemkl::transpose trans,
                 std::int64_t m,
                 std::int64_t n,
                 std::int64_t kl,
                 std::int64_t ku,
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
       void gbmv(sycl::queue &queue,
                 onemkl::transpose trans,
                 std::int64_t m,
                 std::int64_t n,
                 std::int64_t kl,
                 std::int64_t ku,
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

   trans
      Specifies op(``A``), the transposition operation applied to ``A``.
      See
      :ref:`onemkl_datatypes` for more
      details.

   m
      Number of rows of ``A``. Must be at least zero.

   n
      Number of columns of ``A``. Must be at least zero.

   kl
      Number of sub-diagonals of the matrix ``A``. Must be at least
      zero.

   ku
      Number of super-diagonals of the matrix ``A``. Must be at least
      zero.

   alpha
      Scaling factor for the matrix-vector product.

   a
      Buffer holding input matrix ``A``. Must have size at least ``lda``\ \*\ ``n``
      if column major layout is used or at least ``lda``\ \*\ ``m``
      if row major layout is used. See :ref:`matrix-storage` for more details.

   lda
      Leading dimension of matrix ``A``. Must be at least (``kl`` +
      ``ku`` + 1), and positive.

   x
      Buffer holding input vector ``x``. The length ``len`` of vector
      ``x`` is ``n`` if ``A`` is not transposed, and ``m`` if ``A`` is
      transposed. The buffer must be of size at least (1 + (``len`` -
      1)*abs(``incx``)). See :ref:`matrix-storage` for
      more details.

   incx
      Stride of vector ``x``.

   beta
      Scaling factor for vector ``y``.

   y
      Buffer holding input/output vector ``y``. The length ``len`` of
      vector ``y`` is ``m``, if ``A`` is not transposed, and ``n`` if
      ``A`` is transposed. The buffer must be of size at least (1 +
      (``len`` - 1)*abs(``incy``)) where ``len`` is this length. See
      :ref:`matrix-storage` for
      more details.

   incy
      Stride of vector ``y``.

.. container:: section

   .. rubric:: Output Parameters

   y
      Buffer holding the updated vector ``y``.


.. _onemkl_blas_gbmv_usm:

gbmv (USM Version)
------------------

.. rubric:: Syntax

.. code-block:: cpp

   namespace oneapi::mkl::blas::column_major {
       sycl::event gbmv(sycl::queue &queue,
                        onemkl::transpose trans,
                        std::int64_t m,
                        std::int64_t n,
                        std::int64_t kl,
                        std::int64_t ku,
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
       sycl::event gbmv(sycl::queue &queue,
                        onemkl::transpose trans,
                        std::int64_t m,
                        std::int64_t n,
                        std::int64_t kl,
                        std::int64_t ku,
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

   trans
      Specifies op(``A``), the transposition operation applied to
      ``A``. See
      :ref:`onemkl_datatypes` for
      more details.

   m
      Number of rows of ``A``. Must be at least zero.

   n
      Number of columns of ``A``. Must be at least zero.

   kl
      Number of sub-diagonals of the matrix ``A``. Must be at least
      zero.

   ku
      Number of super-diagonals of the matrix ``A``. Must be at least
      zero.

   alpha
      Scaling factor for the matrix-vector product.

   a
      Pointer to input matrix ``A``. The array holding input matrix
      ``A`` must have size at least ``lda``\ \*\ ``n`` if column
      major layout is used or at least ``lda``\ \*\ ``m`` if row
      major layout is used. See :ref:`matrix-storage` for more details.

   lda
      Leading dimension of matrix ``A``. Must be at least (``kl`` +
      ``ku`` + 1), and positive.

   x
      Pointer to input vector ``x``. The length ``len`` of vector
      ``x`` is ``n`` if ``A`` is not transposed, and ``m`` if ``A``
      is transposed. The array holding input vector ``x`` must be of
      size at least (1 + (``len`` - 1)*abs(``incx``)). See 
      :ref:`matrix-storage` for more details.

   incx
      Stride of vector ``x``.

   beta
      Scaling factor for vector ``y``.

   y
      Pointer to input/output vector ``y``. The length ``len`` of
      vector ``y`` is ``m``, if ``A`` is not transposed, and ``n`` if
      ``A`` is transposed. The array holding input/output vector
      ``y`` must be of size at least (1 + (``len`` -
      1)*abs(``incy``)) where ``len`` is this length. 
      See :ref:`matrix-storage` for more details.

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

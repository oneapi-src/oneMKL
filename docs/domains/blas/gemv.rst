.. _onemkl_blas_gemv:

gemv
====


.. container::


   Computes a matrix-vector product using a general matrix.



      ``gemv`` supports the following precisions.


      .. list-table:: 
         :header-rows: 1

         * -  T 
         * -  ``float`` 
         * -  ``double`` 
         * -  ``std::complex<float>`` 
         * -  ``std::complex<double>`` 




.. container:: section


   .. rubric:: Description
      :class: sectiontitle


   The ``gemv`` routines compute a scalar-matrix-vector product and add the
   result to a scalar-vector product, with a general matrix. The
   operation is defined as


  


      y  <- alpha*op(A)*x + beta*y


   where:


   op(``A``) is one of op(``A``) = ``A``, or op(``A``) =
   ``A``\ :sup:`T`, or op(``A``) = ``A``\ :sup:`H`,


   ``alpha`` and ``beta`` are scalars,


   ``A`` is an ``m``-by-``n`` matrix, and ``x``, ``y`` are vectors.


gemv (Buffer Version)
---------------------

.. container::

   .. container:: section


      .. rubric:: Syntax
         :class: sectiontitle


      .. cpp:function::  void oneapi::mkl::blas::gemv(sycl::queue &queue, transpose trans, std::int64_t m, std::int64_t n, T alpha, sycl::buffer<T,1> &a, std::int64_t lda, sycl::buffer<T,1> &x, std::int64_t incx, T beta, sycl::buffer<T,1> &y, std::int64_t incy)
.. container:: section


   .. rubric:: Input Parameters
      :class: sectiontitle


   queue
      The queue where the routine should be executed.


   trans
      Specifies ``op(A)``, the transposition operation applied to ``A``.


   m
      Specifies the number of rows of the matrix ``A``. The value of
      ``m`` must be at least zero.


   n
      Specifies the number of columns of the matrix ``A``. The value of
      ``n`` must be at least zero.


   alpha
      Scaling factor for the matrix-vector product.


   a
      The buffer holding the input matrix ``A``. Must have a size of at
      least ``lda``\ \*``n``. See `Matrix and Vector
      Storage <../matrix-storage.html>`__ for
      more details.


   lda
      The leading dimension of matrix ``A``. It must be at least m, and
      positive.


   x
      Buffer holding input vector ``x``. The length ``len`` of vector
      ``x`` is ``n`` if ``A`` is not transposed, and ``m`` if ``A`` is
      transposed. The buffer must be of size at least (1 + (``len`` -
      1)*abs(``incx``)). See `Matrix and Vector
      Storage <../matrix-storage.html>`__ for
      more details.


   incx
      The stride of vector ``x``.


   beta
      The scaling factor for vector ``y``.


   y
      Buffer holding input/output vector ``y``. The length ``len`` of
      vector ``y`` is ``m``, if ``A`` is not transposed, and ``n`` if
      ``A`` is transposed. The buffer must be of size at least (1 +
      (``len`` - 1)*abs(``incy``)) where ``len`` is this length. See
      `Matrix and Vector
      Storage <../matrix-storage.html>`__ for
      more details.


   incy
      The stride of vector ``y``.


.. container:: section


   .. rubric:: Output Parameters
      :class: sectiontitle


   y
      The buffer holding updated vector ``y``.


gemv (USM Version)
------------------

.. container::

   .. container:: section


      .. rubric:: Syntax
         :class: sectiontitle


      .. container:: dlsyntaxpara


         .. cpp:function::  sycl::event oneapi::mkl::blas::gemv(sycl::queue &queue, transpose trans, std::int64_t m, std::int64_t n, T alpha, const T *a, std::int64_t lda, const T *x, std::int64_t incx, T beta, T *y, std::int64_t incy, const sycl::vector_class<sycl::event> &dependencies = {})
   .. container:: section


      .. rubric:: Input Parameters
         :class: sectiontitle


      queue
         The queue where the routine should be executed.


      trans
         Specifies ``op(A)``, the transposition operation applied to
         ``A``. See
         :ref:`onemkl_datatypes` for
         more details.



      m
         Specifies the number of rows of the matrix ``A``. The value of
         ``m`` must be at least zero.


      n
         Specifies the number of columns of the matrix ``A``. The value
         of ``n`` must be at least zero.


      alpha
         Scaling factor for the matrix-vector product.


      a
         The pointer to the input matrix ``A``. Must have a size of at
         least ``lda``\ \*``n``. See `Matrix and Vector
         Storage <../matrix-storage.html>`__ for
         more details.


      lda
         The leading dimension of matrix ``A``. It must be at least m,
         and positive.


      x
         Pointer to the input vector ``x``. The length ``len`` of vector
         ``x`` is ``n`` if ``A`` is not transposed, and ``m`` if ``A``
         is transposed. The array holding vector ``x`` must be of size
         at least (1 + (``len`` - 1)*abs(``incx``)). See `Matrix and
         Vector
         Storage <../matrix-storage.html>`__ for
         more details.


      incx
         The stride of vector ``x``.


      beta
         The scaling factor for vector ``y``.


      y
         Pointer to input/output vector ``y``. The length ``len`` of
         vector ``y`` is ``m``, if ``A`` is not transposed, and ``n`` if
         ``A`` is transposed. The array holding input/output vector
         ``y`` must be of size at least (1 + (``len`` -
         1)*abs(``incy``)) where ``len`` is this length. See `Matrix and
         Vector
         Storage <../matrix-storage.html>`__ for
         more details.


      incy
         The stride of vector ``y``.


      dependencies
         List of events to wait for before starting computation, if any.
         If omitted, defaults to no dependencies.


   .. container:: section


      .. rubric:: Output Parameters
         :class: sectiontitle


      y
         The pointer to updated vector ``y``.


   .. container:: section


      .. rubric:: Return Values
         :class: sectiontitle


      Output event to wait on to ensure computation is complete.


.. container:: familylinks


   .. container:: parentlink


      **Parent topic:** :ref:`blas-level-2-routines`

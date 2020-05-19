.. _onemkl_blas_syr2:

syr2
====


.. container::


   Computes a rank-2 update of a symmetric matrix.



      ``syr2`` supports the following precisions.


      .. list-table:: 
         :header-rows: 1

         * -  T 
         * -  ``float`` 
         * -  ``double`` 


.. container:: section


   .. rubric:: Description
      :class: sectiontitle


   The ``syr2`` routines compute two scalar-vector-vector product add them
   and add the result to a matrix, with a symmetric matrix. The
   operation is defined as


      A <- alpha*x*y :sup:`T` + alpha*y*x :sup:`T` + A


   where:


   ``alpha`` is a scalar,


   ``A`` is an ``n``-by-``n`` symmetric matrix,


   ``x`` and ``y`` are vectors of length ``n``.


syr2 (Buffer Version)
---------------------

.. container::

   .. container:: section


      .. rubric:: Syntax
         :class: sectiontitle


      .. cpp:function::  void onemkl::blas::syr2(sycl::queue &queue, uplo upper_lower, std::int64_t n, T alpha, sycl::buffer<T,1> &x, std::int64_t incx, sycl::buffer<T,1> &y, std::int64_t incy, sycl::buffer<T,1> &a, std::int64_t lda)

.. container:: section


   .. rubric:: Input Parameters
      :class: sectiontitle


   queue
      The queue where the routine should be executed.


   upper_lower
      Specifies whether ``A`` is upper or lower triangular. See :ref:`onemkl_datatypes` for more details.


   n
      Number of columns of ``A``. Must be at least zero.


   alpha
      Scaling factor for the matrix-vector product.


   x
      Buffer holding input vector ``x``. The buffer must be of size at
      least (1 + (``n`` - 1)*abs(``incx``)). See `Matrix and Vector
      Storage <../matrix-storage.html>`__ for
      more details.


   incx
      Stride of vector ``x``.


   y
      Buffer holding input/output vector ``y``. The buffer must be of
      size at least (1 + (``n`` - 1)*abs(``incy``)). See `Matrix and
      Vector Storage <../matrix-storage.html>`__
      for more details.


   incy
      Stride of vector ``y``.


   a
      Buffer holding input matrix ``A``. Must have size at least
      ``lda``\ \*\ ``n``. See `Matrix and Vector
      Storage <../matrix-storage.html>`__ for
      more details.


   lda
      Leading dimension of matrix ``A``. Must be at least ``n``, and
      positive.


.. container:: section


   .. rubric:: Output Parameters
      :class: sectiontitle


   a
      Buffer holding the updated upper triangular part of the symmetric
      matrix ``A`` if ``upper_lower =upper``, or the updated lower
      triangular part of the symmetric matrix ``A`` if
      ``upper_lower =lower``.


syr2 (USM Version)
------------------

.. container::

   .. container:: section


      .. rubric:: Syntax
         :class: sectiontitle


      .. container:: dlsyntaxpara


         .. cpp:function::  sycl::event onemkl::blas::syr2(sycl::queue &queue, uplo upper_lower, std::int64_t n, T alpha, const T *x, std::int64_t incx, const T *y, std::int64_t incy, T *a, std::int64_t lda, const sycl::vector_class<sycl::event> &dependencies = {})
   .. container:: section


      .. rubric:: Input Parameters
         :class: sectiontitle


      queue
         The queue where the routine should be executed.


      upper_lower
         Specifies whether ``A`` is upper or lower triangular. See :ref:`onemkl_datatypes` for more details.


      n
         Number of columns of ``A``. Must be at least zero.


      alpha
         Scaling factor for the matrix-vector product.


      x
         Pointer to input vector ``x``. The array holding input vector
         ``x`` must be of size at least (1 + (``n`` - 1)*abs(``incx``)).
         See `Matrix and Vector
         Storage <../matrix-storage.html>`__ for
         more details.


      incx
         Stride of vector ``x``.


      y
         Pointer to input/output vector ``y``. The array holding
         input/output vector ``y`` must be of size at least (1 + (``n``
         - 1)*abs(``incy``)). See `Matrix and Vector
         Storage <../matrix-storage.html>`__ for
         more details.


      incy
         Stride of vector ``y``.


      a
         Pointer to input matrix ``A``. The array holding input matrix
         ``A`` must have size at least ``lda``\ \*\ ``n``. See `Matrix
         and Vector
         Storage <../matrix-storage.html>`__ for
         more details.


      lda
         Leading dimension of matrix ``A``. Must be at least ``n``, and
         positive.


      dependencies
         List of events to wait for before starting computation, if any.
         If omitted, defaults to no dependencies.


   .. container:: section


      .. rubric:: Output Parameters
         :class: sectiontitle


      a
         Pointer to the updated upper triangular part of the symmetric
         matrix ``A`` if ``upper_lower =upper``, or the updated lower
         triangular part of the symmetric matrix ``A`` if
         ``upper_lower =lower``.


   .. container:: section


      .. rubric:: Return Values
         :class: sectiontitle


      Output event to wait on to ensure computation is complete.


.. container:: familylinks


   .. container:: parentlink


      **Parent topic:** :ref:`blas-level-2-routines`

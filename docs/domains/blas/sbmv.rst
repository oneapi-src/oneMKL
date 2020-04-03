.. _sbmv:

sbmv
====


.. container::


   Computes a matrix-vector product with a symmetric band matrix.


   .. container:: section
      :name: GUID-BEDE7E82-C168-498D-BF65-085BBCEF9A27


      .. rubric:: Syntax
         :name: syntax
         :class: sectiontitle


      .. cpp:function::  void sbmv(queue &exec_queue, uplo upper_lower,      std::int64_t n, std::int64_t k, T alpha, buffer<T,1> &a,      std::int64_t lda, buffer<T,1> &x, std::int64_t incx, T beta,      buffer<T,1> &y, std::int64_t incy)

      ``sbmv`` supports the following precisions.


      .. list-table:: 
         :header-rows: 1

         * -  T 
         * -  ``float`` 
         * -  ``double`` 




.. container:: section
   :name: GUID-4F227157-1724-4D1F-AFAB-58C722CA8D08


   .. rubric:: Description
      :name: description
      :class: sectiontitle


   The sbmv routines compute a scalar-matrix-vector product and add the
   result to a scalar-vector product, with a symmetric band matrix. The
   operation is defined as


  


      y <- alpha*A*x + beta*y


   where:


   ``alpha`` and ``beta`` are scalars,


   ``A`` is an ``n``-by-``n`` symmetric matrix with ``k``
   super-diagonals,


   ``x`` and ``y`` are vectors of length ``n``.


.. container:: section
   :name: GUID-E1436726-01FE-4206-871E-B905F59A96B4


   .. rubric:: Input Parameters
      :name: input-parameters
      :class: sectiontitle


   exec_queue
      The queue where the routine should be executed.


   upper_lower
      Specifies whether ``A`` is upper or lower triangular. See
      :ref:`onemkl_datatypes` for more
      details.


   n
      Number of rows and columns of ``A``. Must be at least zero.


   k
      Number of super-diagonals of the matrix ``A``. Must be at least
      zero.


   alpha
      Scaling factor for the matrix-vector product.


   a
      Buffer holding input matrix ``A``. Must have size at least
      ``lda``\ \*\ ``n``. See `Matrix and Vector
      Storage <../matrix-storage.html>`__ for
      more details.


   lda
      Leading dimension of matrix ``A``. Must be at least (``k`` + 1),
      and positive.


   x
      Buffer holding input vector ``x``. The buffer must be of size at
      least (1 + (``n`` - 1)*abs(``incx``)). See `Matrix and Vector
      Storage <../matrix-storage.html>`__ for
      more details.


   incx
      Stride of vector ``x``.


   beta
      Scaling factor for vector ``y``.


   y
      Buffer holding input/output vector ``y``. The buffer must be of
      size at least (1 + (``n`` - 1)*abs(``incy``)). See `Matrix and
      Vector Storage <../matrix-storage.html>`__
      for more details.


   incy
      Stride of vector ``y``.


.. container:: section
   :name: GUID-ABBEA4DA-7B4C-489A-8063-BDC09FBB1ADD


   .. rubric:: Output Parameters
      :name: output-parameters
      :class: sectiontitle


   y
      Buffer holding the updated vector ``y``.


.. container:: familylinks


   .. container:: parentlink


      **Parent topic:** :ref:`blas-level-2-routines`
      


.. container::

